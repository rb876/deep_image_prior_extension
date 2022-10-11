# -*- coding: utf-8 -*-
"""
Provides the dataset base classes.
"""
from functools import partial
import numpy as np

class Dataset():
    """
    Dataset base class.
    """

    def __init__(self, space=None, name=''):
        self.name = name or self.__class__.__name__
        self.space = space

    def generator(self, fold='train'):
        """
        Yield data.
        """
        raise NotImplementedError

    def get_train_generator(self):
        return self.generator(fold='train')

    def get_validation_generator(self):
        return self.generator(fold='validation')

    def get_test_generator(self):
        return self.generator(fold='test')

    def get_len(self, fold='train'):
        """
        Return the number of elements the generator will yield.
        """
        if fold == 'train':
            return self.get_train_len()
        elif fold == 'validation':
            return self.get_validation_len()
        elif fold == 'test':
            return self.get_test_len()
        raise ValueError("dataset fold must be 'train', "
                         "'validation' or 'test', not '{}'".format(fold))

    def get_train_len(self):
        """
        Return the number of samples the train generator will yield.
        """
        try:
            return self.train_len
        except AttributeError:
            raise NotImplementedError

    def get_validation_len(self):
        """
        Return the number of samples the validation generator will yield.
        """
        try:
            return self.validation_len
        except AttributeError:
            raise NotImplementedError

    def get_test_len(self):
        """
        Return the number of samples the test generator will yield.
        """
        try:
            return self.test_len
        except AttributeError:
            raise NotImplementedError

    def get_shape(self):
        """
        Return the shape of each element.
        """
        try:
            return self.shape
        except AttributeError:
            if self.space is not None:
                if self.get_num_elements_per_sample() == 1:
                    return self.space.shape
                else:
                    return tuple(s.shape for s in self.space)
            raise NotImplementedError

    def get_num_elements_per_sample(self):
        """
        Return number of elements per sample.
        """
        try:
            return self.num_elements_per_sample
        except AttributeError:
            if self.space is not None:
                return len(self.space) if isinstance(self.space, tuple) else 1
            raise NotImplementedError

    def create_torch_dataset(self, fold='train', reshape=None):
        """
        Create a torch dataset wrapper for one fold of this dataset.
        """
        from torch.utils.data import Dataset as TorchDataset
        import torch
        class GeneratorTorchDataset(TorchDataset):
            def __init__(self, dataset, fold, reshape=None):
                self.fold = fold
                self.dataset = dataset
                self.generator = self.dataset.generator(self.fold)
                self.length = self.dataset.get_len(self.fold)
                self.reshape = reshape or (
                    (None,) * dataset.get_num_elements_per_sample())

            def __len__(self):
                return self.length

            def __getitem__(self, idx):
                try:
                    arrays = next(self.generator)
                except StopIteration:
                    self.generator = self.dataset.generator(self.fold)
                    arrays = next(self.generator)
                mult_elem = isinstance(arrays, tuple)
                if not mult_elem:
                    arrays = (arrays,)
                tensors = []
                for arr, s in zip(arrays, self.reshape):
                    t = torch.from_numpy(np.asarray(arr))
                    if s is not None:
                        t = t.view(*s)
                    tensors.append(t)
                return tuple(tensors) if mult_elem else tensors[0]

        dataset = GeneratorTorchDataset(self, fold, reshape=reshape)
        return dataset

class ObservationGroundTruthPairDataset(Dataset):
    """
    Dataset of pairs generated from a ground truth generator by applying a
    forward operator and noise.
    """
    def __init__(self, ground_truth_gen, ray_trafo, pinv_ray_trafo,
                 train_len=None, validation_len=None, test_len=None,
                 domain=None, proj_space=None, noise_type=None,
                 specs_kwargs=None, noise_seeds=None):

        self.ground_truth_gen = ground_truth_gen
        self.ray_trafo = ray_trafo
        self.pinv_ray_trafo = pinv_ray_trafo
        if train_len is not None:
            self.train_len = train_len
        if validation_len is not None:
            self.validation_len = validation_len
        if test_len is not None:
            self.test_len = test_len
        if domain is None:
            domain = self.ray_trafo.domain
        self.specs_kwargs = specs_kwargs
        self.noise_type = noise_type
        self.noise_seeds = noise_seeds or {}
        if proj_space is None:
            proj_space = self.ray_trafo.range
        super().__init__(space=(proj_space, domain))
        self.shape = (self.space[0].shape, self.space[1].shape)
        self.num_elements_per_sample = 3

    def ground_truth_to_obs(self, ground_truth, random_gen=None):

        def white_forward_func(ground_truth, random_gen, stddev):

            # apply forward operator
            obs = np.asarray(self.ray_trafo(ground_truth))
            # noise model
            relative_stddev = np.mean(np.abs(obs))
            noisy_obs = obs + random_gen.normal(size=self.space[0].shape) \
                * relative_stddev * stddev
            return noisy_obs

        def poisson_forward_func(ground_truth, random_gen,
            photons_per_pixel,
            mu_max):

            # apply forward operator
            obs = np.asarray(self.ray_trafo(ground_truth))
            # noise model
            obs *= mu_max
            obs *= -1
            np.exp(obs, out=obs)
            obs *= photons_per_pixel
            noisy_obs = random_gen.poisson(obs)
            noisy_obs = np.maximum(1, noisy_obs) / photons_per_pixel
            post_log_noisy_obs = np.log(noisy_obs) * (-1. / mu_max)
            return post_log_noisy_obs

        if random_gen is None:
            random_gen = np.random.default_rng()

        if self.noise_type == 'white':
            forward_func = partial(white_forward_func, random_gen=random_gen,
                                  stddev=self.specs_kwargs['stddev'])
        elif self.noise_type == 'poisson':
            forward_func = partial(poisson_forward_func, random_gen=random_gen,
                                  photons_per_pixel=self.specs_kwargs['photons_per_pixel'],
                                  mu_max=self.specs_kwargs['mu_max'])
        else:
            raise NotImplementedError

        return forward_func(ground_truth)

    def generator(self, fold='train'):
        # construct noise
        random_gen = np.random.default_rng(self.noise_seeds.get(fold))
        gt_gen_instance = self.ground_truth_gen(fold=fold)
        for ground_truth in gt_gen_instance:
            noisy_obs = self.ground_truth_to_obs(ground_truth, random_gen)
            fbp_reco = self.pinv_ray_trafo(noisy_obs)
            yield (noisy_obs, fbp_reco, ground_truth)

class GroundTruthDataset(Dataset):
    """
    Ground truth dataset base class.
    """
    def __init__(self, space=None):

        self.num_elements_per_sample = 1
        super().__init__(space=space)

    def create_pair_dataset(self, ray_trafo, pinv_ray_trafo, domain=None, proj_space=None, noise_type=None, specs_kwargs=None, noise_seeds=None):

        try:
            train_len = self.get_train_len()
        except NotImplementedError:
            train_len = None
        try:
            validation_len = self.get_validation_len()
        except NotImplementedError:
            validation_len = None
        try:
            test_len = self.get_test_len()
        except NotImplementedError:
            test_len = None
        dataset = ObservationGroundTruthPairDataset(
                self.generator, ray_trafo, pinv_ray_trafo,
                train_len=train_len, validation_len=validation_len,
                test_len=test_len, domain=domain, proj_space=proj_space,
                noise_type=noise_type, specs_kwargs=specs_kwargs,
                noise_seeds=noise_seeds)
        return dataset
