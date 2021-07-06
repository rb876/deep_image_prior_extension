"""
Provides the EllipsesDataset.
"""
import odl
import numpy as np
from itertools import repeat
from odl import uniform_discr
from functools import partial
from odl.phantom import ellipsoid_phantom
from .dataset import Dataset
from util.transforms_np import deform_random_grid

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
            mu_water):

            # apply forward operator
            obs = np.asarray(self.ray_trafo(ground_truth))
            # noise model
            obs *= mu_water
            obs *= -1
            np.exp(obs, out=obs)
            obs *= photons_per_pixel
            noisy_obs = random_gen.poisson(obs)
            noisy_obs = np.maximum(1, noisy_obs) / photons_per_pixel
            post_log_noisy_obs = np.log(noisy_obs) * (-1. / mu_water)
            return post_log_noisy_obs

        if random_gen is None:
            random_gen = np.random.default_rng()

        if self.noise_type == 'white':
            forward_func = partial(white_forward_func, random_gen=random_gen,
                                  stddev=self.specs_kwargs['stddev'])
        elif self.noise_type == 'poisson':
            forward_func = partial(poisson_forward_func, random_gen=random_gen,
                                  photons_per_pixel=self.specs_kwargs['photons_per_pixel'],
                                  mu_water=self.specs_kwargs['mu_water'])
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

class EllipsesDataset(GroundTruthDataset):
    """
    Dataset with images of multiple random ellipses.
    This dataset uses :meth:`odl.phantom.ellipsoid_phantom` to create
    the images. The images are normalized to have a value range of ``[0., 1.]`` with a
    background value of ``0.``.
    """
    def __init__(self, image_size=128, min_pt=None, max_pt=None,
                 train_len=32000, validation_len=3200, test_len=3200,
                 fixed_seeds=True, deform=False, deform_kwargs=None):

        self.shape = (image_size, image_size)
        # defining discretization space ODL
        if min_pt is None:
            min_pt = [-self.shape[0]/2, -self.shape[1]/2]
        if max_pt is None:
            max_pt = [self.shape[0]/2, self.shape[1]/2]
        space = uniform_discr(min_pt, max_pt, self.shape, dtype=np.float32)
        self.train_len = train_len
        self.validation_len = validation_len
        self.test_len = test_len
        if isinstance(fixed_seeds, bool):
            if fixed_seeds:
                self.fixed_seeds = {'train': 1, 'validation': 2, 'test': 3,
                                    'train_deform': 11,
                                    'validation_deform': 12,
                                    'test_deform': 13}
            else:
                self.fixed_seeds = {}
        else:
            self.fixed_seeds = fixed_seeds.copy()
        self.deform = deform
        self.deform_kwargs = deform_kwargs or {}
        super().__init__(space=space)

    def generator(self, fold='train'):
        """
        Yield random ellipse phantom images using
        :meth:`odl.phantom.ellipsoid_phantom`.
        """
        seed = self.fixed_seeds.get(fold)
        r = np.random.RandomState(seed)
        if self.deform:
            seed_deform = self.fixed_seeds.get(fold + '_deform')
            r_deform = np.random.RandomState(seed_deform)
        max_n_ellipse = 70
        ellipsoids = np.empty((max_n_ellipse, 6))
        n = self.get_len(fold=fold)
        it = repeat(None, n) if n is not None else repeat(None)
        for _ in it:
            v = (r.uniform(-0.4, 1.0, (max_n_ellipse,)))
            a1 = .2 * r.exponential(1., (max_n_ellipse,))
            a2 = .2 * r.exponential(1., (max_n_ellipse,))
            x = r.uniform(-0.9, 0.9, (max_n_ellipse,))
            y = r.uniform(-0.9, 0.9, (max_n_ellipse,))
            rot = r.uniform(0., 2 * np.pi, (max_n_ellipse,))
            n_ellipse = min(r.poisson(40), max_n_ellipse)
            v[n_ellipse:] = 0.
            ellipsoids = np.stack((v, a1, a2, x, y, rot), axis=1)
            image = ellipsoid_phantom(self.space, ellipsoids)
            # normalize the foreground (all non-zero pixels) to [0., 1.]
            image[np.array(image) != 0.] -= np.min(image)
            image /= np.max(image)
            if self.deform:
                image[:] = deform_random_grid(np.asarray(image),
                                              **self.deform_kwargs,
                                              random_gen=r_deform)
            yield image
