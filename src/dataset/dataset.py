# -*- coding: utf-8 -*-
"""
Provides the dataset base classes.
"""
from itertools import islice
from math import ceil
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
