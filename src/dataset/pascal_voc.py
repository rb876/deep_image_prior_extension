"""
Provides the PascalVOCDataset.
"""

import numpy as np
from odl import uniform_discr
import torch
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import Grayscale, RandomCrop, PILToTensor, Lambda, Compose
from .dataset import GroundTruthDataset


class PascalVOCDataset(GroundTruthDataset):
    """
    Dataset with randomly cropped patches from Pascal VOC2012
    (http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html)
    """
    def __init__(self,
            data_path, year='2012', shuffle=True,
            image_size=128, min_pt=None, max_pt=None,
            train_len=32000, validation_len=3200, test_len=0,
            fixed_seeds=True):
        """
        shuffle : bool, optional
            Whether to shuffle the images.
            The default is ``True``.
        image_size : int, optional
            Image size (side length). The default is ``128``.
        train_len : int, optional
            Number of images in the training fold.
            The default is ``32000``.
        validation_len : int, optional
            Number of images in the validation fold.
            The default is ``3200``.
        fixed_seed : bool or dict, optional
            If ``True``, use the fixed random seeds ``{'train': 1, 'validation': 2}``.
            A custom dict can also be specified.
            The default is ``True``.
        """

        self.shape = (image_size, image_size)
        # defining discretization space ODL
        if min_pt is None:
            min_pt = [-self.shape[0]/2, -self.shape[1]/2]
        if max_pt is None:
            max_pt = [self.shape[0]/2, self.shape[1]/2]
        space = uniform_discr(min_pt, max_pt, self.shape, dtype=np.float32)
        self.transform = Compose(
                [Grayscale(),
                 RandomCrop(
                        size=image_size, padding=True, pad_if_needed=True, padding_mode='reflect'),
                 PILToTensor(),
                 Lambda(lambda x: ((x.to(torch.float32) + torch.rand(*x.shape)) / 256).numpy()),
                ])
        self.datasets = {
            'train': VOCSegmentation(root=data_path, year=year, image_set='train'),
            'validation': VOCSegmentation(root=data_path, year=year, image_set='val')}
        self.train_len = train_len
        self.validation_len = validation_len
        assert test_len == 0, 'PascalVOCDataset does not support the test fold, must pass test_len=0'
        self.test_len = 0
        if isinstance(shuffle, bool):
            self.shuffle = {
                    'train': shuffle, 'validation': shuffle}
        else:
            self.shuffle = shuffle.copy()
        if isinstance(fixed_seeds, bool):
            if fixed_seeds:
                self.fixed_seeds = {'train': 1, 'validation': 2}
            else:
                self.fixed_seeds = {}
        else:
            self.fixed_seeds = fixed_seeds.copy()
        super().__init__(space=space)

    def _generate_item(self, fold, idx, rng):
        image = self.datasets[fold][idx][0]
        seed = rng.randint(np.iinfo(np.int64).max)
        with torch.random.fork_rng():
            torch.random.manual_seed(seed)
            image = self.transform(image)[0, :, :]
        image -= image.min()
        image /= image.max()
        return image

    def generator(self, fold='train'):
        rng = np.random.RandomState(self.fixed_seeds.get(fold, None))
        idx_list = rng.randint(len(self.datasets[fold]), size=self.get_len(fold))
        if self.shuffle[fold]:
            rng.shuffle(idx_list)
        for idx in idx_list:
            yield self.space.element(self._generate_item(fold, idx, rng))
