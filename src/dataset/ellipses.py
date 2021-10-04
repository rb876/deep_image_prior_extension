"""
Provides the EllipsesDataset.
"""
import odl
import numpy as np
from itertools import repeat
from odl import uniform_discr
from odl.phantom import ellipsoid_phantom
from .dataset import GroundTruthDataset

class EllipsesDataset(GroundTruthDataset):
    """
    Dataset with images of multiple random ellipses.
    This dataset uses :meth:`odl.phantom.ellipsoid_phantom` to create
    the images. The images are normalized to have a value range of ``[0., 1.]`` with a
    background value of ``0.``.
    """
    def __init__(self, image_size=128, min_pt=None, max_pt=None,
                 train_len=32000, validation_len=3200, test_len=3200,
                 fixed_seeds=True):

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
                self.fixed_seeds = {'train': 1, 'validation': 2, 'test': 3}
            else:
                self.fixed_seeds = {}
        else:
            self.fixed_seeds = fixed_seeds.copy()
        super().__init__(space=space)

    def generator(self, fold='train'):
        """
        Yield random ellipse phantom images using
        :meth:`odl.phantom.ellipsoid_phantom`.
        """
        seed = self.fixed_seeds.get(fold)
        r = np.random.RandomState(seed)
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
            yield image

class DiskDistributedEllipsesDataset(GroundTruthDataset):
    """
    Dataset with images of multiple random ellipses with centers sampled from a
    disk distribution.
    This dataset uses :meth:`odl.phantom.ellipsoid_phantom` to create
    the images. The images are normalized to have a value range of ``[0., 1.]`` with a
    background value of ``0.``.
    """
    def __init__(self, diameter=1., image_size=128, min_pt=None, max_pt=None,
                 train_len=32000, validation_len=3200, test_len=3200,
                 fixed_seeds=True):

        self.diameter = diameter
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
                self.fixed_seeds = {'train': 1, 'validation': 2, 'test': 3}
            else:
                self.fixed_seeds = {}
        else:
            self.fixed_seeds = fixed_seeds.copy()
        super().__init__(space=space)

    def generator(self, fold='train'):
        """
        Yield random ellipse phantom images with centers sampled from a disk
        distribution using :meth:`odl.phantom.ellipsoid_phantom`.
        """
        seed = self.fixed_seeds.get(fold)
        r = np.random.RandomState(seed)
        max_n_ellipse = 70
        ellipsoids = np.empty((max_n_ellipse, 6))
        n = self.get_len(fold=fold)
        it = repeat(None, n) if n is not None else repeat(None)
        for _ in it:
            v = (r.uniform(-0.4, 1.0, (max_n_ellipse,)))
            a1 = 0.2 * self.diameter * r.exponential(1., (max_n_ellipse,))
            a2 = 0.2 * self.diameter * r.exponential(1., (max_n_ellipse,))
            c_r = r.triangular(0., self.diameter, self.diameter,
                               size=(max_n_ellipse,))
            c_a = r.uniform(0., 2 * np.pi, (max_n_ellipse,))
            x = np.cos(c_a) * c_r
            y = np.sin(c_a) * c_r
            rot = r.uniform(0., 2 * np.pi, (max_n_ellipse,))
            n_ellipse = min(r.poisson(40), max_n_ellipse)
            v[n_ellipse:] = 0.
            ellipsoids = np.stack((v, a1, a2, x, y, rot), axis=1)
            image = ellipsoid_phantom(self.space, ellipsoids)
            # normalize the foreground (all non-zero pixels) to [0., 1.]
            image[np.array(image) != 0.] -= np.min(image)
            image /= np.max(image)
            yield image
