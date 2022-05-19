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

class EllipsoidsInBallDataset(GroundTruthDataset):
    """
    Dataset with images of multiple random ellipses distributed in a ball
    (approximately; it is only checked that the end points of each axis lie in
    the ball, so in order to ensure the full ellipses are inside, choose a
    slightly smaller value for `in_ball_axis`).
    This dataset uses :meth:`odl.phantom.ellipsoid_phantom` to create
    the images. The images are normalized to have a value range of ``[0., 1.]`` with a
    background value of ``0.``.
    """
    def __init__(self, image_size=128, min_pt=None, max_pt=None, in_ball_axis=1.,
                 train_len=32000, validation_len=3200, test_len=3200,
                 fixed_seeds=True):

        self.shape = (image_size, image_size, image_size)
        # defining discretization space ODL
        if min_pt is None:
            min_pt = [-self.shape[0]/2, -self.shape[1]/2, -self.shape[2]/2]
        if max_pt is None:
            max_pt = [self.shape[0]/2, self.shape[1]/2, self.shape[2]/2]
        self.in_ball_axis = (in_ball_axis,) * 3 if np.isscalar(in_ball_axis) else in_ball_axis
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

    def random_ellipsoid_spec_in_ball(self, rng):
        in_ball = False
        while not in_ball:
            axis = rng.exponential(size=3) * 0.2 * min(self.in_ball_axis)
            center = rng.uniform(-1., 1., size=3) * self.in_ball_axis
            rotation = rng.uniform(0., 2. * np.pi, size=3)

            phi = rotation[0]
            theta = rotation[1]
            psi = rotation[2]

            cphi = np.cos(phi)
            sphi = np.sin(phi)
            ctheta = np.cos(theta)
            stheta = np.sin(theta)
            cpsi = np.cos(psi)
            spsi = np.sin(psi)

            # check that end points are in ball (not sure if angles are used correctly)
            rot_mat = np.array(
                    [[cpsi * cphi - ctheta * sphi * spsi,
                    cpsi * sphi + ctheta * cphi * spsi,
                    spsi * stheta],
                    [-spsi * cphi - ctheta * sphi * cpsi,
                    -spsi * sphi + ctheta * cphi * cpsi,
                    cpsi * stheta],
                    [stheta * sphi,
                    -stheta * cphi,
                    ctheta]]).T
            in_ball = all(np.sum((p / self.in_ball_axis)**2) < 1. for p in [
                    center + rot_mat[:, 0] * axis[0],
                    center - rot_mat[:, 0] * axis[0],
                    center + rot_mat[:, 1] * axis[1],
                    center - rot_mat[:, 1] * axis[1],
                    center + rot_mat[:, 2] * axis[2],
                    center - rot_mat[:, 2] * axis[2]])
            # if not in_ball:
            #     print('rejecting')

        v = rng.uniform(-0.4, 1.0)

        return (v, *axis, *center, *rotation)

    def generator(self, fold='train'):
        """
        Yield random ellipse phantom images using
        :meth:`odl.phantom.ellipsoid_phantom`.
        """
        seed = self.fixed_seeds.get(fold)
        r = np.random.RandomState(seed)
        max_n_ellipse = 210
        ellipsoids = np.empty((max_n_ellipse, 6))
        n = self.get_len(fold=fold)
        it = repeat(None, n) if n is not None else repeat(None)
        for _ in it:
            n_ellipse = min(r.poisson(120), max_n_ellipse)
            ellipsoids = [self.random_ellipsoid_spec_in_ball(rng=r)
                          for _ in range(n_ellipse)]
            image = ellipsoid_phantom(self.space, ellipsoids)
            # normalize the foreground (all non-zero pixels) to [0., 1.]
            image[np.array(image) != 0.] -= np.min(image)
            image /= np.max(image)
            yield image


class DiskDistributedNoiseMasksDataset(GroundTruthDataset):
    """
    Dataset with images of multiple random noise masks with centers sampled from a
    disk distribution.
    This dataset uses :meth:`np.random.rand to create
    the images. The images are normalized to have a value range of ``[0., 1.]`` with a
    background value of ``0.``.
    """
    def __init__(self, in_circle_axis=1., use_mask=False, image_size=128, min_pt=None, max_pt=None,
                 train_len=32000, validation_len=3200, test_len=3200,
                 fixed_seeds=True):

        self.in_circle_axis = in_circle_axis
        self.shape = (image_size, image_size)
        self.mask = self._create_disk_mask() if use_mask else None
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

    def _create_disk_mask(self):
        h, w = self.shape
        center = ((w-1)/2, (h-1)/2)
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        return dist_from_center <= (w * self.in_circle_axis/2)

    def generator(self, fold='train'):
        """
        Yield random ellipse phantom images with centers sampled from a disk
        distribution using :meth:`np.random.rand`.
        """
        seed = self.fixed_seeds.get(fold)
        r = np.random.RandomState(seed)
        n = self.get_len(fold=fold)
        it = repeat(None, n) if n is not None else repeat(None)
        for _ in it:
            image = r.rand(self.shape[0], self.shape[0]).astype(np.float32)
            if self.mask is not None:
                image[~self.mask] = 0
            yield image
