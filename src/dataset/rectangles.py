"""
Provides the RectanglesDataset.
"""

import numpy as np
from odl import uniform_discr
from skimage.draw import polygon
from skimage.transform import downscale_local_mean
from .dataset import GroundTruthDataset

def _rect_coords(shape, a1, a2, x, y, rot):
    # convert [-1., 1.]^2 coordinates to [0., shape[0]] x [0., shape[1]]
    x, y = 0.5 * shape[0] * (x + 1.), 0.5 * shape[1] * (y + 1.)
    a1, a2 = 0.5 * shape[0] * a1, 0.5 * shape[1] * a2
    # rotate side vector [a1, a2] to rot_mat @ [a1, a2]
    rot_mat = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    coord_diffs = np.array([
            rot_mat @ [a1, a2],
            rot_mat @ [a1, -a2],
            rot_mat @ [-a1, -a2],
            rot_mat @ [-a1, a2]])
    coords = np.array([x, y])[None, :] + coord_diffs
    return coords


def _rect_phantom(shape, rects, smooth_sr_fact=2, blend_mode='add'):
    sr_shape = (shape[0] * smooth_sr_fact, shape[1] * smooth_sr_fact)
    img = np.zeros(sr_shape, dtype='float32')
    for rect in rects:
        v, a1, a2, x, y, rot = rect
        coords = _rect_coords(sr_shape, a1, a2, x, y, rot)
        p_rr, p_cc = polygon(coords[:, 1], coords[:, 0], shape=sr_shape)
        if blend_mode == 'add':
            img[p_rr, p_cc] += v
        elif blend_mode == 'set':
            img[p_rr, p_cc] = v
    if smooth_sr_fact != 1:
        img = downscale_local_mean(img, (smooth_sr_fact, smooth_sr_fact))
    return img


class RectanglesDataset(GroundTruthDataset):
    """
    Dataset with images of multiple random rectangles.
    The images are normalized to have a value range of ``[0., 1.]`` with a
    background value of ``0.``. Each image has shape ``(image_size, image_size)``.
    """
    def __init__(self,
            image_size=128, min_pt=None, max_pt=None, num_rects=3,
            num_angle_modes=1, angle_modes_sigma=0.05,
            train_len=32000, validation_len=3200, test_len=3200,
            fixed_seeds=True, smooth_sr_fact=2):
        """
        image_size : int, optional
            Image size (side length). The default is ``128``.
        num_rects : int, optional
            Number of rectangles (overlayed additively).
            The default is ``3``.
        num_angle_modes : int, optional
            Number of Gaussian modes from which angles can be sampled.
            For each rectangle, one of the modes is selected (with equal
            probability for each mode).
            The default is ``1``.
        angle_modes_sigma : float, optional
            Scale of each Gaussian mode.
            The default is ``0.05``.
        train_len : int, optional
            Number of images in the training fold.
            The default is ``32000``.
        validation_len : int, optional
            Number of images in the validation fold.
            The default is ``3200``.
        test_len : int, optional
            Number of images in the test fold.
            The default is ``3200``.
        fixed_seed : bool or dict, optional
            If ``True``, use the fixed random seeds ``{'train': 1, 'validation': 2, 'test': 3}``.
            A custom dict can also be specified.
            The default is ``True``.
        smooth_sr_fact : int, optional
            Super-resolution factor for the image generation.
            A higher factor leads to smoother edges (if not aligned with the
            pixel grid).
            The default is ``2``.
        """

        self.shape = (image_size, image_size)
        # defining discretization space ODL
        if min_pt is None:
            min_pt = [-self.shape[0]/2, -self.shape[1]/2]
        if max_pt is None:
            max_pt = [self.shape[0]/2, self.shape[1]/2]
        space = uniform_discr(min_pt, max_pt, self.shape, dtype=np.float32)
        self.num_rects = num_rects
        self.num_angle_modes = num_angle_modes or num_rects
        self.angle_modes_sigma = angle_modes_sigma
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
        self.smooth_sr_fact = smooth_sr_fact
        super().__init__(space=space)

        self.rects_data = {'train': [], 'validation': [], 'test': []}
        for fold in self.rects_data.keys():
            self._create_rects_data(fold)

    def _create_rects_data(self, fold):
        rng = np.random.RandomState(self.fixed_seeds.get(fold, None))
        for _ in range(self.get_len(fold)):
            v = rng.uniform(0.5, 1.0, (self.num_rects,))
            a1 = rng.uniform(0.1, .8, (self.num_rects,))
            a2 = rng.uniform(0.1, .8, (self.num_rects,))
            x = rng.uniform(-.75, .75, (self.num_rects,))
            y = rng.uniform(-.75, .75, (self.num_rects,))
            angle_modes = rng.uniform(0., np.pi, (self.num_angle_modes,))
            angle_modes_per_rect = angle_modes[rng.randint(
                    0, self.num_angle_modes, (self.num_rects,))]
            rot = rng.normal(angle_modes_per_rect, self.angle_modes_sigma)
            rot = np.mod(rot, np.pi)
            rects = np.stack((v, a1, a2, x, y, rot), axis=1)
            self.rects_data[fold].append(rects)

    def _generate_item(self, fold, idx):
        image = _rect_phantom(self.shape, self.rects_data[fold][idx], self.smooth_sr_fact)
        # normalize the foreground (all non-zero pixels) to [0., 1.]
        image[np.array(image) != 0.] -= np.min(image)
        image /= np.max(image)
        return image

    def generator(self, fold='train'):
        for idx in range(self.get_len(fold)):
            yield self.space.element(self._generate_item(fold, idx))
