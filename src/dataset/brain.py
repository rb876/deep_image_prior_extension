"""
Provides the ACRINFMISOBrainDataset.
"""
import os
import json
import odl
import numpy as np
from itertools import repeat
from odl import uniform_discr
from odl.phantom import ellipsoid_phantom
from pydicom.filereader import dcmread
from scipy.ndimage import affine_transform, rotate
from .dataset import GroundTruthDataset

class ACRINFMISOBrainDataset(GroundTruthDataset):
    """
    Dataset with images of the human brain from the ACRIN-FMISO-Brain dataset.
    https://doi.org/10.7937/K9/TCIA.2018.vohlekok
    https://doi.org/10.1158%2F1078-0432.CCR-15-2529
    https://doi.org/10.1371%2Fjournal.pone.0198548
    https://doi.org/10.1007/s10278-013-9622-7

    To create the dcm file list, see ``examples/create_brain_file_list.py``.

    The images are normalized to have a values within the range ``[0., 1.]``.
    """
    def __init__(
            self, data_path=None, shuffle=True,
            zoom=1., zoom_fit=True,
            random_rotation=True, fixed_seeds=True,
            dcm_file_list_path='acrin_fmiso_brain_file_list.json',
            min_pt=None, max_pt=None):

        self.data_path = data_path

        self.shape = (501, 501)
        # defining discretization space ODL
        # a common reconstruction diameter is 240 mm
        if min_pt is None:
            min_pt = [-0.12, -0.12]
        if max_pt is None:
            max_pt = [0.12, 0.12]
        space = uniform_discr(min_pt, max_pt, self.shape, dtype=np.float32)
        with open(os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'data', 'brain', dcm_file_list_path), 'r') as f:
            self.dcm_file_list = json.load(f)

        self.train_len = len(self.dcm_file_list['train'])
        self.validation_len = len(self.dcm_file_list['validation'])
        self.test_len = len(self.dcm_file_list['test'])

        if isinstance(shuffle, bool):
            self.shuffle = {
                    'train': shuffle, 'validation': shuffle, 'test': shuffle}
        else:
            self.shuffle = shuffle.copy()

        self.zoom = zoom
        self.zoom_fit = zoom_fit

        if isinstance(random_rotation, bool):
            self.random_rotation = {
                    'train': random_rotation, 'validation': random_rotation,
                    'test': random_rotation}
        else:
            self.random_rotation = random_rotation.copy()

        if isinstance(fixed_seeds, bool):
            if fixed_seeds:
                self.fixed_seeds = {'train': 1, 'validation': 2, 'test': 3}
            else:
                self.fixed_seeds = {}
        else:
            self.fixed_seeds = fixed_seeds.copy()

        super().__init__(space=space)

    def generator(self, fold='train'):
        seed = self.fixed_seeds.get(fold)
        r = np.random.default_rng(seed)
        dcm_files = self.dcm_file_list[fold].copy()
        if self.shuffle[fold]:
            r.shuffle(dcm_files)
        for dcm_file in dcm_files:
            dcm_dataset = dcmread(os.path.join(self.data_path, dcm_file))
            image = dcm_dataset.pixel_array.astype(np.float32).T

            # add noise to get continuous values from discrete ones
            image += r.uniform(0., 1., size=image.shape)

            # no need to rescale by dicom meta info (if present) because we are
            # going to normalize to range [0., 1.] anyways
            # image *= dcm_dataset.get('RescaleSlope', 1.)
            # image += dcm_dataset.get('RescaleIntercept', 0.)

            # normalize to [0., 1.]
            image -= np.min(image)
            image /= np.max(image)

            # apply zoom and rotation, combined if both are requested
            affine_mat = np.eye(2)
            zoom_factor = float(self.zoom)
            if self.zoom_fit:
                zoom_factor *= min(self.shape[0] / image.shape[0],
                                   self.shape[1] / image.shape[1])
            affine_mat /= zoom_factor
            if self.random_rotation[fold]:
                theta = r.uniform(0., 2*np.pi)
                rot_mat = np.array([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]])
                affine_mat = rot_mat @ affine_mat
            if np.any(affine_mat != np.eye(2)):
                in_center = np.array([(image.shape[0] - 1) / 2,
                                    (image.shape[1] - 1) / 2])
                out_center = affine_mat @ in_center
                offset = in_center - out_center
                if affine_mat[0, 1] == 0. and affine_mat[1, 0] == 0.:
                    # inform affine_transform that the matrix is diagonal
                    affine_mat = (affine_mat[0, 0], affine_mat[1, 1])
                image = affine_transform(image, affine_mat, offset=offset)

            # crop to central part if image is too large, zero-pad if too small
            pad_width0 = None
            if image.shape[0] > self.shape[0]:
                i0 = (image.shape[0]-self.shape[0])//2
                image = image[i0:i0+self.shape[0], :]
            elif image.shape[0] < self.shape[0]:
                before0 = (self.shape[0] - image.shape[0]) // 2
                pad_width0 = (before0, self.shape[0] - image.shape[0] - before0)
            pad_width1 = None
            if image.shape[1] > self.shape[1]:
                j0 = (image.shape[1]-self.shape[1])//2
                image = image[:, j0:j0+self.shape[1]]
            elif image.shape[1] < self.shape[1]:
                before1 = (self.shape[1] - image.shape[1]) // 2
                pad_width1 = (before1, self.shape[1] - image.shape[1] - before1)
            if pad_width0 is not None or pad_width1 is not None:
                image = np.pad(image, pad_width=(
                        pad_width0 if pad_width0 is not None else (0, 0),
                        pad_width1 if pad_width1 is not None else (0, 0)))

            yield image
