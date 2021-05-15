"""
Interface to the dataset "Tomographic X-ray data of a lotus root filled with
attenuating objects".
https://www.fips.fi/dataset.php#lotus
https://zenodo.org/record/1254204
https://arxiv.org/abs/1609.07299
"""
import numpy as np
import odl
from scipy.io import loadmat

SRC_RADIUS = 540
DET_RADIUS = 90
NUM_ANGLES = 120
NUM_DET_PIXELS = 429
NUM_DET_PIXELS128 = NUM_DET_PIXELS


def get_ray_trafo_matrix(filename, normalize=False):
    """
    Interface to the `filled lotus root data <https://zenodo.org/record/1254204>`_.
    """
    matrix = loadmat(filename, variable_names=['A'])['A'].astype('float32')
    if normalize:
        matrix /= np.squeeze(
                loadmat(filename, variable_names=['normA'])['normA'])
    return matrix


def get_domain128():
    cell_side = 0.627  # pixel size in mm
    size = 128
    domain = odl.uniform_discr(
            [-0.5*size*cell_side, -0.5*size*cell_side],
            [0.5*size*cell_side, 0.5*size*cell_side],
            [size, size],
            dtype='float32')
    return domain


def get_proj_space128():
    angle_step = 2.*np.pi / NUM_ANGLES
    det_extent = 132.41564427  # detector size in mm computed by:
    # det_extent = odl.tomo.cone_beam_geometry(get_domain128(),
    #         src_radius=SRC_RADIUS,
    #         det_radius=DET_RADIUS,
    #         num_angles=NUM_ANGLES,
    #         det_shape=NUM_DET_PIXELS128).detector.partition.extent
    # In https://arxiv.org/abs/1609.07299 the detector is reported to have
    # extent 120 mm, however the sinogram m in LotusData128.mat seems to
    # include some more space at both ends. Above ODL geometry with
    # autocomputed detector extent of 132.42 mm seems to match the provided
    # operator A approximately.
    domain = odl.uniform_discr(
            [-0.5*angle_step, -0.5*det_extent],
            [2.*np.pi - 0.5*angle_step, 0.5*det_extent],
            [NUM_ANGLES, NUM_DET_PIXELS128],
            dtype='float32')
    return domain


def get_sinogram_full(filename):
    sinogram = loadmat(filename, variable_names=['sinogram'])['sinogram']
    return sinogram


def get_sinogram(filename):
    sinogram = loadmat(filename, variable_names=['m'])['m'].T
    return sinogram
