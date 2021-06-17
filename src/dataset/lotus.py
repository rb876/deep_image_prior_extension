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

NUM_ANGLES_FULL = 360
NUM_DET_PIXELS_FULL = 2221
NUM_ANGLES_FULL_UNCROPPED = 366
NUM_DET_PIXELS_FULL_UNCROPPED = 2240

SCALE_TO_FBP_MAX_1_FACTOR = 21.4543004236
"""
Factor by which image and sinogram can be multiplied to reach a range of
approximately ``[0, 1]``.
It is computed as ``1./np.max(fbp_reco)``, where `fbp_reco` is the Ram-Lak
filtered back-projection computed as in the script ``examples/lotus_fbp.py``.
"""

def get_ray_trafo_matrix(filename, normalize=False):
    """
    Return the matrix `A` implementing the ray transform.

    Parameters
    ----------
    filename : str
        Filename (including path) of the Matlab file containing `A`, i.e. one
        of the files ``LotusData128.mat`` and ``LotusData256.mat`` (which in
        https://arxiv.org/abs/1609.07299 are called ``Data128.mat`` and
        ``Data256.mat``, respectively).
    normalize : bool, optional
        Whether to divide by the scalar `normA` included in the Matlab file.
        The default is `False`.

    Returns
    -------
    A : array
        Numpy array of shape ``(120 * 429, size * size)``,
        where ``size=128`` or ``size=256`` depending on `filename`.
        It is converted to ``dtype='float32'``.
    """
    matrix = loadmat(filename, variable_names=['A'])['A'].astype('float32')
    if normalize:
        matrix /= get_norm_ray_trafo(filename)
    return matrix

def get_ground_truth(filename, scale_to_fbp_max_1=False):
    """
    Return the virtual ground truth `recon`.

    Parameters
    ----------
    filename : str
        Filename (including path) of a Matlab file containing virtual ground
        truth.
    scale_to_fbp_max_1 : bool, optional
        Whether to scale by `SCALE_TO_FBP_MAX_1_FACTOR`.
        The default is `False`.

    Returns
    -------
    recon : array
        Numpy array with a shape depending on `filename`.
        It is the transposed of the Matlab array and converted to
        ``dtype='float32'``.
    """
    ground_truth = loadmat(
            filename, variable_names=['recon'])['recon'].astype('float32').T
    if scale_to_fbp_max_1:
        ground_truth *= SCALE_TO_FBP_MAX_1_FACTOR
    return ground_truth

def get_domain128():
    """
    Return an :class:`odl.DiscretizedSpace` describing the image domain
    corresponding to the ray transform `A` from ``LotusData128.mat``.
    """
    cell_side = 0.627  # pixel size in mm
    size = 128
    domain = odl.uniform_discr(
            [-0.5*size*cell_side, -0.5*size*cell_side],
            [0.5*size*cell_side, 0.5*size*cell_side],
            [size, size],
            dtype='float32')
    return domain


def get_proj_space128(det_extent=None, det_shift=0.):
    """
    Return an :class:`odl.DiscretizedSpace` describing the projection space
    corresponding to the ray transform `A` from ``LotusData128.mat``.

    Note that the exact geometry of `A` is not known (to me), but the returned
    space seems to match approximately.
    """
    angle_step = 2.*np.pi / NUM_ANGLES
    if det_extent is None:
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
            [-0.5*angle_step, -0.5*det_extent + det_shift],
            [2.*np.pi - 0.5*angle_step, 0.5*det_extent + det_shift],
            [NUM_ANGLES, NUM_DET_PIXELS128],
            dtype='float32')
    return domain


def get_sinogram_full(filename, crop=True):
    """
    Return the full measured sinogram (of the 2D slice).

    Parameters
    ----------
    filename : str
        Filepath of the Matlab file ``sinogram.mat``, containing `sinogram`.
        In https://arxiv.org/abs/1609.07299 the file is called
        ``FullSizeSinograms.mat`` and the variable is named `sinogram360`,
        having size ``2221 x 360``, which according to ``Lotus_FBP.m``
        corresponds to the first rows and columns of `sinogram` in
        ``sinogram.mat`` (which has size ``2240 x 366``).
    crop : bool, optional
        Whether to return a cropped array of shape ``(360, 2221)`` (instead of
        shape ``(366, 2240)``), analogously to ``Lotus_FBP.m``.
        The default is `True`.

    Returns
    -------
    sinogram : array
        Numpy array of shape ``(360, 2221)``, or ``(366, 2240)`` if
        ``crop=False``.
        It is the transposed of the Matlab array.
    """
    sinogram = loadmat(filename, variable_names=['sinogram'])['sinogram'].T
    if crop:
        sinogram = sinogram[:NUM_ANGLES_FULL, :NUM_DET_PIXELS_FULL]
    return sinogram


def get_sinogram(filename, normalize=False, scale_to_fbp_max_1=False):
    """
    Return the down-sampled measured sinogram `m`.

    Parameters
    ----------
    filename : str
        Filename (including path) of the Matlab file containing `m`, i.e. one
        of the files ``LotusData128.mat`` and ``LotusData256.mat`` (which in
        https://arxiv.org/abs/1609.07299 are called ``Data128.mat`` and
        ``Data256.mat``, respectively).
    normalize : bool, optional
        Whether to divide by the scalar `normA` included in the Matlab file.
        The default is `False`.
    scale_to_fbp_max_1 : bool, optional
        Whether to scale by `SCALE_TO_FBP_MAX_1_FACTOR`.
        The default is `False`.

    Returns
    -------
    m : array
        Numpy array of shape ``(120, 429)``.
        It is the transposed of the Matlab array.
    """
    sinogram = loadmat(filename, variable_names=['m'])['m'].T
    factor = 1.
    if normalize:
        factor /= get_norm_ray_trafo(filename)
    if scale_to_fbp_max_1:
        factor *= SCALE_TO_FBP_MAX_1_FACTOR
    sinogram *= factor
    return sinogram


def get_norm_ray_trafo(filename, upper_bound=False):
    """
    Return (an upper bound to) the norm of the ray transform.

    Parameters
    ----------
    filename : str
        Filename (including path) of the Matlab file containing `normA` and
        `normA_est`, i.e. one of the files ``LotusData128.mat`` and
        ``LotusData256.mat`` (which in https://arxiv.org/abs/1609.07299 are
        called ``Data128.mat`` and ``Data256.mat``, respectively).
    upper_bound : bool, optional
        Whether to return the upper bound `normA_est` instead of `normA`.
        Note that the numbers are numerically almost indistinguishable (with
        absolute differences smaller than 2e-12; in fact for
        ``LotusData128.mat`` the "upper bound" `normA_est` is even slightly
        smaller than `normA`).
        The default is `False`.

    Returns
    -------
    norm : float
        Norm of the ray transform, or upper bound to it.
    """
    var_name = 'normA_est' if upper_bound else 'normA'
    norm = np.squeeze(loadmat(filename, variable_names=[var_name])[var_name])
    return norm
