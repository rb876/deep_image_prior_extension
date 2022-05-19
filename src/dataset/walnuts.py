"""
Interface to the dataset "A Cone-Beam X-Ray CT Data Collection Designed for
Machine Learning".
https://doi.org/10.1038/s41597-019-0235-y
https://zenodo.org/record/2686726
https://arxiv.org/abs/1905.04787

Based on https://github.com/cicwi/WalnutReconstructionCodes .

Features restriction to a z-slice of the volume.
"""
import numpy as np
import astra
import os
import imageio
from math import ceil
import torch
import scipy.sparse
import scipy.io
import scipy.interpolate
from tqdm import tqdm

VOXEL_PER_MM = 10
DEFAULT_ANGULAR_SUB_SAMPLING = 10
DEFAULT_PROJ_COL_SUB_SAMPLING = 1
DEFAULT_PROJ_ROW_SUB_SAMPLING = 1
DEFAULT_VOL_DOWN_SAMPLING = (1, 1, 1)
PROJS_NAME = 'scan_{:06}.tif'
DARK_NAME = 'di000000.tif'
FLAT_NAME = ['io000000.tif', 'io000001.tif']
VECS_NAME = 'scan_geom_corrected.geom'
PROJS_ROWS = 972
PROJS_COLS = 768
MAX_NUM_ANGLES = 1200
VOL_SZ  = 3 * (50 * VOXEL_PER_MM + 1,)
VOX_SZ  = 1. / VOXEL_PER_MM

GT_NB_ITER = 50

SINGLE_SLICE_CONFIGS = {  # first key: walnut_id; second key: orbit_id
    1: {  # walnut_id
        2: {  # orbit_id
            'num_slices': 9,
            'slice_offset': 3,  # source is shifted by ~ 0.3 mm = 3 px
            'num_proj_rows': 9,
            'first_proj_row': 474,
        }
    },
    2: {  # walnut_id
        2: {  # orbit_id
            'num_slices': 25,
            'slice_offset': -11,  # source is shifted by ~ -1.1 mm = -11 px
            'num_proj_rows': 9,
            'first_proj_row': 474,
        }
    },
    3: {  # walnut_id
        2: {  # orbit_id
            'num_slices': 25,
            'slice_offset': -11,  # source is shifted by ~ -1.1 mm = -11 px
            'num_proj_rows': 9,
            'first_proj_row': 474,
        }
    },
    4: {  # walnut_id
        2: {  # orbit_id
            'num_slices': 25,
            'slice_offset': -11,  # source is shifted by ~ -1.1 mm = -11 px
            'num_proj_rows': 9,
            'first_proj_row': 475,
        }
    },
    5: {  # walnut_id
        2: {  # orbit_id
            'num_slices': 25,
            'slice_offset': -11,  # source is shifted by ~ -1.1 mm = -11 px
            'num_proj_rows': 9,
            'first_proj_row': 474,
        }
    },
}

DEFAULT_SINGLE_SLICE_WALNUT_ID = 1
DEFAULT_SINGLE_SLICE_ORBIT_ID = 2

def get_first_proj_col_for_sub_sampling(factor=DEFAULT_PROJ_COL_SUB_SAMPLING):
    num_proj_cols = ceil(PROJS_COLS / factor)
    first_proj_col = (PROJS_COLS - ((num_proj_cols - 1) * factor + 1)) // 2
    return first_proj_col

def get_first_proj_row_for_sub_sampling(factor=DEFAULT_PROJ_ROW_SUB_SAMPLING, num_orig=PROJS_ROWS, num=-1):
    max_num_proj_rows = ceil(num_orig / factor)
    if num == -1:
        num_proj_rows = max_num_proj_rows
    else:
        assert num <= max_num_proj_rows
        num_proj_rows = num
    first_proj_row = (num_orig - ((num_proj_rows - 1) * factor + 1)) // 2
    return first_proj_row

def sub_sample_proj(
        projs,
        factor_row=DEFAULT_PROJ_ROW_SUB_SAMPLING, first_row=-1, num_rows=-1,
        factor_col=DEFAULT_PROJ_COL_SUB_SAMPLING, first_col=-1):

    if first_row == -1:
        first_row = get_first_proj_row_for_sub_sampling(factor=factor_row, num_orig=projs.shape[0])
    max_num_proj_rows = len(range(
            first_row, PROJS_ROWS, factor_row))
    if num_rows == -1:
        num_rows = max_num_proj_rows
    else:
        assert num_rows <= max_num_proj_rows

    if first_col == -1:
        first_col = get_first_proj_col_for_sub_sampling(factor=factor_col)

    out = projs[first_row:(first_row+num_rows*factor_row):factor_row, :, first_col::factor_col]
    return out

def up_sample_proj(
        projs,
        factor_row=DEFAULT_PROJ_ROW_SUB_SAMPLING, first_row=-1,
        factor_col=DEFAULT_PROJ_COL_SUB_SAMPLING, first_col=-1,
        num_rows_orig=None,
        kind='linear'):

    if factor_row != 1:
        if first_row == -1:
            assert num_rows_orig is not None, (
                'either first_row or num_rows_orig must be specified if '
                'sub-sampling rows')
            first_row = get_first_proj_row_for_sub_sampling(factor=factor_row, num_orig=num_rows_orig)
        x_row = np.arange(first_row, PROJS_ROWS, factor_row)
        projs_interp1d_row = scipy.interpolate.interp1d(
                x_row, projs, kind=kind, axis=0, bounds_error=False,
                fill_value=(projs[0, :, :], projs[-1, :, :]), assume_sorted=True)
        projs = projs_interp1d_row(np.arange(PROJS_ROWS))

    if factor_col != 1:
        if first_col == -1:
            first_col = get_first_proj_col_for_sub_sampling(factor=factor_col)
        x_col = np.arange(first_col, PROJS_COLS, factor_col)
        projs_interp1d_col = scipy.interpolate.interp1d(
                x_col, projs, kind=kind, axis=2, bounds_error=False,
                fill_value=(projs[:, :, 0], projs[:, :, -1]), assume_sorted=True)
        projs = projs_interp1d_col(np.arange(PROJS_COLS))

    return projs

# Note from ASTRA documentation:
#   For usage with GPU code, the volume must be centered around the origin
#   and pixels must be square. This is not always explicitly checked in all
#   functions, so not following these requirements may have unpredictable
#   results.
# We therefore always select a centered volume down-sampling, letting
# ``WindowMinX == -WindowMaxX``, and so on.
# In order to obtain an aligned pixel grid (under the constraint to be central),
# only odd down-sampling factors are supported (because the image shape is odd,
# 501^3), and the number of voxels for each dimension is selected to be odd, so
# the central pixel is always located at the origin.

def get_down_sampled_vol_shape(
        down_sampling=DEFAULT_VOL_DOWN_SAMPLING):

    if np.isscalar(down_sampling):
        down_sampling = (down_sampling,) * 3
    down_sampling = np.asarray(down_sampling)
    # must be odd since VOL_SZ is odd for an aligned pixel grid
    assert np.all(np.mod(down_sampling, 2) == 1)

    down_sampled_vol_shape = np.floor(
            np.asarray(VOL_SZ) / down_sampling).astype(int)
    # make down_sampled_vol_shape odd, yielding an aligned pixel grid
    down_sampled_vol_shape -= np.mod(down_sampled_vol_shape + 1, 2)

    return tuple(down_sampled_vol_shape.tolist())

def down_sample_vol(
        vol_x, down_sampling=DEFAULT_VOL_DOWN_SAMPLING, kind='mean_pooling'):

    if np.isscalar(down_sampling):
        down_sampling = (down_sampling,) * 3
    down_sampling = np.asarray(down_sampling)
    assert np.array_equal(vol_x.shape, VOL_SZ)
    assert np.all(np.mod(down_sampling, 2) == 1)

    down_sampled_vol_shape = np.asarray(get_down_sampled_vol_shape(
            down_sampling=down_sampling))

    margin = vol_x.shape - down_sampled_vol_shape * down_sampling
    assert np.all(np.mod(margin, 2) == 0), 'cannot split margin evenly'

    if kind == 'mean_pooling':
        vol_x_cropped = vol_x[
                margin[0]//2:vol_x.shape[0]-(margin[0]//2),
                margin[1]//2:vol_x.shape[1]-(margin[1]//2),
                margin[2]//2:vol_x.shape[2]-(margin[2]//2)]
        vol_x_cropped_reshaped = vol_x_cropped.reshape(
                (down_sampled_vol_shape[0], down_sampling[0],
                 down_sampled_vol_shape[1], down_sampling[1],
                 down_sampled_vol_shape[2], down_sampling[2]))
        vol_x_down_sampled = np.mean(vol_x_cropped_reshaped, axis=(1, 3, 5))

    else:
        raise NotImplementedError(
                'Unknown down-sampling method \'{}\''.format(kind))

    return vol_x_down_sampled

def get_vol_geom(down_sampling=DEFAULT_VOL_DOWN_SAMPLING, num_slices=-1):

    if np.isscalar(down_sampling):
        down_sampling = (down_sampling,) * 3
    down_sampled_vol_shape = np.asarray(get_down_sampled_vol_shape(
            down_sampling=down_sampling))

    vol_geom = astra.create_vol_geom(
            (down_sampled_vol_shape[1],
             down_sampled_vol_shape[2],
             down_sampled_vol_shape[0] if num_slices == -1 else num_slices))
    vol_geom['option']['WindowMinX'] = vol_geom['option']['WindowMinX'] * VOX_SZ * down_sampling[2]
    vol_geom['option']['WindowMaxX'] = vol_geom['option']['WindowMaxX'] * VOX_SZ * down_sampling[2]
    vol_geom['option']['WindowMinY'] = vol_geom['option']['WindowMinY'] * VOX_SZ * down_sampling[1]
    vol_geom['option']['WindowMaxY'] = vol_geom['option']['WindowMaxY'] * VOX_SZ * down_sampling[1]
    vol_geom['option']['WindowMinZ'] = vol_geom['option']['WindowMinZ'] * VOX_SZ * down_sampling[0]
    vol_geom['option']['WindowMaxZ'] = vol_geom['option']['WindowMaxZ'] * VOX_SZ * down_sampling[0]

    return vol_geom

def get_proj_geom(data_path, walnut_id, orbit_id,
                  angular_sub_sampling=DEFAULT_ANGULAR_SUB_SAMPLING,
                  proj_row_sub_sampling=DEFAULT_PROJ_ROW_SUB_SAMPLING,
                  proj_col_sub_sampling=DEFAULT_PROJ_COL_SUB_SAMPLING,
                  first_proj_row=-1, first_proj_col=-1,
                  num_proj_rows=-1,
                  rotation=None, shift_z=0., return_vecs=False):
    data_path_full = os.path.join(data_path, 'Walnut{}'.format(walnut_id),
                                  'Projections', 'tubeV{}'.format(orbit_id))

    if first_proj_row == -1:
        first_proj_row = get_first_proj_row_for_sub_sampling(
                factor=proj_row_sub_sampling, num=num_proj_rows)
    max_num_proj_rows = len(range(
            first_proj_row, PROJS_ROWS, proj_row_sub_sampling))
    if num_proj_rows == -1:
        num_proj_rows = max_num_proj_rows
    else:
        assert num_proj_rows <= max_num_proj_rows

    if first_proj_col == -1:
        first_proj_col = get_first_proj_col_for_sub_sampling(
                factor=proj_col_sub_sampling)

    num_proj_cols = len(range(
            first_proj_col, PROJS_COLS, proj_col_sub_sampling))

    vecs_all = np.loadtxt(os.path.join(data_path_full, VECS_NAME))
    vecs = vecs_all[range(0, MAX_NUM_ANGLES, angular_sub_sampling)]

    # determine the detector center, such that the first detector row in the
    # sub-sampled geometry coincides with row `first_proj_row` of the full
    # geometry with PROJS_ROWS rows
    row_margin_end = (PROJS_ROWS - 1) - (
            first_proj_row + (num_proj_rows - 1) * proj_row_sub_sampling)
    vecs[:, 3:6] += (first_proj_row - row_margin_end) / 2 * vecs[:, 9:12]

    # determine the detector center, such that the first detector column in the
    # sub-sampled geometry coincides with column `first_proj_col` of the full
    # geometry with PROJS_COLS columns
    col_margin_end = (PROJS_COLS - 1) - (
            first_proj_col + (num_proj_cols - 1) * proj_col_sub_sampling)
    vecs[:, 3:6] += (first_proj_col - col_margin_end) / 2 * vecs[:, 6:9]

    # multiply step between detector rows by proj_row_sub_sampling
    vecs[:, 9:12] *= proj_row_sub_sampling

    # multiply step between detector columns by proj_col_sub_sampling
    vecs[:, 6:9] *= proj_col_sub_sampling

    # apply a scipy rotation (globally) if specified
    if rotation is not None:
        for i in range(0, 12, 3):
            vecs[:, i:i+3] = rotation.apply(vecs[:, i:i+3])

    # apply a shift in z direction if specified
    vecs[:, 2] += shift_z
    vecs[:, 5] += shift_z

    proj_geom = astra.create_proj_geom(
            'cone_vec', num_proj_rows, num_proj_cols, vecs)

    return proj_geom if not return_vecs else (proj_geom, vecs)

def get_projection_data(data_path, walnut_id, orbit_id,
                        angular_sub_sampling=DEFAULT_ANGULAR_SUB_SAMPLING,
                        proj_row_sub_sampling=DEFAULT_PROJ_ROW_SUB_SAMPLING,
                        proj_col_sub_sampling=DEFAULT_PROJ_COL_SUB_SAMPLING,
                        first_proj_row=-1, first_proj_col=-1,
                        num_proj_rows=-1):
    data_path_full = os.path.join(data_path, 'Walnut{}'.format(walnut_id),
                                  'Projections', 'tubeV{}'.format(orbit_id))

    # projection file indices, we need to read in the projection in reverse
    # order due to the portrait mode acquision
    projs_idx  = range(MAX_NUM_ANGLES, 0, -angular_sub_sampling)

    num_angles = ceil(MAX_NUM_ANGLES / angular_sub_sampling)

    # create the numpy array which will receive projection data from tiff files
    projs = np.zeros((num_angles, PROJS_ROWS, PROJS_COLS), dtype=np.float32)

    # transformation to apply to each image, we need to get the image from
    # the way the scanner reads it out into to way described in the projection
    # geometry
    trafo = lambda image : np.transpose(np.flipud(image))

    # load flat-field and dark-fields
    # there are two flat-field images (taken before and after acquisition), we
    # simply average them
    dark = trafo(imageio.imread(os.path.join(data_path_full, DARK_NAME)))
    flat = np.zeros((2, PROJS_ROWS, PROJS_COLS), dtype=np.float32)

    for i, fn in enumerate(FLAT_NAME):
        flat[i] = trafo(imageio.imread(os.path.join(data_path_full, fn)))
    flat =  np.mean(flat,axis=0)

    # load projection data
    for i in range(num_angles):
        projs[i] = trafo(imageio.imread(
                os.path.join(data_path_full, PROJS_NAME.format(projs_idx[i]))))

    # subtract the dark field, divide by the flat field, and take the negative
    # log to linearize the data according to the Beer-Lambert law
    projs -= dark
    projs /= (flat - dark)
    np.log(projs, out=projs)
    np.negative(projs, out=projs)
    # permute data to ASTRA convention
    projs = np.transpose(projs, (1, 0, 2))
    # sub-sample
    projs = sub_sample_proj(
            projs,
            factor_row=proj_row_sub_sampling, first_row=first_proj_row, num_rows=num_proj_rows,
            factor_col=proj_col_sub_sampling, first_col=first_proj_col)
    projs = np.ascontiguousarray(projs)

    return projs

def get_ground_truth(data_path, walnut_id, orbit_id, slice_ind):
    slice_path = os.path.join(
            data_path, 'Walnut{}'.format(walnut_id), 'Reconstructions',
            'full_AGD_{}_{:06}.tiff'.format(GT_NB_ITER, slice_ind))
    gt = imageio.imread(slice_path)

    return gt

def get_ground_truth_3d(data_path, walnut_id, orbit_id):
    gt_slices = [get_ground_truth(data_path, walnut_id, orbit_id, slice_ind)
            for slice_ind in range(VOL_SZ[0])]
    gt_3d = np.stack(gt_slices, axis=0)

    return gt_3d

def get_single_slice_ind(
        data_path,
        walnut_id=DEFAULT_SINGLE_SLICE_WALNUT_ID,
        orbit_id=DEFAULT_SINGLE_SLICE_ORBIT_ID):

    single_slice_config = SINGLE_SLICE_CONFIGS.get(walnut_id, {}).get(orbit_id)
    if single_slice_config is None:
        raise ValueError('No single slice ray trafo configuration known for '
                         'walnut_id={:d}, orbit_id={:d}'.format(
                                 walnut_id, orbit_id))
    slice_offset = single_slice_config['slice_offset']
    slice_ind = (VOL_SZ[0] - 1) // 2 + slice_offset

    return slice_ind

def get_single_slice_ray_trafo(
        data_path,
        walnut_id=DEFAULT_SINGLE_SLICE_WALNUT_ID,
        orbit_id=DEFAULT_SINGLE_SLICE_ORBIT_ID,
        angular_sub_sampling=DEFAULT_ANGULAR_SUB_SAMPLING,
        proj_col_sub_sampling=DEFAULT_PROJ_COL_SUB_SAMPLING):

    single_slice_config = SINGLE_SLICE_CONFIGS.get(walnut_id, {}).get(orbit_id)
    if single_slice_config is None:
        raise ValueError('No single slice ray trafo configuration known for '
                         'walnut_id={:d}, orbit_id={:d}'.format(
                                 walnut_id, orbit_id))
    num_slices = single_slice_config['num_slices']
    slice_offset = single_slice_config['slice_offset']
    num_proj_rows = single_slice_config['num_proj_rows']
    first_proj_row = single_slice_config['first_proj_row']

    walnut_ray_trafo = MaskedWalnutRayTrafo(
            data_path=data_path,
            walnut_id=walnut_id, orbit_id=orbit_id,
            angular_sub_sampling=angular_sub_sampling,
            proj_col_sub_sampling=proj_col_sub_sampling,
            num_slices=num_slices, num_proj_rows=num_proj_rows,
            first_proj_row=first_proj_row,
            vol_mask_slice=(num_slices - 1) // 2 + slice_offset,
            proj_mask_select_k_rows=1,
            )

    return walnut_ray_trafo

def astra_fp3d_cuda(vol_x, vol_geom, proj_geom, projs_out):

    vol_x = np.ascontiguousarray(vol_x, dtype=np.float32)
    vol_id = astra.data3d.link('-vol', vol_geom, vol_x)
    proj_id = astra.data3d.link('-sino', proj_geom, projs_out)

    cfg_fp = astra.astra_dict('FP3D_CUDA')
    cfg_fp['VolumeDataId'] = vol_id
    cfg_fp['ProjectionDataId'] = proj_id
    alg_id = astra.algorithm.create(cfg_fp)

    astra.algorithm.run(alg_id)

    astra.algorithm.delete(alg_id)
    astra.data3d.delete(proj_id)
    astra.data3d.delete(vol_id)

def astra_bp3d_cuda(projs, vol_geom, proj_geom, vol_x_out):

    projs = np.ascontiguousarray(projs, dtype=np.float32)
    proj_id = astra.data3d.link('-sino', proj_geom, projs)
    vol_id = astra.data3d.link('-vol', vol_geom, vol_x_out)

    cfg_bp = astra.astra_dict('BP3D_CUDA')
    cfg_bp['ReconstructionDataId'] = vol_id
    cfg_bp['ProjectionDataId'] = proj_id
    alg_id = astra.algorithm.create(cfg_bp)

    astra.algorithm.run(alg_id)

    astra.algorithm.delete(alg_id)
    astra.data3d.delete(proj_id)
    astra.data3d.delete(vol_id)

def astra_fdk_cuda(projs, vol_geom, proj_geom, vol_x_out):

    projs = np.ascontiguousarray(projs, dtype=np.float32)
    proj_id = astra.data3d.link('-sino', proj_geom, projs)
    vol_id = astra.data3d.link('-vol', vol_geom, vol_x_out)

    cfg_fdk = astra.astra_dict('FDK_CUDA')
    cfg_fdk['ReconstructionDataId'] = vol_id
    cfg_fdk['ProjectionDataId'] = proj_id
    cfg_fdk['option'] = {}
    cfg_fdk['option']['ShortScan'] = False
    alg_id = astra.algorithm.create(cfg_fdk)

    astra.algorithm.run(alg_id, 1)

    astra.algorithm.delete(alg_id)
    astra.data3d.delete(proj_id)
    astra.data3d.delete(vol_id)

class WalnutRayTrafo:
    def __init__(self, data_path, walnut_id, orbit_id,
                 angular_sub_sampling=DEFAULT_ANGULAR_SUB_SAMPLING,
                 proj_row_sub_sampling=DEFAULT_PROJ_ROW_SUB_SAMPLING,
                 proj_col_sub_sampling=DEFAULT_PROJ_COL_SUB_SAMPLING,
                 vol_down_sampling=DEFAULT_VOL_DOWN_SAMPLING,
                 first_proj_row=-1, first_proj_col=-1,
                 rotation=None, shift_z=0.,
                 proj_sub_sampling_via_geom=True,
                 proj_up_sampling_via_geom=True,
                 proj_up_sampling_kind_if_not_via_geom='linear'):
        self.data_path = data_path
        self.walnut_id = walnut_id
        self.orbit_id = orbit_id
        self.angular_sub_sampling = angular_sub_sampling
        self.proj_row_sub_sampling = proj_row_sub_sampling
        self.proj_col_sub_sampling = proj_col_sub_sampling
        self.vol_down_sampling = (
                (vol_down_sampling,) * 3 if np.isscalar(vol_down_sampling)
                else vol_down_sampling)
        self.first_proj_row = (
                get_first_proj_row_for_sub_sampling(
                        factor=self.proj_row_sub_sampling)
                if first_proj_row == -1 else first_proj_row)
        self.first_proj_col = (
                get_first_proj_col_for_sub_sampling(
                        factor=self.proj_col_sub_sampling)
                if first_proj_col == -1 else first_proj_col)
        self.rotation = rotation
        self.shift_z = shift_z
        self.proj_sub_sampling_via_geom = (proj_sub_sampling_via_geom or
                (proj_row_sub_sampling == 1 and proj_col_sub_sampling == 1))
        self.proj_up_sampling_via_geom = (proj_up_sampling_via_geom or
                (proj_row_sub_sampling == 1 and proj_col_sub_sampling == 1))
        self.proj_up_sampling_kind_if_not_via_geom = (
                proj_up_sampling_kind_if_not_via_geom)

        self.num_angles = ceil(MAX_NUM_ANGLES / self.angular_sub_sampling)
        self.num_proj_rows = len(range(
                self.first_proj_row, PROJS_ROWS, self.proj_row_sub_sampling))
        self.num_proj_cols = len(range(
                self.first_proj_col, PROJS_COLS, self.proj_col_sub_sampling))

        self.vol_shape = get_down_sampled_vol_shape(
                down_sampling=self.vol_down_sampling)
        self.proj_shape = (
                self.num_proj_rows, self.num_angles, self.num_proj_cols)

        self.vol_geom = get_vol_geom(down_sampling=self.vol_down_sampling)

        proj_geom_kwargs = dict(
                data_path=self.data_path,
                walnut_id=self.walnut_id, orbit_id=self.orbit_id,
                angular_sub_sampling=self.angular_sub_sampling,
                rotation=self.rotation, shift_z=self.shift_z,
        )

        # geometry with projection sub-sampling
        self.proj_geom, self.vecs = get_proj_geom(
                **proj_geom_kwargs,
                proj_row_sub_sampling=self.proj_row_sub_sampling,
                proj_col_sub_sampling=self.proj_col_sub_sampling,
                first_proj_row=self.first_proj_row,
                first_proj_col=self.first_proj_col,
                num_proj_rows=self.num_proj_rows,
                return_vecs=True)

        # geometry with all projection values (no sub-sampling)
        if self.proj_row_sub_sampling != 1 or self.proj_col_sub_sampling != 1:
            self.proj_geom_no_sub_sampling, self.vecs_no_sub_sampling = get_proj_geom(
                    **proj_geom_kwargs,
                    proj_row_sub_sampling=1,
                    proj_col_sub_sampling=1,
                    return_vecs=True)
            self.proj_shape_no_sub_sampling = (PROJS_ROWS, self.proj_shape[1], PROJS_COLS)
        else:
            self.proj_geom_no_sub_sampling = None
            self.vecs_no_sub_sampling = None
            self.proj_shape_no_sub_sampling = None

    def fp3d(self, vol_x, vol_geom=None):
        if vol_geom is None:
            vol_geom = self.vol_geom

        if self.proj_sub_sampling_via_geom:
            proj_shape = self.proj_shape
            proj_geom = self.proj_geom
        else:
            proj_shape = self.proj_shape_no_sub_sampling
            proj_geom = self.proj_geom_no_sub_sampling
        projs = np.zeros(proj_shape, dtype=np.float32)

        astra_fp3d_cuda(vol_x=vol_x, vol_geom=vol_geom, proj_geom=proj_geom,
                projs_out=projs)

        if not self.proj_sub_sampling_via_geom:
            projs = sub_sample_proj(
                    projs,
                    factor_row=self.proj_row_sub_sampling,
                    factor_col=self.proj_col_sub_sampling,
                    first_row=self.first_proj_row,
                    first_col=self.first_proj_col)

        return projs

    def bp3d(self, projs, proj_geom=None, proj_geom_no_sub_sampling=None):

        if self.proj_up_sampling_via_geom:
            proj_geom = proj_geom if proj_geom is not None else self.proj_geom
        else:
            proj_geom = (proj_geom_no_sub_sampling if proj_geom_no_sub_sampling is not None
                         else self.proj_geom_no_sub_sampling)
            projs = up_sample_proj(
                    projs,
                    factor_row=self.proj_row_sub_sampling,
                    factor_col=self.proj_col_sub_sampling,
                    first_row=self.first_proj_row,
                    first_col=self.first_proj_col,
                    num_rows_orig=self.proj_shape_no_sub_sampling[0],
                    kind=self.proj_up_sampling_kind_if_not_via_geom)
        vol_x = np.zeros(self.vol_shape, dtype=np.float32)

        astra_bp3d_cuda(projs=projs, vol_geom=self.vol_geom, proj_geom=proj_geom,
                vol_x_out=vol_x)

        return vol_x

    def fdk(self, projs, proj_geom=None, proj_geom_no_sub_sampling=None):

        if self.proj_up_sampling_via_geom:
            proj_geom = proj_geom if proj_geom is not None else self.proj_geom
        else:
            proj_geom = (proj_geom_no_sub_sampling if proj_geom_no_sub_sampling is not None
                         else self.proj_geom_no_sub_sampling)
            projs = up_sample_proj(
                    projs,
                    factor_row=self.proj_row_sub_sampling,
                    factor_col=self.proj_col_sub_sampling,
                    first_row=self.first_proj_row,
                    first_col=self.first_proj_col,
                    num_rows_orig=self.proj_shape_no_sub_sampling[0],
                    kind=self.proj_up_sampling_kind_if_not_via_geom)
        vol_x = np.zeros(self.vol_shape, dtype=np.float32)

        astra_fdk_cuda(projs=projs, vol_geom=self.vol_geom, proj_geom=proj_geom,
                vol_x_out=vol_x)

        return vol_x

    # alternative function names
    apply = fp3d
    apply_adjoint = bp3d
    apply_fdk = fdk


class MaskedWalnutRayTrafo:
    def __init__(self, data_path, walnut_id, orbit_id,
                 angular_sub_sampling=DEFAULT_ANGULAR_SUB_SAMPLING,
                 proj_row_sub_sampling=DEFAULT_PROJ_ROW_SUB_SAMPLING,
                 proj_col_sub_sampling=DEFAULT_PROJ_COL_SUB_SAMPLING,
                 first_proj_row=-1, first_proj_col=-1, num_proj_rows=-1,
                 num_slices=VOL_SZ[0],
                 rotation=None, shift_z=0.,
                 vol_mask_slice=None, proj_mask_select_k_rows=None,
                 proj_sub_sampling_via_geom=True,
                 proj_up_sampling_via_geom=True,
                 proj_up_sampling_kind_if_not_via_geom='linear'):

        assert num_slices % 2 == 1  # each slice then matches one in full volume

        self.data_path = data_path
        self.walnut_id = walnut_id
        self.orbit_id = orbit_id
        self.angular_sub_sampling = angular_sub_sampling
        self.proj_row_sub_sampling = proj_row_sub_sampling
        self.proj_col_sub_sampling = proj_col_sub_sampling

        self.rotation = rotation
        self.shift_z = shift_z

        # here we support restriction to some rows via first_proj_row and
        # num_proj_rows, so it makes sense for first_proj_row to take values
        # greater than or equal to proj_row_sub_sampling; for WalnutRayTrafo on
        # the other hand, we start with first_proj_row % proj_row_sub_sampling.
        self.first_proj_row = (
                get_first_proj_row_for_sub_sampling(
                        factor=proj_row_sub_sampling, num=num_proj_rows)
                if first_proj_row == -1 else first_proj_row)
        # choose matching row grid for full (non-restricted) geometry
        first_proj_row_full = self.first_proj_row % proj_row_sub_sampling

        self.ray_trafo_full = WalnutRayTrafo(data_path, walnut_id, orbit_id,
                 angular_sub_sampling=angular_sub_sampling,
                 proj_row_sub_sampling=proj_row_sub_sampling,
                 proj_col_sub_sampling=proj_col_sub_sampling,
                 first_proj_row=first_proj_row_full,
                 first_proj_col=first_proj_col,
                 rotation=rotation, shift_z=shift_z,
                 proj_sub_sampling_via_geom=proj_sub_sampling_via_geom,
                 proj_up_sampling_via_geom=proj_up_sampling_via_geom,
                 proj_up_sampling_kind_if_not_via_geom=proj_up_sampling_kind_if_not_via_geom)

        self.first_proj_col = (  # same as self.ray_trafo_full.first_proj_col
                get_first_proj_col_for_sub_sampling(
                        factor=self.proj_col_sub_sampling)
                if first_proj_col == -1 else first_proj_col)
        self.proj_sub_sampling_via_geom = (proj_sub_sampling_via_geom or
                (proj_row_sub_sampling == 1 and proj_col_sub_sampling == 1))
        self.proj_up_sampling_via_geom = (proj_up_sampling_via_geom or
                (proj_row_sub_sampling == 1 and proj_col_sub_sampling == 1))
        self.proj_up_sampling_kind_if_not_via_geom = (
                proj_up_sampling_kind_if_not_via_geom)

        self.num_angles = ceil(MAX_NUM_ANGLES / self.angular_sub_sampling)

        max_num_proj_rows = len(range(
                self.first_proj_row, PROJS_ROWS, self.proj_row_sub_sampling))
        self.num_proj_rows = max_num_proj_rows if num_proj_rows == -1 else num_proj_rows
        assert self.num_proj_rows <= max_num_proj_rows

        self.num_slices = num_slices

        self.vol_shape = (self.num_slices,) + VOL_SZ[1:]

        self.num_proj_cols = len(range(
                self.first_proj_col, PROJS_COLS, self.proj_col_sub_sampling))
        self.proj_shape = (
                self.num_proj_rows, self.num_angles, self.num_proj_cols)

        if isinstance(vol_mask_slice, int):
            self.vol_mask_slice = slice(vol_mask_slice, vol_mask_slice+1)
        else:
            if vol_mask_slice is not None:
                assert vol_mask_slice.step is None or vol_mask_slice.step == 1
            self.vol_mask_slice = vol_mask_slice
        self.proj_mask_select_k_rows = proj_mask_select_k_rows

        self.vol_geom = get_vol_geom(num_slices=self.num_slices)

        proj_geom_kwargs = dict(
                data_path=self.data_path,
                walnut_id=self.walnut_id, orbit_id=self.orbit_id,
                angular_sub_sampling=self.angular_sub_sampling,
                rotation=self.rotation, shift_z=self.shift_z,
        )

        ## geometries with projection sub-sampling
        self.proj_geom, self.vecs = get_proj_geom(
                **proj_geom_kwargs,
                proj_row_sub_sampling=self.proj_row_sub_sampling,
                proj_col_sub_sampling=self.proj_col_sub_sampling,
                first_proj_row=self.first_proj_row,
                first_proj_col=self.first_proj_col,
                num_proj_rows=self.num_proj_rows,
                return_vecs=True)

        ## geometries with dense projection values (no sub-sampling)
        if self.proj_row_sub_sampling != 1 or self.proj_col_sub_sampling != 1:
            num_proj_rows_no_sub_sampling = (
                    (self.num_proj_rows - 1) * self.proj_row_sub_sampling + 1)
            self.proj_geom_no_sub_sampling, self.vecs_no_sub_sampling = (
                    get_proj_geom(
                            **proj_geom_kwargs,
                            proj_row_sub_sampling=1,
                            proj_col_sub_sampling=1,
                            first_proj_row=self.first_proj_row,
                            num_proj_rows=num_proj_rows_no_sub_sampling,
                            return_vecs=True))
            self.proj_shape_no_sub_sampling = (
                    num_proj_rows_no_sub_sampling, self.proj_shape[1], PROJS_COLS)
        else:
            self.proj_geom_no_sub_sampling = None
            self.vecs_no_sub_sampling = None
            self.proj_shape_no_sub_sampling = None

        self.build_proj_mask()

        self.num_projs_in_mask = np.count_nonzero(self.proj_mask)

        self.assert_proj_rows_suffice()
        self.assert_vol_slices_suffice()

    def build_proj_mask(self):
        if self.vol_mask_slice is None:
            self.proj_mask = None
        else:
            proj_mask = np.zeros(self.proj_shape, dtype=bool)

            vol_test = np.zeros(self.vol_shape, dtype=np.float32)
            vol_test[self.vol_mask_slice] = 1.
            projs = self.fp3d(vol_test)

            if self.proj_mask_select_k_rows is not None:
                vol_test_full = np.zeros(VOL_SZ, dtype=np.float32)

                vol_test_full[:] = 1.
                projs_sum = self.fp3d(vol_test_full,
                        vol_geom=self.ray_trafo_full.vol_geom)

                fraction = np.zeros(self.proj_shape)
                valid = projs_sum > 0.
                fraction[valid] = projs[valid] / projs_sum[valid]
                for _ in range(self.proj_mask_select_k_rows):
                    index_array = np.expand_dims(
                            np.argmax(fraction, axis=0), axis=0)
                    assert np.all(
                            np.take_along_axis(projs, index_array, axis=0) > 0.)
                    np.put_along_axis(proj_mask, index_array, True, axis=0)
                    np.put_along_axis(fraction, index_array, 0., axis=0)
            else:
                proj_mask[:] = projs > 0.

            assert np.all(np.any(proj_mask, axis=0)), (
                    'The projection mask should select at least one row at each'
                    '(angle, column)-position.')

            self.proj_mask = proj_mask

            self.proj_mask_first_row_inds = np.argmax(self.proj_mask, axis=0)
            self.proj_mask_last_row_inds = (
                    self.num_proj_rows - 1 -
                    np.argmax(self.proj_mask[::-1], axis=0))

    def assert_proj_rows_suffice(self):
        projs_test_full = np.ones(self.ray_trafo_full.proj_shape, dtype=np.float32)
        first_proj_row_in_full = (  # offset on sub-sampled row grid
                (self.first_proj_row - self.ray_trafo_full.first_proj_row) //
                self.proj_row_sub_sampling)
        projs_test_full[
                first_proj_row_in_full:first_proj_row_in_full+self.num_proj_rows] = 0.
        vol_x = self.bp3d(projs_test_full,
                proj_geom=self.ray_trafo_full.proj_geom,
                proj_geom_no_sub_sampling=self.ray_trafo_full.proj_geom_no_sub_sampling)
        assert np.all(vol_x[self.vol_mask_slice] == 0.)

    def assert_vol_slices_suffice(self):
        vol_test_full = np.ones(VOL_SZ, dtype=np.float32)
        first = (VOL_SZ[0] - self.num_slices) // 2
        vol_test_full[first:first+self.num_slices] = 0.
        projs = self.fp3d(vol_test_full, vol_geom=self.ray_trafo_full.vol_geom)
        assert np.all(projs[self.proj_mask] == 0.)

    def get_proj_slice_contributing_to_masked_vol(self):
        proj_test = np.zeros(self.proj_shape, dtype=np.float32)

        i = 0
        while i < self.num_proj_rows:
            proj_test[:] = 0.
            proj_test[i] = 1.
            vol_x = self.bp3d(proj_test)
            if np.any(vol_x[self.vol_mask_slice] > 0.):
                break
            i += 1
        first_contributing = i

        i = self.num_proj_rows - 1
        while i >= 0:
            proj_test[:] = 0.
            proj_test[i] = 1.
            vol_x = self.bp3d(proj_test)
            if np.any(vol_x[self.vol_mask_slice] > 0.):
                break
            i -= 1
        last_contributing = i

        return slice(first_contributing, last_contributing + 1)

    def get_vol_slice_contributing_to_masked_projs(self):
        if self.proj_mask is None:
            return None

        vol_test = np.zeros(self.vol_shape, dtype=np.float32)

        i = 0
        while i < self.num_slices:
            vol_test[:] = 0.
            vol_test[i] = 1.
            projs = self.fp3d(vol_test)
            if np.any(projs[self.proj_mask] > 0.):
                break
            i += 1
        first_contributing = i

        i = self.num_slices - 1
        while i >= 0:
            vol_test[:] = 0.
            vol_test[i] = 1.
            projs = self.fp3d(vol_test)
            if np.any(projs[self.proj_mask] > 0.):
                break
            i -= 1
        last_contributing = i

        return slice(first_contributing, last_contributing + 1)

    def vol_from_full(self, vol_x_full):
        first = (VOL_SZ[0] - self.num_slices) // 2
        vol_x = vol_x_full[first:first+self.num_slices]
        return vol_x

    def projs_from_full(self, projs_full):
        first_proj_row_in_full = (  # offset on sub-sampled row grid
                (self.first_proj_row - self.ray_trafo_full.first_proj_row) //
                self.proj_row_sub_sampling)
        projs = projs_full[
                first_proj_row_in_full:first_proj_row_in_full+self.num_proj_rows]
        return projs

    def vol_in_mask(self, vol_x, full_input=False, squeeze=False):
        if full_input:
            vol_x = self.vol_from_full(vol_x)
        vol_in_mask = vol_x[self.vol_mask_slice]
        if squeeze:
            vol_in_mask = np.squeeze(vol_in_mask, axis=0)
        return vol_in_mask

    def flat_projs_in_mask(self, projs, full_input=False):
        """
        The entries are selected by boolean index :attr:`proj_mask`,
        which selects entries in row-major order.
        NB: It would *not* be meaningful to reshape the result to a
        multidimensional array.
        """
        if full_input:
            projs = self.projs_from_full(projs)
        flat_projs_in_mask = projs[self.proj_mask]
        return flat_projs_in_mask

    def vol_from_vol_in_mask(self, vol_in_mask, out=None,
                             padding_mode='edge'):
        vol_padded = (np.zeros(self.vol_shape, dtype=np.float32)
                      if out is None else out)
        vol_padded[self.vol_mask_slice] = vol_in_mask

        if padding_mode == 'edge':
            vol_mask_inds = range(self.num_slices)[self.vol_mask_slice]
            vol_padded[:vol_mask_inds[0]] = vol_padded[vol_mask_inds[0]]
            vol_padded[vol_mask_inds[-1]+1:] = vol_padded[vol_mask_inds[-1]]
        elif padding_mode == 'zeros':
            pass
        else:
            raise ValueError('Unknown padding mode \'{}\''.format(padding_mode))

        return vol_padded

    def projs_from_flat_projs_in_mask(self, flat_projs_in_mask, out=None,
                                      padding_mode='edge'):
        assert self.proj_mask is not None

        projs_padded = (np.zeros(self.proj_shape, dtype=np.float32)
                        if out is None else out)
        projs_padded[self.proj_mask] = flat_projs_in_mask

        if padding_mode == 'edge':
            if self.proj_mask_select_k_rows == 1:
                # in this case, self.proj_mask selects exactly one row at each
                # (angle, column)-position; we just broadcast it to all rows
                projs_padded[:] = np.take_along_axis(
                        projs_padded, self.proj_mask_first_row_inds[None],
                        axis=0)
            else:
                # TODO more efficient implementation
                for i in range(self.proj_shape[1]):
                    for j in range(self.proj_shape[2]):
                        first_ind = self.proj_mask_first_row_inds[i, j]
                        last_ind = self.proj_mask_last_row_inds[i, j]
                        projs_padded[:first_ind, i, j] = projs_padded[
                                first_ind, i, j]
                        projs_padded[last_ind + 1:, i, j] = projs_padded[
                                last_ind, i, j]
        elif padding_mode == 'zeros':
            pass
        else:
            raise ValueError('Unknown padding mode \'{}\''.format(padding_mode))

        return projs_padded

    def fp3d(self, vol_x, vol_geom=None):
        if vol_geom is None:
            vol_geom = self.vol_geom

        if self.proj_sub_sampling_via_geom:
            proj_shape = self.proj_shape
            proj_geom = self.proj_geom
        else:
            proj_shape = self.proj_shape_no_sub_sampling
            proj_geom = self.proj_geom_no_sub_sampling
        projs = np.zeros(proj_shape, dtype=np.float32)

        astra_fp3d_cuda(vol_x=vol_x, vol_geom=vol_geom, proj_geom=proj_geom,
                projs_out=projs)

        if not self.proj_sub_sampling_via_geom:
            projs = sub_sample_proj(
                    projs,
                    factor_row=self.proj_row_sub_sampling,
                    factor_col=self.proj_col_sub_sampling,
                    first_row=0,  # rows in self.proj_geom_no_sub_sampling are
                                  # bounding tightly
                    first_col=self.first_proj_col)

        return projs

    def bp3d(self, projs, proj_geom=None, proj_geom_no_sub_sampling=None):

        if self.proj_up_sampling_via_geom:
            proj_geom = proj_geom if proj_geom is not None else self.proj_geom
        else:
            proj_geom = (proj_geom_no_sub_sampling if proj_geom_no_sub_sampling is not None
                         else self.proj_geom_no_sub_sampling)
            projs = up_sample_proj(
                    projs,
                    factor_row=self.proj_row_sub_sampling,
                    factor_col=self.proj_col_sub_sampling,
                    first_row=0,  # rows in self.proj_geom_no_sub_sampling are
                                  # bounding tightly
                    first_col=self.first_proj_col,
                    num_rows_orig=self.proj_shape_no_sub_sampling[0],
                    kind=self.proj_up_sampling_kind_if_not_via_geom)
        vol_x = np.zeros(self.vol_shape, dtype=np.float32)

        astra_bp3d_cuda(projs=projs, vol_geom=self.vol_geom, proj_geom=proj_geom,
                vol_x_out=vol_x)

        return vol_x

    def fdk(self, projs, proj_geom=None, proj_geom_no_sub_sampling=None):

        if self.proj_up_sampling_via_geom:
            proj_geom = proj_geom if proj_geom is not None else self.proj_geom
        else:
            proj_geom = (proj_geom_no_sub_sampling if proj_geom_no_sub_sampling is not None
                         else self.proj_geom_no_sub_sampling)
            projs = up_sample_proj(
                    projs,
                    factor_row=self.proj_row_sub_sampling,
                    factor_col=self.proj_col_sub_sampling,
                    first_row=0,  # rows in self.proj_geom_no_sub_sampling are
                                  # bounding tightly
                    first_col=self.first_proj_col,
                    num_rows_orig=self.proj_shape_no_sub_sampling[0],
                    kind=self.proj_up_sampling_kind_if_not_via_geom)
        vol_x = np.zeros(self.vol_shape, dtype=np.float32)

        astra_fdk_cuda(projs=projs, vol_geom=self.vol_geom, proj_geom=proj_geom,
                vol_x_out=vol_x)

        return vol_x

    def apply(self, vol_in_mask, padding_mode='edge'):
        vol_x = self.vol_from_vol_in_mask(vol_in_mask,
                                          padding_mode=padding_mode)
        projs = self.fp3d(vol_x)
        flat_projs_in_mask = self.flat_projs_in_mask(projs)
        return flat_projs_in_mask

    def apply_adjoint(self, flat_projs_in_mask, padding_mode='edge',
                      squeeze=False):
        projs = self.projs_from_flat_projs_in_mask(flat_projs_in_mask,
                                                   padding_mode=padding_mode)
        vol_x = self.bp3d(projs)
        vol_in_mask = self.vol_in_mask(vol_x, squeeze=squeeze)
        return vol_in_mask

    def apply_fdk(self, flat_projs_in_mask, padding_mode='edge', squeeze=False):
        projs = self.projs_from_flat_projs_in_mask(flat_projs_in_mask,
                                                   padding_mode=padding_mode)
        vol_x = self.fdk(projs)
        vol_in_mask = self.vol_in_mask(vol_x, squeeze=squeeze)
        return vol_in_mask

# based on
# https://github.com/odlgroup/odl/blob/25ec783954a85c2294ad5b76414f8c7c3cd2785d/odl/contrib/torch/operator.py#L33
class NumpyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, forward_fun, backward_fun):
        ctx.forward_fun = forward_fun
        ctx.backward_fun = backward_fun

        x_np = x.detach().cpu().numpy()
        # y_np = np.stack([ctx.forward_fun(x_np_i) for x_np_i in x_np])
        y_np = ctx.forward_fun(x_np)
        y = torch.from_numpy(y_np).to(x.device)
        return y

    @staticmethod
    def backward(ctx, y):
        y_np = y.detach().cpu().numpy()
        # x_np = np.stack([ctx.backward_fun(y_np_i) for y_np_i in y_np])
        x_np = ctx.backward_fun(y_np)
        x = torch.from_numpy(x_np).to(y.device)
        return x, None, None

class WalnutRayTrafoModule(torch.nn.Module):
    def __init__(self, walnut_ray_trafo, adjoint=False):
        super().__init__()
        self.walnut_ray_trafo = walnut_ray_trafo
        self.adjoint = adjoint

    def forward(self, x):
        # x.shape: (N, C, H, W) or (N, C, D, H, W)
        forward_fun = (self.walnut_ray_trafo.apply_adjoint if self.adjoint else
                       self.walnut_ray_trafo.apply)
        # note: backward_fun is only an approximation to the transposed jacobian
        backward_fun = (self.walnut_ray_trafo.apply if self.adjoint else
                        self.walnut_ray_trafo.apply_adjoint)
        x_nc_flat = x.view(-1, *x.shape[2:])
        y_nc_flat = []
        for x_i in x_nc_flat:
            y_i = NumpyFunction.apply(x_i, forward_fun, backward_fun)
            y_nc_flat.append(y_i)
        y = torch.stack(y_nc_flat)
        y = y.view(*x.shape[:2], *y.shape[1:])
        return y

def save_masked_ray_trafo_matrix(file_path, walnut_ray_trafo):
    assert walnut_ray_trafo.rotation is None
    assert walnut_ray_trafo.shift_z == 0.

    vol_in_mask = np.zeros(VOL_SZ[1:])

    ray_trafo_matrix = scipy.sparse.dok_matrix(
        (walnut_ray_trafo.num_projs_in_mask, VOL_SZ[1] * VOL_SZ[2]),
        dtype=np.float32)
    for i in tqdm(range(VOL_SZ[1] * VOL_SZ[2])):
        vol_in_mask[:] = 0.
        vol_in_mask[np.unravel_index(i, VOL_SZ[1:])] = 1.

        flat_projs_in_mask = walnut_ray_trafo.apply(vol_in_mask)
        non_zero_mask = flat_projs_in_mask != 0.
        ray_trafo_matrix[non_zero_mask, i] = flat_projs_in_mask[non_zero_mask]

    # matlab appears to load garbage values if stored as float32, related issue:
    # https://github.com/scipy/scipy/issues/4826#issuecomment-120951128
    ray_trafo_matrix = ray_trafo_matrix.astype(np.float64)

    scipy.io.savemat(
            file_path,
            {
                'ray_trafo_matrix': ray_trafo_matrix,
                'walnut_id': walnut_ray_trafo.walnut_id,
                'orbit_id': walnut_ray_trafo.orbit_id,
                'angular_sub_sampling': walnut_ray_trafo.angular_sub_sampling,
                'proj_col_sub_sampling': walnut_ray_trafo.proj_col_sub_sampling,
                'first_proj_col': walnut_ray_trafo.first_proj_col + 1,  # matlab ind
                'num_slices': walnut_ray_trafo.num_slices,
                'num_proj_rows': walnut_ray_trafo.num_proj_rows,
                'first_proj_row': walnut_ray_trafo.first_proj_row + 1,  # matlab ind
                'vol_mask_slice': np.array([  # matlab inds, first and last included
                        range(walnut_ray_trafo.num_slices)[
                                walnut_ray_trafo.vol_mask_slice][0] + 1,
                        range(walnut_ray_trafo.num_slices)[
                                walnut_ray_trafo.vol_mask_slice][-1] + 1]),
                'proj_mask': walnut_ray_trafo.proj_mask,
            })

def get_masked_ray_trafo_matrix(file_path):
    matrix = scipy.io.loadmat(
            file_path, variable_names=['ray_trafo_matrix'])[
                    'ray_trafo_matrix'].astype('float32')
    return matrix

def get_single_slice_ray_trafo_matrix_filename(
        walnut_id, orbit_id,
        angular_sub_sampling=DEFAULT_ANGULAR_SUB_SAMPLING,
        proj_col_sub_sampling=DEFAULT_PROJ_COL_SUB_SAMPLING):
    filename = (
            'single_slice_ray_trafo_matrix_walnut{:d}_orbit{:d}_ass{:d}'
                    .format(walnut_id, orbit_id, angular_sub_sampling))
    if proj_col_sub_sampling != 1:
        filename = filename + '_css{:d}'.format(proj_col_sub_sampling)
    filename = filename + '.mat'
    return filename

def save_single_slice_ray_trafo_matrix(
        output_path, data_path,
        walnut_id=DEFAULT_SINGLE_SLICE_WALNUT_ID,
        orbit_id=DEFAULT_SINGLE_SLICE_ORBIT_ID,
        angular_sub_sampling=DEFAULT_ANGULAR_SUB_SAMPLING,
        proj_col_sub_sampling=DEFAULT_PROJ_COL_SUB_SAMPLING):

    walnut_ray_trafo = get_single_slice_ray_trafo(
            data_path=data_path, walnut_id=walnut_id, orbit_id=orbit_id,
            angular_sub_sampling=angular_sub_sampling,
            proj_col_sub_sampling=proj_col_sub_sampling)

    filename = get_single_slice_ray_trafo_matrix_filename(
            walnut_id=walnut_id, orbit_id=orbit_id,
            angular_sub_sampling=angular_sub_sampling,
            proj_col_sub_sampling=proj_col_sub_sampling)

    save_masked_ray_trafo_matrix(os.path.join(output_path, filename),
                          walnut_ray_trafo)

def get_single_slice_ray_trafo_matrix(
        path, walnut_id, orbit_id,
        angular_sub_sampling=DEFAULT_ANGULAR_SUB_SAMPLING,
        proj_col_sub_sampling=DEFAULT_PROJ_COL_SUB_SAMPLING):

    filename = get_single_slice_ray_trafo_matrix_filename(
            walnut_id, orbit_id,
            angular_sub_sampling=angular_sub_sampling,
            proj_col_sub_sampling=proj_col_sub_sampling)

    matrix = get_masked_ray_trafo_matrix(os.path.join(path, filename))
    return matrix

# def get_src_z(data_path, walnut_id, orbit_id):
#     data_path_full = os.path.join(data_path, 'Walnut{}'.format(walnut_id),
#                                   'Projections', 'tubeV{}'.format(orbit_id))
#     vecs = np.loadtxt(os.path.join(data_path_full, VECS_NAME))

#     src_z = vecs[0, 2]
#     assert np.all(vecs[:, 2] == src_z)

#     return src_z

# def get_det_z(data_path, walnut_id, orbit_id):
#     data_path_full = os.path.join(data_path, 'Walnut{}'.format(walnut_id),
#                                   'Projections', 'tubeV{}'.format(orbit_id))
#     vecs = np.loadtxt(os.path.join(data_path_full, VECS_NAME))

#     det_z = vecs[0, 5]
#     assert np.all(vecs[:, 5] == det_z)

#     return det_z

# def get_det_z_spacing(data_path, walnut_id, orbit_id):
#     data_path_full = os.path.join(data_path, 'Walnut{}'.format(walnut_id),
#                                   'Projections', 'tubeV{}'.format(orbit_id))
#     vecs = np.loadtxt(os.path.join(data_path_full, VECS_NAME))

#     det_z_spacing = vecs[0, 11]
#     assert np.all(vecs[:, 11] == det_z_spacing)

#     return det_z_spacing
