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
import scipy
from tqdm import tqdm

VOXEL_PER_MM = 10
DEFAULT_ANGULAR_SUB_SAMPLING = 10
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
    }
}

DEFAULT_SINGLE_SLICE_WALNUT_ID = 1
DEFAULT_SINGLE_SLICE_ORBIT_ID = 2


def get_vol_geom(num_slices=-1):
    vol_geom = astra.create_vol_geom(
            (VOL_SZ[1],
             VOL_SZ[2],
             VOL_SZ[0] if num_slices == -1 else num_slices))
    vol_geom['option']['WindowMinX'] = vol_geom['option']['WindowMinX'] * VOX_SZ
    vol_geom['option']['WindowMaxX'] = vol_geom['option']['WindowMaxX'] * VOX_SZ
    vol_geom['option']['WindowMinY'] = vol_geom['option']['WindowMinY'] * VOX_SZ
    vol_geom['option']['WindowMaxY'] = vol_geom['option']['WindowMaxY'] * VOX_SZ
    vol_geom['option']['WindowMinZ'] = vol_geom['option']['WindowMinZ'] * VOX_SZ
    vol_geom['option']['WindowMaxZ'] = vol_geom['option']['WindowMaxZ'] * VOX_SZ

    return vol_geom

def get_proj_geom(data_path, walnut_id, orbit_id,
                  angular_sub_sampling=DEFAULT_ANGULAR_SUB_SAMPLING,
                  num_proj_rows=-1, first_proj_row=-1, rotation=None,
                  shift_z=0., return_vecs=False):
    data_path_full = os.path.join(data_path, 'Walnut{}'.format(walnut_id),
                                  'Projections', 'tubeV{}'.format(orbit_id))

    if num_proj_rows == -1:
        num_proj_rows = PROJS_ROWS

    if first_proj_row == -1:
        first_proj_row = (PROJS_ROWS - num_proj_rows) // 2

    vecs_all = np.loadtxt(os.path.join(data_path_full, VECS_NAME))
    vecs = vecs_all[range(0, MAX_NUM_ANGLES, angular_sub_sampling)]

    # determine the detector center, such that the first detector row in this
    # geometry coincides with row `first_proj_row` of the full geometry with
    # num_proj_rows=PROJS_ROWS
    vecs[:, 3:6] += (-(PROJS_ROWS - 1) / 2 + first_proj_row -
                     (-(num_proj_rows - 1) / 2)) * vecs[:, 9:12]

    # apply a scipy rotation (globally) if specified
    if rotation is not None:
        for i in range(0, 12, 3):
            vecs[:, i:i+3] = rotation.apply(vecs[:, i:i+3])

    # apply a shift in z direction if specified
    vecs[:, 2] += shift_z
    vecs[:, 5] += shift_z

    proj_geom = astra.create_proj_geom(
            'cone_vec', num_proj_rows, PROJS_COLS, vecs)

    return proj_geom if not return_vecs else (proj_geom, vecs)

def get_projection_data(data_path, walnut_id, orbit_id,
                        angular_sub_sampling=DEFAULT_ANGULAR_SUB_SAMPLING):
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
    projs = np.ascontiguousarray(projs)

    return projs

def get_ground_truth(data_path, walnut_id, orbit_id, slice_ind):
    slice_path = os.path.join(
            data_path, 'Walnut{}'.format(walnut_id), 'Reconstructions',
            'full_AGD_{}_{:06}.tiff'.format(GT_NB_ITER, slice_ind))
    gt = imageio.imread(slice_path)

    return gt

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
        angular_sub_sampling=DEFAULT_ANGULAR_SUB_SAMPLING):

    single_slice_config = SINGLE_SLICE_CONFIGS.get(walnut_id, {}).get(orbit_id)
    if single_slice_config is None:
        raise ValueError('No single slice ray trafo configuration known for '
                         'walnut_id={:d}, orbit_id={:d}'.format(
                                 walnut_id, orbit_id))
    num_slices = single_slice_config['num_slices']
    slice_offset = single_slice_config['slice_offset']
    num_proj_rows = single_slice_config['num_proj_rows']
    first_proj_row = single_slice_config['first_proj_row']

    walnut_ray_trafo = WalnutRayTrafo(
            data_path=data_path,
            walnut_id=walnut_id, orbit_id=orbit_id,
            angular_sub_sampling=angular_sub_sampling,
            num_slices=num_slices, num_proj_rows=num_proj_rows,
            first_proj_row=first_proj_row,
            vol_mask_slice=(num_slices - 1) // 2 + slice_offset,
            proj_mask_select_k_rows=1,
            )

    return walnut_ray_trafo

class WalnutRayTrafo:
    def __init__(self, data_path, walnut_id, orbit_id,
                 angular_sub_sampling=DEFAULT_ANGULAR_SUB_SAMPLING,
                 num_slices=VOL_SZ[0], num_proj_rows=PROJS_ROWS,
                 first_proj_row=0, rotation=None, shift_z=0.,
                 vol_mask_slice=None, proj_mask_select_k_rows=None):
        self.data_path = data_path
        self.walnut_id = walnut_id
        self.orbit_id = orbit_id
        self.angular_sub_sampling = angular_sub_sampling
        assert num_slices % 2 == 1  # each slice then matches one in full volume
        self.num_slices = num_slices
        self.num_proj_rows = num_proj_rows
        self.first_proj_row = first_proj_row
        self.rotation = rotation
        self.shift_z = shift_z
        if isinstance(vol_mask_slice, int):
            self.vol_mask_slice = slice(vol_mask_slice, vol_mask_slice+1)
        else:
            assert vol_mask_slice.step is None or vol_mask_slice.step == 1
            self.vol_mask_slice = vol_mask_slice
        self.proj_mask_select_k_rows = proj_mask_select_k_rows

        self.num_angles = ceil(MAX_NUM_ANGLES / self.angular_sub_sampling)

        self.vol_geom = get_vol_geom(num_slices=self.num_slices)
        self.proj_geom, self.vecs = get_proj_geom(
                data_path=self.data_path,
                walnut_id=self.walnut_id, orbit_id=self.orbit_id,
                angular_sub_sampling=self.angular_sub_sampling,
                num_proj_rows=self.num_proj_rows,
                first_proj_row=self.first_proj_row,
                rotation=self.rotation, shift_z=self.shift_z,
                return_vecs=True)

        self.vol_geom_full = get_vol_geom(num_slices=VOL_SZ[0])
        self.proj_geom_full = get_proj_geom(
                data_path=self.data_path,
                walnut_id=self.walnut_id, orbit_id=self.orbit_id,
                angular_sub_sampling=self.angular_sub_sampling,
                num_proj_rows=PROJS_ROWS,
                first_proj_row=0,
                rotation=self.rotation, shift_z=self.shift_z)

        self.vol_shape = (self.num_slices,) + VOL_SZ[1:]
        self.proj_shape = (self.num_proj_rows, self.num_angles, PROJS_COLS)

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
                                                    full_input=True)

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
        projs_test_full = np.ones((PROJS_ROWS, self.num_angles, PROJS_COLS),
                                  dtype=np.float32)
        projs_test_full[
                self.first_proj_row:self.first_proj_row+self.num_proj_rows] = 0.
        vol_x = self.bp3d(projs_test_full, full_input=True)
        assert np.all(vol_x[self.vol_mask_slice] == 0.)

    def assert_vol_slices_suffice(self):
        vol_test_full = np.ones(VOL_SZ, dtype=np.float32)
        first = (VOL_SZ[0] - self.num_slices) // 2
        vol_test_full[first:first+self.num_slices] = 0.
        projs = self.fp3d(vol_test_full, full_input=True)
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
        projs = projs_full[
                self.first_proj_row:self.first_proj_row+self.num_proj_rows]
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

    def fp3d(self, vol_x, full_input=False):
        vol_geom = self.vol_geom_full if full_input else self.vol_geom

        vol_x = np.ascontiguousarray(vol_x, dtype=np.float32)
        vol_id = astra.data3d.link('-vol', vol_geom, vol_x)

        projs = np.zeros(self.proj_shape, dtype=np.float32)
        proj_id = astra.data3d.link('-sino', self.proj_geom, projs)

        cfg_fp = astra.astra_dict('FP3D_CUDA')
        cfg_fp['VolumeDataId'] = vol_id
        cfg_fp['ProjectionDataId'] = proj_id
        alg_id = astra.algorithm.create(cfg_fp)

        astra.algorithm.run(alg_id)

        astra.algorithm.delete(alg_id)
        astra.data3d.delete(proj_id)
        astra.data3d.delete(vol_id)

        return projs

    def bp3d(self, projs, full_input=False):
        proj_geom = self.proj_geom_full if full_input else self.proj_geom

        projs = np.ascontiguousarray(projs, dtype=np.float32)
        proj_id = astra.data3d.link('-sino', proj_geom, projs)

        vol_x = np.zeros(self.vol_shape, dtype=np.float32)
        vol_id = astra.data3d.link('-vol', self.vol_geom, vol_x)

        cfg_bp = astra.astra_dict('BP3D_CUDA')
        cfg_bp['ReconstructionDataId'] = vol_id
        cfg_bp['ProjectionDataId'] = proj_id
        alg_id = astra.algorithm.create(cfg_bp)

        astra.algorithm.run(alg_id)

        astra.algorithm.delete(alg_id)
        astra.data3d.delete(proj_id)
        astra.data3d.delete(vol_id)

        return vol_x

    def fdk(self, projs, full_input=False):
        proj_geom = self.proj_geom_full if full_input else self.proj_geom

        projs = np.ascontiguousarray(projs, dtype=np.float32)
        proj_id = astra.data3d.link('-sino', proj_geom, projs)

        vol_x = np.zeros(self.vol_shape, dtype=np.float32)
        vol_id = astra.data3d.link('-vol', self.vol_geom, vol_x)

        cfg_fdk = astra.astra_dict('FDK_CUDA')
        cfg_fdk['ReconstructionDataId'] = vol_id
        cfg_fdk['ProjectionDataId'] = proj_id
        cfg_fdk['option'] = {}
        cfg_fdk['option']['ShortScan'] = False
        alg_id = astra.algorithm.create(cfg_fdk)

        # run FDK algorithm
        astra.algorithm.run(alg_id, 1)

        # release memory allocated by ASTRA structures
        astra.algorithm.delete(alg_id)
        astra.data3d.delete(proj_id)
        astra.data3d.delete(vol_id)

        return vol_x

    def apply(self, vol_in_mask, padding_mode='edge'):
        vol_x = self.vol_from_vol_in_mask(vol_in_mask,
                                          padding_mode=padding_mode)
        projs = self.fp3d(vol_x)
        flat_projs_in_mask = self.flat_projs_in_mask(projs)
        return flat_projs_in_mask

    def apply_adjoint(self, flat_projs_in_mask, padding_mode='edge', squeeze=False):
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

def save_ray_trafo_matrix(file_path, walnut_ray_trafo):
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

def get_ray_trafo_matrix(file_path):
    matrix = scipy.io.loadmat(
            file_path, variable_names=['ray_trafo_matrix'])[
                    'ray_trafo_matrix'].astype('float32')
    return matrix

def get_single_slice_ray_trafo_matrix_filename(
        walnut_id, orbit_id, angular_sub_sampling=DEFAULT_ANGULAR_SUB_SAMPLING):
    filename = (
            'single_slice_ray_trafo_matrix_walnut{:d}_orbit{:d}_ass{:d}.mat'
                    .format(walnut_id, orbit_id, angular_sub_sampling))
    return filename

def save_single_slice_ray_trafo_matrix(
        output_path, data_path,
        walnut_id=DEFAULT_SINGLE_SLICE_WALNUT_ID,
        orbit_id=DEFAULT_SINGLE_SLICE_ORBIT_ID,
        angular_sub_sampling=DEFAULT_ANGULAR_SUB_SAMPLING):

    walnut_ray_trafo = get_single_slice_ray_trafo(
            data_path=data_path, walnut_id=walnut_id, orbit_id=orbit_id,
            angular_sub_sampling=angular_sub_sampling)

    filename = get_single_slice_ray_trafo_matrix_filename(
            walnut_id=walnut_id, orbit_id=orbit_id,
            angular_sub_sampling=angular_sub_sampling)

    save_ray_trafo_matrix(os.path.join(output_path, filename),
                          walnut_ray_trafo)

def get_single_slice_ray_trafo_matrix(
        path, walnut_id, orbit_id,
        angular_sub_sampling=DEFAULT_ANGULAR_SUB_SAMPLING):

    filename = get_single_slice_ray_trafo_matrix_filename(
            walnut_id, orbit_id, angular_sub_sampling)

    matrix = get_ray_trafo_matrix(os.path.join(path, filename))
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
