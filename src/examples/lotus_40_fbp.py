import numpy as np
import odl
from dataset.lotus import (
        get_ray_trafo_matrix, get_sinogram, get_proj_space128, NUM_ANGLES,
        NUM_DET_PIXELS128)
from util.matrix_ray_trafo import MatrixRayTrafo
from util.fbp import get_fbp_filter_op
import matplotlib.pyplot as plt

size = 128
num_angles = 40
angles_subsampling = {
    'start': 0,
    'stop': None,
    'step': 3
}
num_angles_orig = NUM_ANGLES
num_det_pixels = NUM_DET_PIXELS128

normalize = False

matrix = get_ray_trafo_matrix(
        '/localdata/data/FIPS_Lotus/LotusData128.mat', normalize=normalize)
matrix = matrix.reshape(
        (num_angles_orig, num_det_pixels * size * size),
        order='C').tocsc()
matrix = matrix[angles_subsampling['start']:
                angles_subsampling['stop']:
                angles_subsampling['step'], :]
matrix = matrix.reshape((num_angles * num_det_pixels, size * size),
                        order='C').tocsc()
matrix_ray_trafo = MatrixRayTrafo(matrix,
        (size, size), (num_angles, num_det_pixels))

proj_space_orig = get_proj_space128()
angles_coord_vector = proj_space_orig.grid.coord_vectors[0][
        angles_subsampling['start']:
        angles_subsampling['stop']:
        angles_subsampling['step']]
proj_space = odl.uniform_discr_frompartition(
        odl.uniform_partition_fromgrid(
                odl.RectGrid(angles_coord_vector,
                             proj_space_orig.grid.coord_vectors[1])))
# scaling_factor = 1.
scaling_factor = 1.11379899631 / 115.664419887  # = 0.009629573185929967
                                                # empirical factor, replace by
                                                # correct one if known

filter_op = get_fbp_filter_op(proj_space, scaling_factor=scaling_factor,
        padding=True, filter_type='Ram-Lak', frequency_scaling=1.0)

sinogram = get_sinogram(
        '/localdata/data/FIPS_Lotus/LotusData128.mat',
        normalize=normalize,
        scale_to_fbp_max_1=False)
sinogram = sinogram[angles_subsampling['start']:
                    angles_subsampling['stop']:
                    angles_subsampling['step'], :]

y = np.asarray(filter_op(sinogram))
reco = matrix_ray_trafo.apply_adjoint(y)

print('scale_to_fbp_max_1_factor:', 1./np.max(reco))

sinogram2 = matrix_ray_trafo.apply(reco)

plt.figure()
plt.imshow(sinogram)
plt.figure()
plt.imshow(y)
plt.figure()
plt.imshow(reco)
plt.figure()
plt.imshow(sinogram2)

print('mean of sinogram:\n', np.mean(sinogram))
print('mean of filtered sinogram:\n', np.mean(y))
print('mean of Ram-Lak FBP reconstruction:\n', np.mean(reco))
print('mean of forward projection of Ram-Lak FBP reconstruction:\n',
      np.mean(sinogram2))
