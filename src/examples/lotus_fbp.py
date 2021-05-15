import numpy as np
from dataset.lotus import (
        get_ray_trafo_matrix, get_sinogram, get_proj_space128, NUM_ANGLES,
        NUM_DET_PIXELS128)
from util.matrix_ray_trafo import MatrixRayTrafo
from util.fbp import get_fbp_filter_op
import matplotlib.pyplot as plt

size = 128
num_angles = NUM_ANGLES
num_det_pixels = NUM_DET_PIXELS128

matrix = get_ray_trafo_matrix(
        '/localdata/data/FIPS_Lotus/LotusData128.mat', normalize=False)
matrix_ray_trafo = MatrixRayTrafo(matrix,
        (size, size), (num_angles, num_det_pixels))

proj_space = get_proj_space128()
scaling_factor = 1.11382176502 / 347.334  # empirical factor, replace by
                                          # correct one if known

filter_op = get_fbp_filter_op(proj_space, scaling_factor=scaling_factor,
        padding=True, filter_type='Ram-Lak', frequency_scaling=1.0)

sinogram = get_sinogram(
        '/localdata/data/FIPS_Lotus/LotusData128.mat')

y = np.asarray(filter_op(sinogram))
reco = matrix_ray_trafo.apply_adjoint(y)

sinogram2 = matrix_ray_trafo.apply(reco)

plt.figure()
plt.imshow(sinogram)
plt.figure()
plt.imshow(y)
plt.figure()
plt.imshow(reco)
plt.figure()
plt.imshow(sinogram2)

print(np.mean(sinogram))
print(np.mean(y))
print(np.mean(reco))
print(np.mean(sinogram2))
