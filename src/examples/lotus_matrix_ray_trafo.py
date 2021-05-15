import numpy as np
import torch
from tqdm import tqdm
from dataset.lotus import (
        get_ray_trafo_matrix, NUM_ANGLES, NUM_DET_PIXELS128)
from util.matrix_ray_trafo import MatrixRayTrafo
from util.matrix_ray_trafo_torch import get_matrix_ray_trafo_module

size = 128
num_angles = NUM_ANGLES
num_det_pixels = NUM_DET_PIXELS128
matrix = get_ray_trafo_matrix(
        '/localdata/data/FIPS_Lotus/LotusData128.mat').astype('float32')

sparse = True
matrix_ray_trafo = MatrixRayTrafo(matrix if sparse else matrix.todense(),
        (size, size), (num_angles, num_det_pixels))
matrix_ray_trafo_mod = get_matrix_ray_trafo_module(matrix,
        (size, size), (num_angles, num_det_pixels), sparse=sparse)

x_np = np.random.random((1, 1, size, size)).astype('float32')
x = torch.from_numpy(x_np)

matrix_ray_trafo_mod.cuda()
x = x.cuda()

# scipy/numpy is faster on CPU, torch is faster on GPU
for i in tqdm(range(1000), desc='scipy'):
    y_np = matrix_ray_trafo.apply(x_np)
for i in tqdm(range(1000), desc='torch'):
    y = matrix_ray_trafo_mod(x)
