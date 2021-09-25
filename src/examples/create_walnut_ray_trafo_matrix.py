import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy
from dataset.walnuts import save_single_slice_ray_trafo_matrix

ANGULAR_SUB_SAMPLING = 10

WALNUT_ID = 1
ORBIT_ID = 2

DATA_PATH = '/localdata/Walnuts/'
OUTPUT_PATH = DATA_PATH

save_single_slice_ray_trafo_matrix(
        output_path=OUTPUT_PATH, data_path=DATA_PATH,
        walnut_id=WALNUT_ID, orbit_id=ORBIT_ID,
        angular_sub_sampling=ANGULAR_SUB_SAMPLING)
