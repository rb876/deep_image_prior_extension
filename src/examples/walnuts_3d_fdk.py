import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio
from dataset.walnuts import (
        VOL_SZ, PROJS_COLS, get_projection_data, WalnutRayTrafo,
        get_ground_truth_3d, down_sample_vol)

ANGULAR_SUB_SAMPLING = 10
PROJ_ROW_SUB_SAMPLING = 3
PROJ_COL_SUB_SAMPLING = 3
VOL_DOWN_SAMPLING = 3

WALNUT_ID = 1
ORBIT_ID = 2

DATA_PATH = '/localdata/Walnuts/'

walnut_ray_trafo = WalnutRayTrafo(
        data_path=DATA_PATH, walnut_id=WALNUT_ID, orbit_id=ORBIT_ID,
        angular_sub_sampling=ANGULAR_SUB_SAMPLING,
        proj_row_sub_sampling=PROJ_ROW_SUB_SAMPLING,
        proj_col_sub_sampling=PROJ_COL_SUB_SAMPLING,
        vol_down_sampling=VOL_DOWN_SAMPLING)

# test FDK reconstruction
projs = get_projection_data(
        data_path=DATA_PATH, walnut_id=WALNUT_ID, orbit_id=ORBIT_ID,
        angular_sub_sampling=ANGULAR_SUB_SAMPLING,
        proj_row_sub_sampling=PROJ_ROW_SUB_SAMPLING,
        proj_col_sub_sampling=PROJ_COL_SUB_SAMPLING)
reco = walnut_ray_trafo.fdk(projs)

gt_3d = get_ground_truth_3d(
        data_path=DATA_PATH, walnut_id=WALNUT_ID, orbit_id=ORBIT_ID)
gt_3d = down_sample_vol(gt_3d, down_sampling=VOL_DOWN_SAMPLING)
slice_ind = gt_3d.shape[0] // 2

gt_slice = gt_3d[slice_ind]
reco_slice = reco[slice_ind]

fig, axs = plt.subplots(1, 3, figsize=(9, 3))

im0 = axs[0].imshow(reco_slice, cmap='gray')
fig.colorbar(im0, ax=axs[0])
axs[0].set_title('FDK')
axs[0].set_xlabel('PSNR: {:.4f} dB'.format(peak_signal_noise_ratio(gt_slice, reco_slice, data_range=gt_slice.max()-gt_slice.min())))
im1 = axs[1].imshow(reco_slice-gt_slice, cmap='gray')
fig.colorbar(im1, ax=axs[1])
axs[1].set_title('diff')
axs[1].set_xlabel('mae: {:.7f}'.format(np.mean(np.abs(reco_slice-gt_slice))))
im2 = axs[2].imshow(gt_slice, cmap='gray')
fig.colorbar(im2, ax=axs[2])
axs[2].set_title('ground truth')

plt.show()
