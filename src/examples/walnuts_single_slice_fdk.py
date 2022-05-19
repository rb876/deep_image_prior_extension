import numpy as np
import matplotlib.pyplot as plt
from dataset.walnuts import (
        VOL_SZ, PROJS_COLS, get_projection_data,
        get_single_slice_ray_trafo, get_single_slice_ind, get_ground_truth)

ANGULAR_SUB_SAMPLING = 10

WALNUT_ID = 1
ORBIT_ID = 2

DATA_PATH = '/localdata/Walnuts/'

walnut_ray_trafo = get_single_slice_ray_trafo(
        data_path=DATA_PATH, walnut_id=WALNUT_ID, orbit_id=ORBIT_ID,
        angular_sub_sampling=ANGULAR_SUB_SAMPLING)

print('vol_slice_contributing_to_masked_projs',
        walnut_ray_trafo.get_vol_slice_contributing_to_masked_projs())
print('proj_slice_contributing_to_masked_vol',
        walnut_ray_trafo.get_proj_slice_contributing_to_masked_vol())

vol_in_mask = np.ones((1,) + VOL_SZ[1:])
vol_x = np.zeros((walnut_ray_trafo.num_slices,) + VOL_SZ[1:])
vol_x[walnut_ray_trafo.vol_mask_slice] = vol_in_mask
projs = walnut_ray_trafo.fp3d(vol_x)
backprojection_mask = walnut_ray_trafo.bp3d(
        walnut_ray_trafo.proj_mask.astype(np.float32))

# visualize restriction of geometry
angle_indices = range(0, walnut_ray_trafo.num_angles,
                      walnut_ray_trafo.num_angles // 4)
view_proj_cols_values = [
        slice(None, 40),
        slice(PROJS_COLS//2-20, PROJS_COLS//2+20),
        slice(-40, None)]
view_vol_cols_values = [
        slice(None, 40),
        slice(VOL_SZ[2]//2-20, VOL_SZ[2]//2+20),
        slice(-40, None)]

fig, ax = plt.subplots(
        len(angle_indices),
        2 * len(view_proj_cols_values) + len(view_vol_cols_values),
        gridspec_kw={'hspace': 0.5})

for i, angle_index in enumerate(angle_indices):
    col = 0
    for view_proj_cols in view_proj_cols_values:
        ax[i, col].imshow(projs[:, angle_index, view_proj_cols])
        ax[i, col].set_title('FP of vol mask\ncolumn\n{}'.format(
                range(PROJS_COLS)[view_proj_cols]))
        col += 1
    for view_proj_cols in view_proj_cols_values:
        ax[i, col].imshow(
                walnut_ray_trafo.proj_mask[:, angle_index, view_proj_cols])
        ax[i, col].set_title('proj mask\ncolumn\n{}'.format(
                range(PROJS_COLS)[view_proj_cols]))
        col += 1
    for view_vol_cols in view_vol_cols_values:
        ax[i, col].imshow(backprojection_mask[:, VOL_SZ[1] // 2, view_vol_cols])
        ax[i, col].set_title('BP of proj mask\ncolumn\n{}'.format(
                range(VOL_SZ[2])[view_vol_cols]))
        col += 1
    ax[i, 0].set_ylabel('angle index {:d}'.format(angle_index))

plt.show()


# test FDK reconstruction from restricted projections
projs_full = get_projection_data(data_path=DATA_PATH,
                                 walnut_id=WALNUT_ID, orbit_id=ORBIT_ID,
                                 angular_sub_sampling=ANGULAR_SUB_SAMPLING)
projs = walnut_ray_trafo.projs_from_full(projs_full)
flat_projs_in_mask = walnut_ray_trafo.flat_projs_in_mask(projs)
projs_padded = walnut_ray_trafo.projs_from_flat_projs_in_mask(
        flat_projs_in_mask)
reco_projs_full = walnut_ray_trafo.vol_in_mask(
        walnut_ray_trafo.fdk(projs_full, proj_geom=walnut_ray_trafo.ray_trafo_full.proj_geom, proj_geom_no_sub_sampling=walnut_ray_trafo.ray_trafo_full.proj_geom_no_sub_sampling))[0]
reco_projs = walnut_ray_trafo.vol_in_mask(
        walnut_ray_trafo.fdk(projs))[0]
reco_projs_padded = walnut_ray_trafo.vol_in_mask(
        walnut_ray_trafo.fdk(projs_padded))[0]
# reco_projs_padded = walnut_ray_trafo.apply_fdk(
#         flat_projs_in_mask)[0]

slice_ind = get_single_slice_ind(
        data_path=DATA_PATH, walnut_id=WALNUT_ID, orbit_id=ORBIT_ID)
gt = get_ground_truth(
        data_path=DATA_PATH, walnut_id=WALNUT_ID, orbit_id=ORBIT_ID,
        slice_ind=slice_ind)

recos = [reco_projs_padded,
         reco_projs,
         reco_projs_full]
titles = ['masked and padded',
          'rows subset',
          'full']
reference_idx = 2

fig, ax = plt.subplots(2, len(recos))

for i, (reco, title) in enumerate(zip(recos, titles)):
    im = ax[0, i].imshow(reco.T, cmap='gray')
    ax[0, i].set_title(title)
    plt.colorbar(im, ax=ax[0, i])
    if i != reference_idx:
        im = ax[1, i].imshow((reco - recos[reference_idx]).T, cmap='gray')
        ax[1, i].set_title('diff to {}'.format(titles[reference_idx]))
        plt.colorbar(im, ax=ax[1, i])
    else:
        im = ax[1, i].imshow(gt.T, cmap='gray')
        ax[1, i].set_title('ground truth')
        plt.colorbar(im, ax=ax[1, i])

plt.show()
