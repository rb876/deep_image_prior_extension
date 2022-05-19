import unittest
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio
import astra
from dataset.walnuts import (
        get_projection_data, WalnutRayTrafo, get_ground_truth_3d,
        sub_sample_proj, down_sample_vol)

DATA_PATH = '/localdata/Walnuts/'

class TestDatasetWalnuts(unittest.TestCase):
    def __init__(self, *args,
            data_path=DATA_PATH, plot_images_before_failing=True,
            plot_images_always=False, **kwargs):
        super().__init__(*args, **kwargs)

        self.data_path = data_path

        self.walnut_id = 1
        self.orbit_id = 2

        self.plot_images_before_failing = plot_images_before_failing
        self.plot_images_always = plot_images_always

    def plot_images(self, im, im_ref, inds=(None, None, None)):
        import matplotlib.pyplot as plt
        if im.ndim == 2:
            fix, axs = plt.subplots(1, 3, figsize=(10, 3.5))
            axs[0].imshow(im)
            axs[1].imshow(im_ref)
            axs[2].imshow(im-im_ref)
            plt.show()
        else:
            ind2 = im.shape[2] // 2 if inds[2] is None else inds[2]
            ind1 = im.shape[1] // 2 if inds[1] is None else inds[1]
            ind0 = im.shape[0] // 2 if inds[0] is None else inds[0]
            fix, axs = plt.subplots(3, 3, figsize=(10, 6))
            axs[0, 0].imshow(im[:, :, ind2])
            axs[0, 1].imshow(im_ref[:, :, ind2])
            axs[0, 2].imshow((im-im_ref)[:, :, ind2])
            axs[1, 0].imshow(im[:, ind1, :].T)
            axs[1, 1].imshow(im_ref[:, ind1, :].T)
            axs[1, 2].imshow((im-im_ref)[:, ind1, :].T)
            axs[2, 0].imshow(im[ind0, :, :].T)
            axs[2, 1].imshow(im_ref[ind0, :, :].T)
            axs[2, 2].imshow((im-im_ref)[ind0, :, :].T)
            plt.show()

    def assertImagesEqual(self, im, im_ref, rtol=1e-05, atol=1e-08):
        success = np.allclose(im, im_ref, rtol=rtol, atol=atol)  # TODO

        if self.plot_images_always or (not success and self.plot_images_before_failing):
            if im.ndim == 3:
                fail_mask = np.abs(im-im_ref) / (atol + rtol * np.abs(im_ref)) > 1.
                ind2 = np.argmax(np.sum(fail_mask, axis=(0, 1)))
                ind1 = np.argmax(np.sum(fail_mask, axis=(0, 2)))
                ind0 = np.argmax(np.sum(fail_mask, axis=(1, 2)))
            self.plot_images(im, im_ref, inds=(ind0, ind1, ind2))

        self.assertTrue(success)

    def assertMSELessEqual(self, im, im_ref, max_mse):
        mse = np.mean(np.square(im - im_ref))
        success = mse <= max_mse

        if self.plot_images_always or (not success and self.plot_images_before_failing):
            self.plot_images(im, im_ref)

        self.assertLessEqual(mse, max_mse)

    def test_walnut_ray_trafo_apply_vs_walnut_reconstruction_codes_geometry(self):

            angular_sub_sampling = 10

            kwargs = dict(
                    data_path=self.data_path, walnut_id=self.walnut_id,
                    orbit_id=self.orbit_id,
                    angular_sub_sampling=angular_sub_sampling,
                    proj_row_sub_sampling=1, proj_col_sub_sampling=1)

            walnut_ray_trafo = WalnutRayTrafo(**kwargs)

            gt = get_ground_truth_3d(  # could be random as well
                    self.data_path, self.walnut_id, orbit_id=self.orbit_id)
            gt = np.ascontiguousarray(gt, dtype=np.float32)

            fp = walnut_ray_trafo.apply(gt)

            # the following code is adapted from https://github.com/cicwi/WalnutReconstructionCodes/blob/master/GroundTruthReconstruction.py
            voxel_per_mm = 10
            vecs_name = 'scan_geom_corrected.geom'
            projs_rows = 972
            projs_cols = 768
            vecs_orbit = np.loadtxt(
                    os.path.join(os.path.join(self.data_path,
                            'Walnut{:d}'.format(self.walnut_id),
                            'Projections', 'tubeV{:d}'.format(self.orbit_id)),
                            vecs_name))
            vecs = vecs_orbit[range(0, 1200, angular_sub_sampling)]

            # size of the reconstruction volume in voxels
            vol_sz  = 3*(50 * voxel_per_mm + 1,)
            # size of a cubic voxel in mm
            vox_sz  = 1/voxel_per_mm

            # we need to specify the details of the reconstruction space to ASTRA
            # this is done by a "volume geometry" type of structure, in the form of a Python dictionary
            # by default, ASTRA assumes a voxel size of 1, we need to scale the reconstruction space here by the actual voxel size
            vol_geom = astra.create_vol_geom(vol_sz)
            vol_geom['option']['WindowMinX'] = vol_geom['option']['WindowMinX'] * vox_sz
            vol_geom['option']['WindowMaxX'] = vol_geom['option']['WindowMaxX'] * vox_sz
            vol_geom['option']['WindowMinY'] = vol_geom['option']['WindowMinY'] * vox_sz
            vol_geom['option']['WindowMaxY'] = vol_geom['option']['WindowMaxY'] * vox_sz
            vol_geom['option']['WindowMinZ'] = vol_geom['option']['WindowMinZ'] * vox_sz
            vol_geom['option']['WindowMaxZ'] = vol_geom['option']['WindowMaxZ'] * vox_sz

            # we need to specify the details of the projection space to ASTRA
            # this is done by a "projection geometry" type of structure, in the form of a Python dictionary
            proj_geom = astra.create_proj_geom('cone_vec', projs_rows, projs_cols, vecs)

            fp_walnut_reconstruction_codes = np.zeros((projs_rows, len(vecs), projs_cols), dtype=np.float32)

            # register both volume and projection geometries and arrays to ASTRA
            vol_id = astra.data3d.link('-vol', vol_geom, gt)
            sino_id = astra.data3d.link('-sino', proj_geom, fp_walnut_reconstruction_codes)

            cfg_fp = astra.astra_dict('FP3D_CUDA')
            cfg_fp['VolumeDataId'] = vol_id
            cfg_fp['ProjectionDataId'] = sino_id
            alg_id = astra.algorithm.create(cfg_fp)

            astra.algorithm.run(alg_id)

            astra.algorithm.delete(alg_id)
            astra.data3d.delete(sino_id)
            astra.data3d.delete(vol_id)

            self.assertImagesEqual(fp, fp_walnut_reconstruction_codes)

    def test_walnut_ray_trafo_proj_sub_sampling_apply(self):

        gt = get_ground_truth_3d(  # could be random as well
                self.data_path, self.walnut_id, orbit_id=self.orbit_id)

        angular_sub_sampling = 10

        # for reference 1: dense measured projections
        projs_no_sub_sampling = get_projection_data(
                data_path=self.data_path, walnut_id=self.walnut_id,
                orbit_id=self.orbit_id,
                angular_sub_sampling=angular_sub_sampling,
                proj_row_sub_sampling=1, proj_col_sub_sampling=1)

        # for reference 2: dense forward projection of ground truth
        walnut_ray_trafo_no_sub_sampling = WalnutRayTrafo(
                data_path=self.data_path, walnut_id=self.walnut_id,
                orbit_id=self.orbit_id,
                angular_sub_sampling=angular_sub_sampling,
                proj_row_sub_sampling=1, proj_col_sub_sampling=1)
        fp_no_sub_sampling = walnut_ray_trafo_no_sub_sampling.apply(gt)

        for proj_row_sub_sampling, proj_col_sub_sampling in [
                (6, 6), (4, 3), (1, 1)]:

            walnut_ray_trafo = WalnutRayTrafo(
                    data_path=self.data_path, walnut_id=self.walnut_id,
                    orbit_id=self.orbit_id,
                    angular_sub_sampling=angular_sub_sampling,
                    proj_row_sub_sampling=proj_row_sub_sampling,
                    proj_col_sub_sampling=proj_col_sub_sampling)

            fp = walnut_ray_trafo.apply(gt)

            # reference 1: sub-sampled dense measured projections
            projs = sub_sample_proj(projs_no_sub_sampling,
                    factor_row=proj_row_sub_sampling,
                    factor_col=proj_col_sub_sampling)

            # reference 2: sub-sampled dense forward projection of ground truth
            fp_post = sub_sample_proj(fp_no_sub_sampling,
                    factor_row=proj_row_sub_sampling,
                    factor_col=proj_col_sub_sampling)

            self.assertMSELessEqual(fp, projs, 1.5e-4)
            self.assertImagesEqual(fp, fp_post, atol=6e-3)

    def test_walnut_ray_trafo_vol_down_sampling_apply(self):

        gt = get_ground_truth_3d(
                self.data_path, self.walnut_id, orbit_id=self.orbit_id)

        angular_sub_sampling = 10

        # reference 1: measured projections
        projs = get_projection_data(
                data_path=self.data_path, walnut_id=self.walnut_id,
                orbit_id=self.orbit_id,
                angular_sub_sampling=angular_sub_sampling,
                proj_row_sub_sampling=1, proj_col_sub_sampling=1)

        # reference 2: forward projection of full resolution ground truth
        walnut_ray_trafo_orig_res = WalnutRayTrafo(
                data_path=self.data_path, walnut_id=self.walnut_id,
                orbit_id=self.orbit_id,
                angular_sub_sampling=angular_sub_sampling,
                proj_row_sub_sampling=1, proj_col_sub_sampling=1)
        fp_orig_res = walnut_ray_trafo_orig_res.apply(gt)

        for vol_down_sampling, max_mse in [
                (5, 4.5e-4), (3, 2.5e-4), (1, 1.5e-4)]:

            walnut_ray_trafo = WalnutRayTrafo(
                    data_path=self.data_path, walnut_id=self.walnut_id,
                    orbit_id=self.orbit_id,
                    angular_sub_sampling=angular_sub_sampling,
                    proj_row_sub_sampling=1, proj_col_sub_sampling=1,
                    vol_down_sampling=vol_down_sampling)

            gt_down_sampled = down_sample_vol(
                    gt, down_sampling=vol_down_sampling)

            fp = walnut_ray_trafo.apply(gt_down_sampled)

            self.assertMSELessEqual(fp, projs, max_mse)
            self.assertImagesEqual(fp, fp_orig_res, rtol=4e-2, atol=2e-1)

    def test_walnut_ray_trafo_vol_down_sampling_fdk(self):

        angular_sub_sampling = 10

        gt = get_ground_truth_3d(
                self.data_path, self.walnut_id, orbit_id=self.orbit_id)

        # for input 1: measurements
        projs_no_sub_sampling = get_projection_data(
                data_path=self.data_path, walnut_id=self.walnut_id,
                orbit_id=self.orbit_id,
                angular_sub_sampling=angular_sub_sampling,
                proj_row_sub_sampling=1, proj_col_sub_sampling=1)

        # for input 2: forward projection of ground truth
        walnut_ray_trafo_orig_res = WalnutRayTrafo(
                data_path=self.data_path, walnut_id=self.walnut_id,
                orbit_id=self.orbit_id,
                angular_sub_sampling=angular_sub_sampling,
                proj_row_sub_sampling=1, proj_col_sub_sampling=1)
        fp_no_sub_sampling = walnut_ray_trafo_orig_res.apply(gt)

        for (proj_row_sub_sampling, proj_col_sub_sampling, vol_down_sampling,
             max_mse) in [
                (6, 6, 5, 2.1e-5), (4, 3, 3, 3.1e-5), (1, 1, 1, 6e-5)]:

            # input 1: sub-sampled measurements
            projs = sub_sample_proj(projs_no_sub_sampling,
                    factor_row=proj_row_sub_sampling,
                    factor_col=proj_col_sub_sampling)

            # input 2: sub-sampled forward projection of ground truth
            fp = sub_sample_proj(fp_no_sub_sampling,
                    factor_row=proj_row_sub_sampling,
                    factor_col=proj_col_sub_sampling)

            walnut_ray_trafo = WalnutRayTrafo(
                    data_path=self.data_path, walnut_id=self.walnut_id,
                    orbit_id=self.orbit_id,
                    angular_sub_sampling=angular_sub_sampling,
                    proj_row_sub_sampling=proj_row_sub_sampling,
                    proj_col_sub_sampling=proj_col_sub_sampling,
                    vol_down_sampling=vol_down_sampling)

            fdk_projs = walnut_ray_trafo.fdk(projs)
            fdk_fp = walnut_ray_trafo.fdk(fp)

            # reference
            gt_down_sampled = down_sample_vol(
                    gt, down_sampling=vol_down_sampling)

            self.assertMSELessEqual(fdk_projs, gt_down_sampled, max_mse)
            self.assertImagesEqual(fdk_fp, gt_down_sampled, atol=6e-2)


if __name__ == '__main__':
    unittest.main()
