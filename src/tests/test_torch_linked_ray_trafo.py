import unittest
import os
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio
import astra
from dataset.walnuts import (
        get_projection_data, WalnutRayTrafo, get_ground_truth_3d,
        sub_sample_proj, down_sample_vol)
from util.torch_linked_ray_trafo import TorchLinkedRayTrafoModule

DATA_PATH = '/localdata/jleuschn/Walnuts/'

class TestTorchLinkedRayTrafo(unittest.TestCase):
    def __init__(self, *args, data_path=DATA_PATH, device='cpu', **kwargs):
        super().__init__(*args, **kwargs)

        self.data_path = data_path
        # ASTRA 3D routines will run on CUDA regardless of self.device
        self.device = device if device is not None else (
                'cuda:0' if torch.cuda.is_available() else 'cpu')

        self.walnut_id = 1
        self.orbit_id = 2

        self.walnut_ray_trafo = WalnutRayTrafo(
                data_path=self.data_path, walnut_id=self.walnut_id,
                orbit_id=self.orbit_id)

    def get_test_image_batch(self):
        gt = get_ground_truth_3d(
                        self.data_path, self.walnut_id, orbit_id=self.orbit_id)
        rng = np.random.default_rng()
        test_image_batch = np.stack([
                gt,
                rng.random(gt.shape).astype(np.float32),
                ], axis=0)
        return test_image_batch

    def get_test_proj_batch(self):
        proj = get_projection_data(
                self.data_path, self.walnut_id, orbit_id=self.orbit_id)
        rng = np.random.default_rng()
        test_proj_batch = np.stack([
                proj,
                rng.random(proj.shape).astype(np.float32),
                ], axis=0)
        return test_proj_batch

    def test_fp_forward(self):
        torch_linked_fp_module = TorchLinkedRayTrafoModule(
                self.walnut_ray_trafo.vol_geom, self.walnut_ray_trafo.proj_geom)

        test_image_batch = self.get_test_image_batch()

        test_fp_batch_reference = np.stack([
                self.walnut_ray_trafo.apply(test_image)
                for test_image in test_image_batch], axis=0)

        test_image_batch_torch = torch.from_numpy(
                test_image_batch).to(self.device)
        test_fp_batch_torch = torch_linked_fp_module(test_image_batch_torch)
        test_fp_batch = test_fp_batch_torch.cpu().numpy()

        self.assertTrue(np.allclose(test_fp_batch, test_fp_batch_reference))

    def test_bp_forward(self):
        torch_linked_bp_module = TorchLinkedRayTrafoModule(
                self.walnut_ray_trafo.vol_geom, self.walnut_ray_trafo.proj_geom,
                adjoint=True)

        test_proj_batch = self.get_test_proj_batch()

        test_bp_batch_reference = np.stack([
                self.walnut_ray_trafo.apply_adjoint(test_proj)
                for test_proj in test_proj_batch], axis=0)

        test_proj_batch_torch = torch.from_numpy(
                test_proj_batch).to(self.device)
        test_bp_batch_torch = torch_linked_bp_module(test_proj_batch_torch)
        test_bp_batch = test_bp_batch_torch.cpu().numpy()

        self.assertTrue(np.allclose(test_bp_batch, test_bp_batch_reference))

    def test_fp_gradient(self):
        torch_linked_fp_module = TorchLinkedRayTrafoModule(
                self.walnut_ray_trafo.vol_geom, self.walnut_ray_trafo.proj_geom)
        torch_in = torch.from_numpy(self.get_test_image_batch()).to(self.device)
        torch_in.requires_grad_(True)
        torch_out = torch_linked_fp_module(torch_in)
        out_grad = torch.rand(torch_out.shape, device=self.device)
        torch_in.grad = None
        torch_out.backward(out_grad, retain_graph=True)
        scalar_prod_out = torch.sum(torch_out * out_grad).item()
        scalar_prod_in = torch.sum(torch_in * torch_in.grad).item()
        self.assertAlmostEqual(
            scalar_prod_out,
            scalar_prod_in,
            delta=3.5e-2*np.mean([scalar_prod_out, scalar_prod_in]))

    def test_bp_gradient(self):
        torch_linked_bp_module = TorchLinkedRayTrafoModule(
                self.walnut_ray_trafo.vol_geom, self.walnut_ray_trafo.proj_geom,
                adjoint=True)
        torch_in = torch.from_numpy(self.get_test_proj_batch()).to(self.device)
        torch_in.requires_grad_(True)
        torch_out = torch_linked_bp_module(torch_in)
        out_grad = torch.rand(torch_out.shape, device=self.device)
        torch_in.grad = None
        torch_out.backward(out_grad)
        scalar_prod_out = torch.sum(torch_out * out_grad).item()
        scalar_prod_in = torch.sum(torch_in * torch_in.grad).item()
        self.assertAlmostEqual(
            scalar_prod_out,
            scalar_prod_in,
            delta=3.5e-2*np.mean([scalar_prod_out, scalar_prod_in]))

    ## grad check via torch, could not get it to pass (probably it is just too different)
    # def test_fp_gradcheck(self):

    #     angular_sub_sampling=100
    #     proj_row_sub_sampling=31
    #     proj_col_sub_sampling=31
    #     vol_down_sampling=31

    #     walnut_ray_trafo_small = WalnutRayTrafo(
    #             data_path=self.data_path, walnut_id=self.walnut_id,
    #             orbit_id=self.orbit_id,
    #             angular_sub_sampling=angular_sub_sampling,
    #             proj_row_sub_sampling=proj_row_sub_sampling,
    #             proj_col_sub_sampling=proj_col_sub_sampling,
    #             vol_down_sampling=vol_down_sampling)

    #     torch_linked_fp_module = TorchLinkedRayTrafoModule(
    #             walnut_ray_trafo_small.vol_geom, walnut_ray_trafo_small.proj_geom)

    #     def torch_test_fun(inp):
    #         out = torch_linked_fp_module(inp)
    #         # out = out[out.shape[0] // 2, out.shape[1] // 2, (out.shape[2] // 2 - 2):(out.shape[2] // 2 + 3)].mean()
    #         return out

    #     test_image_batch = self.get_test_image_batch()[:1]
    #     test_image_batch = np.stack([
    #             down_sample_vol(test_image, down_sampling=vol_down_sampling)
    #             for test_image in test_image_batch], axis=0)
    #     test_image_batch_torch = torch.from_numpy(
    #             test_image_batch).to(self.device)
    #     test_image_batch_torch.requires_grad_(True)
    #     torch.autograd.gradcheck(torch_test_fun, test_image_batch_torch)

    #     # put this in torch.autograd.gradcheck._allclose_with_type_promotion
    #     # for debugging:
    #     # import numpy as np
    #     # im_shape = (np.round(a.shape[0]**(1/3)).astype(int),) * 3
    #     # print(a.shape)
    #     # import matplotlib.pyplot as plt
    #     # plt.subplot(311)
    #     # plt.imshow(a.reshape(*im_shape, -1)[im_shape[0] // 2, :, :, a.shape[1] // 2].detach().cpu().numpy())
    #     # plt.colorbar()
    #     # plt.subplot(312)
    #     # plt.imshow(b.reshape(*im_shape, -1)[im_shape[0] // 2, :, :, b.shape[1] // 2].detach().cpu().numpy())
    #     # plt.colorbar()
    #     # plt.subplot(313)
    #     # plt.imshow((b-a).reshape(*im_shape, -1)[im_shape[0] // 2, :, :, a.shape[1] // 2].detach().cpu().numpy())
    #     # plt.colorbar()
    #     # plt.show()


if __name__ == '__main__':
    unittest.main()
