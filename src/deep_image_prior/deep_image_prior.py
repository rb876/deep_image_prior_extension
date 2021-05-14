import os
import torch
import numpy as np
import torch.nn as nn
import tensorboardX
from torch.optim import Adam
from torch.nn import MSELoss
from odl.contrib.torch import OperatorModule
from warnings import warn
from functools import partial
from tqdm import tqdm

from .network import UNet
from .utils import poisson_loss, tv_loss, PSNR

class DeepImagePriorReconstructor():
    """
    CT reconstructor applying DIP with TV regularization (see [2]_).
    The DIP was introduced in [1].
    .. [1] V. Lempitsky, A. Vedaldi, and D. Ulyanov, 2018, "Deep Image Prior".
           IEEE/CVF Conference on Computer Vision and Pattern Recognition.
           https://doi.org/10.1109/CVPR.2018.00984
    .. [2] D. Otero Baguer, J. Leuschner, M. Schmidt, 2020, "Computed
           Tomography Reconstruction Using Deep Image Prior and Learned
           Reconstruction Methods". Inverse Problems.
           https://doi.org/10.1088/1361-6420/aba415
    """

    def __init__(self, ray_trafo, reco_space, observation_space, arch_cfg):

        self.ray_trafo = ray_trafo
        self.reco_space = reco_space
        self.observation_space = observation_space
        self.arch_cfg = arch_cfg
        self.ray_trafo_module = OperatorModule(self.ray_trafo)
        self.device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.init_model()

    def init_model(self):

        input_depth = 1
        output_depth = 1

        self.model = UNet(
            input_depth,
            output_depth,
            channels=self.arch_cfg.channels[:self.arch_cfg.scales],
            skip_channels=self.arch_cfg.skip_channels[:self.arch_cfg.scales],
            use_sigmoid=self.arch_cfg.use_sigmoid,
            use_norm=self.arch_cfg.use_norm,
            ).to(self.device)
        self.writer = tensorboardX.SummaryWriter(comment='DIP+TV')

    def reconstruct(self, cfg, noisy_observation, fbp=None, ground_truth=None):

        if cfg.torch_manual_seed:
            torch.random.manual_seed(cfg.torch_manual_seed)

        self.init_model()

        if cfg.load_pretain_model:
            path = cfg.learned_params_path if cfg.learned_params_path.endswith('.pt') else cfg.learned_params_path + '.pt'
            path = os.path.join(os.getcwd(), path)
            self.model.load_state_dict(torch.load(path, map_location=self.device))
        else:
            self.model.to(self.device)

        self.model.train()

        if cfg.recon_from_randn:
            input_depth = 1
            self.net_input = 0.1 * \
                torch.randn(input_depth, *self.reco_space.shape)[None].to(self.device)
        else:
            self.net_input = fbp.to(self.device)

        self.optimizer = Adam(self.model.parameters(), lr=cfg.loss.lr)
        y_delta = noisy_observation.to(self.device)

        if cfg.loss.loss_function == 'mse':
            criterion = MSELoss()
        elif cfg.loss.loss_function == 'poisson':
            criterion = partial(poisson_loss,
                                photons_per_pixel=cfg.loss.photons_per_pixel,
                                mu_water=cfg.loss.mu_water)
        else:
            warn('Unknown loss function, falling back to MSE')
            criterion = MSELoss()

        best_loss = np.inf
        best_output = self.model(self.net_input).detach()
        # saving ground_truth and filter-backprojection
        self.writer.add_image('ground_truth', ground_truth[0, ...].cpu().clone().numpy(), 0)
        self.writer.add_image('initial_guess', fbp[0, ...].cpu().clone().numpy(), 0)
        with tqdm(range(cfg.loss.iterations), desc='DIP', disable= not cfg.show_pbar) as pbar:
            for i in pbar:
                self.optimizer.zero_grad()
                output = self.model(self.net_input)
                loss = criterion(self.ray_trafo_module(output), y_delta) + cfg.loss.gamma * tv_loss(output)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                self.optimizer.step()

                for p in self.model.parameters():
                    p.data.clamp_(-1000, 1000) # MIN,MAX

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_output = output.detach()

                self.writer.add_scalar('loss', loss.item(),  i)

                if ground_truth is not None:
                    psnr = PSNR(best_output.detach().cpu(), ground_truth)
                    pbar.set_postfix({"psnr": psnr})
                    self.writer.add_scalar('psnr', psnr,i)

                if i % 100:
                    self.writer.add_image('reco', best_output[0, ...].cpu().numpy(), i)

        self.writer.close()

        return best_output[0, 0, ...].cpu().numpy()
