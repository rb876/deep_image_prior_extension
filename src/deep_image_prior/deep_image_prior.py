import torch
import numpy as np
import torch.nn as nn
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

    def __init__(self, ray_trafo):

        self.ray_trafo = ray_trafo
        self.reco_space = ray_trafo.domain
        self.observation_space = ray_trafo.range
        self.ray_trafo_module = OperatorModule(self.ray_trafo)

    def reconstruct(self, cfg, noisy_observation, ground_truth=None):
        if cfg.torch_manual_seed:
            torch.random.manual_seed(cfg.torch_manual_seed)

        output_depth = 1
        input_depth = 1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.net_input = 0.1 * \
            torch.randn(input_depth, *self.reco_space.shape)[None].to(device)
        self.model = UNet(
            input_depth,
            output_depth,
            channels=cfg.arch.channels[:cfg.arch.scales],
            skip_channels=cfg.arch.skip_channels[:cfg.arch.scales],
            use_sigmoid=True,
            use_norm=True).to(device)

        self.optimizer = Adam(self.model.parameters(), lr=cfg.loss.lr)
        y_delta = noisy_observation.to(device)

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

                if ground_truth is not None:
                    pbar.set_postfix(
                        {"psnr": PSNR(best_output.detach().cpu(), ground_truth)}
                    )

        return best_output[0, 0, ...].cpu().numpy()
