import os
import socket
import datetime
import torch
import numpy as np
import tensorboardX
from torch.optim import Adam
from torch.nn import MSELoss
from warnings import warn
from functools import partial
from tqdm import tqdm

from .network import UNet
from .utils import poisson_loss, tv_loss, PSNR, normalize

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

    def __init__(self, ray_trafo_module, reco_space, observation_space, cfg):

        self.reco_space = reco_space
        self.observation_space = observation_space
        self.cfg = cfg
        self.device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.ray_trafo_module = ray_trafo_module.to(self.device)
        self.init_model()

    def init_model(self):

        input_depth = 1 if not self.cfg.add_init_reco else 2
        output_depth = 1

        self.model = UNet(
            input_depth,
            output_depth,
            channels=self.cfg.arch.channels[:self.cfg.arch.scales],
            skip_channels=self.cfg.arch.skip_channels[:self.cfg.arch.scales],
            use_sigmoid=self.cfg.arch.use_sigmoid,
            use_norm=self.cfg.arch.use_norm,
            ).to(self.device)
        current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
        comment = 'DIP+TV'
        logdir = os.path.join(
            self.cfg.log_path,
            current_time + '_' + socket.gethostname() + comment)
        self.writer = tensorboardX.SummaryWriter(logdir=logdir)

    def reconstruct(self, noisy_observation, fbp=None, ground_truth=None):

        if self.cfg.torch_manual_seed:
            torch.random.manual_seed(self.cfg.torch_manual_seed)

        self.init_model()
        if self.cfg.load_pretrain_model:
            path = \
                self.cfg.learned_params_path if self.cfg.learned_params_path.endswith('.pt') \
                    else self.cfg.learned_params_path + '.pt'
            path = os.path.join(os.getcwd().partition('src')[0], path)
            self.model.load_state_dict(torch.load(path, map_location=self.device))
        else:
            self.model.to(self.device)

        self.model.train()

        if self.cfg.recon_from_randn:
            self.net_input = 0.1 * \
                torch.randn(1, *self.reco_space.shape)[None].to(self.device)
            if self.cfg.add_init_reco:
                self.net_input = \
                    torch.cat([fbp.to(self.device), self.net_input], dim=1)
        else:
            self.net_input = fbp.to(self.device)

        self.optimizer = Adam(self.model.parameters(), lr=self.cfg.loss.lr)
        y_delta = noisy_observation.to(self.device)

        if self.cfg.loss.loss_function == 'mse':
            criterion = MSELoss()
        elif self.cfg.loss.loss_function == 'poisson':
            criterion = partial(poisson_loss,
                                photons_per_pixel=self.cfg.loss.photons_per_pixel,
                                mu_water=self.cfg.loss.mu_water)
        else:
            warn('Unknown loss function, falling back to MSE')
            criterion = MSELoss()

        best_loss = np.inf
        best_output = self.model(self.net_input).detach()

        with tqdm(range(self.cfg.loss.iterations), desc='DIP', disable= not self.cfg.show_pbar) as pbar:
            for i in pbar:
                self.optimizer.zero_grad()
                output = self.model(self.net_input)
                loss = criterion(self.ray_trafo_module(output), y_delta) + self.cfg.loss.gamma * tv_loss(output)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                self.optimizer.step()

                for p in self.model.parameters():
                    p.data.clamp_(-1000, 1000) # MIN,MAX

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_output = output.detach()

                if ground_truth:
                    psnr = PSNR(best_output.detach().cpu(), ground_truth[0])
                    pbar.set_postfix({"psnr": psnr})
                    self.writer.add_scalar('psnr', psnr, i)

                self.writer.add_scalar('loss', loss.item(),  i)
                if i % 1000 == 0:
                    self.writer.add_image('reco', normalize(best_output[0, ...]).cpu().numpy(), i)

        self.writer.close()

        return best_output[0, 0, ...].cpu().numpy()
