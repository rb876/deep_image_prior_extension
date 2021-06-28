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
from .utils import poisson_loss, tv_loss, PSNR, normalize, get_learnable_params

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
        scaling_kwargs = {
            'mean_in': self.cfg.stats.mean_fbp,
            'mean_out': self.cfg.stats.mean_gt,
            'std_in': self.cfg.stats.std_fbp,
            'std_out': self.cfg.stats.std_gt
            } if self.cfg.normalize_by_stats else {}

        self.model = UNet(
            input_depth,
            output_depth,
            channels=self.cfg.arch.channels[:self.cfg.arch.scales],
            skip_channels=self.cfg.arch.skip_channels[:self.cfg.arch.scales],
            use_sigmoid=self.cfg.arch.use_sigmoid,
            use_norm=self.cfg.arch.use_norm,
            use_scale_layer = self.cfg.normalize_by_stats,
            scaling_kwargs = scaling_kwargs
            ).to(self.device)

        current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
        comment = 'DIP+TV'
        logdir = os.path.join(
            self.cfg.log_path,
            current_time + '_' + socket.gethostname() + comment)
        self.writer = tensorboardX.SummaryWriter(logdir=logdir)

    def apply_model_on_test_data(self, net_input):
        test_scaling = self.cfg.get('implicit_scaling_except_for_test_data')
        if test_scaling is not None and test_scaling != 1.:
            if self.cfg.add_init_reco:
                net_input = torch.cat(
                        (test_scaling * net_input[:, 0].unsqueeze(dim=1),
                         net_input[:, 1].unsqueeze(dim=1)), dim=1)
            output = self.model(net_input)
            output = output / test_scaling
        else:
            output = self.model(net_input)

        return output

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

        self.init_optimizer()
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
        best_output = self.apply_model_on_test_data(self.net_input).detach()

        with tqdm(range(self.cfg.loss.iterations), desc='DIP', disable= not self.cfg.show_pbar) as pbar:
            for i in pbar:
                self.optimizer.zero_grad()
                output = self.apply_model_on_test_data(self.net_input)
                loss = criterion(self.ray_trafo_module(output), y_delta) + self.cfg.loss.gamma * tv_loss(output)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                self.optimizer.step()

                for p in self.model.parameters():
                    p.data.clamp_(-1000, 1000) # MIN,MAX

                if self.cfg.freeze and i == self.cfg.loss.num_warmup_iter:
                    self.add_params_group(['down', 'inc'])
                    
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_output = output.detach()

                if ground_truth is not None:
                    psnr = PSNR(best_output.detach().cpu(), ground_truth.cpu())
                    pbar.set_postfix({"psnr": psnr})
                    self.writer.add_scalar('psnr', psnr, i)

                self.writer.add_scalar('loss', loss.item(),  i)
                if i % 1000 == 0:
                    self.writer.add_image('reco', normalize(best_output[0, ...]).cpu().numpy(), i)

        self.writer.close()

        return best_output[0, 0, ...].cpu().numpy()

    def init_optimizer(self):
        """
        Initialize the optimizer.
        """

        encoder_params = get_learnable_params(self.model, ['down', 'inc'])
        decoder_params = get_learnable_params(self.model, ['up', 'scale',
                'outc'])
        import pdb; pdb.set_trace()
        if self.cfg.use_different_lr:
            if self.cfg.freeze:
                self._optimizer = \
                    torch.optim.Adam([{'params': decoder_params,
                                     'lr': self.cfg.loss.lr.decoder}])
            else:
                self._optimizer = \
                    torch.optim.Adam([{'params': encoder_params,
                                     'lr': self.cfg.loss.lr.encoder},
                                     {'params': decoder_params,
                                     'lr': self.cfg.loss.lr.decoder}])
        else:
            self._optimizer = torch.optim.Adam(self.model.parameters(),
                    lr=self.cfg.loss.lr.coupled)

    def add_params_group(self, add_params_list):

        encoder_params = get_learnable_params(self.model, add_params_list)
        self._optimizer.param_groups.append({
            'params': encoder_params,
            'lr': self.cfg.loss.lr.encoder,
            'betas': (0.9, 0.999),
            'eps': 1e-08,
            'weight_decay': 0,
            'amsgrad': False,
            })

    @property
    def optimizer(self):
        """
        :class:`torch.optim.Optimizer` :
        The optimizer, usually set by :meth:`init_optimizer`, which gets called
        in :meth:`train`.
        """
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value
