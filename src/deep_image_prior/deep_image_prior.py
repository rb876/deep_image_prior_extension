import os
import socket
import datetime
import contextlib
import torch
import numpy as np
import tensorboardX
from torch.optim import Adam
from torch.nn import MSELoss
from torch.cuda.amp import autocast, GradScaler
from warnings import warn
from functools import partial
from copy import deepcopy
from tqdm import tqdm

from .network import UNet, UNet3D
from .utils import poisson_loss, tv_loss, tv_loss_3d, PSNR, normalize, extract_learnable_params


class LRPolicy():
    """
    Learning rate policy implementing linear warmup phase(s), for use with
    the `LambdaLR` scheduler.
    """
    def __init__(self, init_lr, lr, num_warmup_iter, num_iterations):
        self.init_lr = init_lr
        self.lr = lr
        self.num_warmup_iter = num_warmup_iter
        self.num_iterations = num_iterations
        self.lambda_fct = np.ones(self.num_iterations + 1)
        self.restart(0, init_lr, lr, preserve_initial_warmup=False)

    def restart(self, epoch, init_lr, lr, preserve_initial_warmup=True):
        """
        Add a linear warmup phase starting at `epoch`, linearly increasing
        from `init_lr` to `lr`. After the warmup, `lr` is used. Optionally, the
        initial warmup phase is respected by taking the pointwise minimum of
        both rates.
        Note: `epoch` specifies the iteration.
        """
        n = min(self.num_warmup_iter, max(0, self.num_iterations + 1 - epoch))

        self.lambda_fct[epoch:epoch+n] = np.linspace(
                init_lr/self.lr if self.lr != 0. else 1.,
                lr/self.lr if self.lr != 0. else 1.,
                self.num_warmup_iter)[:n]

        self.lambda_fct[epoch+n:] = lr/self.lr if self.lr != 0. else 1.

        if preserve_initial_warmup:
            self.lambda_fct[epoch:self.num_warmup_iter] = np.minimum(
                    self.lambda_fct[epoch:self.num_warmup_iter],
                    np.linspace(
                            self.init_lr/self.lr if self.lr != 0. else 1., 1.,
                            self.num_warmup_iter)[epoch:])

    def __call__(self, epoch):
        """Note: `epoch` specifies the iteration."""
        return self.lambda_fct[epoch]


def get_iterates_iters(cfg, iterations):
    s = []

    if cfg.mode == 'standard_sequence':
        s += range(100)
        s += range(100, 250, 5)
        s += range(250, 1000, 25)
        s += range(1000, 5000, 100)
        s += range(5000, 10000, 500)
        s += range(10000, iterations, 2500)
    elif cfg.mode == 'manual':
        pass
    else:
        raise ValueError('Unknown iterates selection mode \'{}\''
                         .format(cfg.mode))

    if cfg.manual_iters is not None:
        s += cfg.manual_iters

    s = [i for i in s if i < iterations]
    s = sorted(list(set(s)))

    return s

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

        if len(self.reco_space.shape) == 2:
            self.model = UNet(
                input_depth,
                output_depth,
                channels=self.cfg.arch.channels[:self.cfg.arch.scales],
                skip_channels=self.cfg.arch.skip_channels[:self.cfg.arch.scales],
                use_sigmoid=self.cfg.arch.use_sigmoid,
                use_norm=self.cfg.arch.use_norm,
                use_scale_in_layer = (self.cfg.normalize_by_stats and
                        ((not self.cfg.recon_from_randn) or
                        self.cfg.add_init_reco)),
                use_scale_out_layer = self.cfg.normalize_by_stats,
                scaling_kwargs = scaling_kwargs
                ).to(self.device)
        elif len(self.reco_space.shape) == 3:
            self.model = UNet3D(
                input_depth,
                output_depth,
                channels=self.cfg.arch.channels[:self.cfg.arch.scales],
                skip_channels=self.cfg.arch.skip_channels[:self.cfg.arch.scales],
                down_channel_overrides=self.cfg.arch.down_channel_overrides,
                down_single_conv=self.cfg.arch.down_single_conv,
                use_sigmoid=self.cfg.arch.use_sigmoid,
                use_norm=self.cfg.arch.use_norm,
                use_relu_out=self.cfg.arch.use_relu_out == 'model',
                out_kernel_size=self.cfg.arch.out_kernel_size,
                pre_out_channels=self.cfg.arch.pre_out_channels,
                pre_out_kernel_size=self.cfg.arch.pre_out_kernel_size,
                insert_res_blocks_before=self.cfg.arch.insert_res_blocks_before,
                approx_conv3d_at_scales=self.cfg.arch.approx_conv3d_at_scales,
                approx_conv3d_low_rank_dim=self.cfg.arch.approx_conv3d_low_rank_dim
                ).to(self.device)
        else:
            raise ValueError

        current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
        comment = 'DIP+TV'
        logdir = os.path.join(
            self.cfg.log_path,
            current_time + '_' + socket.gethostname() + comment)
        self.writer = tensorboardX.SummaryWriter(logdir=logdir)

    def apply_model_on_test_data(self, net_input):
        test_scaling = self.cfg.get('implicit_scaling_except_for_test_data')
        if test_scaling is not None and test_scaling != 1.:
            if self.cfg.recon_from_randn:
                if self.cfg.add_init_reco:
                    net_input = torch.cat(
                            (test_scaling * net_input[:, 0].unsqueeze(dim=1),
                             net_input[:, 1].unsqueeze(dim=1)), dim=1)
            else:
                net_input = test_scaling * net_input
            output = self.model(net_input)
            output = output / test_scaling
        else:
            output = self.model(net_input)

        return output

    def reconstruct(self, noisy_observation, fbp=None, ground_truth=None,
                    return_histories=False, return_iterates=False,
                    return_iterates_params=False):
        """
        Parameters
        ----------
        noisy_observation : :class:`torch.Tensor`
            Noisy observation.
        fbp : :class:`torch.Tensor`, optional
            Input reconstruction (e.g. filtered backprojection).
        ground_truth : :class:`torch.Tensor`, optional
            Ground truth image.
        return_histories : bool, optional
            Whether to return histories of loss, PSNR and learning rates.
            The default is `False`.
        return_iterates : bool, optional
            Whether to return a selection of iterates, configured via
            ``self.cfg.return_iterates_selection``.
            The default is `False`.
        return_iterates_params : bool, optional
            Whether to return the parameters for a selection of iterates,
            configured via ``self.cfg.return_iterates_params_selection``.
            The default is `False`.

        Returns
        -------
        out : :class:`numpy.ndarray`
            The reconstruction with minimum loss value reached.
        histories : dict, optional
            Histories, contained in a dict under the following keys:
            `'loss'`, `'psnr'`, `'lr_encoder'`, `'lr_decoder'`.
            Each history is a list of scalar values.
            Only provided if ``return_histories=True``.
        iterates : list of :class:`numpy.ndarray`, optional
            Reconstructions at intermediate iterations.
            Only provided if ``return_iterates=True``.
        iterates_iters : list of int, optional
            Iterations corresponding to `iterates`.
            Only provided if ``return_iterates=True``.
        iterates_params : list of :class:`numpy.ndarray`, optional
            Model parameters (state dictionaries) at intermediate iterations.
            Only provided if ``return_iterates_params=True``.
        iterates_params_iters : list of int, optional
            Iterations corresponding to `iterates_params`.
            Only provided if ``return_iterates_params=True``.
        """

        if self.cfg.torch_manual_seed:
            torch.random.manual_seed(self.cfg.torch_manual_seed)

        self.init_model()
        if self.cfg.load_pretrain_model:
            path = \
                self.cfg.learned_params_path if self.cfg.learned_params_path.endswith('.pt') \
                    else self.cfg.learned_params_path + '.pt'
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
        self.init_scheduler()
        y_delta = noisy_observation.to(self.device)

        if self.cfg.use_mixed:
            scaler = GradScaler()

        if self.cfg.optim.loss_function == 'mse':
            criterion = MSELoss()
        elif self.cfg.optim.loss_function == 'poisson':
            criterion = partial(poisson_loss,
                                photons_per_pixel=self.cfg.optim.photons_per_pixel,
                                mu_max=self.cfg.optim.mu_max)
        else:
            warn('Unknown loss function, falling back to MSE')
            criterion = MSELoss()

        tv_loss_fun = tv_loss if len(self.reco_space.shape) == 2 else tv_loss_3d

        iterates_iters = []
        if return_iterates:
            iterates_iters = get_iterates_iters(
                self.cfg.return_iterates_selection,
                self.cfg.optim.iterations)
        iterates_params_iters = []
        if return_iterates_params:
            iterates_params_iters = get_iterates_iters(
                self.cfg.return_iterates_params_selection,
                self.cfg.optim.iterations)

        iterates = []
        iterates_params = []

        best_loss = np.inf
        best_output = self.apply_model_on_test_data(self.net_input).detach()
        if self.cfg.arch.use_relu_out == 'post':
            best_output = torch.nn.functional.relu(best_output)

        loss_history = []
        psnr_history = []
        lr_encoder_history = []
        lr_decoder_history = []
        loss_avg_history = []
        last_lr_adaptation_iter = 0

        with tqdm(range(self.cfg.optim.iterations), desc='DIP', disable= not self.cfg.show_pbar) as pbar:
            for i in pbar:
                self.optimizer.zero_grad()
                with autocast() if self.cfg.use_mixed else contextlib.nullcontext():
                    output = self.apply_model_on_test_data(self.net_input)
                    loss = criterion(self.ray_trafo_module(output), y_delta) + self.cfg.optim.gamma * tv_loss_fun(output)

                if i in iterates_params_iters:
                    iterates_params.append(deepcopy(self.model.state_dict()))

                if self.cfg.use_mixed:
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                else:
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                if self.cfg.use_mixed:
                    scaler.step(self.optimizer)
                    scale = scaler.get_scale()
                    scaler.update()
                else:
                    self.optimizer.step()

                if return_histories:
                    lr_encoder_history.append(self.optimizer.param_groups[0]['lr'])
                    lr_decoder_history.append(self.optimizer.param_groups[1]['lr'])
                self.writer.add_scalar('lr_encoder', self.optimizer.param_groups[0]['lr'], i)
                self.writer.add_scalar('lr_decoder', self.optimizer.param_groups[1]['lr'], i)

                if self.cfg.use_mixed:
                    # avoid calling scheduler before optimizer in case of nan/inf values
                    # https://discuss.pytorch.org/t/optimizer-step-before-lr-scheduler-step-error-using-gradscaler/92930/8
                    if scaler.get_scale() == scale:
                        self.scheduler.step()
                else:
                    self.scheduler.step()
                for p in self.model.parameters():
                    p.data.clamp_(-1000, 1000) # MIN,MAX

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_output = output.detach()
                    if self.cfg.arch.use_relu_out == 'post':
                        best_output = torch.nn.functional.relu(best_output)

                if self.cfg.optim.use_adaptive_lr or return_histories:
                    loss_history.append(loss.item())

                if self.cfg.optim.use_adaptive_lr:
                    loss_avg_history.append(np.inf if i+1 < self.cfg.optim.adaptive_lr.num_avg_iter else
                                            np.mean(loss_history[-self.cfg.optim.adaptive_lr.num_avg_iter:]))

                    if (i >= last_lr_adaptation_iter + self.cfg.optim.adaptive_lr.num_avg_iter and
                            loss_avg_history[-1] > loss_avg_history[-self.cfg.optim.adaptive_lr.num_avg_iter-1] * (1. - self.cfg.optim.adaptive_lr.min_rel_loss_decrease)):
                        self._adapt_lr(i)
                        last_lr_adaptation_iter = i

                if ground_truth is not None:
                    best_output_psnr = PSNR(best_output.detach().cpu(), ground_truth.cpu())
                    output_psnr = PSNR((torch.nn.functional.relu(output) if self.cfg.arch.use_relu_out == 'post' else output).detach().cpu(), ground_truth.cpu())
                    if return_histories:
                        psnr_history.append(output_psnr)
                    pbar.set_postfix({'output_psnr': output_psnr})
                    self.writer.add_scalar('best_output_psnr', best_output_psnr, i)
                    self.writer.add_scalar('output_psnr', output_psnr, i)

                self.writer.add_scalar('loss', loss.item(),  i)
                if i in iterates_iters:
                    iterates.append((torch.nn.functional.relu(output) if self.cfg.arch.use_relu_out == 'post' else output)[0, ...].detach().cpu().numpy())
                if i % 1000 == 0:
                    if len(self.reco_space.shape) == 2:
                        self.writer.add_image('reco', normalize(best_output[0, ...]).cpu().numpy(), i)
                    else:  # 3d
                        self.writer.add_image('reco_mid_slice',
                                normalize(best_output[0, :, best_output.shape[2] // 2, ...]).cpu().numpy(), i)

        self.writer.close()

        out = best_output[0, 0, ...].cpu().numpy()

        optional_out = []
        if return_histories:
            histories = {'loss': loss_history,
                         'psnr': psnr_history,
                         'lr_encoder': lr_encoder_history,
                         'lr_decoder': lr_decoder_history}
            optional_out.append(histories)
        if return_iterates:
            optional_out.append(iterates)
            optional_out.append(iterates_iters)
        if return_iterates_params:
            optional_out.append(iterates_params)
            optional_out.append(iterates_params_iters)

        return (out, *optional_out) if optional_out else out

    def init_optimizer(self):
        """
        Initialize the optimizer.
        """

        self._optimizer = \
            torch.optim.Adam([{'params': extract_learnable_params(self.model,
                             ['down', 'inc']),
                             'lr': self.cfg.optim.encoder.lr},
                             {'params': extract_learnable_params(self.model,
                             ['up', 'scale', 'outc']),
                             'lr': self.cfg.optim.decoder.lr}])

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

    def init_scheduler(self):

        # always use `self.scheduler` to enable lr changes on checkpoint
        # returns, but set init_lr=lr and num_warmup_iter=0 if
        # `not self.cfg.optim.use_scheduler`, effectively disabling warmups
        if self.cfg.optim.use_scheduler:
            init_lr_encoder = self.cfg.optim.encoder.init_lr
            init_lr_decoder = self.cfg.optim.decoder.init_lr
            num_warmup_iter_encoder = self.cfg.optim.encoder.num_warmup_iter
            num_warmup_iter_decoder = self.cfg.optim.decoder.num_warmup_iter
        else:
            init_lr_encoder = self.cfg.optim.encoder.lr
            init_lr_decoder = self.cfg.optim.decoder.lr
            num_warmup_iter_encoder = 0
            num_warmup_iter_decoder = 0

        self._lr_policy_encoder = LRPolicy(
                init_lr=init_lr_encoder,
                lr=self.cfg.optim.encoder.lr,
                num_warmup_iter=num_warmup_iter_encoder,
                num_iterations=self.cfg.optim.iterations)
        self._lr_policy_decoder = LRPolicy(
                init_lr=init_lr_decoder,
                lr=self.cfg.optim.decoder.lr,
                num_warmup_iter=num_warmup_iter_decoder,
                num_iterations=self.cfg.optim.iterations)

        self._scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                lr_lambda=[self._lr_policy_encoder, self._lr_policy_decoder])

        self.init_lr_fct_encoder = 1.
        self.init_lr_fct_decoder = 1.
        self.lr_fct_encoder = 1.
        self.lr_fct_decoder = 1.

    def _adapt_lr(self, iteration):

        self.init_lr_fct_encoder *= self.cfg.optim.adaptive_lr.get('multiply_init_lr_by', 1.)
        self.init_lr_fct_decoder *= self.cfg.optim.adaptive_lr.get('multiply_init_lr_by', 1.)
        self.lr_fct_encoder *= self.cfg.optim.adaptive_lr.get('multiply_lr_by', 1.)
        self.lr_fct_decoder *= self.cfg.optim.adaptive_lr.get('multiply_lr_by', 1.)

        lr_encoder = self.cfg.optim.encoder.lr * self.lr_fct_encoder
        lr_decoder = self.cfg.optim.decoder.lr * self.lr_fct_decoder
        if (self.cfg.optim.use_scheduler and
                self.cfg.optim.adaptive_lr.restart_scheduler):
            init_lr_encoder = self.cfg.optim.encoder.init_lr * self.init_lr_fct_encoder
            init_lr_decoder = self.cfg.optim.decoder.init_lr * self.init_lr_fct_decoder
        else:
            init_lr_encoder = lr_encoder
            init_lr_decoder = lr_decoder

        self._lr_policy_encoder.restart(iteration, init_lr_encoder, lr_encoder,
                preserve_initial_warmup=not self.cfg.optim.adaptive_lr.restart_scheduler)
        self._lr_policy_decoder.restart(iteration, init_lr_decoder, lr_decoder,
                preserve_initial_warmup=not self.cfg.optim.adaptive_lr.restart_scheduler)

    @property
    def scheduler(self):
        """
        torch learning rate scheduler :
        The scheduler, usually set by :meth:`init_scheduler`, which gets called
        in :meth:`train`.
        """
        return self._scheduler

    @scheduler.setter
    def scheduler(self, value):
        self._scheduler = value
