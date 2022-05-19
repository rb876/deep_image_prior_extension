import os
import json
import socket
import datetime
import contextlib
import odl
import h5py
import torch
import numpy as np
import torch.nn.functional as F
import tensorboardX
import matplotlib.pyplot as plt
from copy import deepcopy
from math import ceil
from tqdm import tqdm, trange
from inspect import signature, Parameter
from warnings import warn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CyclicLR, OneCycleLR
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.cuda.amp import autocast, GradScaler
from deep_image_prior import PSNR, SSIM
from util.transforms import random_brightness_contrast
from functools import partial
from .adversarial_attacks import PGDAttack

class Trainer():

    """
    Wrapper for pre-trainig a model.
    """
    def __init__(self, model, ray_trafos, cfg):
        self.model = model
        self.ray_trafos = ray_trafos
        self.cfg = cfg
        self.device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))
        current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
        comment = 'trainer.train'
        logdir = os.path.join(
            cfg.log_path,
            current_time + '_' + socket.gethostname() + comment)
        self.writer = tensorboardX.SummaryWriter(logdir=logdir)
        if self.cfg.use_adversarial_attacks:
            self.adversarial_attack = PGDAttack(model=model, ray_trafos=ray_trafos,
                                                steps=cfg.adversarial_attacks.steps,
                                                eps=cfg.adversarial_attacks.eps,
                                                alpha=cfg.adversarial_attacks.alpha)

    def train(self, dataset):
        if self.cfg.torch_manual_seed:
            torch.random.manual_seed(self.cfg.torch_manual_seed)
        # create PyTorch datasets
        dataset_train = dataset.create_torch_dataset(
            fold='train', reshape=((1,) + dataset.space[0].shape,
                                   (1,) + dataset.space[1].shape,
                                   (1,) + dataset.space[1].shape))

        dataset_validation = dataset.create_torch_dataset(
            fold='validation', reshape=((1,) + dataset.space[0].shape,
                                        (1,) + dataset.space[1].shape,
                                        (1,) + dataset.space[1].shape))

        criterion = torch.nn.MSELoss()
        self.init_optimizer()

        transforms = []
        for transform in self.cfg.get('transforms', []):
            if transform.name == 'random_brightness_contrast':
                transforms.append(
                        partial(
                            random_brightness_contrast,
                            brightness_shift_range=(
                                transform.brightness_shift_min,
                                transform.brightness_shift_max),
                            contrast_factor_range=(
                                transform.contrast_factor_min,
                                transform.contrast_factor_max),
                            clip_range=(transform.clip_min, transform.clip_max)
                        ))
            else:
                raise ValueError(
                        'Unknown transform \'{}\''.format(transform.name))

        # create PyTorch dataloaders
        data_loaders = {'train': DataLoader(dataset_train, batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_data_loader_workers, shuffle=True,
            pin_memory=True),
                        'validation': DataLoader(dataset_validation, batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_data_loader_workers,
            shuffle=True, pin_memory=True)}

        dataset_sizes = {'train': len(dataset_train), 'validation': len(dataset_validation)}

        self.init_scheduler()
        if self._scheduler is not None:
            schedule_every_batch = isinstance(
                self._scheduler, (CyclicLR, OneCycleLR))

        if self.cfg.perform_swa:
            self.swa_scheduler = SWALR(
                    self._optimizer,
                    anneal_strategy=self.cfg.swa.anneal_strategy,
                    anneal_epochs=self.cfg.swa.anneal_epochs,
                    swa_lr=self.cfg.swa.swa_lr)

        best_model_wts = deepcopy(self.model.state_dict())
        best_psnr = -np.inf

        self.model.to(self.device)
        self.model.train()

        if self.cfg.perform_swa:
            self.swa_model = AveragedModel(self.model)

        assert not (self.cfg.perform_swa and self.cfg.use_mixed)

        if self.cfg.use_mixed:
            scaler = GradScaler()

        num_iter = 0
        for epoch in range(self.cfg.epochs):
            # Each epoch has a training and validation phase
            for phase in ['train', 'validation']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                running_psnr = 0.0
                running_loss = 0.0
                running_size = 0
                with tqdm(data_loaders[phase],
                          desc='epoch {:d}'.format(epoch + 1),
                          disable=not self.cfg.show_pbar) as pbar:
                    for obs, fbp, gt in pbar:

                        if phase == 'train':
                            for transform in transforms:
                                fbp, gt = transform([fbp, gt])

                            if self.cfg.use_adversarial_attacks:
                                tmp_fbp = fbp.clone()
                                fbp, costs = self.adversarial_attack(obs, gt)

                                if (self.cfg.adversarial_attacks.log_interval and
                                            num_iter % self.cfg.adversarial_attacks.log_interval == 0):
                                    self.log_adversarial(
                                            num_iter=num_iter, adv_fbp=fbp,
                                            orig_fbp=tmp_fbp, gt=gt,
                                            costs=costs)

                        fbp = fbp.to(self.device)
                        gt = gt.to(self.device)

                        if self.cfg.add_randn_mask:
                            fbp = torch.cat([fbp, 0.1*torch.randn(*fbp.shape).to(self.device)], dim=1)

                        # zero the parameter gradients
                        self._optimizer.zero_grad()

                        # forward
                        # track gradients only if in train phase
                        with torch.set_grad_enabled(phase == 'train'):
                            with autocast() if self.cfg.use_mixed else contextlib.nullcontext():
                                outputs = self.model(fbp)
                                loss = criterion(outputs, gt)

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                if self.cfg.use_mixed:
                                    scaler.scale(loss).backward()
                                    scaler.unscale_(self._optimizer)
                                else:
                                    loss.backward()
                                torch.nn.utils.clip_grad_norm_(
                                    self.model.parameters(), max_norm=1)
                                if self.cfg.use_mixed:
                                    scaler.step(self._optimizer)
                                    scale = scaler.get_scale()
                                    scaler.update()
                                else:
                                    self._optimizer.step()
                                if (self._scheduler is not None and
                                        schedule_every_batch and
                                        not (self.cfg.perform_swa and
                                            epoch >= self.cfg.swa.start_epoch)):
                                    if self.cfg.use_mixed:
                                        # avoid calling scheduler before optimizer in case of nan/inf values
                                        # https://discuss.pytorch.org/t/optimizer-step-before-lr-scheduler-step-error-using-gradscaler/92930/8
                                        if scaler.get_scale() == scale:
                                            self._scheduler.step()
                                    else:
                                        self._scheduler.step()

                        for i in range(outputs.shape[0]):
                            gt_ = gt[i, 0].detach().cpu().numpy()
                            outputs_ = outputs[i, 0].detach().cpu().numpy()
                            running_psnr += PSNR(outputs_, gt_, data_range=1)

                        # statistics
                        running_loss += loss.item() * outputs.shape[0]
                        running_size += outputs.shape[0]

                        pbar.set_postfix({'phase': phase,
                                          'loss': running_loss/running_size,
                                          'psnr': running_psnr/running_size})

                        if phase == 'train':
                            num_iter += 1
                            self.writer.add_scalar('loss', running_loss/running_size, num_iter)
                            self.writer.add_scalar('psnr', running_psnr/running_size, num_iter)
                            self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], num_iter)

                        if (phase == 'train' and
                                self.cfg.get('save_learned_params_base_path') is not None and
                                (ceil(running_size / self.cfg.batch_size)) % (self.cfg.get('save_learned_params_during_epoch_interval', np.inf) // self.cfg.batch_size) == 0):
                            self.save_learned_params(
                                '{}_epochs{:d}_steps{:d}'.format(self.cfg.save_learned_params_base_path, epoch, ceil(running_size / self.cfg.batch_size)))

                    if phase == 'train':
                        if (self.cfg.perform_swa
                                and epoch >= self.cfg.swa.start_epoch):
                            self.swa_model.update_parameters(self.model)
                            self.swa_scheduler.step()
                        elif (self._scheduler is not None
                                and not schedule_every_batch):
                            if self.cfg.use_mixed:
                                # avoid calling scheduler before optimizer in case of nan/inf values
                                # https://discuss.pytorch.org/t/optimizer-step-before-lr-scheduler-step-error-using-gradscaler/92930/8
                                if scaler.get_scale() == scale:
                                    self._scheduler.step()
                            else:
                                self._scheduler.step()

                    epoch_loss = running_loss / dataset_sizes[phase]
                    epoch_psnr = running_psnr / dataset_sizes[phase]

                    if phase == 'validation':
                        self.writer.add_scalar('val_loss', epoch_loss, num_iter)
                        self.writer.add_scalar('val_psnr', epoch_psnr, num_iter)

                    # deep copy the model (if it is the best one seen so far)
                    if phase == 'validation' and epoch_psnr > best_psnr:
                        best_psnr = epoch_psnr
                        best_model_wts = deepcopy(self.model.state_dict())
                        if self.cfg.save_best_learned_params_path is not None:
                            self.save_learned_params(
                                self.cfg.save_best_learned_params_path)
                    if (phase =='validation' and
                            self.cfg.get('save_learned_params_base_path') is not None and
                            (epoch + 1) % self.cfg.get('save_learned_params_interval', self.cfg.epochs) == 0):
                        self.save_learned_params(
                            '{}_epochs{:d}'.format(self.cfg.save_learned_params_base_path, epoch + 1))

        print('Best val psnr: {:4f}'.format(best_psnr))
        if self.cfg.perform_swa:
            self.best_model = deepcopy(self.model)
            self.best_model.load_state_dict(best_model_wts)
            self.model.load_state_dict(
                    deepcopy(self.swa_model.module.state_dict()))
            if self.cfg.save_swa_learned_params_path is not None:
                self.save_learned_params(
                    self.cfg.save_swa_learned_params_path)
        else:
            self.model.load_state_dict(best_model_wts)
        self.writer.close()

    def log_adversarial(self, num_iter, adv_fbp, orig_fbp, gt, costs):
        fig, ax = plt.subplots()
        ax.plot(costs)
        ax.set_xlabel('adversarial steps')
        ax.set_ylabel('cost')
        fig.tight_layout()
        self.writer.add_figure('cost_convergence', fig, num_iter)

        fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
        im = ax[0].imshow(adv_fbp.cpu()[0].numpy().T, cmap='gray')
        fig.colorbar(im, ax=ax[0])
        ax[0].set_title('adv_fbp')
        im = ax[1].imshow(orig_fbp.cpu()[0].numpy().T, cmap='gray')
        fig.colorbar(im, ax=ax[1])
        ax[1].set_title('orig_fbp')
        im = ax[2].imshow((adv_fbp.cpu()-orig_fbp.cpu())[0].numpy().T, cmap='gray')
        fig.colorbar(im, ax=ax[2])
        ax[2].set_title('adv_fbp-orig_fbp')
        fig.tight_layout()
        self.writer.add_figure('adversarial_fbp', fig, num_iter)

        mode = self.model.training
        self.model.eval()
        with torch.no_grad():
            adv_outputs = self.model(adv_fbp.to(self.device))
            orig_outputs = self.model(orig_fbp.to(self.device))
        self.model.train(mode)

        fig, ax = plt.subplots(1, 6, figsize=(24, 3.5))
        im = ax[0].imshow(adv_outputs.cpu()[0].numpy().T, cmap='gray')
        fig.colorbar(im, ax=ax[0])
        ax[0].set_title('adv_output')
        im = ax[1].imshow(orig_outputs.cpu()[0].numpy().T, cmap='gray')
        fig.colorbar(im, ax=ax[1])
        ax[1].set_title('orig_output')
        im = ax[2].imshow((adv_outputs.cpu()-orig_outputs.cpu())[0].numpy().T, cmap='gray')
        fig.colorbar(im, ax=ax[2])
        ax[2].set_title('adv_output-orig_output')
        im = ax[3].imshow(gt.cpu()[0].numpy().T, cmap='gray')
        fig.colorbar(im, ax=ax[3])
        ax[3].set_title('gt')
        im = ax[4].imshow((adv_outputs.cpu()-gt.cpu())[0].numpy().T, cmap='gray')
        fig.colorbar(im, ax=ax[4])
        ax[4].set_title('adv_output-gt')
        im = ax[5].imshow((orig_outputs.cpu()-gt.cpu())[0].numpy().T, cmap='gray')
        fig.colorbar(im, ax=ax[5])
        ax[5].set_title('orig_output-gt')
        fig.tight_layout()
        self.writer.add_figure('adversarial_output', fig, num_iter)


    def init_optimizer(self):
        """
        Initialize the optimizer.
        """
        self._optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay)

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
        if self.cfg.scheduler.lower() == 'cosine':
            self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.cfg.epochs,
                eta_min=self.cfg.lr_min)
        elif self.cfg.scheduler.lower() == 'onecyclelr':
            self._scheduler = OneCycleLR(
                self.optimizer,
                steps_per_epoch=ceil(self.cfg.train_len / self.cfg.batch_size),
                max_lr=self.cfg.max_lr,
                epochs=self.cfg.epochs)
        else:
            raise KeyError

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

    def save_learned_params(self, path):
        """
        Save learned parameters from file.
        """
        path = path if path.endswith('.pt') else path + '.pt'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load_learned_params(self, path):
        """
        Load learned parameters from file.
        """
        # TODO: not suitable for nn.DataParallel
        path = path if path.endswith('.pt') else path + '.pt'
        map_location = ('cuda:0' if self.use_cuda and torch.cuda.is_available()
                        else 'cpu')
        state_dict = torch.load(path, map_location=map_location)
        self.model.load_state_dict(state_dict)
