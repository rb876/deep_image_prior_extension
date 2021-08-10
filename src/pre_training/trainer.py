import os
import json
import socket
import datetime
import odl
import h5py
import torch
import numpy as np
import torch.nn.functional as F
import tensorboardX
from copy import deepcopy
from math import ceil
from tqdm import tqdm, trange
from inspect import signature, Parameter
from warnings import warn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CyclicLR, OneCycleLR
from torch.optim.swa_utils import AveragedModel, SWALR
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
            self.adversarial_attack = PGDAttack(model=model, ray_trafos=ray_trafos)

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
        best_psnr = 0

        self.model.to(self.device)
        self.model.train()

        if self.cfg.perform_swa:
            self.swa_model = AveragedModel(self.model)

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
                                if num_iter%500 == 0:
                                    self.writer.add_image('original_fbp', tmp_fbp[0, ...].cpu().numpy(), num_iter)
                                    self.writer.add_image('adversarial_fbp', fbp[0, ...].cpu().numpy(), num_iter)
                                    self.writer.add_image('diff.', (fbp[0, ...] - tmp_fbp[0, ...]).cpu().numpy(), num_iter)

                        fbp = fbp.to(self.device)
                        gt = gt.to(self.device)

                        if self.cfg.add_randn_mask:
                            fbp = torch.cat([fbp, 0.1*torch.randn(*fbp.shape).to(self.device)], dim=1)

                        # zero the parameter gradients
                        self._optimizer.zero_grad()

                        # forward
                        # track gradients only if in train phase
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = self.model(fbp)
                            loss = criterion(outputs, gt)

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                torch.nn.utils.clip_grad_norm_(
                                    self.model.parameters(), max_norm=1)
                                self._optimizer.step()
                                if (self._scheduler is not None and
                                        schedule_every_batch and
                                        not (self.cfg.perform_swa and
                                             epoch >= self.cfg.swa.start_epoch)):
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

                    if phase == 'train':
                        if (self.cfg.perform_swa
                                and epoch >= self.cfg.swa.start_epoch):
                            self.swa_model.update_parameters(self.model)
                            self.swa_scheduler.step()
                        elif (self._scheduler is not None
                                and not schedule_every_batch):
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
        path = os.path.join(os.getcwd().partition('src')[0], path)
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
