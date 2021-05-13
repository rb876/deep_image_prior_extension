import os
import json
import odl
import h5py
import torch
import numpy as np
import torch.nn.functional as F
from copy import deepcopy
from math import ceil
from tqdm import tqdm, trange
from inspect import signature, Parameter
from warnings import warn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CyclicLR, OneCycleLR
from deep_image_prior import PSNR, SSIM

class Trainer():

    """
    Wrapper for pre-trainig a model.
    """
    def __init__(self, model, configs):
        self.model = model
        self.configs = configs
        self.device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))

    def train(self, dataset):
        if self.configs.torch_manual_seed:
            torch.random.manual_seed(self.configs.torch_manual_seed)
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

        # create PyTorch dataloaders
        data_loaders = {'train': DataLoader(dataset_train, batch_size=self.configs.batch_size,
            num_workers=self.configs.num_data_loader_workers, shuffle=True,
            pin_memory=True),
                        'validation': DataLoader(dataset_validation, batch_size=self.configs.batch_size,
            num_workers=self.configs.num_data_loader_workers,
            shuffle=True, pin_memory=True)}

        dataset_sizes = {'train': len(dataset_train), 'validation': len(dataset_validation)}

        self.init_scheduler()
        if self._scheduler is not None:
            schedule_every_batch = isinstance(
                self._scheduler, (CyclicLR, OneCycleLR))

        best_model_wts = deepcopy(self.model.state_dict())
        best_psnr = 0

        self.model.to(self.device)
        self.model.train()

        for epoch in range(self.configs.epochs):
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
                          disable=not self.configs.show_pbar) as pbar:
                    for _, fbp, gt in pbar:

                        fbp = fbp.to(self.device)
                        gt = gt.to(self.device)

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
                                        schedule_every_batch):
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

                    if (self._scheduler is not None
                            and not schedule_every_batch):
                        self._scheduler.step()

                    epoch_loss = running_loss / dataset_sizes[phase]
                    epoch_psnr = running_psnr / dataset_sizes[phase]

                    # deep copy the model (if it is the best one seen so far)
                    if phase == 'validation' and epoch_psnr > best_psnr:
                        best_psnr = epoch_psnr
                        best_model_wts = deepcopy(self.model.state_dict())
                        if self.configs.save_best_learned_params_path is not None:
                            self.save_learned_params(
                                self.configs.save_best_learned_params_path)

        print('Best val psnr: {:4f}'.format(best_psnr))
        self.model.load_state_dict(best_model_wts)

    def init_optimizer(self):
        """
        Initialize the optimizer.
        """
        self._optimizer = torch.optim.Adam(self.model.parameters(), lr=self.configs.lr, weight_decay=self.configs.weight_decay)

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
        if self.configs.scheduler.lower() == 'cosine':
            self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.configs.epochs,
                eta_min=self.configs.lr_min)
        else:
            raise NotImplementedError

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
        path = os.path.abspath(path)
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
