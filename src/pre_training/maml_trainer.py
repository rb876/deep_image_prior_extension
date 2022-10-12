import os
import random
import socket
import datetime
from itertools import islice
import torch
import numpy as np
import functorch as ftch
import tensorboardX
from copy import deepcopy
from math import ceil
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CyclicLR, OneCycleLR
from deep_image_prior import PSNR
from .maml_utils import one_step_gd_update_wtups

# taken from https://gist.githubusercontent.com/MFreidank/821cc87b012c53fade03b0c7aba13958/raw/41ad2c08a019c72b278866e1b02b355f1fce44a4/infinite_dataloader.py
class InfiniteDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize an iterator over the dataset.
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            # Dataset exhausted, use a new fresh iterator.
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch

class MetaTrainer():

    """
    Wrapper for meta-pre-trainig a model.
    """
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))
        current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
        comment = 'trainer.train'
        logdir = os.path.join(
            cfg.log_path,
            current_time + '_' + socket.gethostname() + comment)
        self.writer = tensorboardX.SummaryWriter(logdir=logdir)
        self.func_model_with_input, self.func_params = ftch.make_functional(self.model)
        ftch._src.make_functional.extract_weights(self.model) # self.model.parameters() will be empty

    def metatrain(self, list_of_datasets):
        if self.cfg.torch_manual_seed:
            torch.random.manual_seed(self.cfg.torch_manual_seed)
        # create PyTorch datasets
        list_datasets_train = [dataset.create_torch_dataset(
            fold='train', reshape=((1,) + dataset.space[0].shape,
                                   (1,) + dataset.space[1].shape,
                                   (1,) + dataset.space[1].shape)) for dataset in list_of_datasets]

        list_datasets_validation = [dataset.create_torch_dataset(
            fold='validation', reshape=((1,) + dataset.space[0].shape,
                                        (1,) + dataset.space[1].shape,
                                        (1,) + dataset.space[1].shape)) for dataset in list_of_datasets]


        num_tasks = len(list_datasets_train)

        def _create_data_loader(dset):
            return InfiniteDataLoader(dset, batch_size=self.cfg.batch_size,
                    num_workers=self.cfg.num_data_loader_workers, shuffle=True,
                    pin_memory=True)
        # create PyTorch dataloaders
        data_loaders = {'train': [_create_data_loader(dataset_train) for dataset_train in list_datasets_train],
                        'validation': [_create_data_loader(dataset_validation) for dataset_validation in list_datasets_validation]}

        self.init_optimizer()
        self.init_scheduler()
        if self._scheduler is not None:
            schedule_every_batch = isinstance(
                self._scheduler, (CyclicLR, OneCycleLR))
        criterion = torch.nn.MSELoss()

        best_model_func_params = deepcopy(self.func_params)
        best_psnr = -np.inf
        with tqdm(range(int(self.cfg.meta_trainer.num_iters))) as pbar:
            for it in pbar:
                id_tasks = [
                        random.randint(0, num_tasks-1) for _ in range(
                            self.cfg.meta_trainer.num_tasks_per_iter)
                    ]
                picked_dataloaders = [
                    data_loaders['train'][id_task] for id_task in id_tasks]

                self._optimizer.zero_grad()
                # inner loop
                inn_loop_data = []
                for dataset in picked_dataloaders:
                    _, fbp, gt = next(dataset)
                    fbp, gt= fbp.to(self.device), gt.to(self.device)
                    outputs = self.func_model_with_input(self.func_params, fbp)
                    loss = criterion(outputs, gt)
                    ftch_grads = torch.autograd.grad(loss, self.func_params, create_graph=True)
                    inn_loop_data.append(
                        (one_step_gd_update_wtups(
                            self.func_params, ftch_grads, self.cfg.meta_trainer.inner_loop_optim.lr), fbp, gt) )
                # outer loop
                all_loss = torch.tensor([0.0]).to(self.device)
                for data in inn_loop_data:
                    one_step_gd_update_func_params, fbp, gt = data
                    outputs = self.func_model_with_input(one_step_gd_update_func_params, fbp)
                    loss = criterion(outputs, gt)
                    all_loss += loss

                all_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1)
                self._optimizer.step()

                # overall training stats
                all_psnrs = []
                for data in inn_loop_data:
                    _, fbp, gt = data
                    outputs = self.func_model_with_input(self.func_params, fbp)
                    all_psnrs.append(PSNR(outputs.detach().cpu(), gt.detach().cpu(), data_range=1))
                if (self._scheduler is not None and
                        schedule_every_batch):
                    self._scheduler.step()

                self.writer.add_scalar('loss', all_loss.item(), it)
                self.writer.add_scalar('psnr', np.mean(all_psnrs), it)
                self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], it)

                pbar.set_description(f'loss: {all_loss.item()}, psnr: {np.mean(all_psnrs)}')
                if ( (it + 1) % self.cfg.meta_trainer.eval_every_num_iters) == 0:
                    all_val_loss, all_val_psnrs = [], []
                    for val_loader in data_loaders['validation']:
                        for _, fbp, gt in islice(val_loader, self.cfg.meta_trainer.num_val_iters_per_task):
                            fbp, gt = fbp.to(self.device), gt.to(self.device)
                            outputs = self.func_model_with_input(self.func_params, fbp)
                            loss = criterion(outputs, gt)
                            all_val_loss.append(loss.item())
                            all_val_psnrs.append(PSNR(outputs.detach().cpu(), gt.detach().cpu(), data_range=1))
                    self.writer.add_scalar('val_loss', np.mean(all_val_loss), it)
                    self.writer.add_scalar('val_psnr', np.mean(all_val_psnrs), it)

                    pbar.set_description(
                            f'valloss: {np.mean(all_val_loss)}, valpsnr: {np.mean(all_val_psnrs)}'
                        )
                    # deep copy the model (if it is the best one seen so far)
                    if np.mean(all_val_psnrs) > best_psnr:
                        best_psnr = np.mean(all_val_psnrs)
                        best_model_func_params = deepcopy(self.func_params)
                        if self.cfg.save_best_learned_params_path is not None:
                            ftch._src.make_functional.load_state(
                                self.model,
                                best_model_func_params,
                                [name for name, _ in self.model.named_parameters()]
                                )
                            self.save_learned_params(
                                self.cfg.save_best_learned_params_path)

        ftch._src.make_functional.load_state(
                self.model,
                best_model_func_params,
                [name for name, _ in self.model.named_parameters()]
            )

        print('Best val psnr: {:4f}'.format(best_psnr))
        self.writer.close()

    def init_optimizer(self):
        """
        Initialize the optimizer.
        """
        self._optimizer = torch.optim.Adam(
                self.func_params,
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
        self.func_model_with_input, self.func_params = ftch.make_functional(self.model)
        ftch._src.make_functional.extract_weights(self.model) # self.model.parameters() will be empty
