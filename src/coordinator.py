import hydra
import os
import h5py
import numpy as np
from omegaconf import DictConfig
from dataset import get_standard_dataset, get_test_data, get_validation_data
import torch
from torch.utils.data import DataLoader
from deep_image_prior import DeepImagePriorReconstructor
from pre_training import Trainer
from copy import deepcopy

@hydra.main(config_path='cfgs', config_name='config')
def coordinator(cfg : DictConfig) -> None:

    dataset, ray_trafos = get_standard_dataset(cfg.data.name, cfg.data)

    obs_shape = dataset.space[0].shape
    im_shape = dataset.space[1].shape

    if cfg.validation_run:
        if cfg.data.validation_data:
            dataset_test = get_validation_data(cfg.data.name, cfg.data)
        else:
            dataset_test = dataset.create_torch_dataset(
                fold='validation', reshape=((1,) + obs_shape,
                                            (1,) + im_shape,
                                            (1,) + im_shape))
    else:
        if cfg.data.test_data:
            dataset_test = get_test_data(cfg.data.name, cfg.data)
        else:
            dataset_test = dataset.create_torch_dataset(
                fold='test', reshape=((1,) + obs_shape,
                                      (1,) + im_shape,
                                      (1,) + im_shape))

    ray_trafo = {'ray_trafo_module': ray_trafos['ray_trafo_module'],
                 'reco_space': dataset.space[1],
                 'observation_space': dataset.space[0]
                 }

    if cfg.torch_manual_seed_pretrain_init_model:
        torch.random.manual_seed(cfg.torch_manual_seed_pretrain_init_model)

    reconstructor = DeepImagePriorReconstructor(**ray_trafo, cfg=cfg.mdl)
    model = deepcopy(reconstructor.model)
    if cfg.pretraining:
        trn_ray_trafos = ({'smooth_pinv_ray_trafo_module':
                               ray_trafos['smooth_pinv_ray_trafo_module']}
                          if cfg.trn.use_adversarial_attacks else {})
        Trainer(model=model,
                ray_trafos=trn_ray_trafos,
                cfg=cfg.trn).train(dataset)

    os.makedirs(cfg.save_reconstruction_path, exist_ok=True)
    if cfg.save_histories_path is not None:
        os.makedirs(cfg.save_histories_path, exist_ok=True)
    if cfg.save_iterates_path is not None:
        os.makedirs(cfg.save_iterates_path, exist_ok=True)
    if cfg.save_iterates_params_path is not None:
        os.makedirs(cfg.save_iterates_params_path, exist_ok=True)

    filename = os.path.join(cfg.save_reconstruction_path,'recos.hdf5')
    file = h5py.File(filename, 'w')
    recos_dataset = file.create_dataset('recos',
            shape=(1,) + im_shape, maxshape=(1,) + im_shape, dtype=np.float32,
            chunks=True)

    dataloader = DataLoader(dataset_test, batch_size=1, num_workers=0,
                            shuffle=True, pin_memory=True)

    for i, (noisy_obs, fbp, *gt) in enumerate(dataloader):
        gt = gt[0] if gt else None
        reco, *optional_out = reconstructor.reconstruct(
                noisy_obs.float(), fbp, gt,
                return_histories=cfg.save_histories_path is not None,
                return_iterates=cfg.save_iterates_path is not None,
                return_iterates_params=cfg.save_iterates_params_path is not None)
        recos_dataset[i] = reco
        if cfg.save_histories_path is not None:
            histories = optional_out.pop(0)
            histories = {k: np.array(v, dtype=np.float32)
                         for k, v in histories.items()}
            np.savez(os.path.join(cfg.save_histories_path, 'histories.npz'),
                     **histories)
        if cfg.save_iterates_path is not None:
            iterates = optional_out.pop(0)
            iterates_iters = optional_out.pop(0)
            np.savez_compressed(
                    os.path.join(cfg.save_iterates_path, 'iterates.npz'),
                    iterates=np.asarray(iterates),
                    iterates_iters=iterates_iters)
        if cfg.save_iterates_params_path is not None:
            iterates_params = optional_out.pop(0)
            iterates_params_iters = optional_out.pop(0)
            for params, iters in zip(iterates_params, iterates_params_iters):
                torch.save(params,
                           os.path.join(cfg.save_iterates_params_path,
                                        'params_iters{:d}.pt'.format(iters)))

if __name__ == '__main__':
    coordinator()
