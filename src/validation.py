import hydra
import os
import os.path
import h5py
import numpy as np
from omegaconf import DictConfig
from dataset import get_validation_data, get_standard_dataset
import torch
from torch.utils.data import DataLoader
from deep_image_prior import DeepImagePriorReconstructor
from pre_training import Trainer
from copy import deepcopy

def print_dct(dct):
    for (item, values) in dct.items():
        print(item)
        for value in values:
            print(value)

def collect_runs_paths(base_path):
    paths = {}
    path = os.path.join(os.getcwd().partition('src')[0], base_path)
    for dirpath, dirnames, filenames in os.walk(path):
        paths[dirpath] = [f for f in filenames if f.endswith(".pt")]
    paths = {k:v for k, v in paths.items() if v}
    return paths

@hydra.main(config_path='cfgs', config_name='config')
def coordinator(cfg : DictConfig) -> None:

    assert cfg.mdl.load_pretrain_model, \
    'load_pretrain_model is False, assertion failed'

    runs = collect_runs_paths(cfg.val.multirun_base_path)
    print_dct(runs) # visualise runs and models checkpoints
    dataset, ray_trafos = get_standard_dataset(cfg.data.name, cfg.data)
    ray_trafo = {'ray_trafo_module': ray_trafos['ray_trafo_module'],
                 'reco_space': dataset.space[1],
                 'observation_space': dataset.space[0]
                 }
    val_datatset = get_validation_data(cfg.data.name, cfg.data)

    infos = {}
    seed = cfg.mdl.torch_manual_seed
    for (directory_path, checkpoints_paths) in runs.items():
        for filename in checkpoints_paths:
            print('loading model:\n{}\nfrom path:\n{}'.format(filename, directory_path))
            cfg.mdl.learned_params_path = os.path.join(directory_path, filename)
            tmp = []
            for i in range(cfg.val.num_repeats):
                for i, (noisy_obs, fbp, *gt) in enumerate(val_datatset):
                    gt = gt[0] if gt else None
                    cfg.mdl.torch_manual_seed = seed + i
                    reconstructor = DeepImagePriorReconstructor(**ray_trafo, cfg=cfg.mdl)
                    reco, *optional_out = reconstructor.reconstruct(
                            noisy_obs.float().unsqueeze(dim=0), fbp.unsqueeze(dim=0), gt.unsqueeze(dim=0),
                            return_histories=cfg.save_histories_path is not None,
                            return_iterates=cfg.save_iterates_path is not None)
                tmp.append(optional_out[0]['psnr'])
            mean_psnr_output = np.mean(tmp, axis=0)
            infos[os.path.join(directory_path, filename)] = {'rise_time': None,
                    'PSNR_steady': None, 'PSNR_0': mean_psnr_output[0]}

if __name__ == '__main__':
    coordinator()
