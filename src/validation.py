import hydra
import os
import os.path
import json
import h5py
import numpy as np
from omegaconf import DictConfig, OmegaConf
from dataset import get_validation_data, get_standard_dataset
import torch
from torch.utils.data import DataLoader
from deep_image_prior import DeepImagePriorReconstructor
from pre_training import Trainer
from copy import deepcopy
import difflib

def print_dct(dct):
    for (item, values) in dct.items():
        print(item)
        for value in values:
            print(value)

def collect_runs_paths(base_paths):
    paths = {}
    if isinstance(base_paths, str):
        base_paths = [base_paths]
    for base_path in base_paths:
        path = os.path.join(os.getcwd().partition('src')[0], base_path)
        for dirpath, dirnames, filenames in os.walk(path):
            paths[dirpath] = sorted([f for f in filenames if f.endswith(".pt")])
    paths = {k:v for k, v in sorted(paths.items()) if v}
    return paths

def val_sub_path(i_run, i_ckpt, i, i_sample):
    sub_path = os.path.join('run_{:d}'.format(i_run),
                            'ckpt_{:d}'.format(i_ckpt),
                            'rep_{:d}'.format(i),
                            'sample_{:d}'.format(i_sample))
    return sub_path

@hydra.main(config_path='cfgs', config_name='config')
def coordinator(cfg : DictConfig) -> None:

    assert cfg.mdl.load_pretrain_model, \
    'load_pretrain_model is False, assertion failed'

    runs = collect_runs_paths(cfg.val.multirun_base_paths)
    print_dct(runs) # visualise runs and models checkpoints

    if cfg.val.load_histories_from_run_path is not None:
        orig_cfg = OmegaConf.load(os.path.join(
                cfg.val.load_histories_from_run_path, '.hydra', 'config.yaml'))

        with open(os.path.join(cfg.val.load_histories_from_run_path,
                               cfg.val.run_paths_filename), 'r') as f:
            assert json.load(f) == runs

        print('Diff of config in run path from which histories are loaded and '
              'current config:')
        differ = difflib.Differ()
        diff = differ.compare(OmegaConf.to_yaml(orig_cfg).splitlines(),
                              OmegaConf.to_yaml(cfg).splitlines())
        print('\n'.join(diff))

    dataset, ray_trafos = get_standard_dataset(cfg.data.name, cfg.data)
    ray_trafo = {'ray_trafo_module': ray_trafos['ray_trafo_module'],
                 'reco_space': dataset.space[1],
                 'observation_space': dataset.space[0]
                 }
    val_dataset = get_validation_data(cfg.data.name, cfg.data)

    os.makedirs(os.path.dirname(os.path.abspath(cfg.val.run_paths_filename)), exist_ok=True)
    with open(cfg.val.run_paths_filename, 'w') as f:
        json.dump(runs, f, indent=1)

    os.makedirs(os.path.dirname(os.path.abspath(cfg.val.results_filename)), exist_ok=True)

    infos = {}
    log_path_base = cfg.mdl.log_path
    seed = cfg.mdl.torch_manual_seed
    cfg_mdl_val = deepcopy(cfg.mdl)
    for k, v in cfg.val.mdl_overrides.items():
        OmegaConf.update(cfg_mdl_val, k, v, merge=False)
    for i_run, (directory_path, checkpoints_paths) in enumerate(runs.items()):
        for i_ckpt, filename in enumerate(checkpoints_paths):
            print('model:\n{}\nfrom path:\n{}'.format(filename, directory_path))
            cfg_mdl_val.learned_params_path = os.path.join(directory_path, filename)
            psnr_histories = []
            for i in range(cfg.val.num_repeats):
                for i_sample, (noisy_obs, fbp, *gt) in enumerate(val_dataset):
                    gt = gt[0] if gt else None

                    if cfg.val.load_histories_from_run_path is not None:
                        load_histories_path = os.path.join(
                                cfg.val.load_histories_from_run_path,
                                cfg.save_histories_path,
                                val_sub_path(i_run=i_run, i_ckpt=i_ckpt, i=i, i_sample=i_sample))
                        psnr_history = np.load(os.path.join(load_histories_path, 'histories.npz'))['psnr'].tolist()
                    else:
                        cfg_mdl_val.torch_manual_seed = seed + i
                        cfg_mdl_val.log_path = os.path.join(
                                log_path_base, val_sub_path(i_run=i_run, i_ckpt=i_ckpt, i=i, i_sample=i_sample))
                        reconstructor = DeepImagePriorReconstructor(**ray_trafo, cfg=cfg_mdl_val)
                        reco, *optional_out = reconstructor.reconstruct(
                                noisy_obs.float().unsqueeze(dim=0), fbp.unsqueeze(dim=0), gt.unsqueeze(dim=0),
                                return_histories=True,
                                return_iterates=cfg.save_iterates_path is not None)
                        psnr_history = optional_out[0]['psnr']

                        if cfg.save_histories_path is not None:
                            histories = {k: np.array(v, dtype=np.float32)
                                         for k, v in optional_out[0].items()}
                            save_histories_path = os.path.join(
                                    cfg.save_histories_path,
                                    val_sub_path(i_run=i_run, i_ckpt=i_ckpt, i=i, i_sample=i_sample))
                            os.makedirs(save_histories_path, exist_ok=True)
                            np.savez(os.path.join(save_histories_path, 'histories.npz'),
                                     **histories)
                        if cfg.save_iterates_path is not None:
                            iterates = optional_out[1]
                            iterates_iters = optional_out[2]
                            save_iterates_path = os.path.join(
                                    cfg.save_iterates_path,
                                    val_sub_path(i_run=i_run, i_ckpt=i_ckpt, i=i, i_sample=i_sample))
                            os.makedirs(save_iterates_path, exist_ok=True)
                            np.savez_compressed(
                                    os.path.join(save_iterates_path, 'iterates.npz'),
                                    iterates=np.asarray(iterates),
                                    iterates_iters=iterates_iters)

                    psnr_histories.append(psnr_history)

            mean_psnr_output = np.median(psnr_histories, axis=0)
            psnr_steady = np.median(mean_psnr_output[
                    cfg.val.psnr_steady_start:cfg.val.psnr_steady_stop])
            rise_time = int(np.argwhere(
                mean_psnr_output > psnr_steady - cfg.val.rise_time_remaining_psnr)[0][0])

            infos[os.path.join(directory_path, filename)] = {
                    'rise_time': rise_time,
                    'PSNR_steady': psnr_steady, 'PSNR_0': mean_psnr_output[0]}

            with open(cfg.val.results_filename, 'w') as f:
                json.dump(infos, f, indent=1)

    infos_unfiltered = infos
    infos = {}
    max_psnr_steady = max(v['PSNR_steady'] for v in infos_unfiltered.values())
    for model_path, info in infos_unfiltered.items():
        if info['PSNR_steady'] < max_psnr_steady - cfg.val.tolerated_diff_to_max_psnr_steady:
            print('Excluding model \'{}\' because its steady PSNR differs too '
                  'much from the maximum steady PSNR: {:f} < {:f} - {:f}.'
                  .format(model_path, info['PSNR_steady'], max_psnr_steady,
                          cfg.val.tolerated_diff_to_max_psnr_steady))
        else:
            infos[model_path] = infos_unfiltered[model_path]

    def key(info):
        return info['rise_time']

    infos = {k: v for k, v in sorted(infos.items(), key=lambda item: key(item[1]))}

    os.makedirs(os.path.dirname(os.path.abspath(cfg.val.results_sorted_filename)), exist_ok=True)
    with open(cfg.val.results_sorted_filename, 'w') as f:
        json.dump(infos, f, indent=1)

    print('best model(s):\n{}'.format('\n'.join([k for k, v in infos.items() if key(v) == key(list(infos.values())[0])])))

if __name__ == '__main__':
    coordinator()
