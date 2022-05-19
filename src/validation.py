import hydra
import os
import os.path
import re
import json
from omegaconf import DictConfig, OmegaConf
from dataset import get_validation_data, get_standard_dataset
from validation import validate_model, val_sub_sub_path
from copy import deepcopy
import difflib

def print_dct(dct):
    for (item, values) in dct.items():
        print(item)
        for value in values:
            print(value)

def collect_runs_paths(base_paths, exclude_re=None):
    paths = {}
    if isinstance(base_paths, str):
        base_paths = [base_paths]
    for base_path in base_paths:
        path = os.path.join(os.getcwd().partition('src')[0], base_path)
        for dirpath, dirnames, filenames in os.walk(path):
            paths[dirpath] = sorted(
                    [f for f in filenames if f.endswith(".pt") and (
                            exclude_re is None or
                            not re.fullmatch(exclude_re, f))])
    paths = {k:v for k, v in sorted(paths.items()) if v}
    return paths

def val_sub_path_ckpt(i_run, i_ckpt):
    sub_path_ckpt = os.path.join('run_{:d}'.format(i_run),
                                 'ckpt_{:d}'.format(i_ckpt))
    return sub_path_ckpt

def val_sub_path(i_run, i_ckpt, i, i_sample):
    sub_path = os.path.join(val_sub_path_ckpt(i_run, i_ckpt),
                            val_sub_sub_path(i, i_sample))
    return sub_path

@hydra.main(config_path='cfgs', config_name='config')
def coordinator(cfg : DictConfig) -> None:

    assert cfg.mdl.load_pretrain_model, \
    'load_pretrain_model is False, assertion failed'

    runs = collect_runs_paths(cfg.val.multirun_base_paths,
                              exclude_re=cfg.val.get('exclude_re'))
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

    infos = {}
    log_path_base = cfg.mdl.log_path
    seed = cfg.mdl.torch_manual_seed
    cfg_mdl_val = deepcopy(cfg.mdl)

    # load baseline
    baseline_cfg = OmegaConf.load(os.path.join(
            cfg.val.baseline_run_path, '.hydra', 'config.yaml'))
    print('Diff of config in baseline run path and current config:')
    differ = difflib.Differ()
    diff = differ.compare(OmegaConf.to_yaml(baseline_cfg).splitlines(),
                          OmegaConf.to_yaml(cfg).splitlines())
    print('\n'.join(diff))
    dummy_baseline_PSNR_steady = cfg.val.get('dummy_baseline_PSNR_steady', None)
    if dummy_baseline_PSNR_steady is None:
        with open(os.path.join(cfg.val.baseline_run_path, baseline_cfg.val.results_filename), 'r') as f:
            infos['baseline'] = json.load(f)['baseline']
        baseline_psnr_steady = infos['baseline']['PSNR_steady']
    else:
        baseline_psnr_steady = dummy_baseline_PSNR_steady

    os.makedirs(os.path.dirname(os.path.abspath(cfg.val.results_filename)), exist_ok=True)

    with open(cfg.val.results_filename, 'w') as f:
        json.dump(infos, f, indent=1)

    # validate pretrained models
    for k, v in cfg.val.mdl_overrides.items():
        OmegaConf.update(cfg_mdl_val, k, v, merge=False)
    for i_run, (directory_path, checkpoints_paths) in enumerate(runs.items()):
        for i_ckpt, filename in enumerate(checkpoints_paths):
            print('model:\n{}\nfrom path:\n{}'.format(filename, directory_path))
            cfg_mdl_val.learned_params_path = os.path.join(directory_path, filename)
            _, info = validate_model(
                    val_dataset=val_dataset, ray_trafo=ray_trafo,
                    val_sub_path_mdl=val_sub_path_ckpt(i_run=i_run, i_ckpt=i_ckpt),
                    baseline_psnr_steady=baseline_psnr_steady, seed=seed,
                    log_path_base=log_path_base,
                    cfg=cfg, cfg_mdl_val=cfg_mdl_val)

            infos[os.path.join(directory_path, filename)] = info

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
        return (
                info['rise_time_to_baseline']
                if info['rise_time_to_baseline'] is not None else float('inf'))

    infos = {k: v for k, v in sorted(infos.items(), key=lambda item: key(item[1]))}

    os.makedirs(os.path.dirname(os.path.abspath(cfg.val.results_sorted_filename)), exist_ok=True)
    with open(cfg.val.results_sorted_filename, 'w') as f:
        json.dump(infos, f, indent=1)

    print('best model(s):\n{}'.format('\n'.join([k for k, v in infos.items() if key(v) == key(list(infos.values())[0])])))

if __name__ == '__main__':
    coordinator()
