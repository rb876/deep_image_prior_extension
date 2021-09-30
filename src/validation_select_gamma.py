import hydra
import os
import os.path
import json
import numpy as np
from omegaconf import DictConfig, OmegaConf
import copy
import difflib

def print_dct(dct):
    for (item, values) in dct.items():
        print(item)
        for value in values:
            print(value)

# from https://stackoverflow.com/a/47882384
def sorted_dict(d):
    return {k: sorted_dict(v) if isinstance(v, dict) else v
            for k, v in sorted(d.items())}

def collect_runs_paths_per_gamma(base_paths, raise_on_cfg_diff=True):
    paths = {}
    if isinstance(base_paths, str):
        base_paths = [base_paths]
    ref_cfg = None
    ignore_keys_in_cfg_diff = [
            'mdl.optim.gamma', 'mdl.torch_manual_seed']
    for base_path in base_paths:
        path = os.path.join(os.getcwd().partition('src')[0], base_path)
        for dirpath, dirnames, filenames in os.walk(path):
            if '.hydra' in dirnames:
                cfg = OmegaConf.load(
                        os.path.join(dirpath, '.hydra', 'config.yaml'))
                paths.setdefault(cfg.mdl.optim.gamma, []).append(dirpath)

                if ref_cfg is None:
                    ref_cfg = copy.deepcopy(cfg)
                    for k in ignore_keys_in_cfg_diff:
                        OmegaConf.update(ref_cfg, k, None)
                    ref_cfg_yaml = OmegaConf.to_yaml(sorted_dict(OmegaConf.to_object(ref_cfg)))
                    ref_dirpath = dirpath
                else:
                    cur_cfg = copy.deepcopy(cfg)
                    for k in ignore_keys_in_cfg_diff:
                        OmegaConf.update(cur_cfg, k, None)
                    cur_cfg_yaml = OmegaConf.to_yaml(sorted_dict(OmegaConf.to_object(cur_cfg)))
                    try:
                        assert cur_cfg_yaml == ref_cfg_yaml
                    except AssertionError:
                        print('Diff between config at path {} and config at path {}'.format(ref_dirpath, dirpath))
                        differ = difflib.Differ()
                        diff = differ.compare(ref_cfg_yaml.splitlines(),
                                              cur_cfg_yaml.splitlines())
                        print('\n'.join(diff))
                        # print('\n'.join([d for d in diff if d.startswith('-') or d.startswith('+')]))
                        if raise_on_cfg_diff:
                            raise

    paths = {k:sorted(v) for k, v in sorted(paths.items()) if v}
    return paths

@hydra.main(config_path='cfgs', config_name='config')
def coordinator(cfg : DictConfig) -> None:

    if not cfg.val.select_gamma_multirun_base_paths:
        raise ValueError('Please specify one or multiple base path(s) from '
                         'which to load the PSNR histories for different '
                         'values of gamma; e.g., specify the path of a '
                         'multirun that was started like this:\n'
                         '`python coordinator.py --multirun +experiment=no_pretrain data=standard_ellipses_lotus_20 \'mdl.optim.gamma=1e-5,2e-5,4e-5,6.5e-5,1e-4,2e-4,4e-4,6.5e-4,1e-3\' \'mdl.torch_manual_seed=range(10,15)\'`')

    runs = collect_runs_paths_per_gamma(
            cfg.val.select_gamma_multirun_base_paths)  # , raise_on_cfg_diff=False)  # -> check diff output manually
    print_dct(runs) # visualise runs and models checkpoints

    os.makedirs(os.path.dirname(os.path.abspath(cfg.val.select_gamma_run_paths_filename)), exist_ok=True)
    with open(cfg.val.run_paths_filename, 'w') as f:
        json.dump(runs, f, indent=1)

    os.makedirs(os.path.dirname(os.path.abspath(cfg.val.select_gamma_results_filename)), exist_ok=True)

    infos = {}
    for i_run, (gamma, histories_paths) in enumerate(runs.items()):
        psnr_histories = []
        for i_hist, dirpath in enumerate(histories_paths):
            print('history from path:\n{}\nfor gamma:\n{}'.format(dirpath, gamma))
            psnr_history = np.load(os.path.join(dirpath, cfg.save_histories_path, 'histories.npz'))['psnr'].tolist()
            psnr_histories.append(psnr_history)

        median_psnr_output = np.median(psnr_histories, axis=0)
        psnr_steady = np.median(median_psnr_output[
                cfg.val.psnr_steady_start:cfg.val.psnr_steady_stop])
        rise_time = int(np.argwhere(
            median_psnr_output > psnr_steady - cfg.val.rise_time_remaining_psnr)[0][0])

        infos[gamma] = {
                'rise_time': rise_time,
                'PSNR_steady': psnr_steady, 'PSNR_0': median_psnr_output[0]}

        with open(cfg.val.select_gamma_results_filename, 'w') as f:
            json.dump(infos, f, indent=1)

    def key(info):
        return -info['PSNR_steady']

    infos = {k: v for k, v in sorted(infos.items(), key=lambda item: key(item[1]))}

    os.makedirs(os.path.dirname(os.path.abspath(cfg.val.select_gamma_results_sorted_filename)), exist_ok=True)
    with open(cfg.val.select_gamma_results_sorted_filename, 'w') as f:
        json.dump(infos, f, indent=1)

    print('best gamma(s):\n{}'.format('\n'.join(['{:g}'.format(k) for k, v in infos.items() if key(v) == key(list(infos.values())[0])])))

if __name__ == '__main__':
    coordinator()
