"""
Utility functions for accessing the results of the hydra runs.
"""
import os
import yaml
from omegaconf import OmegaConf
import h5py
import numpy as np

def get_run_cfg(run_path):
    cfg = OmegaConf.load(os.path.join(run_path, '.hydra', 'config.yaml'))
    return cfg

def get_run_experiment_name(run_path):
    with open(os.path.join(run_path, '.hydra', 'overrides.yaml'), 'r') as f:
        overrides = yaml.load(f, Loader=yaml.FullLoader)
        experiment_override = next(
                (ov for ov in overrides if ov.startswith('+experiment=')),
                None)
        experiment_name = experiment_override.split('=')[1]

    return experiment_name

def get_run_reconstruction(run_path):
    cfg = get_run_cfg(run_path)

    with h5py.File(os.path.join(
            run_path, cfg['save_reconstruction_path'], 'recos.hdf5'), 'r') as f:
        recos = np.asarray(f['recos'])

    return recos

def get_run_histories(run_path):
    cfg = get_run_cfg(run_path)

    histories = np.load(os.path.join(
            run_path, cfg['save_histories_path'], 'histories.npz'))

    return histories

def get_run_iterates(run_path):
    cfg = get_run_cfg(run_path)

    iterates_dict = np.load(os.path.join(
            run_path, cfg['save_iterates_path'], 'iterates.npz'))
    iterates = iterates_dict['iterates']
    iterates_iters = iterates_dict['iterates_iters']

    return iterates, iterates_iters

def get_multirun_num_runs(run_path_multirun):
    num_runs = 0
    while os.path.isdir(os.path.join(run_path_multirun,
                                     '{:d}'.format(num_runs))):
        num_runs += 1
    return num_runs

def get_multirun_cfgs(run_path_multirun, sub_runs=None):
    if sub_runs is None:
        sub_runs = range(get_multirun_num_runs(run_path_multirun))

    cfgs = [get_run_cfg(os.path.join(run_path_multirun,
                                     '{:d}'.format(i)))
            for i in sub_runs]

    return cfgs

def get_multirun_experiment_names(run_path_multirun, sub_runs=None):
    if sub_runs is None:
        sub_runs = range(get_multirun_num_runs(run_path_multirun))

    experiment_names = [get_run_experiment_name(os.path.join(run_path_multirun,
                                                             '{:d}'.format(i)))
                        for i in sub_runs]

    return experiment_names

def get_multirun_reconstructions(run_path_multirun, sub_runs=None):
    if sub_runs is None:
        sub_runs = range(get_multirun_num_runs(run_path_multirun))

    cfgs = get_multirun_cfgs(run_path_multirun, sub_runs=sub_runs)
    assert len(sub_runs) == len(cfgs)

    recos_list = [get_run_reconstruction(os.path.join(run_path_multirun,
                                                      '{:d}'.format(i)))
                  for i in sub_runs]

    return recos_list

def get_multirun_histories(run_path_multirun, sub_runs=None):
    if sub_runs is None:
        sub_runs = range(get_multirun_num_runs(run_path_multirun))

    cfgs = get_multirun_cfgs(run_path_multirun, sub_runs=sub_runs)
    assert len(sub_runs) == len(cfgs)

    histories_list = [get_run_histories(os.path.join(run_path_multirun,
                                                     '{:d}'.format(i)))
                      for i, cfg in zip(sub_runs, cfgs)]

    return histories_list

def get_multirun_iterates(run_path_multirun, sub_runs=None):
    if sub_runs is None:
        sub_runs = range(get_multirun_num_runs(run_path_multirun))

    cfgs = get_multirun_cfgs(run_path_multirun, sub_runs=sub_runs)
    assert len(sub_runs) == len(cfgs)

    iterates_list, iterates_iters_list = zip(
            *[get_run_iterates(os.path.join(run_path_multirun,
                                            '{:d}'.format(i)))
              for i, cfg in zip(sub_runs, cfgs)])
    iterates_list = list(iterates_list)
    iterates_iters_list = list(iterates_iters_list)

    return iterates_list, iterates_iters_list

def uses_swa_weights(cfg):
     return (os.path.splitext(cfg['mdl']['learned_params_path'])[0]
             .endswith('_swa'))
