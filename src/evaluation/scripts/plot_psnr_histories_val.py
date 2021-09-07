import os
import json
from warnings import warn
import numpy as np
import yaml
from dataset import get_validation_data
from evaluation.utils import get_run_cfg
from evaluation.display_utils import data_title_dict, get_title_from_run_spec

import matplotlib.pyplot as plt

PATH = '/localdata/jleuschn/experiments/deep_image_prior_extension/'

FIG_PATH = '.'

save_fig = True

with open('../runs.yaml', 'r') as f:
    runs = yaml.load(f, Loader=yaml.FullLoader)

data = 'ellipses_lotus_20'

val_run_spec = {
    'experiment': 'pretrain_only_fbp',
    'name': 'lr0.0001',
    'name_title': None,
    'only_run_indices': range(0, 1),
}

val_run_title = ''  # falsy -> auto from run_spec
val_run_filename = ''  # falsy -> auto from run_spec

num_iters_dict = {
    'ellipses_lotus_20': 10000,
    'ellipses_lotus_limited_30': 5000,
}

data_title = data_title_dict[data]
num_iters = num_iters_dict[data]

fig, ax = plt.subplots(figsize=(10, 6))

def get_auto_xlim_with_padding(ax, xmin, xmax):
    hmin = ax.axvline(xmin, linestyle='')
    hmax = ax.axvline(xmax, linestyle='')
    xlim = ax.get_xlim()
    hmin.remove()
    hmax.remove()
    del hmin
    del hmax
    return xlim

xlim = get_auto_xlim_with_padding(ax, 0, num_iters)

experiment = val_run_spec['experiment']
available_runs = runs['validation'][data][experiment]
if val_run_spec.get('name') is not None:
    run = next((r for r in available_runs if
                r.get('name') == val_run_spec['name']),
               None)
    if run is None:
        raise ValueError('Unknown validation run name "{}" for reconstruction '
                         'with data={}, experiment={}'.format(
                                 val_run_spec['name'], data, experiment))
else:
    if len(available_runs) > 1:
        warn('There are multiple validation runs listed for reconstruction '
             'with data={}, experiment={}, selecting the first one.'.format(
                     data, experiment))
    run = available_runs[0]

if run.get('single_run_path') is None:
    raise ValueError('Missing single_run_path specification for '
                     'reconstruction with data={}, experiment={}. Please '
                     'insert it in `runs.yaml`.'.format(data, experiment))

run_path = os.path.join(PATH, run['single_run_path'])

cfg = get_run_cfg(run_path)

val_dataset = get_validation_data(cfg.data.name, cfg.data)
num_val_samples = len(val_dataset)

with open(os.path.join(run_path, 'val_run_paths.json'), 'r') as f:
    val_run_paths = json.load(f)

with open(os.path.join(run_path, 'val_results.json'), 'r') as f:
    val_results = json.load(f)
psnrs_steady = {ckpt_path: result['PSNR_steady']
                for ckpt_path, result in val_results.items()}
min_psnr_steady = np.min(list(psnrs_steady.values()))
max_psnr_steady = np.max(list(psnrs_steady.values()))
if max_psnr_steady - min_psnr_steady > 1.:
    warn('Steady PSNRs differ by more than 1 dB')

common_path = os.path.commonpath(val_run_paths.keys())
cycle_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

for i_run, (directory_path, checkpoints_paths) in enumerate(
        val_run_paths.items()):
    if i_run not in val_run_spec.get('only_run_indices',
                                     range(len(val_run_paths.items()))):
        continue
    for i_ckpt, filename in enumerate(checkpoints_paths):
        psnr_histories = []
        for i in range(cfg.val.num_repeats):
            for i_sample in range(num_val_samples):
                histories_path = os.path.join(
                        cfg.save_histories_path,
                        'run_{:d}'.format(i_run),
                        'ckpt_{:d}'.format(i_ckpt),
                        'rep_{:d}'.format(i),
                        'sample_{:d}'.format(i_sample))
                psnr_histories.append(
                        np.load(os.path.join(run_path, histories_path,
                                             'histories.npz'))['psnr'])

        mean_psnr_history = np.median(psnr_histories, axis=0)

        label = os.path.relpath(os.path.join(directory_path, filename),
                                common_path)
        color = cycle_colors[
                np.ravel_multi_index(
                        (i_run, i_ckpt),
                        (len(val_run_paths), len(checkpoints_paths))) %
                len(cycle_colors)]

        for psnr_history in psnr_histories:
            ax.plot(psnr_history, color=color, alpha=0.1)

        ax.plot(mean_psnr_history, label=label, color=color)

ax.grid(True)
ax.set_xlim(xlim)
ax.set_xlabel('Iteration')
ax.set_ylabel('PSNR [dB]')
ax.set_ylim([15, 28])
# ax.set_xscale('log')

ax.legend(loc='lower right')

if not val_run_title:
    val_run_title = get_title_from_run_spec(val_run_spec)
title = '{} on {}'.format(val_run_title, data_title)
fig.suptitle(title)

if save_fig:
    if not val_run_filename:
        val_run_filename = (
                val_run_spec['experiment'] if val_run_spec.get('name') is None
                else '{}_{}'.format(
                        val_run_spec['experiment'], val_run_spec['name']))
    filename = '{}_on_{}.pdf'.format(val_run_filename, data)
    fig.savefig(os.path.join(FIG_PATH, filename), bbox_inches='tight')
