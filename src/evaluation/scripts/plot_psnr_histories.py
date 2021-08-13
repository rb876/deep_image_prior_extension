import os
from warnings import warn
import numpy as np
import yaml
from evaluation.utils import (
        get_multirun_cfgs, get_multirun_experiment_names,
        get_multirun_histories, uses_swa_weights)
from evaluation.display_utils import data_title_dict, get_title_from_run_spec

import matplotlib.pyplot as plt

PATH = '/localdata/jleuschn/experiments/deep_image_prior_extension/'

FIG_PATH = '.'

save_fig = False

with open('../runs.yaml', 'r') as f:
    runs = yaml.load(f, Loader=yaml.FullLoader)

data = 'ellipses_lotus_20'

# Additional `run_spec` dict fields:
# 
#     color : str, optional
#         Line color.

# runs_to_compare = [
#     {
#       'experiment': 'pretrain_noise',
#     },
#     {
#       'experiment': 'no_pretrain_2inputs',
#     },
# ]
# runs_to_compare = [
#     {
#       'experiment': 'pretrain_noise',
#       'name': 'run1',
#       'name_title': 're-run',
#     },
#     {
#       'experiment': 'no_pretrain_2inputs',
#     },
# ]

# runs_to_compare = [
#     {
#       'experiment': 'pretrain',
#     },
#     {
#       'experiment': 'no_pretrain',
#     },
# ]
# runs_to_compare = [
#     {
#       'experiment': 'pretrain',
#       'name': 'run1',
#       'name_title': 're-run',
#     },
#     {
#       'experiment': 'no_pretrain',
#     },
# ]

runs_to_compare = [
    {
      'experiment': 'pretrain_only_fbp',
      # 'sub_runs': range(5, 10),
    },
    {
      'experiment': 'no_pretrain_fbp',
    },
]

# runs_to_compare = [
#     {
#       'experiment': 'pretrain_only_fbp',
#       'sub_runs': range(5, 10),
#     },
#     {
#       'experiment': 'pretrain_only_fbp',
#       'name': 'run1',
#       'name_title': 're-run',
#       'color': 'cyan',
#     },
#     {
#       'experiment': 'pretrain_only_fbp',
#       'name': 'adversarial_epochs40',
#       'name_title': 'adversarial, after 40 epochs',
#       'color': 'lime',
#     },
# ]

num_iters_dict = {
    'ellipses_lotus_20': 10000,
    'ellipses_lotus_limited_30': 5000,
}

data_title = data_title_dict[data]
num_iters = num_iters_dict[data]

fig, ax = plt.subplots(figsize=(10, 6))

auto_collapse_legend = True
seed_in_legend = False

def get_color(run_spec, cfg):
    color = run_spec.get('color')

    if color is None:
        if cfg['mdl']['load_pretrain_model']:
            color = 'red' if uses_swa_weights(cfg) else 'blue'
        else:
            color = 'gray'

    return color

def get_label(run_spec, cfg, include_seed=False):
    label_parts = [get_title_from_run_spec(run_spec)]

    if cfg['mdl']['load_pretrain_model']:
        if uses_swa_weights(cfg):
            label_parts.append('SWA weights')

    if include_seed:
        label_parts.append(
                'seed={:d}'.format(cfg['mdl']['torch_manual_seed']))

    label = ', '.join(label_parts)
    return label

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

for run_spec in runs_to_compare:
    experiment = run_spec['experiment']
    available_runs = runs['reconstruction'][data][experiment]
    if run_spec.get('name') is not None:
        run = next((r for r in available_runs if
                    r.get('name') == run_spec['name']),
                   None)
        if run is None:
            raise ValueError('Unknown multirun name "{}" for reconstruction '
                             'with data={}, experiment={}'.format(
                                     run_spec['name'], data, experiment))
    else:
        if len(available_runs) > 1:
            warn('There are multiple multiruns listed for reconstruction with '
                 'data={}, experiment={}, selecting the first one.'.format(
                         data, experiment))
        run = available_runs[0]

    if run.get('run_path') is None:
        raise ValueError('Missing run_path specification for reconstruction '
                         'with data={}, experiment={}. Please insert it in '
                         '`runs.yaml`.'.format(data, experiment))

    run_path_multirun = os.path.join(PATH, run['run_path'])
    sub_runs = run_spec.get('sub_runs')

    cfgs = get_multirun_cfgs(
            run_path_multirun, sub_runs=sub_runs)
    experiment_names = get_multirun_experiment_names(
            run_path_multirun, sub_runs=sub_runs)
    histories = get_multirun_histories(
            run_path_multirun, sub_runs=sub_runs)
    assert all((cfg['data']['name'] == data for cfg in cfgs))
    assert all((en == experiment for en in experiment_names))
    
    num_runs = len(cfgs)
    print('Found {:d} runs at path "{}".'.format(num_runs, run_path_multirun))

    psnr_histories = [h['psnr'] for h in histories]

    prev_label = None
    prev_color = None

    for i, (cfg, psnr_history) in enumerate(zip(cfgs, psnr_histories)):
        color = get_color(run_spec, cfg)

        label = get_label(run_spec, cfg, include_seed=seed_in_legend)

        if auto_collapse_legend and (
                label == prev_label and color == prev_color):
            label = None
        else:
            prev_label = label
            prev_color = color

        ax.plot(psnr_history, label=label, color=color)

ax.grid(True)
ax.set_xlim(xlim)
ax.set_xlabel('Iteration')
ax.set_ylabel('PSNR [dB]')

ax.legend(loc='lower right')
run_titles = [get_title_from_run_spec(r) for r in runs_to_compare]
title = '{} on {}'.format(' vs '.join(run_titles), data_title)
fig.suptitle(title)

if save_fig:
    filename = '{}_on_{}.pdf'.format(
            '_vs_'.join([(r['experiment'] if r.get('name') is None else
                          '{}_{}'.format(r['experiment'], r['name']))
                         for r in runs_to_compare]),
            data)
    fig.savefig(os.path.join(FIG_PATH, filename), bbox_inches='tight')
