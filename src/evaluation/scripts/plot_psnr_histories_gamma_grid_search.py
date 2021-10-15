import os
from warnings import warn
import numpy as np
import yaml
from evaluation.utils import (
        get_multirun_cfgs, get_multirun_experiment_names,
        get_multirun_histories, get_run_cfg)
from evaluation.display_utils import get_data_title_full

import matplotlib.pyplot as plt

# PATH = '/media/chen/Res/deep_image_prior_extension/'
# PATH = '/localdata/jleuschn/experiments/deep_image_prior_extension/'
PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

FIG_PATH = '.'

save_fig = True

with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'runs.yaml'),
        'r') as f:
    runs = yaml.load(f, Loader=yaml.FullLoader)

experiment = 'no_pretrain'

data = 'ellipses_lotus_20'
# data = 'brain_walnut_120'
validation_run = False


# runs_to_combine = [
#     {
#         'experiment': experiment,
#         'name': 'gamma_grid_search',
#         'assert_val_settings_are_like_in_validation_select_gamma_run': True,
#     }
# ]



if data == 'ellipses_lotus_20':
    if validation_run:
        runs_to_combine = [
            {
                'experiment': experiment,
                'name': 'gamma_grid_search',
                # 'name': 'no_stats_no_sigmoid_gamma_grid_search',
            },
        ]
    else:
        runs_to_combine = [
            {
                'experiment': experiment,
                'name': 'gamma_grid_search',
            },
            {
                'experiment': experiment,
                'name': 'gamma_grid_search_extra',
            },
        ]
        # runs_to_combine = [
        #     {
        #         'experiment': experiment,
        #         'name': 'no_stats_no_sigmoid_gamma_grid_search',
        #     },
        # ]

if data == 'brain_walnut_120':
    if validation_run:
        runs_to_combine = [
            {
                'experiment': experiment,
                'name': 'gamma_1e-6',
            },
            {
                'experiment': experiment,
                'name': 'gamma_6.5e-7',
            },
            {
                'experiment': experiment,
                'name': 'gamma_2e-6',
            },
            {
                'experiment': experiment,
                'name': 'gamma_4e-7',
            },
            {
                'experiment': experiment,
                'name': 'gamma_2e-7',
            },
            {
                'experiment': experiment,
                'name': 'gamma_1e-7',
            },
        ]
    else:
        runs_to_combine = [
            {
                'experiment': experiment,
                'name': 'gamma_1e-6_2e-6',
            },
            {
                'experiment': experiment,
                'name': 'gamma_6.5e-7_4e-7',
            },
            {
                'experiment': experiment,
                'name': 'gamma_2e-7',
            },
            {
                'experiment': experiment,
                'name': 'gamma_1e-7',
            },
        ]


NUM_REPEATS = 5

val_settings_dict = {
    'ellipses_lotus_20': {
        'val': {
            'psnr_steady_start': -5000,
            'psnr_steady_stop': None
        },
        'test': {
            'psnr_steady_start': -5000,
            'psnr_steady_stop': None
        },
    },
    'ellipses_lotus_limited_30': {
        'val': {
            'psnr_steady_start': 10000,
            'psnr_steady_stop': 15000
        },
        'test': {
            'psnr_steady_start': 10000,
            'psnr_steady_stop': 15000
        },
    },
    'brain_walnut_120': {
        'val': {
            'psnr_steady_start': -5000,
            'psnr_steady_stop': None
        },
        'test': {
            'psnr_steady_start': -5000,
            'psnr_steady_stop': None
        },
    },
}

if validation_run:
    psnr_steady_start = val_settings_dict[data]['val']['psnr_steady_start']
    psnr_steady_stop = val_settings_dict[data]['val']['psnr_steady_stop']
else:
    psnr_steady_start = val_settings_dict[data]['test']['psnr_steady_start']
    psnr_steady_stop = val_settings_dict[data]['test']['psnr_steady_stop']

plot_settings_dict = {
    'ellipses_lotus_20': {
        'val': {
            'num_iters': None,
            'ylim': (14., None),
        },
        'test': {
            'num_iters': None,
            'ylim': (22., None),
        },
    },
    'ellipses_lotus_limited_30': {
        'val': {
            'num_iters': None,
            'ylim': None,
        },
        'test': {
            'num_iters': None,
            'ylim': None,
        },
    },
    'brain_walnut_120': {
        'val': {
            'num_iters': None,
            'ylim': None,
        },
        'test': {
            'num_iters': None,
            'ylim': None,
        },
    },
}

data_title_full = get_data_title_full(data, validation_run)

if validation_run:
    num_iters = plot_settings_dict[data]['val'].get('num_iters')
    ylim = plot_settings_dict[data]['val'].get('ylim')
else:
    num_iters = plot_settings_dict[data]['test'].get('num_iters')
    ylim = plot_settings_dict[data]['test'].get('ylim')

fig, ax = plt.subplots(figsize=(10, 8))

def get_auto_xlim_with_padding(ax, xmin, xmax):
    hmin = ax.axvline(xmin, linestyle='')
    hmax = ax.axvline(xmax, linestyle='')
    xlim = ax.get_xlim()
    hmin.remove()
    hmax.remove()
    del hmin
    del hmax
    return xlim

xlim = (get_auto_xlim_with_padding(ax, 0, num_iters) if num_iters is not None
        else None)

cfgs = []
histories = []
for run_spec in runs_to_combine:
    assert run_spec['experiment'] == experiment
    available_runs = runs['reconstruction_validation' if validation_run else
                          'reconstruction'][data][experiment]
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

    cur_cfgs = get_multirun_cfgs(
            run_path_multirun, sub_runs=sub_runs)
    cur_experiment_names = get_multirun_experiment_names(
            run_path_multirun, sub_runs=sub_runs)
    cur_histories = get_multirun_histories(
            run_path_multirun, sub_runs=sub_runs)
    cur_num_runs = len(cur_cfgs)
    if cur_num_runs == 0:
        warn('No runs found at path "{}", skipping.'.format(run_path_multirun))
        continue
    print('Found {:d} runs at path "{}".'.format(
            cur_num_runs, run_path_multirun))
    assert len(cur_experiment_names) == cur_num_runs
    assert len(cur_histories) == cur_num_runs
    assert all((cfg['data']['name'] == data for cfg in cur_cfgs))
    assert all((cfg['validation_run'] == validation_run for cfg in cur_cfgs))
    assert all((en == experiment for en in cur_experiment_names))
    assert_like_select_gamma_run = run_spec.get(
            'assert_val_settings_are_like_in_validation_select_gamma_run',
            False)
    if assert_like_select_gamma_run:
        select_gamma_run_name = (None if assert_like_select_gamma_run == True
                                 else assert_like_select_gamma_run)
        available_select_gamma_runs = runs[
                'validation_select_gamma'][data][experiment]
        if select_gamma_run_name is not None:
            select_gamma_run = next((r for r in available_select_gamma_runs if
                        r.get('name') == select_gamma_run_name),
                       None)
            if select_gamma_run is None:
                raise ValueError('Unknown multirun name "{}" for '
                                 '\'validation_select_gamma\' with data={}, '
                                 'experiment={}'.format(
                                         select_gamma_run_name, data,
                                         experiment))
        else:
            if len(available_select_gamma_runs) > 1:
                warn('There are multiple multiruns listed for reconstruction '
                     'with data={}, experiment={}, selecting the first one.'
                     .format(data, experiment))
            select_gamma_run = available_select_gamma_runs[0]
        select_gamma_run_path = select_gamma_run['single_run_path']
        if validation_run:
            iterations = cur_cfgs[0].mdl.optim.iterations
            assert all((cfg.mdl.optim.iterations == iterations
                        for cfg in cur_cfgs))
            select_gamma_cfg = get_run_cfg(
                    os.path.join(PATH, select_gamma_run_path))
            select_gamma_cfg_psnr_steady_range = range(iterations)[
                    select_gamma_cfg.val.psnr_steady_start:
                    select_gamma_cfg.val.psnr_steady_stop]
            psnr_steady_range = range(iterations)[
                    psnr_steady_start:psnr_steady_stop]
            assert select_gamma_cfg_psnr_steady_range == psnr_steady_range
        else:
            warn('Cannot assert validation configuration for grid search on '
                 'test data (yet).')
    cfgs += cur_cfgs
    histories += cur_histories

num_runs = len(cfgs)
print('Found in total {:d} runs.'.format(num_runs))

all_psnr_histories = [h['psnr'] for h in histories]

gammas = np.unique([cfg.mdl.optim.gamma for cfg in cfgs]).tolist()
cycle_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

psnr_histories_per_gamma = {}
for cfg, psnr_history in zip(cfgs, all_psnr_histories):
    psnr_histories_per_gamma.setdefault(
            float(cfg.mdl.optim.gamma), []).append(psnr_history)

psnr_histories_per_gamma = {
        k: v for k, v in sorted(psnr_histories_per_gamma.items())}

assert all(len(p) == NUM_REPEATS
           for p in psnr_histories_per_gamma.values())

for gamma, psnr_histories in psnr_histories_per_gamma.items():

    median_psnr_history = np.median(psnr_histories, axis=0)
    psnr_steady = np.median(median_psnr_history[
            psnr_steady_start:psnr_steady_stop])

    label = 'gamma={:.1e}'.format(gamma)
    color = cycle_colors[gammas.index(gamma)]

    for psnr_history in psnr_histories:
        ax.plot(psnr_history, color=color, alpha=0.1)

    ax.plot(median_psnr_history, label=label, color=color)

    ax.axhline(psnr_steady, color=color, linestyle='--')

ax.grid(True)
ax.set_xlim(xlim)
ax.set_xlabel('Iteration')
ax.set_ylabel('PSNR [dB]')
ax.legend()
ax.set_ylim(ylim)

title = 'Gamma grid search on {}'.format(data_title_full)
ax.set_title(title)

if save_fig:
    filename = 'gamma_grid_search_on_{}_{}.pdf'.format(
            data, 'val' if validation_run else 'test')
    fig.savefig(os.path.join(FIG_PATH, filename), bbox_inches='tight')

plt.show()
