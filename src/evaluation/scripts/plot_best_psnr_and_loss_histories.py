import os
import json
from warnings import warn
import numpy as np
import yaml
from evaluation.utils import (
        get_multirun_cfgs, get_multirun_experiment_names,
        get_multirun_histories, uses_swa_weights)
from evaluation.evaluation import (
        get_median_psnr_history, get_psnr_steady)
from evaluation.display_utils import (
    data_title_dict, experiment_color_dict, get_title_from_run_spec)
from copy import copy
from math import ceil

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerLine2D

# PATH = '/media/chen/Res/deep_image_prior_extension/'
# PATH = '/localdata/jleuschn/experiments/deep_image_prior_extension/'
PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

FIG_PATH = os.path.dirname(__file__)
EVAL_RESULTS_PATH = os.path.dirname(__file__)

save_fig = True
save_eval_results = True
formats = ('pdf', 'png')

with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'runs.yaml'),
        'r') as f:
    runs = yaml.load(f, Loader=yaml.FullLoader)

data = 'ellipses_lotus_20'
# data = 'ellipses_lotus_limited_45'
# data = 'brain_walnut_120'
# data = 'ellipses_walnut_120'

variant = ''
# variant = 'all'
# variant = 'checkpoints'
# variant = 'checkpoints_epochs'

# Additional `run_spec` dict fields:
# 
#     color : str, optional
#         Line color.

if data == 'ellipses_lotus_20':
    if (not variant) or variant == 'all':
        runs_to_compare = [
            {
            'experiment': 'no_pretrain',
            'name': 'no_stats_no_sigmoid',
            'name_title': '',
            },
            {
            'experiment': 'no_pretrain_fbp',
            'name': 'no_stats_no_sigmoid',
            'name_title': '',
            },
            *([{
            'experiment': 'no_pretrain',
            'name': 'no_stats_no_sigmoid_fixed_encoder',
            'experiment_title': 'DIP-FE (noise)',
            'name_title': '',
            'color': 'gray',
            }] if variant == 'all' else []),
            *([{
            'experiment': 'no_pretrain_fbp',
            'name': 'no_stats_no_sigmoid_fixed_encoder',
            'experiment_title': 'DIP-FE (FBP)',
            'name_title': '',
            'color': '#00AAFF',
            }] if variant == 'all' else []),
            {
            'experiment': 'pretrain_only_fbp',
            'name': 'no_stats_no_sigmoid_train_run2_epochs100',
            'name_title': '',
            },
            {
            'experiment': 'pretrain',
            'name': 'no_stats_no_sigmoid_train_run2_epochs100',
            'name_title': '',
            },
            *([{
            'experiment': 'pretrain_only_fbp',
            'name': 'no_stats_no_sigmoid_train_run2_epochs100_fixed_encoder',
            'experiment_title': 'EDIP-FE (FBP)',
            'name_title': '',
            'color': '#EC2215',
            }] if variant == 'all' else []),
            *([{
            'experiment': 'pretrain',
            'name': 'no_stats_no_sigmoid_train_run2_epochs100_fixed_encoder',
            'experiment_title': 'EDIP-FE (noise)',
            'name_title': '',
            'color': '#B15CD1',
            }] if variant == 'all' else []),
        ]
    elif variant == 'checkpoints':
        runs_to_compare = [
            {
            'experiment': 'pretrain_only_fbp',
            'name': 'no_stats_no_sigmoid_train_run0_epochs100',
            'experiment_title': 'Run 0: 100 epochs',
            'name_title': '',
            'color': '#404099',
            },
            {
            'experiment': 'pretrain_only_fbp',
            'name': 'no_stats_no_sigmoid_train_run0_epochs20',
            'experiment_title': 'Run 0: 20 epochs',
            'name_title': '',
            'color': '#8080BB',
            },
            {
            'experiment': 'pretrain_only_fbp',
            'name': 'no_stats_no_sigmoid_train_run1_epochs100',
            'experiment_title': 'Run 1: 100 epochs',
            'name_title': '',
            'color': '#994040',
            },
            {
            'experiment': 'pretrain_only_fbp',
            'name': 'no_stats_no_sigmoid_train_run1_epochs20',
            'experiment_title': 'Run 1: 20 epochs',
            'name_title': '',
            'color': '#BB8080',
            },
            {
            'experiment': 'pretrain_only_fbp',
            'name': 'no_stats_no_sigmoid_train_run2_epochs100',
            'experiment_title': 'Run 2: 100 epochs',
            'name_title': '',
            'color': '#409940',
            },
            {
            'experiment': 'pretrain_only_fbp',
            'name': 'no_stats_no_sigmoid_train_run2_epochs20',
            'experiment_title': 'Run 2: 20 epochs',
            'name_title': '',
            'color': '#80BB80',
            },
            {
            'experiment': 'no_pretrain',
            'name': 'no_stats_no_sigmoid',
            'name_title': '',
            },
        ]

# elif data == 'ellipses_lotus_limited_45':
#     if (not variant) or variant == 'all':
#         runs_to_compare = [
#             {
#             'experiment': 'no_pretrain',
#             },
#             {
#             'experiment': 'no_pretrain_fbp',
#             },
#             *([{
#             'experiment': 'no_pretrain',
#             'name': 'fixed_encoder',
#             'experiment_title': 'DIP-FE (noise)',
#             'name_title': '',
#             'color': 'gray',
#             }] if variant == 'all' else []),
#             *([{
#             'experiment': 'no_pretrain_fbp',
#             'name': 'fixed_encoder',
#             'experiment_title': 'DIP-FE (FBP)',
#             'name_title': '',
#             'color': '#00AAFF',
#             }] if variant == 'all' else []),
#             {
#             'experiment': 'pretrain_only_fbp',
#             },
#             {
#             'experiment': 'pretrain',
#             },
#             *([{
#             'experiment': 'pretrain_only_fbp',
#             'name': 'train_run2_epochs40_fixed_encoder',
#             'experiment_title': 'EDIP-FE (FBP)',
#             'name_title': '',
#             'color': '#EC2215',
#             }] if variant == 'all' else []),
#             *([{
#             'experiment': 'pretrain',
#             'name': 'train_run2_epochs40_fixed_encoder',
#             'experiment_title': 'EDIP-FE (noise)',
#             'name_title': '',
#             'color': '#B15CD1',
#             }] if variant == 'all' else []),
#         ]

elif data == 'brain_walnut_120':
    if (not variant) or variant == 'all':
        runs_to_compare = [
            {
            'experiment': 'no_pretrain',
            'name': 'no_stats_no_sigmoid',
            'name_title': '',
            },
            {
            'experiment': 'no_pretrain_fbp',
            'name': 'no_stats_no_sigmoid',
            'name_title': '',
            },
            *([{
            'experiment': 'no_pretrain',
            'name': 'no_stats_no_sigmoid_fixed_encoder',
            'experiment_title': 'DIP-FE (noise)',
            'name_title': '',
            'color': 'gray',
            }] if variant == 'all' else []),
            *([{
            'experiment': 'no_pretrain_fbp',
            'name': 'no_stats_no_sigmoid_fixed_encoder',
            'experiment_title': 'DIP-FE (FBP)',
            'name_title': '',
            'color': '#00AAFF',
            }] if variant == 'all' else []),
            {
            'experiment': 'pretrain_only_fbp',
            'name': 'no_stats_no_sigmoid_train_run1',
            'name_title': '',
            },
            {
            'experiment': 'pretrain',
            'name': 'no_stats_no_sigmoid_train_run1',
            'name_title': '',
            },
            *([{
            'experiment': 'pretrain_only_fbp',
            'name': 'no_stats_no_sigmoid_train_run1_fixed_encoder',
            'experiment_title': 'EDIP-FE (FBP)',
            'name_title': '',
            'color': '#EC2215',
            }] if variant == 'all' else []),
            *([{
            'experiment': 'pretrain',
            'name': 'no_stats_no_sigmoid_train_run1_fixed_encoder',
            'experiment_title': 'EDIP-FE (noise)',
            'name_title': '',
            'color': '#B15CD1',
            }] if variant == 'all' else []),
        ]
    elif variant == 'checkpoints':
        runs_to_compare = [
            {
            'experiment': 'pretrain_only_fbp',
            'name': 'no_stats_no_sigmoid_train_run0',
            'experiment_title': 'Run 0: min. val. loss',
            'name_title': '',
            'color': '#404099',
            },
            {
            'experiment': 'pretrain_only_fbp',
            'name': 'no_stats_no_sigmoid_train_run1',
            'experiment_title': 'Run 1: min. val. loss',
            'name_title': '',
            'color': '#994040',
            },
            {
            'experiment': 'pretrain_only_fbp',
            'name': 'no_stats_no_sigmoid_train_run2',
            'experiment_title': 'Run 2: min. val. loss',
            'name_title': '',
            'color': '#409940',
            },
            {
            'experiment': 'no_pretrain',
            },
        ]
    elif variant == 'checkpoints_epochs':
        runs_to_compare = [
            {
            'experiment': 'pretrain_only_fbp',
            'name': 'no_stats_no_sigmoid_repeated_epochs1',
            'experiment_title': '1 epoch',
            'name_title': '',
            'color': plt.get_cmap('magma')(80),
            },
            {
            'experiment': 'pretrain_only_fbp',
            'name': 'no_stats_no_sigmoid_repeated_epochs2',
            'experiment_title': '2 epochs',
            'name_title': '',
            'color': plt.get_cmap('magma')(128),
            },
            {
            'experiment': 'pretrain_only_fbp',
            'name': 'no_stats_no_sigmoid_repeated_epochs3',
            'experiment_title': '3 epochs',
            'name_title': '',
            'color': plt.get_cmap('magma')(160),
            },
            {
            'experiment': 'pretrain_only_fbp',
            'name': 'no_stats_no_sigmoid_repeated_epochs4',
            'experiment_title': '4 epochs',
            'name_title': '',
            'color': plt.get_cmap('magma')(200),
            },
            {
            'experiment': 'pretrain_only_fbp',
            'name': 'no_stats_no_sigmoid_repeated_epochs20',
            'experiment_title': '20 epochs',
            'name_title': '',
            'color': plt.get_cmap('magma')(232),
            },
            {
            'experiment': 'no_pretrain',
            'name': 'no_stats_no_sigmoid',
            'name_title': '',
            },
        ]

elif data == 'ellipses_walnut_120':
    if (not variant) or variant == 'all':
        runs_to_compare = [
            {
            'experiment': 'no_pretrain',
            'name': 'no_stats_no_sigmoid',
            'name_title': '',
            },
            {
            'experiment': 'no_pretrain_fbp',
            'name': 'no_stats_no_sigmoid',
            'name_title': '',
            },
            *([{
            'experiment': 'no_pretrain',
            'name': 'no_stats_no_sigmoid_fixed_encoder',
            'experiment_title': 'DIP-FE (noise)',
            'name_title': '',
            'color': 'gray',
            }] if variant == 'all' else []),
            *([{
            'experiment': 'no_pretrain_fbp',
            'name': 'no_stats_no_sigmoid_fixed_encoder',
            'experiment_title': 'DIP-FE (FBP)',
            'name_title': '',
            'color': '#00AAFF',
            }] if variant == 'all' else []),
            {
            'experiment': 'pretrain_only_fbp',
            'name': 'no_stats_no_sigmoid_train_run1',
            'name_title': '',
            },
            {
            'experiment': 'pretrain',
            'name': 'no_stats_no_sigmoid_train_run1_warmup5000_init5e-4',
            'name_title': 'warm-up',
            },
            *([{
            'experiment': 'pretrain_only_fbp',
            'name': 'no_stats_no_sigmoid_train_run1_fixed_encoder',
            'experiment_title': 'EDIP-FE (FBP)',
            'name_title': '',
            'color': '#EC2215',
            }] if variant == 'all' else []),
            *([{
            'experiment': 'pretrain',
            'name': 'no_stats_no_sigmoid_train_run1_warmup5000_init5e-4_fixed_encoder',
            'experiment_title': 'EDIP-FE (noise)',
            'name_title': 'warm-up',
            'color': '#B15CD1',
            }] if variant == 'all' else []),
        ]
    elif variant == 'checkpoints':
        runs_to_compare = [
            {
            'experiment': 'pretrain_only_fbp',
            'name': 'no_stats_no_sigmoid_train_run0',
            'experiment_title': 'Run 0: min. val. loss',
            'name_title': '',
            'color': '#404099',
            },
            {
            'experiment': 'pretrain_only_fbp',
            'name': 'no_stats_no_sigmoid_train_run1',
            'experiment_title': 'Run 1: min. val. loss',
            'name_title': '',
            'color': '#994040',
            },
            {
            'experiment': 'pretrain_only_fbp',
            'name': 'no_stats_no_sigmoid_train_run2',
            'experiment_title': 'Run 2: min. val. loss',
            'name_title': '',
            'color': '#409940',
            },
            {
            'experiment': 'no_pretrain',
            },
        ]


# if not variant or variant == 'all':
#     baseline_run_idx = 0
# elif variant == 'checkpoints' or variant == 'checkpoints_epochs':
#     baseline_run_idx = -1

title = None
runs_title = ''  # None -> auto from run_specs
if variant == 'checkpoints':
    # runs_title = 'The perks of being a validated EDIP'
    runs_title = 'Comparing training runs for EDIP'
elif variant == 'checkpoints_epochs':
    runs_title = 'Comparing training epochs for EDIP'
runs_filename = 'comparison'  # None -> auto from run_specs

plot_settings_dict = {
    'ellipses_lotus_20': {
        'xlim': (-60, 6000),
        'ylim': (None, 35.95),
        'ylim_loss': None,
        'stop_time_y_pos': 32.5,
        'stop_time_y_shift_per_run_idx': {
            0: 3.,
            1: 1.,
            2: 2.,
            3: 0.,
            4: 7.,
            5: 5.,
            6: 6.,
            7: 4.,
        } if variant == 'all' else ({
            0: 1.,
            1: 0.,
            2: 3.,
            3: 2.,
        } if (not variant) else {}),
        'zorder_per_run_idx': {
        },
        'run_legend_bbox_to_anchor': (0.5, -0.125),
        'run_legend_loc': 'upper center',
        'run_legend_ncol': (
            len(runs_to_compare) if not variant else
            len(runs_to_compare) // 2),
        'symbol_legend_bbox_to_anchor': (1., 0.825),
        'symbol_legend_loc': 'upper right',
    },
    'ellipses_lotus_limited_30': {
    },
    'brain_walnut_120': {
    },
    'ellipses_walnut_120': {
        'xlim': (-300, 30000),
        'ylim': (None, 39.25),
        'ylim_loss': None,
        'stop_time_y_pos': 35.5,
        'stop_time_y_shift_per_run_idx': {
            0: 3.,
            1: 1.,
            2: 2.,
            3: 0.,
            4: 7.,
            5: 5.,
            6: 6.,
            7: 4.,
        } if variant == 'all' else ({
            0: 1.,
            1: 0.,
            2: 3.,
            3: 2.,
        } if (not variant) else {}),
        'zorder_per_run_idx': {
        },
        'run_legend_bbox_to_anchor': (0.5, -0.125),
        'run_legend_loc': 'upper center',
        'run_legend_ncol': (
            len(runs_to_compare) if not variant else
            len(runs_to_compare) // 2),
        'symbol_legend_bbox_to_anchor': (1., 0.85),
        'symbol_legend_loc': 'upper right',
    }
}

eval_settings_dict = {
    'ellipses_lotus_20': {
        'psnr_steady_start': -5000,
        'psnr_steady_stop': None,
        'stop_threshold': 5e-6,
        'stop_avg_interval': 100,
    },
    'ellipses_lotus_limited_30': {
        'psnr_steady_start': -5000,
        'psnr_steady_stop': None,
        'stop_threshold': 1e-8,
        'stop_avg_interval': 100,
    },
    'brain_walnut_120': {
        'psnr_steady_start': -5000,
        'psnr_steady_stop': None,
        'stop_threshold': 1e-8,
        'stop_avg_interval': 100,
    },
    'ellipses_walnut_120': {
        'psnr_steady_start': -5000,
        'psnr_steady_stop': None,
        'stop_threshold': 5e-8,
        'stop_avg_interval': 100,
    },
}

data_title = data_title_dict[data]

fig, ax = plt.subplots(figsize=plot_settings_dict[data].get('figsize', (8, 5)),
                       gridspec_kw={'bottom': 0.2})

ax_loss = ax.twinx()

def get_color(run_spec, cfg):
    color = run_spec.get('color')

    if color is None:
        color = experiment_color_dict.get(run_spec['experiment'])
        if color is None:
            color = 'gray'

    return color

def get_label(run_spec, cfg):
    label_parts = [get_title_from_run_spec(run_spec)]

    if cfg['mdl']['load_pretrain_model']:
        if uses_swa_weights(cfg):
            label_parts.append('SWA weights')

    label = ', '.join(label_parts)
    return label

xlim = plot_settings_dict[data]['xlim']

cfgs_list = []
experiment_names_list = []
histories_list = []

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
    if len(cfgs) == 0:
        warn('No runs found at path "{}", skipping.'.format(run_path_multirun))
        continue
    try:
        assert all((cfg['data']['name'] == data for cfg in cfgs))
    except AssertionError:
        data_name_valid = (
                data in ['ellipses_walnut_120', 'brain_walnut_120'] and
                all((not cfg['mdl']['load_pretrain_model']) and
                     cfg['data']['name'] in [
                            'ellipses_walnut_120', 'brain_walnut_120']
                    for cfg in cfgs))
        if not data_name_valid:
            raise
    swa = cfgs[0]['mdl']['load_pretrain_model'] and uses_swa_weights(cfgs[0])
    assert all(((cfg['mdl']['load_pretrain_model'] and uses_swa_weights(cfg))
                == swa) for cfg in cfgs)
    assert all((en == experiment for en in experiment_names))

    num_runs = len(cfgs)
    print('Found {:d} runs at path "{}".'.format(num_runs, run_path_multirun))

    cfgs_list.append(cfgs)
    experiment_names_list.append(experiment_names)
    histories_list.append(histories)


def get_best_output_psnr_history(histories):
    best_loss_history = np.minimum.accumulate(histories['loss'])
    _, unique_indices, unique_inverse = np.unique(
            best_loss_history, return_index=True, return_inverse=True)
    best_output_psnr_history = histories['psnr'][unique_indices][unique_inverse]
    return best_output_psnr_history

def get_best_loss_history(histories):
    best_loss_history = np.minimum.accumulate(histories['loss'])
    return best_loss_history

def get_stop_time(histories, threshold, avg_interval=1):
    loss_abs_diff = np.abs(np.diff(histories['loss']))
    loss_abs_diff_running_avg = np.pad(
            np.convolve(loss_abs_diff, np.ones(avg_interval),
                        mode='valid') / avg_interval,
            (avg_interval - 1, 0), constant_values=np.inf)
    argwhere_stop = np.argwhere(loss_abs_diff_running_avg < threshold)
    stop_time = int(argwhere_stop[0][0]) if len(argwhere_stop) > 0 else None
    return stop_time

# baseline_histories = histories_list[baseline_run_idx]
# baseline_best_psnr_histories = [
#         get_best_output_psnr_history(h) for h in baseline_histories]
# baseline_psnr_steady = get_psnr_steady(
#         baseline_best_psnr_histories,
#         start=eval_settings_dict[data]['psnr_steady_start'],
#         stop=eval_settings_dict[data]['psnr_steady_stop'])

# print('baseline steady PSNR (using running best loss output)',
#         baseline_psnr_steady)

# h = ax.axhline(baseline_psnr_steady, color='gray', linestyle='--', zorder=1.5)
# baseline_handle = h

run_handles = []
stop_time_handles = []

eval_results_list = []

for i, (run_spec, cfgs, experiment_names, histories) in enumerate(zip(
        runs_to_compare, cfgs_list, experiment_names_list, histories_list)):

    best_psnr_histories = [get_best_output_psnr_history(h) for h in histories]
    loss_histories = [h['loss'] for h in histories]
    best_loss_histories = [get_best_loss_history(h) for h in histories]

    stop_times = [
            get_stop_time(h,
                    threshold=eval_settings_dict[data]['stop_threshold'],
                    avg_interval=eval_settings_dict[data]['stop_avg_interval'])
            for h in histories]

    eval_results = {}
    eval_results['run_spec'] = run_spec
    eval_results['stop_times'] = stop_times

    eval_results_list.append(eval_results)

    label = get_label(run_spec, cfgs[0])
    color = get_color(run_spec, cfgs[0])
    linestyle = run_spec.get('linestyle', 'solid')

    zorder = plot_settings_dict[data].get('zorder_per_run_idx', {}).get(
            2 + 0.1 * i)

    for best_psnr_history in best_psnr_histories:
        h = ax.plot(best_psnr_history, label=label, color=color,
                        linestyle=linestyle, linewidth=2,
                        zorder=zorder)
    run_handles += h
    for best_loss_history in best_loss_histories:
        h = ax_loss.plot(best_loss_history, label=label, color=color,
                        linestyle=linestyle, linewidth=2,
                        zorder=zorder)
    for loss_history in loss_histories:
        h = ax_loss.plot(loss_history, label=label, color=color,
                        linestyle=linestyle, linewidth=2,
                        zorder=zorder, alpha=0.075)
    x_stop_times = sorted([
            (stop_time if stop_time is not None else
             cfgs[0].mdl.optim.iterations + 1)
            for stop_time in stop_times])
    stop_time_y_pos_shifted = (
            plot_settings_dict[data]['stop_time_y_pos'] +
                    plot_settings_dict[data][
                            'stop_time_y_shift_per_run_idx']
                            .get(i, 0))
    h = ax.plot(x_stop_times,
            [stop_time_y_pos_shifted] * len(x_stop_times),
            color=color, linestyle='-', solid_capstyle='butt', linewidth=2,
            marker='|', markersize=4, zorder=zorder)
    stop_time_handles += h

ax.grid(True, linestyle='-')
ax.set_xlim(xlim)
ax.set_xlabel('Iteration')
ax.set_ylabel('PSNR [dB]', labelpad=plot_settings_dict[data].get('ylabel_pad'))
ax.set_ylim(plot_settings_dict[data]['ylim'])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax_loss.set_ylabel('DIP+TV loss $l_t(\\theta)$')
ax_loss.set_yscale('log')
ax_loss.set_ylim(plot_settings_dict[data].get('ylim_loss', (None, None)))
ax_loss.spines['left'].set_visible(False)
ax_loss.spines['top'].set_visible(False)

run_legend = ax.legend(
        handles=run_handles,
        bbox_to_anchor=plot_settings_dict[data].get(
                'run_legend_bbox_to_anchor', (0.5, -0.125)),
        loc=plot_settings_dict[data].get('run_legend_loc', 'upper center'),
        ncol=plot_settings_dict[data].get(
                'run_legend_ncol') or ceil(len(runs_to_compare) / 2),
        framealpha=1.,
        handletextpad=plot_settings_dict[data].get('run_legend_handletextpad'),
        columnspacing=plot_settings_dict[data].get('run_legend_columnspacing'))
ax.add_artist(run_legend)
stop_time_handle = copy(stop_time_handles[0])
stop_time_handle.set_color('gray')
symbol_legend = ax_loss.legend(
        [stop_time_handle],
        ['Stop times'  # (moving avg. size {:d}, threshold {:g})'.format(
                #  eval_settings_dict[data]['stop_avg_interval'],
                #  eval_settings_dict[data]['stop_threshold'])
        ],
        handler_map={stop_time_handle: HandlerLine2D(numpoints=5)},
        bbox_to_anchor=plot_settings_dict[data].get(
                'symbol_legend_bbox_to_anchor', (1., 0.825)),
        loc=plot_settings_dict[data].get('symbol_legend_loc', 'upper right'),
        ncol=1, framealpha=1.,
        )
symbol_legend.set_zorder(50.)

if title is None:
    if runs_title is None:
        runs_title = ' vs '.join(
                [get_title_from_run_spec(r) for r in runs_to_compare])
    title = ('{} on {}'.format(runs_title, data_title) if runs_title != '' else
             data_title)
ax.set_title(title)

if runs_filename is None:
    runs_filename = '_vs_'.join(
            [(r['experiment'] if r.get('name') is None
                else '{}_{}'.format(r['experiment'], r['name']))
                for r in runs_to_compare])
suffix = (
        ('_{}'.format(variant) if variant else '') +
        '_best_psnr_and_loss')
filename = '{}_on_{}{}'.format(runs_filename, data, suffix)

if save_fig:
    for fmt in formats:
        filename_fmt = '{}.{}'.format(filename, fmt)
        fig.savefig(os.path.join(FIG_PATH, filename_fmt), bbox_inches='tight',
                    dpi=200)

if save_eval_results:
    with open(os.path.join(EVAL_RESULTS_PATH,
                           '{}.{}'.format(filename, 'json')), 'w') as f:
        json.dump(eval_results_list, f, indent=4)

plt.show()
