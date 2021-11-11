import os
import json
from warnings import warn
import numpy as np
import yaml
from omegaconf import OmegaConf
from dataset import get_validation_data
from evaluation.utils import get_run_cfg, get_run_experiment_name
from evaluation.display_utils import (
        get_data_title_full, get_title_from_run_spec)
from validation import validate_model, val_sub_sub_path
from copy import copy
from math import ceil

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# PATH = '/media/chen/Res/deep_image_prior_extension/'
# PATH = '/localdata/jleuschn/experiments/deep_image_prior_extension/'
PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

FIG_PATH = os.path.dirname(__file__)

save_fig = True
formats = ('pdf', 'png')

with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'runs.yaml'),
        'r') as f:
    runs = yaml.load(f, Loader=yaml.FullLoader)

data = 'ellipses_lotus_20'
# data = 'ellipses_lotus_limited_45'
# data = 'brain_walnut_120'
# data = 'ellipses_walnut_120'

variant = ''
# variant = 'fewer_ckpts'

use_inset = False #data.startswith('ellipses_lotus_')

# Additional `run_spec` dict fields:
# 
#     color : str, optional
#         Line color.
#     skip_psnr0 : bool, optional
#         Whether to skip the marker indicating the initial PSNR.
#     only_run_indices : collection of int, optional
#         Restrict to the given runs (`i_run` indices).
#     only_ckpt_indices : collection of int, optional
#         Restrict to the given checkpoints (`i_ckpt` indices).
#         The same indices are applied to all runs.

val_run_spec = {
    'experiment': 'pretrain_only_fbp',
    'name': 'no_stats_no_sigmoid',
    # 'name': 'no_stats_no_sigmoid_containing_histories',
    'name_title': '',
    # 'only_run_indices': list(range(0, 3)),
    'only_ckpt_indices': [1, 2, 4] if variant == 'fewer_ckpts' else None,
}

title = None
val_run_filename = 'validation'

plot_settings_dict = {
    'ellipses_lotus_20': {
        'xlim': (-37.5, 37500),
        'xlim_inset': (500, 4250),
        'ylim': (23.5, 32.7) if variant == 'fewer_ckpts' else (23.5, 33.1),
        'ylim_inset': (26.5, 30.35),
        'rise_time_to_baseline_y_pos': 30.5,
        'rise_time_to_baseline_y_shift_i_run': 0.1,
        'rise_time_to_baseline_y_shift_i_plot_ckpt': .45,
        'run_colors': ['#000077', '#770000', '#007700'],
        'zorder_list': [1.51, 1.52, 1.53],
        'inset_axes_rect': [0.125, 0.075, 0.85, 0.35],
        'inset_axes_rect_border': [0.05, 0.0675],
    },
    'ellipses_lotus_limited_45': {
        'xlim': (-37.5, 10000),
        'xlim_inset': (100, 3250),
        'ylim': (17.5, 30) if variant == 'fewer_ckpts' else (17.5, 30),
        'ylim_inset': (22.5, 27),
        'rise_time_to_baseline_y_pos': 27.5,
        'rise_time_to_baseline_y_shift_i_run': 0.01,
        'rise_time_to_baseline_y_shift_i_plot_ckpt': .45,
        'run_colors': ['#000077', '#770000', '#007700'],
        'zorder_list': [1.51, 1.52, 1.53],
        'inset_axes_rect': [0.125, 0.075, 0.85, 0.35],
        'inset_axes_rect_border': [0.05, 0.0675],
    },
    'brain_walnut_120': {
        'xlim': (-50, 50000),
        'ylim': (27.5, 39.),
        'rise_time_to_baseline_y_pos': 38.75,
        'rise_time_to_baseline_y_shift_i_run': 0.,
        'rise_time_to_baseline_y_shift_i_plot_ckpt': .15,
        'run_colors': ['#000077', '#770000', '#007700'],
        'zorder_list': [1.52, 1.53, 1.51],
        'symbol_legend_loc': 'lower center',
    },
    'ellipses_walnut_120': {
        'xlim': (-50, 50000),
        'ylim': (27.5, 39.),
        'rise_time_to_baseline_y_pos': 38.75,
        'rise_time_to_baseline_y_shift_i_run': 0.,
        'rise_time_to_baseline_y_shift_i_plot_ckpt': .15,
        'run_colors': ['#000077', '#770000', '#007700'],
        'zorder_list': [1.52, 1.53, 1.51],
        'symbol_legend_loc': 'lower center',
    },
}

data_title_full = get_data_title_full(data, validation_run=True)

fig, ax = plt.subplots(figsize=plot_settings_dict[data].get('figsize', (8, 5)))

xlim = plot_settings_dict[data]['xlim']
xlim_inset = plot_settings_dict[data].get('xlim_inset', (None, None))

if use_inset:
    axins = ax.inset_axes(plot_settings_dict[data]['inset_axes_rect'])
    axs = [ax, axins]
else:
    axins = None
    axs = [ax]

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

# # one may need to adapt some absolute paths manually here
# cfg.data.geometry_specs.ray_trafo_filename = '/localdata/data/FIPS_Lotus/LotusData128.mat'
# cfg.data.ground_truth_filename = '/localdata/data/FIPS_Lotus/TVGroundTruthLotus128.mat'
# cfg.val.baseline_run_path = os.path.join(PATH, os.path.relpath(
#         cfg.val.baseline_run_path,
#         os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
#                 cfg.val.baseline_run_path.rstrip('/')))))))
# cfg.val.multirun_base_paths = [os.path.join(PATH, os.path.relpath(
#         multirun_run_path,
#         os.path.dirname(os.path.dirname(os.path.dirname(
#                 multirun_run_path.rstrip('/'))))))
#         for multirun_run_path in cfg.val.multirun_base_paths]
# print(cfg.val.multirun_base_paths)
# cfg.val.load_histories_from_run_path = os.path.join(PATH, os.path.relpath(
#         cfg.val.load_histories_from_run_path,
#         os.path.dirname(os.path.dirname(os.path.dirname(
#                 cfg.val.load_histories_from_run_path.rstrip('/'))))))

val_dataset = get_validation_data(cfg.data.name, cfg.data)
num_val_samples = len(val_dataset)  # 1

with open(os.path.join(run_path, 'val_run_paths.json'), 'r') as f:
    val_run_paths = json.load(f)

with open(os.path.join(run_path, 'val_results.json'), 'r') as f:
    val_results = json.load(f)
psnrs_steady = {ckpt_path: result['PSNR_steady']
                for ckpt_path, result in val_results.items()}
min_psnr_steady = np.min(list(psnrs_steady.values()))
max_psnr_steady = np.max(list(psnrs_steady.values()))


zorder_list = plot_settings_dict[data].get(
        'zorder_list', [1.5 for _ in val_run_paths])

baseline_cfg = get_run_cfg(cfg.val.baseline_run_path)
baseline_experiment_name = get_run_experiment_name(cfg.val.baseline_run_path)

# update configs to load the saved histories from the run path
# (or from the originally specified load path if any)
if baseline_cfg.val.load_histories_from_run_path is None:
    baseline_cfg.val.load_histories_from_run_path = cfg.val.baseline_run_path
if cfg.val.load_histories_from_run_path is None:
    cfg.val.load_histories_from_run_path = run_path

# override rise_time_to_baseline_remaining_psnr of baseline to the one used in
# the (actual) validation of pretrained DIPs
baseline_cfg.val.rise_time_to_baseline_remaining_psnr = (
        cfg.val.rise_time_to_baseline_remaining_psnr)

baseline_psnr_histories, baseline_info = validate_model(
        cfg=baseline_cfg, val_dataset=val_dataset,
        baseline_psnr_steady='own_PSNR_steady', val_sub_path_mdl='baseline',
        # unused positional args:
        ray_trafo=None, seed=None, log_path_base=None, cfg_mdl_val=None,
        )

baseline_psnr_steady = baseline_info['PSNR_steady']

psnr_histories_dict = {}
info_dict = {}

for i_run, (directory_path, checkpoints_paths) in enumerate(
        val_run_paths.items()):

    if i_run not in (val_run_spec.get('only_run_indices') or
                             range(len(val_run_paths.items()))):
        continue

    psnr_histories_dict[i_run] = {}
    info_dict[i_run] = {}

    for i_ckpt, filename in enumerate(checkpoints_paths):

        if i_ckpt not in (val_run_spec.get('only_ckpt_indices') or
                                  range(len(checkpoints_paths))):
            continue

        psnr_histories, info = validate_model(
                cfg=cfg, val_dataset=val_dataset,
                baseline_psnr_steady=baseline_psnr_steady,
                val_sub_path_mdl=os.path.join('run_{:d}'.format(i_run),
                                              'ckpt_{:d}'.format(i_ckpt)),
                # unused positional args:
                ray_trafo=None, seed=None, log_path_base=None, cfg_mdl_val=None,
                )

        psnr_histories_dict[i_run][i_ckpt] = psnr_histories
        info_dict[i_run][i_ckpt] = info


def plot_run(ax, psnr_histories, info, label, color, zorder, i_plot_ckpt=None):
    mean_psnr_history = np.mean(psnr_histories, axis=(0, 1))
    std_psnr_history = np.std(psnr_histories, axis=(0, 1))

    median_psnr_history = np.median(psnr_histories, axis=(0, 1))
    rise_time_to_baseline = info['rise_time_to_baseline']

    psnr_steady_handle = ax.axhline(
            info['PSNR_steady'], color=color, linestyle='--',
            zorder=zorder - 0.5)

    ax.fill_between(range(len(mean_psnr_history)),
                    mean_psnr_history - std_psnr_history,
                    mean_psnr_history + std_psnr_history,
                    color=color, alpha=0.1,
                    zorder=zorder - 0.5,
                    # edgecolor=None,
                    )
    run_handle = ax.plot(mean_psnr_history, label=label, color=color, linewidth=1,
                zorder=zorder)[0]
    if rise_time_to_baseline is not None:
        rise_time_to_baseline_y_pos_shifted = (
                plot_settings_dict[data]['rise_time_to_baseline_y_pos'] +
                i_run * (
                        plot_settings_dict[data][
                                'rise_time_to_baseline_y_shift_i_run']) +
                (i_plot_ckpt if i_plot_ckpt is not None else 0) * (
                        plot_settings_dict[data][
                                'rise_time_to_baseline_y_shift_i_plot_ckpt']))
        rise_time_handle = ax.plot(rise_time_to_baseline,
                rise_time_to_baseline_y_pos_shifted,
                '*', color=color, markersize=8,
                zorder=zorder)[0]
        ax.plot([rise_time_to_baseline, rise_time_to_baseline],
                [median_psnr_history[rise_time_to_baseline],
                 rise_time_to_baseline_y_pos_shifted],
                color=color, linestyle='--', zorder=zorder)
    else:
        rise_time_handle = None

    return run_handle, rise_time_handle, psnr_steady_handle


# baseline_label = 'Baseline' 
baseline_label = '{}'.format(
        get_title_from_run_spec({'experiment': baseline_experiment_name}))

(baseline_run_handle,
 baseline_rise_time_handle,
 baseline_psnr_steady_handle) = plot_run(
        ax, psnr_histories=baseline_psnr_histories, info=baseline_info,
        label=baseline_label, color='k', zorder=1.55)
if axins is not None:
    plot_run(
            axins, psnr_histories=baseline_psnr_histories, info=baseline_info,
            label=baseline_label, color='k', zorder=1.55)

run_handles_dict = {}
rise_time_handles_dict = {}
psnr_steady_handles_dict = {}

def get_color(i_run, i_plot_ckpt, max_i_plot_ckpt):
    run_colors = plot_settings_dict[data].get(
            'run_colors', plt.rcParams['axes.prop_cycle'].by_key()['color'])

    run_color = to_rgba(run_colors[i_run])
    bg_color = to_rgba(plt.rcParams['axes.facecolor'])
    mix = 1. - 0.8 * ((max_i_plot_ckpt - i_plot_ckpt) / max(1, max_i_plot_ckpt))
    color = tuple([
            mix * c + (1. - mix) * b for c, b in zip(run_color, bg_color)])

    return color

def get_epochs_from_filename(filename):
    epochs_split = os.path.splitext(filename)[0].split('_epochs')

    if len(epochs_split) == 1:
        epochs = None
    else:
        epochs = int(epochs_split[1])

    return epochs

def get_label(i_run, filename):

    epochs = get_epochs_from_filename(filename)

    if epochs is None:
        ckpt_label = 'min. val. loss'
    else:
        ckpt_label = '{:d} epochs'.format(epochs)

    label = 'Run {:d}: {}'.format(i_run, ckpt_label)

    return label

ckpt_sort_inds_list = []
for i_run, (directory_path, checkpoints_paths) in enumerate(
        val_run_paths.items()):

    ckpt_sort_key = []
    for filename in checkpoints_paths:
        epochs = get_epochs_from_filename(filename)
        if epochs is None:
            key = np.inf
        else:
            key = epochs
        ckpt_sort_key.append(key)
    ckpt_sort_inds = np.argsort(ckpt_sort_key)
    ckpt_sort_inds_list.append(ckpt_sort_inds)

for i_run, (directory_path, checkpoints_paths) in enumerate(
        val_run_paths.items()):

    if i_run not in (val_run_spec.get('only_run_indices') or
                             range(len(val_run_paths.items()))):
        continue

    run_handles_dict[i_run] = {}
    rise_time_handles_dict[i_run] = {}
    psnr_steady_handles_dict[i_run] = {}

    ckpt_sort_inds = ckpt_sort_inds_list[i_run]

    for i_plot_ckpt, i_ckpt in enumerate(ckpt_sort_inds):

        filename = checkpoints_paths[i_ckpt]

        if i_ckpt not in (val_run_spec.get('only_ckpt_indices') or
                                  range(len(checkpoints_paths))):
            continue

        psnr_histories = psnr_histories_dict[i_run][i_ckpt]
        info = info_dict[i_run][i_ckpt]

        label = get_label(i_run, filename)
        color = get_color(i_run, i_plot_ckpt, len(checkpoints_paths) - 1)

        zorder = zorder_list[i_run] + 0.1 * i_plot_ckpt

        run_handle, rise_time_handle, psnr_steady_handle = plot_run(
                ax, psnr_histories=psnr_histories, info=info,
                label=label, color=color, zorder=zorder,
                i_plot_ckpt=i_plot_ckpt)
        if axins is not None:
            plot_run(
                    axins, psnr_histories=psnr_histories, info=info,
                    label=label, color=color, zorder=zorder,
                    i_plot_ckpt=i_plot_ckpt)

        run_handles_dict[i_run][i_ckpt] = run_handle
        rise_time_handles_dict[i_run][i_ckpt] = rise_time_handle
        psnr_steady_handles_dict[i_run][i_ckpt] = psnr_steady_handle

ax.grid(True, linestyle='-')
ax.set_xlim(xlim)
ax.set_xlabel('Iteration')
ax.set_ylabel('PSNR [dB]', labelpad=plot_settings_dict[data].get('ylabel_pad'))
ax.set_ylim(plot_settings_dict[data].get('ylim', (None, None)))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# ax.set_xscale('log')

if axins is not None:
    # axins.grid(True, linestyle=':')
    axins.set_xlim(xlim_inset)
    axins.set_ylim(plot_settings_dict[data]['ylim_inset'])
    axins.spines['right'].set_visible(False)
    axins.spines['top'].set_visible(False)

    ax.add_patch(Rectangle([
            plot_settings_dict[data]['inset_axes_rect'][0] -
            plot_settings_dict[data]['inset_axes_rect_border'][0],
            plot_settings_dict[data]['inset_axes_rect'][1] -
            plot_settings_dict[data]['inset_axes_rect_border'][1]],
            plot_settings_dict[data]['inset_axes_rect'][2] +
            plot_settings_dict[data]['inset_axes_rect_border'][0] +
            0.0025,
            plot_settings_dict[data]['inset_axes_rect'][3] +
            plot_settings_dict[data]['inset_axes_rect_border'][1] +
            0.005,
            transform=ax.transAxes,
            color='#EEEEEE',
            zorder=3,
            ))
    # axins_bbox = axins.get_tightbbox(fig.canvas.get_renderer())  # TODO use this?

run_handles = []
for i_run, ckpt_sort_inds in enumerate(ckpt_sort_inds_list):
    if i_run in run_handles_dict:
        for i_ckpt in reversed(ckpt_sort_inds):
            if i_ckpt in run_handles_dict[i_run]:
                run_handles.append(run_handles_dict[i_run][i_ckpt])
        run_handles.append(
                baseline_run_handle if i_run == len(ckpt_sort_inds_list) - 1
                else Line2D([], [], alpha=0.))
run_legend = ax.legend(
        handles=run_handles,
        loc=plot_settings_dict[data].get('run_legend_loc', 'upper center'),
        bbox_to_anchor=plot_settings_dict[data].get(
                'run_legend_bbox_to_anchor', (0.5, -0.125)),
        ncol=len(val_run_paths), framealpha=1.)
run_legend.set_zorder(50.)
ax.add_artist(run_legend)
rise_time_handle = copy(baseline_rise_time_handle)
rise_time_handle.set_markerfacecolor('gray')
rise_time_handle.set_markeredgecolor('gray')
psnr_steady_handle = copy(baseline_psnr_steady_handle)
psnr_steady_handle.set_color('gray')
symbol_legend = ax.legend(
        [rise_time_handle, psnr_steady_handle],
        ['Rise time (to $-${:g} dB)'.format(
                  cfg.val.rise_time_to_baseline_remaining_psnr),
         'Steady PSNR',
        ],
        loc=plot_settings_dict[data].get('symbol_legend_loc', 'upper right'),
        bbox_to_anchor=plot_settings_dict[data].get(
                'symbol_legend_bbox_to_anchor'),
        ncol=2, framealpha=1.
        )
symbol_legend.set_zorder(50.)

if title is None:
    title = data_title_full
ax.set_title(title)

if save_fig:
    for fmt in formats:
        suffix = '_{}'.format(variant) if variant else ''
        filename = '{}_on_{}{}.{}'.format(val_run_filename, data, suffix, fmt)
        fig.savefig(os.path.join(FIG_PATH, filename), bbox_inches='tight',
                    dpi=200)

plt.show()
