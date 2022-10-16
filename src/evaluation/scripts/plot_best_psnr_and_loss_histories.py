import os
import json
from warnings import warn
import numpy as np
import yaml
from evaluation.utils import (
        get_multirun_cfgs, get_multirun_experiment_names,
        get_multirun_histories, uses_swa_weights)
from evaluation.evaluation import (
        get_rise_time_to_baseline, get_psnr_steady)
from evaluation.display_utils import (
    data_title_dict, experiment_color_dict, get_title_from_run_spec)
from copy import copy
from math import ceil

import matplotlib.pyplot as plt
from matplotlib import gridspec
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

show_stop_times = False
show_rise_times = False
show_baseline_steady_psnr = False
use_only_median_rep = True

with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'runs.yaml'),
        'r') as f:
    runs = yaml.load(f, Loader=yaml.FullLoader)

data = 'ellipses_lotus_20'
# data = 'ellipses_lotus_limited_45'
# data = 'brain_walnut_120'
# data = 'ellipses_walnut_120'
# data = 'ellipsoids_walnut_3d'
# data = 'ellipsoids_walnut_3d_60'

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
            *([{
            'experiment': 'pretrain',
            'name': 'no_stats_no_sigmoid_train_run2_epochs100',
            'name_title': '',
            }] if variant == 'all' else []),
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
            *([{
            'experiment': 'pretrain',
            'name': 'no_stats_no_sigmoid_train_run1_warmup5000_init5e-4',
            'name_title': 'warm-up',
            }] if variant == 'all' else []),
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

elif data in ['ellipsoids_walnut_3d', 'ellipsoids_walnut_3d_60']:
    if (not variant) or variant == 'all':
        runs_to_compare = [
            {
            'experiment': 'no_pretrain',
            'name': 'default',
            'name_title': '',
            },
            {
            'experiment': 'no_pretrain_fbp',
            'name': 'default',
            'name_title': '',
            },
            {
            'experiment': 'pretrain_only_fbp',
            'name': 'epochs0_steps8000',
            'name_title': '',
            },
        ]


if not variant or variant == 'all':
    baseline_run_idx = 0
elif variant == 'checkpoints' or variant == 'checkpoints_epochs':
    baseline_run_idx = -1

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
        'xlim': (-65, 6500),
        'xlim_loss_diff': (-99.99, 9999),
        'ylim_psnr': (14., 36.95),
        'ylim_loss': (2e-3, 2.),
        'ylim_loss_diff': (None, None),
        'stop_time_y_pos': 1.5e-1,
        'stop_time_y_shift_per_run_idx': {
            0: 0.3,
            1: 0.1,
            2: 0.2,
            3: 0.0,
            4: 0.7,
            5: 0.5,
            6: 0.6,
            7: 0.4,
        } if variant == 'all' else ({
            0: 0.2,
            1: 0.0,
            2: 0.6,
            3: 0.4,
        } if (not variant) else {}),
        'rise_time_to_baseline_y_pos': 33.,
        'rise_time_to_baseline_y_shift_per_run_idx': {
            0: 4.5,
            1: 1.5,
            2: 3.0,
            3: 0.0,
            4: 10.5,
            5: 7.5,
            6: 9.0,
            7: 6.0,
        } if variant == 'all' else ({
            0: 2.0,
            1: 0.0,
            2: 3.0,
            3: 1.0,
        } if (not variant) else {}),
        'zorder_per_run_idx': {
            0: 2.2,
            1: 2.1,
            2: 2.4,
            3: 2.3,
        },
        'hist_num_bins': 25,
        'hist_num_k_retained': 5,
        'hist_opacity': 0.3,
        'hist_linewidth': 2.,
        'hist_xticks': [1e-6, 1e-3, 1e0],
        'hist_ylim': [1e-5, 5e5],
        'hist_iteration_slices': [
                slice(0, 2000), slice(2000, 5000), slice(5000, 10000)],
        'run_legend_bbox_to_anchor': (0.5, -0.14),
        'run_legend_loc': 'lower center',
        'run_legend_ncol': (
            len(runs_to_compare) if not variant else
            len(runs_to_compare) // 2),
        'symbol_legend_loss_bbox_to_anchor': (1., 1.),
        'symbol_legend_loss_loc': 'upper right',
        'symbol_legend_psnr_bbox_to_anchor': (1., 0.),
        'symbol_legend_psnr_loc': 'lower right',
    },
    'ellipses_lotus_limited_45': {
    },
    'brain_walnut_120': {
    },
    'ellipses_walnut_120': {
        'xlim': (-300, 30000),
        'xlim_loss_diff': (-299.99, 29999),
        'ylim_psnr': (14., 43.25 if show_rise_times else 35.75),
        'ylim_loss': (1.5e-4, 8e-3),
        'ylim_loss_diff': (None, None),
        'stop_time_y_pos': 1.5e-2,
        'stop_time_y_shift_per_run_idx': {
            0: 0.3,
            1: 0.1,
            2: 0.2,
            3: 0.0,
            4: 0.7,
            5: 0.5,
            6: 0.6,
            7: 0.4,
        } if variant == 'all' else ({
            0: 0.2,
            1: 0.0,
            2: 0.4,
        } if (not variant) else {}),
        'rise_time_to_baseline_y_pos': 36.,
        'rise_time_to_baseline_y_shift_per_run_idx': {
            0: 6.0,
            1: 2.0,
            2: 4.0,
            3: 0.0,
            4: 14.0,
            5: 10.0,
            6: 12.0,
            7: 8.0,
        } if variant == 'all' else ({
            0: 2.0,
            1: 0.0,
            2: 4.0,
        } if (not variant) else {}),
        'zorder_per_run_idx': {
            0: 2.2,
            1: 2.1,
            2: 2.4,
            3: 2.3,
        },
        'hist_num_bins': 25,
        'hist_num_k_retained': 5,
        'hist_opacity': 0.3,
        'hist_linewidth': 2.,
        'hist_xticks': [1e-6, 1e-3, 1e0],
        'hist_ylim': [1e-5, 1e7],
        'hist_iteration_slices': [
                slice(0, 5000), slice(5000, 10000), slice(10000, 30000)],
        'run_legend_bbox_to_anchor': (0.5, -0.14),
        'run_legend_loc': 'lower center',
        'run_legend_ncol': (
            len(runs_to_compare) if not variant else
            len(runs_to_compare) // 2),
        'symbol_legend_loss_bbox_to_anchor': (1., 1.),
        'symbol_legend_loss_loc': 'upper right',
        'symbol_legend_psnr_bbox_to_anchor': (1., 0.),
        'symbol_legend_psnr_loc': 'lower right',
    },
    'ellipsoids_walnut_3d': {
        'xlim': (-300, 30000),
        'xlim_loss_diff': (-299.99, 29999),
        'ylim_psnr': (17.5, 33.25),
        'ylim_loss': (3.5e-4, 3e-3),
        'ylim_loss_diff': (None, None),
        'stop_time_y_pos': 1.25e-3,
        'stop_time_y_shift_per_run_idx': {
            0: 0.3,
            1: 0.1,
            2: 0.2,
            3: 0.0,
            4: 0.7,
            5: 0.5,
            6: 0.6,
            7: 0.4,
        } if variant == 'all' else ({
            0: 0.1,
            1: 0.0,
            2: 0.2,
        } if (not variant) else {}),
        'rise_time_to_baseline_y_pos': 37.,
        'rise_time_to_baseline_y_shift_per_run_idx': {
            0: 6.0,
            1: 2.0,
            2: 4.0,
            3: 0.0,
            4: 14.0,
            5: 10.0,
            6: 12.0,
            7: 8.0,
        } if variant == 'all' else ({
            0: 2.0,
            1: 0.0,
            2: 4.0,
        } if (not variant) else {}),
        'zorder_per_run_idx': {
            0: 2.2,
            1: 2.1,
            2: 2.4,
            3: 2.3,
        },
        'hist_num_bins': 25,
        'hist_num_k_retained': 5,
        'hist_opacity': 0.3,
        'hist_linewidth': 2.,
        'hist_xticks': [1e-6, 1e-3, 1e0],
        'hist_ylim': [1e-5, 1e7],
        'hist_iteration_slices': [
                slice(0, 5000), slice(5000, 10000), slice(10000, 30000)],
        'run_legend_bbox_to_anchor': (0.5, -0.14),
        'run_legend_loc': 'lower center',
        'run_legend_ncol': (
            len(runs_to_compare) if not variant else
            len(runs_to_compare) // 2),
        'symbol_legend_loss_bbox_to_anchor': (1., 1.),
        'symbol_legend_loss_loc': 'upper right',
        'symbol_legend_psnr_bbox_to_anchor': (1., 0.),
        'symbol_legend_psnr_loc': 'lower right',
    },
    'ellipsoids_walnut_3d_60': {
        'xlim': (-600, 60000),
        'xlim_loss_diff': (-599.99, 59999),
        'ylim_psnr': (17.5, 36.25),
        'ylim_loss': (3.5e-4, 3e-3),
        'ylim_loss_diff': (None, None),
        'stop_time_y_pos': 1.25e-3,
        'stop_time_y_shift_per_run_idx': {
            0: 0.3,
            1: 0.1,
            2: 0.2,
            3: 0.0,
            4: 0.7,
            5: 0.5,
            6: 0.6,
            7: 0.4,
        } if variant == 'all' else ({
            0: 0.1,
            1: 0.0,
            2: 0.2,
        } if (not variant) else {}),
        'rise_time_to_baseline_y_pos': 37.,
        'rise_time_to_baseline_y_shift_per_run_idx': {
            0: 6.0,
            1: 2.0,
            2: 4.0,
            3: 0.0,
            4: 14.0,
            5: 10.0,
            6: 12.0,
            7: 8.0,
        } if variant == 'all' else ({
            0: 2.0,
            1: 0.0,
            2: 4.0,
        } if (not variant) else {}),
        'zorder_per_run_idx': {
            0: 2.2,
            1: 2.1,
            2: 2.4,
            3: 2.3,
        },
        'hist_num_bins': 25,
        'hist_num_k_retained': 5,
        'hist_opacity': 0.3,
        'hist_linewidth': 2.,
        'hist_xticks': [1e-6, 1e-3, 1e0],
        'hist_ylim': [1e-5, 1e7],
        'hist_iteration_slices': [
                slice(0, 10000), slice(10000, 30000), slice(30000, 60000)],
        'run_legend_bbox_to_anchor': (0.5, -0.14),
        'run_legend_loc': 'lower center',
        'run_legend_ncol': (
            len(runs_to_compare) if not variant else
            len(runs_to_compare) // 2),
        'symbol_legend_loss_bbox_to_anchor': (1., 1.),
        'symbol_legend_loss_loc': 'upper right',
        'symbol_legend_psnr_bbox_to_anchor': (1., 0.),
        'symbol_legend_psnr_loc': 'lower right',
    }
}

eval_settings_dict = {
    'ellipses_lotus_20': {
        'psnr_steady_start': -5000,
        'psnr_steady_stop': None,
        'filter_kind': 'avg',
        'stop_threshold': 5e-6,
        'stop_avg_interval': 100,
        'rise_time_to_baseline_remaining_psnr': 0.1,
    },
    'ellipses_lotus_limited_45': {
        'psnr_steady_start': -5000,
        'psnr_steady_stop': None,
        'filter_kind': 'avg',
        'stop_threshold': 1e-8,
        'stop_avg_interval': 100,
        'rise_time_to_baseline_remaining_psnr': 0.1,
    },
    'brain_walnut_120': {
        'psnr_steady_start': -5000,
        'psnr_steady_stop': None,
        'filter_kind': 'avg',
        'stop_threshold': 1e-8,
        'stop_avg_interval': 100,
        'rise_time_to_baseline_remaining_psnr': 0.1,
    },
    'ellipses_walnut_120': {
        'psnr_steady_start': -5000,
        'psnr_steady_stop': None,
        'filter_kind': 'avg',
        'stop_threshold': 5e-8,
        'stop_max_interval': 100,
        'stop_avg_interval': 100,
        'rise_time_to_baseline_remaining_psnr': 0.1,
    },
    'ellipsoids_walnut_3d': {
        'psnr_steady_start': -5000,
        'psnr_steady_stop': None,
        'filter_kind': 'avg',
        'stop_threshold': 5e-7,
        'stop_max_interval': 100,
        'stop_avg_interval': 100,
        'rise_time_to_baseline_remaining_psnr': 0.1,
    },
    'ellipsoids_walnut_3d_60': {
        'psnr_steady_start': -5000,
        'psnr_steady_stop': None,
        'filter_kind': 'avg',
        'stop_threshold': 1e-6,
        'stop_max_interval': 100,
        'stop_avg_interval': 100,
        'rise_time_to_baseline_remaining_psnr': 0.1,
    },
}

if use_only_median_rep:
    # run export_images.py first
    MEDIAN_PSNR_REPS_PATH = os.path.join(
            os.path.dirname(__file__), 'images', '{}_median_psnr_reps.json'.format(data))

    with open(MEDIAN_PSNR_REPS_PATH, 'r') as f:
        median_psnr_reps = json.load(f)

data_title = data_title_dict[data]

fig = plt.figure(
        figsize=plot_settings_dict[data].get('figsize', (8, 5)))
gs = gridspec.GridSpec(2, 2, figure=fig,
        **{'bottom': 0.025, 'hspace': 0.5, 'wspace': 0.25, 'top': 0.85})

dummy_gs11 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 1], wspace=0, height_ratios=[0.025, 0.975])
gs11 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=dummy_gs11[1], wspace=0)
fake11 = fig.add_subplot(gs[1, 1])
fake11.set_axis_off()

ax_loss = fig.add_subplot(gs[0, 0])
ax_psnr = fig.add_subplot(gs[1, 0])
ax_loss_diff = fig.add_subplot(gs[0, 1])
ax_loss_diff_hists = [fig.add_subplot(gs11[i]) for i in range(3)]

ax_loss.set_yscale('log')
ax_loss_diff.set_yscale('log')
for i, ax_ in enumerate(ax_loss_diff_hists):
    ax_.grid(alpha=0.3)
    ax_.set_xscale('log')
    ax_.set_yscale('log')
    ax_.set_xticks(plot_settings_dict[data]['hist_xticks'])
    ax_.tick_params(axis='x', labelrotation=45.)
    if i != 0:
        ax_.set_yticklabels([])
    ax_.set_ylim(plot_settings_dict[data]['hist_ylim'])

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

def hex_to_rgb(value, alpha):
    value = value.lstrip('#')
    lv = len(value)
    out = tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
    out = [el / 255 for el in out] + [alpha]
    return tuple(out)

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

def get_loss_abs_diff_running_avg_history(
        histories, avg_interval=1, normalize_loss=False):
    loss_abs_diff = np.abs(
            np.diff(histories['loss'] / histories['loss'][0]
            if normalize_loss else histories['loss']))
    loss_abs_diff_running_avg = np.pad(
            np.convolve(loss_abs_diff, np.ones(avg_interval),
                        mode='valid') / avg_interval,
            (avg_interval - 1, 0), constant_values=np.inf)
    return loss_abs_diff_running_avg

def get_loss_abs_diff_running_max_history(
        histories, max_interval=1, normalize_loss=False):
    loss_abs_diff = np.abs(
            np.diff(histories['loss'] / histories['loss'][0]
            if normalize_loss else histories['loss']))
    loss_abs_diff_running_max = np.zeros_like(histories['loss'])
    for i in range(len(loss_abs_diff)):
        loss_abs_diff_window = loss_abs_diff[max(0, i+1-max_interval):i+1].copy()
        loss_abs_diff_window[loss_abs_diff_window == 0.] = np.nan
        loss_abs_diff_running_max[i] = np.std(loss_abs_diff_window[np.logical_not(np.isnan(loss_abs_diff_window))])
    return loss_abs_diff_running_max

def get_stop_time(histories, threshold, filter_kind='avg', interval=1):
    if filter_kind == 'avg':
        loss_abs_diff_running = get_loss_abs_diff_running_avg_history(
                histories, avg_interval=interval)
    elif filter_kind == 'max':
        loss_abs_diff_running = get_loss_abs_diff_running_max_history(
                histories, max_interval=interval)
    else:
        raise ValueError
    argwhere_stop = np.argwhere(loss_abs_diff_running < threshold)
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

baseline_histories = histories_list[baseline_run_idx]
baseline_psnr_histories = [
        h['psnr'] for h in baseline_histories]
baseline_psnr_steady = get_psnr_steady(
        baseline_psnr_histories,
        start=eval_settings_dict[data]['psnr_steady_start'],
        stop=eval_settings_dict[data]['psnr_steady_stop'])

print('baseline steady PSNR',
        baseline_psnr_steady)

if show_baseline_steady_psnr:
    h = ax_psnr.axhline(baseline_psnr_steady, color='gray', linestyle='--', zorder=1.5)
    baseline_handle = h

run_handles = []
stop_time_handles = []
rise_time_handles = []

eval_results_list = []

for i, (run_spec, cfgs, experiment_names, histories) in enumerate(zip(
        runs_to_compare, cfgs_list, experiment_names_list, histories_list)):

    best_psnr_histories = [get_best_output_psnr_history(h) for h in histories]
    loss_histories = [h['loss'] for h in histories]
    best_loss_histories = [get_best_loss_history(h) for h in histories]
    psnr_histories = [h['psnr'] for h in histories]
    loss_abs_diff_histories = [np.abs(np.diff(h['loss'])) for h in histories]
    mean_loss_abs_diff_history = np.mean(loss_abs_diff_histories, axis=0)
    if use_only_median_rep:
        median_psnr_rep = median_psnr_reps[run_spec['experiment']]['sample_0']
    loss_abs_diff_running_avg_histories = [
            get_loss_abs_diff_running_avg_history(h,
                    avg_interval=eval_settings_dict[data]['stop_avg_interval'],
                    normalize_loss=False)
            for h in histories]
    mean_loss_abs_diff_running_avg_history = np.mean(
            loss_abs_diff_running_avg_histories, axis=0)

    stop_times = [
            get_stop_time(h,
                    threshold=eval_settings_dict[data]['stop_threshold'],
                    filter_kind=eval_settings_dict[data]['filter_kind'],
                    interval=eval_settings_dict[data][{'avg': 'stop_avg_interval', 'max': 'stop_max_interval'}[eval_settings_dict[data]['filter_kind']]])
            for h in histories]

    rise_times_to_baseline = []
    for psnr_history in psnr_histories:
        try:
            rise_times_to_baseline.append(get_rise_time_to_baseline(
                    [psnr_history], baseline_psnr_steady,
                    remaining_psnr=eval_settings_dict[data][
                            'rise_time_to_baseline_remaining_psnr']))
        except IndexError:
            rise_times_to_baseline.append(None)

    eval_results = {}
    eval_results['run_spec'] = run_spec
    eval_results['stop_times'] = stop_times

    eval_results_list.append(eval_results)

    label = get_label(run_spec, cfgs[0])
    color = get_color(run_spec, cfgs[0])
    linestyle = run_spec.get('linestyle', 'solid')

    zorder = plot_settings_dict[data].get('zorder_per_run_idx', {}).get(
            i, 2 + 0.1 * i)

    best_psnr_data = [best_psnr_histories[median_psnr_rep]] if use_only_median_rep else best_psnr_histories
    for best_psnr_history in best_psnr_data:
        h = ax_psnr.plot(best_psnr_history, label=label, color=color,
                        linestyle=(linestyle if show_stop_times else 'dashed'), linewidth=2,
                        zorder=zorder)
    run_handles += h
    best_psnr_handle = h[0]
    psnr_data = [psnr_histories[median_psnr_rep]] if use_only_median_rep else psnr_histories
    for psnr_history in psnr_data:
        h = ax_psnr.plot(psnr_history, label=label, color=color,
                        linestyle='-', linewidth=2,
                        zorder=zorder-0.5, alpha=0.5)
    psnr_handle = h[0]

    best_loss_data = [best_loss_histories[median_psnr_rep]] if use_only_median_rep else best_loss_histories
    for best_loss_history in best_loss_data:
        h = ax_loss.plot(best_loss_history, label=label, color=color,
                        linestyle=(linestyle if show_stop_times else 'dashed'), linewidth=2,
                        zorder=zorder)
    best_loss_handle = h[0]
    loss_data = [loss_histories[median_psnr_rep]] if use_only_median_rep else loss_histories
    for loss_history in loss_data:
        h = ax_loss.plot(loss_history, label=label, color=color,
                        linestyle='-', linewidth=2,
                        zorder=zorder-0.5, alpha=0.5)
    loss_handle = h[0]

    if show_stop_times:
        x_stop_times = sorted([
                (stop_time if stop_time is not None else
                cfgs[0].mdl.optim.iterations + 1)
                for stop_time in stop_times])
        stop_time_y_pos_shifted = ax_loss.transData.inverted().transform(
                ax_loss.transAxes.transform(
                    (0., ax_loss.transAxes.inverted().transform(
                        ax_loss.transData.transform(
                            (0., plot_settings_dict[data]['stop_time_y_pos'])))[1] +
                        plot_settings_dict[data][
                                'stop_time_y_shift_per_run_idx'].get(i, 0))))[1]

        h = ax_loss.plot(x_stop_times,
                [stop_time_y_pos_shifted] * len(x_stop_times),
                color=color, linestyle='-', solid_capstyle='butt', linewidth=2,
                marker='|', markersize=4, zorder=zorder)
        stop_time_handles += h

    if show_rise_times:
        x_rise_times_to_baseline = sorted([
                (rise_time_to_baseline if rise_time_to_baseline is not None else
                cfgs[0].mdl.optim.iterations + 1)
                for rise_time_to_baseline in rise_times_to_baseline])
        rise_time_to_baseline_y_pos_shifted = (
                plot_settings_dict[data]['rise_time_to_baseline_y_pos'] +
                        plot_settings_dict[data][
                                'rise_time_to_baseline_y_shift_per_run_idx']
                                .get(i, 0))
        h = ax_psnr.plot(x_rise_times_to_baseline,
                [rise_time_to_baseline_y_pos_shifted] * len(
                        x_rise_times_to_baseline),
                color=color, linestyle='-', solid_capstyle='butt', linewidth=2,
                marker='*', markersize=8, zorder=zorder)
        rise_time_handles += h
        # ax_psnr.plot([rise_time_to_baseline, rise_time_to_baseline],
        #         [median_psnr_history[rise_time_to_baseline],
        #             rise_time_to_baseline_y_pos_shifted],
        #         color=color, linestyle='--',
        #         zorder=zorder)

    # for loss_abs_diff_running_avg in loss_abs_diff_running_avg_histories:
    #     h = ax_loss_diff.plot(loss_abs_diff_running_avg,
    #             label=label, color=color, linestyle=linestyle, linewidth=2,
    #             zorder=zorder, alpha=0.1)
    loss_abs_diff_data = [loss_abs_diff_running_avg_histories[median_psnr_rep]] if use_only_median_rep else loss_abs_diff_running_avg_histories
    for h in loss_abs_diff_data:
        ax_loss_diff.plot(h,
                label=label, color=color, linestyle=linestyle, linewidth=2,
                zorder=zorder)

    for i, iteration_slice in enumerate(plot_settings_dict[data]['hist_iteration_slices']):
        hist_data = loss_abs_diff_histories[median_psnr_rep] if use_only_median_rep else mean_loss_abs_diff_history
        ax_loss_diff_hists[i].hist(
                mean_loss_abs_diff_history[iteration_slice],
                bins=np.logspace(np.log10(np.min(mean_loss_abs_diff_history[mean_loss_abs_diff_history != 0])),
                                 np.log10(np.max(mean_loss_abs_diff_history)),
                                 plot_settings_dict[data]['hist_num_bins']),
                # bins=plot_settings_dict[data]['hist_num_bins'],
                label=label,
                histtype='stepfilled',
                linewidth=plot_settings_dict[data]['hist_linewidth'],
                linestyle='dashed', density=True, zorder=zorder,
                facecolor=hex_to_rgb(
                        color, alpha=plot_settings_dict[data]['hist_opacity']),
                edgecolor=hex_to_rgb(color, alpha=1))
        ax_loss_diff_hists[i].set_title(('It. ' if i == 0 else '') + '{}$-${}'.format(
                iteration_slice.start, iteration_slice.stop), fontsize=8, pad=3)

for ax in (ax_psnr, ax_loss, ax_loss_diff):
    ax.grid(True, linestyle='-')
    ax.set_xlabel('Iteration')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

ax_psnr.set_xlim(xlim)
ax_loss.set_xlim(xlim)
ax_loss_diff.set_xlim(plot_settings_dict[data].get(
        'xlim_loss_diff', (None, None)))

if show_stop_times:
    ax_psnr.set_title('$\mathrm{PSNR}(\\varphi_{\\theta^{[i]}_{\\mathrm{min\u2010loss}}}(z); x)$')
else:
    ax_psnr.set_title('PSNR')
ax_psnr.set_ylabel('PSNR [dB]')
if show_stop_times:
    ax_loss.set_title('DIP+TV loss $l_t(\\theta^{[i]}_{\\mathrm{min\u2010loss}})$')
else:
    ax_loss.set_title('DIP+TV loss')
ax_loss_diff.set_title(
        'Moving average of $|l_t(\\theta^{[i+1]})-l_t(\\theta^{[i]})|$',
        loc='right')
fake11.set_title(
        'Histogram of $|l_t(\\theta^{[i+1]})-l_t(\\theta^{[i]})|$')
ax_psnr.set_ylim(plot_settings_dict[data].get('ylim_psnr', (None, None)))
ax_loss.set_ylim(plot_settings_dict[data].get('ylim_loss', (None, None)))
ax_loss_diff.set_ylim(plot_settings_dict[data].get(
        'ylim_loss_diff', (None, None)))

run_legend = ax_psnr.legend(
        handles=run_handles,
        bbox_transform=fig.transFigure,
        bbox_to_anchor=plot_settings_dict[data].get(
                'run_legend_bbox_to_anchor', (0.5, -0.125)),
        loc=plot_settings_dict[data].get('run_legend_loc', 'upper center'),
        ncol=plot_settings_dict[data].get(
                'run_legend_ncol') or ceil(len(runs_to_compare) / 2),
        framealpha=1.,
        handletextpad=plot_settings_dict[data].get('run_legend_handletextpad'),
        columnspacing=plot_settings_dict[data].get('run_legend_columnspacing'))
ax_psnr.add_artist(run_legend)
if show_stop_times:
    stop_time_handle = copy(stop_time_handles[0])
    stop_time_handle.set_color('gray')
    symbol_legend_loss = ax_loss.legend(
            [stop_time_handle],
            ['Stop times'  # (moving avg. size {:d}, threshold {:g})'.format(
                    #  eval_settings_dict[data]['stop_avg_interval'],
                    #  eval_settings_dict[data]['stop_threshold'])
            ],
            handler_map={stop_time_handle: HandlerLine2D(numpoints=5)},
            bbox_to_anchor=plot_settings_dict[data].get(
                    'symbol_legend_loss_bbox_to_anchor', (1., 0.825)),
            loc=plot_settings_dict[data].get('symbol_legend_loss_loc', 'upper right'),
            ncol=1, framealpha=1.,
            )
    symbol_legend_loss.set_zorder(50.)
else:
    best_loss_handle = copy(best_loss_handle)
    best_loss_handle.set_color('gray')
    loss_handle = copy(loss_handle)
    loss_handle.set_color('gray')
    symbol_legend_loss = ax_loss.legend(
            [loss_handle, best_loss_handle],
            ['$l_t(\\theta^{[i]})$', '$l_t(\\theta^{[i]}_{\\mathrm{min\u2010loss}})$'],
            bbox_to_anchor=plot_settings_dict[data].get(
                    'symbol_legend_loss_bbox_to_anchor', (1., 0.825)),
            loc=plot_settings_dict[data].get('symbol_legend_loss_loc', 'upper right'),
            ncol=1, framealpha=1.,
            )
    symbol_legend_loss.set_zorder(50.)

if show_rise_times:
    rise_time_handle = copy(rise_time_handles[0])
    rise_time_handle.set_color('gray')
    symbol_legend_psnr = ax_psnr.legend(
            [rise_time_handle],
            ['Rise times'],
            handler_map={rise_time_handle: HandlerLine2D(numpoints=3)},
            bbox_to_anchor=plot_settings_dict[data].get(
                    'symbol_legend_psnr_bbox_to_anchor', (1., 0.825)),
            loc=plot_settings_dict[data].get('symbol_legend_psnr_loc', 'upper right'),
            ncol=1, framealpha=1.,
            )
    symbol_legend_psnr.set_zorder(50.)
else:
    best_psnr_handle = copy(best_psnr_handle)
    best_psnr_handle.set_color('gray')
    psnr_handle = copy(psnr_handle)
    psnr_handle.set_color('gray')
    symbol_legend_psnr = ax_psnr.legend(
            [best_psnr_handle, psnr_handle],
            ['$\mathrm{PSNR}(\\varphi_{\\theta^{[i]}_{\\mathrm{min\u2010loss}}}(z); x)$', '$\mathrm{PSNR}(\\varphi_{\\theta^{[i]}}(z); x)$'],
            bbox_to_anchor=plot_settings_dict[data].get(
                    'symbol_legend_psnr_bbox_to_anchor', (1., 0.825)),
            loc=plot_settings_dict[data].get('symbol_legend_psnr_loc', 'upper right'),
            ncol=1, framealpha=1.,
            )
    symbol_legend_psnr.set_zorder(50.)

if title is None:
    if runs_title is None:
        runs_title = ' vs '.join(
                [get_title_from_run_spec(r) for r in runs_to_compare])
    title = ('{} on {}'.format(runs_title, data_title) if runs_title != '' else
             data_title)
fig.suptitle(title)

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
