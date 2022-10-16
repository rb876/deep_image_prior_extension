import os
import json
from warnings import warn
import numpy as np
import yaml
from evaluation.utils import (
        get_multirun_cfgs, get_multirun_experiment_names,
        get_multirun_histories, uses_swa_weights)
from evaluation.evaluation import (
        get_median_psnr_history, get_psnr_steady, get_rise_time_to_baseline)
from evaluation.display_utils import (
    data_title_dict, experiment_color_dict, get_title_from_run_spec)
from copy import copy
from math import ceil

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# PATH = '/media/chen/Res/deep_image_prior_extension/'
# PATH = '/localdata/jleuschn/experiments/deep_image_prior_extension/'
PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
TVADAM_PSNRS_FILEPATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'baselines', 'tvadam_psnrs.yaml')
        # '../baselines', 'tvadam_psnrs.yaml')

FIG_PATH = os.path.dirname(__file__)
EVAL_RESULTS_PATH = os.path.dirname(__file__)

save_fig = True
save_eval_results = True
formats = ('pdf', 'png')

with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'runs.yaml'),
        'r') as f:
    runs = yaml.load(f, Loader=yaml.FullLoader)

# data = 'ellipses_lotus_20'
# data = 'ellipses_lotus_limited_45'
# data = 'brain_walnut_120'
# data = 'ellipses_walnut_120'
# data = 'ellipsoids_walnut_3d'
# data = 'ellipsoids_walnut_3d_60'
data = 'meta_pretraining_lotus_20'

variant = ''
# variant = 'all'
# variant = 'checkpoints'
# variant = 'checkpoints_epochs'
# variant = 'adversarial'

show_tv_in_inset = (variant == 'all' and not '3d' in data)

use_best_output_psnr = False

# Additional `run_spec` dict fields:
#
#     color : str, optional
#         Line color.
#     skip_psnr0 : bool, optional
#         Whether to skip the marker indicating the initial PSNR.

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
            {
            'experiment': 'no_pretrain',
            'name': 'no_stats_no_sigmoid_fixed_encoder',
            'experiment_title': 'DIP-FE (noise)',
            'name_title': '',
            'color': 'gray',
            'skip_psnr0': True,
            },
            *([{
            'experiment': 'no_pretrain_fbp',
            'name': 'no_stats_no_sigmoid_fixed_encoder',
            'experiment_title': 'DIP-FE (FBP)',
            'name_title': '',
            'color': '#00AAFF',
            'skip_psnr0': True,
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
            {
            'experiment': 'pretrain_only_fbp',
            'name': 'no_stats_no_sigmoid_train_run2_epochs100_fixed_encoder',
            'experiment_title': 'EDIP-FE (FBP)',
            'name_title': '',
            'color': '#EC2215',
            'skip_psnr0': True,
            },
            *([{
            'experiment': 'pretrain',
            'name': 'no_stats_no_sigmoid_train_run2_epochs100_fixed_encoder',
            'experiment_title': 'EDIP-FE (noise)',
            'name_title': '',
            'color': '#B15CD1',
            'skip_psnr0': True,
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
            # {
            # 'experiment': 'pretrain_only_fbp',
            # 'name': 'no_stats_no_sigmoid_train_run0_epochs60',
            # 'experiment_title': 'Run 0: 60 epochs',
            # 'name_title': '',
            # 'color': '#51518C',
            # },
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
            # {
            # 'experiment': 'pretrain_only_fbp',
            # 'name': 'no_stats_no_sigmoid_train_run1_epochs60',
            # 'experiment_title': 'Run 1: 60 epochs',
            # 'name_title': '',
            # 'color': '#8C5151',
            # },
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
            # {
            # 'experiment': 'pretrain_only_fbp',
            # 'name': 'no_stats_no_sigmoid_train_run2_epochs60',
            # 'experiment_title': 'Run 2: 60 epochs',
            # 'name_title': '',
            # 'color': '#518C51',
            # },
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
    elif variant == 'adversarial':
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
            {
            'experiment': 'pretrain_only_fbp',
            'name': 'no_stats_no_sigmoid_train_run2_epochs100',
            'name_title': '',
            },
            {
            'experiment': 'pretrain_only_fbp',
            'name': 'no_stats_no_sigmoid_adversarial_pretraining_epochs100',
            'name_title': 'Adversarial',
            'color': '#ffaa00'
            },
        ]

elif data == 'ellipses_lotus_limited_45':
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
            {
            'experiment': 'no_pretrain',
            'name': 'no_stats_no_sigmoid_fixed_encoder',
            'experiment_title': 'DIP-FE (noise)',
            'name_title': '',
            'color': 'gray',
            'skip_psnr0': True,
            },
            *([{
            'experiment': 'no_pretrain_fbp',
            'name': 'no_stats_no_sigmoid_fixed_encoder',
            'experiment_title': 'DIP-FE (FBP)',
            'name_title': '',
            'color': '#00AAFF',
            'skip_psnr0': True,
            }] if variant == 'all' else []),
            {
            'experiment': 'pretrain_only_fbp',
            'name': 'no_stats_no_sigmoid_train_run1_epochs100',
            'name_title': '',
            },
            {
            'experiment': 'pretrain',
            'name': 'no_stats_no_sigmoid_train_run1_epochs100',
            'name_title': '',
            },
            {
            'experiment': 'pretrain_only_fbp',
            'name': 'no_stats_no_sigmoid_train_run1_epochs100_fixed_encoder',
            'experiment_title': 'EDIP-FE (FBP)',
            'name_title': '',
            'color': '#EC2215',
            'skip_psnr0': True,
            },
            *([{
            'experiment': 'pretrain',
            'name': 'no_stats_no_sigmoid_train_run1_epochs100_fixed_encoder',
            'experiment_title': 'EDIP-FE (noise)',
            'name_title': '',
            'color': '#B15CD1',
            'skip_psnr0': True,
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
            {
            'experiment': 'no_pretrain',
            'name': 'no_stats_no_sigmoid_fixed_encoder',
            'experiment_title': 'DIP-FE (noise)',
            'name_title': '',
            'color': 'gray',
            'skip_psnr0': True,
            },
            *([{
            'experiment': 'no_pretrain_fbp',
            'name': 'no_stats_no_sigmoid_fixed_encoder',
            'experiment_title': 'DIP-FE (FBP)',
            'name_title': '',
            'color': '#00AAFF',
            'skip_psnr0': True,
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
            {
            'experiment': 'pretrain_only_fbp',
            'name': 'no_stats_no_sigmoid_train_run1_fixed_encoder',
            'experiment_title': 'EDIP-FE (FBP)',
            'name_title': '',
            'color': '#EC2215',
            'skip_psnr0': True,
            },
            *([{
            'experiment': 'pretrain',
            'name': 'no_stats_no_sigmoid_train_run1_fixed_encoder',
            'experiment_title': 'EDIP-FE (noise)',
            'name_title': '',
            'color': '#B15CD1',
            'skip_psnr0': True,
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
            {
            'experiment': 'no_pretrain',
            'name': 'no_stats_no_sigmoid_fixed_encoder',
            'experiment_title': 'DIP-FE (noise)',
            'name_title': '',
            'color': 'gray',
            'skip_psnr0': True,
            },
            *([{
            'experiment': 'no_pretrain_fbp',
            'name': 'no_stats_no_sigmoid_fixed_encoder',
            'experiment_title': 'DIP-FE (FBP)',
            'name_title': '',
            'color': '#00AAFF',
            'skip_psnr0': True,
            }] if variant == 'all' else []),
            {
            'experiment': 'pretrain_only_fbp',
            'name': 'no_stats_no_sigmoid_train_run1',
            'name_title': '',
            },
            # {
            # 'experiment': 'pretrain',
            # 'name': 'no_stats_no_sigmoid_train_run1',
            # 'name_title': '',
            # },
            {
            'experiment': 'pretrain',
            'name': 'no_stats_no_sigmoid_train_run1_warmup5000_init5e-4',
            'name_title': 'warm-up',
            },
            {
            'experiment': 'pretrain_only_fbp',
            'name': 'no_stats_no_sigmoid_train_run1_fixed_encoder',
            'experiment_title': 'EDIP-FE (FBP)',
            'name_title': '',
            'color': '#EC2215',
            'skip_psnr0': True,
            },
            # *([{
            # 'experiment': 'pretrain',
            # 'name': 'no_stats_no_sigmoid_train_run1_fixed_encoder',
            # 'experiment_title': 'EDIP-FE (noise)',
            # 'name_title': '',
            # 'color': '#B15CD1',
            # 'skip_psnr0': True,
            # }] if variant == 'all' else []),
            *([{
            'experiment': 'pretrain',
            'name': 'no_stats_no_sigmoid_train_run1_warmup5000_init5e-4_fixed_encoder',
            'experiment_title': 'EDIP-FE (noise)',
            'name_title': 'warm-up',
            'color': '#B15CD1',
            'skip_psnr0': True,
            }] if variant == 'all' else []),
        ]
    elif variant == 'checkpoints':
        runs_to_compare = []

elif data == 'ellipsoids_walnut_3d':
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
            {
            'experiment': 'pretrain_only_fbp',
            'name': 'epochs0_steps8000_fixed_encoder',
            'experiment_title': 'EDIP-FE (FBP)',
            'name_title': '',
            'color': '#EC2215',
            'skip_psnr0': True,
            },
        ]
    elif variant == 'checkpoints_epochs':
        runs_to_compare = [
            {
            'experiment': 'pretrain_only_fbp',
            'name': 'other_checkpoints',
            'sub_runs': [0],
            'experiment_title': '0.125 epochs',
            'name_title': '',
            'color': plt.get_cmap('magma')(80),
            },
            {
            'experiment': 'pretrain_only_fbp',
            'name': 'epochs0_steps8000',
            'sub_runs': [0],
            'experiment_title': '0.250 epochs',
            'name_title': '',
            'color': plt.get_cmap('magma')(128),
            },
            {
            'experiment': 'pretrain_only_fbp',
            'name': 'other_checkpoints',
            'sub_runs': [1],
            'experiment_title': '0.375 epochs',
            'name_title': '',
            'color': plt.get_cmap('magma')(160),
            },
            {
            'experiment': 'pretrain_only_fbp',
            'name': 'other_checkpoints',
            'sub_runs': [2],
            'experiment_title': '0.500 epochs',
            'name_title': '',
            'color': plt.get_cmap('magma')(192),
            },
            {
            'experiment': 'pretrain_only_fbp',
            'name': 'other_checkpoints',
            'sub_runs': [3],
            'experiment_title': '1.000 epochs',
            'name_title': '',
            'color': plt.get_cmap('magma')(208),
            },
            {
            'experiment': 'pretrain_only_fbp',
            'name': 'other_checkpoints',
            'sub_runs': [4],
            'experiment_title': '2.000 epochs',
            'name_title': '',
            'color': plt.get_cmap('magma')(232),
            },
            {
            'experiment': 'no_pretrain',
            'name': 'default',
            # 'sub_runs': [0],
            'name_title': '',
            },
        ]

elif data == 'ellipsoids_walnut_3d_60':
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
            {
            'experiment': 'pretrain_only_fbp',
            'name': 'epochs0_steps8000_fixed_encoder',
            'experiment_title': 'EDIP-FE (FBP)',
            'name_title': '',
            'color': '#EC2215',
            'skip_psnr0': True,
            },
        ]
    elif variant == 'checkpoints_epochs':
        runs_to_compare = [
            {
            'experiment': 'pretrain_only_fbp',
            'name': 'other_checkpoints',
            'sub_runs': [0],
            'experiment_title': '0.125 epochs',
            'name_title': '',
            'color': plt.get_cmap('magma')(80),
            },
            {
            'experiment': 'pretrain_only_fbp',
            'name': 'other_checkpoints',
            'sub_runs': [1],
            'experiment_title': '0.250 epochs',
            'name_title': '',
            'color': plt.get_cmap('magma')(128),
            },
            {
            'experiment': 'pretrain_only_fbp',
            'name': 'other_checkpoints',
            'sub_runs': [2],
            'experiment_title': '0.375 epochs',
            'name_title': '',
            'color': plt.get_cmap('magma')(160),
            },
            {
            'experiment': 'pretrain_only_fbp',
            'name': 'other_checkpoints',
            'sub_runs': [3],
            'experiment_title': '0.500 epochs',
            'name_title': '',
            'color': plt.get_cmap('magma')(192),
            },
            {
            'experiment': 'no_pretrain',
            'name': 'default',
            # 'sub_runs': [0],
            'name_title': '',
            },
        ]

elif data == 'meta_pretraining_lotus_20':
    runs_to_compare = [
        {
        'experiment': 'pretrain_only_fbp',
        'name': 'no_stats_no_sigmoid_adversarial_pretraining_epochs_100',
        'name_title': 'MAML',
        'color': '#00ff00'
        },
    ]


if not variant or variant == 'all':
    baseline_run_idx = 0
elif variant == 'checkpoints' or variant == 'checkpoints_epochs':
    baseline_run_idx = -1
elif variant == 'adversarial':
    baseline_run_idx = 0

title = None
runs_title = ''  # None -> auto from run_specs
if variant == 'checkpoints':
    runs_title = 'Comparing training runs for EDIP'
    # runs_title = 'The perks of being a validated EDIP'
elif variant == 'checkpoints_epochs':
    runs_title = 'Comparing training epochs for EDIP'
runs_filename = 'comparison'  # None -> auto from run_specs

plot_settings_dict = {
    'ellipses_lotus_20': {
        'xlim': (
            (-625, 10000) if (not variant) or variant == 'all' else (
            (-1125, 10000) if variant == 'checkpoints' else
            (None, None))
        ),
        'xlim_inset': (
            (-200, 6750) if (not variant) or variant == 'all' else (
            (-50, 2250) if variant == 'checkpoints' else
            (None, None))
        ),
        'ylim': (
            (None, 34.05) if (not variant) or variant == 'all' else (
            (None, 36.5) if variant == 'checkpoints' else
            (None, None))
        ),
        'ylim_inset': (
            (30.5, 31.85) if (not variant) or variant == 'all' else (
            (30.75, 31.85) if variant == 'checkpoints' else
            (29.25, 31.85))
        ),
        'psnr0_x_pos': -187.5,
        'psnr0_x_shift_per_run_idx': {
            0: -250,
        } if (not variant) or variant == 'all' else ({
            0: -750,
            1: -650,
            2: -425,
            3: -325,
            4: -100,
            5: 0,
        } if variant == 'checkpoints' else {}),
        'rise_time_to_baseline_y_pos': 32.5,
        'rise_time_to_baseline_y_shift_per_run_idx': {
            3: 1.
        } if not variant else ({
            4: 1.,
        } if variant == 'all' else ({
            0: 2.0,
            1: 0.0,
            2: 2.6,
            3: 0.6,
            4: 3.2,
            5: 1.2,
        } if variant == 'checkpoints' else {})),
        'zorder_per_run_idx': {
        } if (not variant) or variant == 'all' else ({
            0: 2.4,
            1: 2.1,
            2: 2.5,
            3: 2.2,
            4: 2.6,
            5: 2.3,
        } if variant == 'checkpoints' else {}),
        'inset_axes_rect': (
            [0.255, 0.175, 0.725, 0.55]
            if (not variant) or variant == 'all' else (
            [0.265, 0.175, 0.715, 0.5]
            if variant == 'checkpoints' else
            [0.255, 0.175, 0.725, 0.55])
        ),
        'inset_axes_rect_border': (
            [0.085, 0.0675] if (not variant) or variant == 'all' else (
            [0.07, 0.0675] if variant == 'checkpoints' else
            [0.085, 0.0675])
        ),
        'tv_text_shift': [30, 0.02],
        'run_legend_bbox_to_anchor': (
            (0.5, -0.125) if (not variant) or variant == 'all' else (
            (0.48, -0.125) if variant == 'checkpoints' else
            (0.5, -0.125))
        ),
        'run_legend_loc': 'upper center',
        'run_legend_handletextpad': (0.6 if variant == 'checkpoints' else None),
        'run_legend_columnspacing': (1.5 if variant == 'checkpoints' else None),
    },
    'ellipses_lotus_limited_45': {
        'xlim': (-625, 10000),
        'xlim_inset': (-200, 4750),
        'ylim': (None, 32.5),
        'ylim_inset': (28.5, 30.1),
        'psnr0_x_pos': -187.5,
        'psnr0_x_shift_per_run_idx': {
            0: -250,
        },
        'rise_time_to_baseline_y_pos': 31,
        'rise_time_to_baseline_y_shift_per_run_idx': {
            1: 0.9,
            3: 0.9,
            4: 0.9
        } if variant == 'all' else ( {
            1: 0.9,
        }),
        'inset_axes_rect': [0.245, 0.2, 0.715, 0.575],
        'inset_axes_rect_border': [0.085, 0.0675],
        'tv_text_shift': [30, -0.1],
        'run_legend_bbox_to_anchor': (
            (0.5, -0.125) if (not variant) or variant == 'all' else (
            (0.48, -0.125) if variant == 'checkpoints' else
            (0.5, -0.125))
        ),
        'run_legend_loc': 'upper center',
        'run_legend_handletextpad': (0.6 if variant == 'checkpoints' else None),
        'run_legend_columnspacing': (1.5 if variant == 'checkpoints' else None),
    },
    'brain_walnut_120': {
        'xlim': (
            (-1875, 30000) if (not variant) or variant == 'all' else (
            # (-20, 100) if (not variant) or variant == 'all' else (
            (-2625, 30000) if variant == 'checkpoints' else (
            (-4125, 30000) if variant == 'checkpoints_epochs' else
            (None, None)))
        ),
        'xlim_inset': (
            (-600, 21250) if (not variant) or variant == 'all' else (
            (-97.5, 9750) if variant == 'checkpoints' else
            (5000, 25250) if variant == 'checkpoints_epochs' else
            (None, None))
        ),
        'ylim': (
            (None, 38.5) if (not variant) or variant == 'all' else (
            (None, 38.5) if variant == 'checkpoints' else (
            (None, 37.) if variant == 'checkpoints_epochs' else
            (None, None)))
        ),
        'ylim_inset': (
            (29.5, 34.75) if (not variant) or variant == 'all' else (
            (25.5, 34.65) if variant == 'checkpoints' else
            (26.5, 35.25) if variant == 'checkpoints_epochs' else
            (None, None))
        ),
        'psnr0_x_pos': -562.5,
        'psnr0_x_shift_per_run_idx': {
            0: -750,
        } if (not variant) or variant == 'all' else (
        {
            0: -1500,
            1: -750,
            2: 0,
        } if variant == 'checkpoints' else ({
            0: -3000,
            1: -2250,
            2: -1500,
            3: -750,
            4: 0,
        } if variant == 'checkpoints_epochs' else {})),
        'rise_time_to_baseline_y_pos': 35.75,
        'rise_time_to_baseline_y_shift_per_run_idx': {
        } if not variant else ({
            1: 1.8,
        } if variant == 'all' else ({
            0: 1.8,
            1: 0.9,
            2: 0.,
        } if variant == 'checkpoints' else {})),
        'zorder_per_run_idx': {
        } if (not variant) or variant == 'all' else ({
            0: 2.6,
            1: 2.3,
            2: 2.5,
            3: 2.2,
            4: 2.4,
            5: 2.1,
        } if variant == 'checkpoints' else ({
            0: 2.6,
            1: 2.5,
            2: 2.4,
            3: 2.3,
            4: 2.2,
            5: 2.1,
        } if variant == 'checkpoints_epochs' else {})),
        'inset_axes_rect': [0.255, 0.175, 0.725, 0.45],
        'inset_axes_rect_border': [0.0625, 0.0675],
        'tv_text_shift': [80, 0.05],
        'ylabel_pad': 0.,
        'run_legend_ncol': (
            len(runs_to_compare) if variant == 'checkpoints' else None),
        'run_legend_bbox_to_anchor': (
            (0.5, -0.125) if (not variant) or variant == 'all' else (
            (0.475, -0.125) if variant == 'checkpoints' else
            (0.5, -0.125))
        ),
        'run_legend_loc': 'upper center',
        'run_legend_handletextpad': (0.4 if variant == 'checkpoints' else None),
        'run_legend_columnspacing': (1.2 if variant == 'checkpoints' else None),
    },
    'ellipses_walnut_120': {
        'xlim': (
            (-1875, 30000) if (not variant) or variant == 'all' else (
            (-2625, 30000) if variant == 'checkpoints' else
            (None, None))
        ),
        'xlim_inset': (
            (-600, 21250) if (not variant) or variant == 'all' else (
            (-62.5, 6250) if variant == 'checkpoints' else
            (None, None))
        ),
        'ylim': (
            (None, 38.5) if (not variant) or variant == 'all' else (
            (None, 38.5) if variant == 'checkpoints' else
            (None, None))
        ),
        'ylim_inset': (
            (29.0, 34.5) if (not variant) or variant == 'all' else (
            (25.5, 34.65) if variant == 'checkpoints' else
            (None, None))
        ),
        'psnr0_x_pos': -562.5,
        'psnr0_x_shift_per_run_idx': {
            0: -750,
        } if (not variant) or variant == 'all' else ({
            0: -1500,
            1: -750,
            2: 0,
        } if variant == 'checkpoints' else {}),
        'rise_time_to_baseline_y_pos': 35.75,
        'rise_time_to_baseline_y_shift_per_run_idx': {
            2: 1.8,
            3: 1.8,
        } if not variant else ({
            2: 1.8,
            3: 1.8,
            4: 1.8,
        } if variant == 'all' else ({
            0: 1.8,
            1: 0.9,
            2: 0.,
        } if variant == 'checkpoints' else {})),
        'zorder_per_run_idx': {
        } if (not variant) or variant == 'all' else ({
            0: 2.6,
            1: 2.3,
            2: 2.5,
            3: 2.2,
            4: 2.4,
            5: 2.1,
        } if variant == 'checkpoints' else {}),
        'inset_axes_rect': [0.255, 0.175, 0.725, 0.475],
        'inset_axes_rect_border': [0.0625, 0.0675],
        'tv_text_shift': [80, 0.05],
        'ylabel_pad': 0.,
        'run_legend_ncol': (
            len(runs_to_compare) if variant == 'checkpoints' else None),
        'run_legend_bbox_to_anchor': (
            (0.5, -0.125) if (not variant) or variant == 'all' else (
            (0.475, -0.125) if variant == 'checkpoints' else
            (0.5, -0.125))
        ),
        'run_legend_loc': 'upper center',
        'run_legend_handletextpad': (0.4 if variant == 'checkpoints' else None),
        'run_legend_columnspacing': (1.2 if variant == 'checkpoints' else None),
    },
    'ellipsoids_walnut_3d': {
        'xlim': (
            (-1875, 30000) if (not variant) or variant == 'all' else (
            (-2625, 30000) if variant == 'checkpoints' else (
            (-4725, 30000) if variant == 'checkpoints_epochs' else
            (None, None)))
        ),
        'xlim_inset': (
            (-600, 18500) if (not variant) or variant == 'all' else (
            (-62.5, 6250) if variant == 'checkpoints' else (
            (0., 11250) if variant == 'checkpoints_epochs' else
            (None, None)))
        ),
        'ylim': (
            (3.75, 34.) if (not variant) or variant == 'all' else (
            (None, 34.) if variant == 'checkpoints' else
            (3.5, 34.5) if variant == 'checkpoints_epochs' else (
            (None, None)))
        ),
        'ylim_inset': (
            (27.0, 32.25) if (not variant) or variant == 'all' else (
            (27.0, 32.25) if variant == 'checkpoints' else (
            (27.5, 31.9) if variant == 'checkpoints_epochs' else
            (None, None)))
        ),
        'psnr0_x_pos': -562.5,
        'psnr0_x_shift_per_run_idx': {
            0: -750,
        } if (not variant) or variant == 'all' else ({
            0: -1500,
            1: -750,
            2: 0,
        } if variant == 'checkpoints' else ({
            0: -3500,
            1: -3000,
            2: -2250,
            3: -1500,
            4: -750,
            5: 0,
        }  if variant == 'checkpoints_epochs' else {})),
        'rise_time_to_baseline_y_pos': 33.,
        'rise_time_to_baseline_y_shift_per_run_idx': {
        } if not variant else ({
        } if variant == 'all' else ({
        } if variant == 'checkpoints' else ({
            0: 1.,
        } if variant == 'checkpoints_epochs' else {
        }))),
        'zorder_per_run_idx': {
        } if (not variant) or variant == 'all' else ({
            0: 2.5,
            1: 2.3,
            2: 2.6,
            3: 2.2,
            4: 2.4,
            5: 2.1,
        } if variant == 'checkpoints' else ({
            0: 2.5,
            1: 2.6,
            2: 2.4,
            3: 2.3,
            4: 2.2,
            5: 2.1,
            6: 2.0,
        } if variant == 'checkpoints_epochs' else {})),
        'inset_axes_rect': [0.255, 0.175, 0.725, 0.475],
        'inset_axes_rect_border': [0.0625, 0.0675],
        'tv_text_shift': [80, 0.05],
        'ylabel_pad': 0.,
        'run_legend_ncol': (
            ceil(len(runs_to_compare) / 2) if variant == 'checkpoints_epochs' else (
                len(runs_to_compare))),
        'run_legend_bbox_to_anchor': (
            (0.5, -0.125) if (not variant) or variant == 'all' else (
            (0.475, -0.125) if variant == 'checkpoints' else
            (0.5, -0.125))
        ),
        'run_legend_loc': 'upper center',
        'run_legend_handletextpad': (0.4 if variant == 'checkpoints' else None),
        'run_legend_columnspacing': (1.2 if variant == 'checkpoints' else None),
    },
    'ellipsoids_walnut_3d_60': {
        'xlim': (
            (-3750, 60000) if (not variant) or variant == 'all' else (
            (-5250, 60000) if variant == 'checkpoints' else (
            (-6500, 60000) if variant == 'checkpoints_epochs' else
            (None, None)))
        ),
        'xlim_inset': (
            (-1200, 37000) if (not variant) or variant == 'all' else (
            (-125, 12500) if variant == 'checkpoints' else (
            (-250, 25250) if variant == 'checkpoints_epochs' else
            (None, None)))
        ),
        'ylim': (
            (3.5, 36.5) if (not variant) or variant == 'all' else (
            (None, 36.5) if variant == 'checkpoints' else
            (3.5, 36.5) if variant == 'checkpoints_epochs' else (
            (None, None)))
        ),
        'ylim_inset': (
            (27.0, 35.25) if (not variant) or variant == 'all' else (
            (27.0, 35.25) if variant == 'checkpoints' else (
            (27.0, 35.25) if variant == 'checkpoints_epochs' else
            (None, None)))
        ),
        'psnr0_x_pos': -1000,
        'psnr0_x_shift_per_run_idx': {
            0: -1500,
        } if (not variant) or variant == 'all' else ({
            0: -3000,
            1: -1500,
            2: 0,
        } if variant == 'checkpoints' else ({
            0: -4500,
            1: -3000,
            2: -1500,
            3: 0,
        }  if variant == 'checkpoints_epochs' else {})),
        'rise_time_to_baseline_y_pos': 35.75,
        'rise_time_to_baseline_y_shift_per_run_idx': {
        } if not variant else ({
        } if variant == 'all' else ({
        } if variant == 'checkpoints' else ({
        } if variant == 'checkpoints_epochs' else {
        }))),
        'zorder_per_run_idx': {
        } if (not variant) or variant == 'all' else ({
            0: 2.5,
            1: 2.3,
            2: 2.6,
            3: 2.2,
            4: 2.4,
            5: 2.1,
        } if variant == 'checkpoints' else ({
            0: 2.5,
            1: 2.6,
            2: 2.4,
            3: 2.3,
            4: 2.2,
            5: 2.1,
            6: 2.0,
        } if variant == 'checkpoints_epochs' else {})),
        'inset_axes_rect': [0.255, 0.175, 0.725, 0.475],
        'inset_axes_rect_border': [0.0625, 0.0675],
        'tv_text_shift': [160, -0.6],
        'ylabel_pad': 0.,
        'run_legend_ncol': (
            ceil(len(runs_to_compare) / 2) if variant == 'checkpoints_epochs' else (
                len(runs_to_compare))),
        'run_legend_bbox_to_anchor': (
            (0.5, -0.125) if (not variant) or variant == 'all' else (
            (0.475, -0.125) if variant == 'checkpoints' else
            (0.5, -0.125))
        ),
        'run_legend_loc': 'upper center',
        'run_legend_handletextpad': (0.4 if variant == 'checkpoints' else None),
        'run_legend_columnspacing': (1.2 if variant == 'checkpoints' else None),
    },
    'meta_pretraining_lotus_20': {
        'xlim': (
            (-625, 10000) if (not variant) or variant == 'all' else (
            (-1125, 10000) if variant == 'checkpoints' else
            (None, None))
        ),
        'xlim_inset': (
            (-200, 6750) if (not variant) or variant == 'all' else (
            (-50, 2250) if variant == 'checkpoints' else
            (None, None))
        ),
        'ylim': (
            (None, 34.05) if (not variant) or variant == 'all' else (
            (None, 36.5) if variant == 'checkpoints' else
            (None, None))
        ),
        'ylim_inset': (
            (30.5, 31.85) if (not variant) or variant == 'all' else (
            (30.75, 31.85) if variant == 'checkpoints' else
            (29.25, 31.85))
        ),
        'psnr0_x_pos': -187.5,
        'psnr0_x_shift_per_run_idx': {
            0: -250,
        } if (not variant) or variant == 'all' else ({
            0: 0,
        } if variant == 'checkpoints' else {}),
        'rise_time_to_baseline_y_pos': 32.5,
        'rise_time_to_baseline_y_shift_per_run_idx': {
        } if not variant else ({
        } if variant == 'all' else ({
            0: 0.,
        } if variant == 'checkpoints' else {})),
        'zorder_per_run_idx': {
        } if (not variant) or variant == 'all' else ({
            0: 2.4,
        } if variant == 'checkpoints' else {}),
        'inset_axes_rect': (
            [0.255, 0.175, 0.725, 0.55]
            if (not variant) or variant == 'all' else (
            [0.265, 0.175, 0.715, 0.5]
            if variant == 'checkpoints' else
            [0.255, 0.175, 0.725, 0.55])
        ),
        'inset_axes_rect_border': (
            [0.085, 0.0675] if (not variant) or variant == 'all' else (
            [0.07, 0.0675] if variant == 'checkpoints' else
            [0.085, 0.0675])
        ),
        'tv_text_shift': [30, 0.02],
        'run_legend_bbox_to_anchor': (
            (0.5, -0.125) if (not variant) or variant == 'all' else (
            (0.48, -0.125) if variant == 'checkpoints' else
            (0.5, -0.125))
        ),
        'run_legend_loc': 'upper center',
        'run_legend_handletextpad': (0.6 if variant == 'checkpoints' else None),
        'run_legend_columnspacing': (1.5 if variant == 'checkpoints' else None),
    },
}

eval_settings_dict = {
    'ellipses_lotus_20': {
        'psnr_steady_start': -5000,
        'psnr_steady_stop': None,
        'rise_time_to_baseline_remaining_psnr': 0.1,
    },
    'ellipses_lotus_limited_45': {
        'psnr_steady_start': -5000,
        'psnr_steady_stop': None,
        'rise_time_to_baseline_remaining_psnr': 0.1,
    },
    'brain_walnut_120': {
        'psnr_steady_start': -5000,
        'psnr_steady_stop': None,
        'rise_time_to_baseline_remaining_psnr': 0.1,
    },
    'ellipses_walnut_120': {
        'psnr_steady_start': -5000,
        'psnr_steady_stop': None,
        'rise_time_to_baseline_remaining_psnr': 0.1,
    },
    'ellipsoids_walnut_3d': {
        'psnr_steady_start': -5000,
        'psnr_steady_stop': None,
        'rise_time_to_baseline_remaining_psnr': 0.1,
    },
    'ellipsoids_walnut_3d_60': {
        'psnr_steady_start': -5000,
        'psnr_steady_stop': None,
        'rise_time_to_baseline_remaining_psnr': 0.1,
    },
    'meta_pretraining_lotus_20': {
        'psnr_steady_start': -5000,
        'psnr_steady_stop': None,
        'rise_time_to_baseline_remaining_psnr': 0.1,
    }
}

data_title = data_title_dict[data]

fig, ax = plt.subplots(figsize=plot_settings_dict[data].get('figsize', (8, 5)),
                       gridspec_kw={'bottom': 0.2})

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
xlim_inset = plot_settings_dict[data]['xlim_inset']

if ((not variant) or variant == 'all' or variant == 'checkpoints' or
        variant == 'checkpoints_epochs'):
    axins = ax.inset_axes(plot_settings_dict[data]['inset_axes_rect'])
    axs = [ax, axins]
else:
    axins = None
    axs = [ax]

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
                data == 'meta_pretraining_lotus_20' or (
                    data in ['ellipses_walnut_120', 'brain_walnut_120'] and
                    all((not cfg['mdl']['load_pretrain_model']) and
                        cfg['data']['name'] in [
                                'ellipses_walnut_120', 'brain_walnut_120']
                        for cfg in cfgs)))
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

baseline_histories = histories_list[baseline_run_idx]
baseline_psnr_histories = [
        (get_best_output_psnr_history(h)
         if use_best_output_psnr else
         h['psnr'])
        for h in baseline_histories]
baseline_psnr_steady = get_psnr_steady(
        baseline_psnr_histories,
        start=eval_settings_dict[data]['psnr_steady_start'],
        stop=eval_settings_dict[data]['psnr_steady_stop'])

print('baseline steady PSNR' + (
        ' (using running best loss output)' if use_best_output_psnr
        else ''),
        baseline_psnr_steady)

# for ax_ in axs:
#     h = ax_.axhline(baseline_psnr_steady, color='gray', linestyle='--',
#                     zorder=1.5)
#     if ax_ is ax:
#         baseline_handle = h
axins_if_exist = axins if axins is not None else ax
h = axins_if_exist.axhline(baseline_psnr_steady, color='gray', linestyle='--', zorder=1.5)
baseline_handle = h

run_handles = []
psnr0_handles = []
rise_time_handles = []

eval_results_list = []

for i, (run_spec, cfgs, experiment_names, histories) in enumerate(zip(
        runs_to_compare, cfgs_list, experiment_names_list, histories_list)):

    psnr_histories = [
            (get_best_output_psnr_history(h)
             if use_best_output_psnr else
             h['psnr'])
            for h in histories]

    mean_psnr_history = np.mean(psnr_histories, axis=0)
    std_psnr_history = np.std(psnr_histories, axis=0)

    median_psnr_history = get_median_psnr_history(psnr_histories)
    try:
        rise_time_to_baseline = get_rise_time_to_baseline(
                psnr_histories, baseline_psnr_steady,
                remaining_psnr=eval_settings_dict[data][
                        'rise_time_to_baseline_remaining_psnr'])
    except IndexError:
        rise_time_to_baseline = None

    try:
        with open(TVADAM_PSNRS_FILEPATH, 'r') as f:
            tv_psnr = yaml.safe_load(f)[data]
    except FileNotFoundError:
        pass
    else:
        argwhere_reached_tv = np.argwhere(
                median_psnr_history > tv_psnr)
        rise_time_to_reach_tv = (
                int(argwhere_reached_tv[0][0])
                if len(argwhere_reached_tv) > 0 else None)

    eval_results = {}
    eval_results['run_spec'] = run_spec
    eval_results['rise_time_to_baseline'] = rise_time_to_baseline
    eval_results['PSNR_best'] = float(np.max(median_psnr_history))
    eval_results['PSNR_best_iter'] = int(np.argmax(median_psnr_history))
    eval_results['PSNR_steady'] = get_psnr_steady(
            psnr_histories,
            start=eval_settings_dict[data]['psnr_steady_start'],
            stop=eval_settings_dict[data]['psnr_steady_stop'])
    eval_results['PSNR_0'] = float(mean_psnr_history[0])
    if rise_time_to_reach_tv is not None:
        eval_results['rise_time_to_reach_tv'] = rise_time_to_reach_tv
    # it = 900
    # print(experiment_names[0], 'median PSNR at iter {:d}'.format(it), median_psnr_history[it])

    eval_results_list.append(eval_results)

    label = get_label(run_spec, cfgs[0])
    color = get_color(run_spec, cfgs[0])
    linestyle = run_spec.get('linestyle', 'solid')

    zorder = plot_settings_dict[data].get('zorder_per_run_idx', {}).get(i)

    # for psnr_history in psnr_histories:
    #     ax.plot(psnr_history, color=color, alpha=0.1)

    for ax_ in axs:
        ax_.fill_between(range(len(mean_psnr_history)),
                         mean_psnr_history - std_psnr_history,
                         mean_psnr_history + std_psnr_history,
                         color=color, alpha=0.1,
                         # edgecolor=None,
                         zorder=zorder,
                         )
        h = ax_.plot(mean_psnr_history, label=label, color=color,
                     linestyle=linestyle, linewidth=2,
                     zorder=zorder)
        if ax_ is ax:
            run_handles += h
        if rise_time_to_baseline is not None:
            rise_time_to_baseline_y_pos_shifted = (
                    plot_settings_dict[data]['rise_time_to_baseline_y_pos'] +
                            plot_settings_dict[data][
                                    'rise_time_to_baseline_y_shift_per_run_idx']
                                    .get(i, 0))
            h = ax_.plot(
                    rise_time_to_baseline, rise_time_to_baseline_y_pos_shifted,
                    '*', color=color, markersize=8,
                    zorder=zorder)
            if ax_ is ax:
                rise_time_handles += h
            ax_.plot([rise_time_to_baseline, rise_time_to_baseline],
                    [median_psnr_history[rise_time_to_baseline],
                     rise_time_to_baseline_y_pos_shifted],
                    color=color, linestyle='--',
                    zorder=zorder)

    h = (ax.plot(
            plot_settings_dict[data]['psnr0_x_pos'] + plot_settings_dict[data][
                    'psnr0_x_shift_per_run_idx'].get(i, 0),
            mean_psnr_history[0],
            '^', color=color, markersize=8,
            zorder=zorder)
            if not run_spec.get('skip_psnr0') else [None])
    psnr0_handles += h


ax.grid(True, linestyle='-')
ax.set_xlim(xlim)
ax.set_xlabel('Iteration')
ax.set_ylabel('PSNR [dB]', labelpad=plot_settings_dict[data].get('ylabel_pad'))
ax.set_ylim(plot_settings_dict[data]['ylim'])
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

    if show_tv_in_inset:
        with open(TVADAM_PSNRS_FILEPATH, 'r') as f:
            tv_psnr = yaml.safe_load(f)[data]
            axins.axhline(tv_psnr, color='#444444', linestyle=':',
                          linewidth=1., zorder=1.4)
            tv_text_pos = axins.get_xlim()[0], tv_psnr
            tv_text_pos_shift = plot_settings_dict[data].get(
                    'tv_text_shift', [30, 0.02])
            tv_text_pos_shifted = (
                    tv_text_pos[0] + tv_text_pos_shift[0],
                    tv_text_pos[1] + tv_text_pos_shift[1])
            axins.text(*tv_text_pos_shifted, 'TV', color='#444444', zorder=1.4,
                       clip_on=True)

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
psnr0_handle = copy(psnr0_handles[0])
psnr0_handle.set_markerfacecolor('gray')
psnr0_handle.set_markeredgecolor('gray')
rise_time_handle = copy(rise_time_handles[0])
rise_time_handle.set_markerfacecolor('gray')
rise_time_handle.set_markeredgecolor('gray')
symbol_legend = ax.legend(
        [psnr0_handle, rise_time_handle, baseline_handle],
        ['Initial PSNR',
         'Rise time (to $-${:g} dB)'.format(eval_settings_dict[data][
                 'rise_time_to_baseline_remaining_psnr']),
         'Steady PSNR of {}'.format(
                 get_label(runs_to_compare[baseline_run_idx], cfgs[0]))
         # 'Baseline steady PSNR'.format(
         #         get_label(runs_to_compare[baseline_run_idx], cfgs[0]))
        ],
        bbox_to_anchor=(0.5, -0.005), loc='lower center',
        ncol=3, framealpha=1.
        )

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
        ('_best_loss_output' if use_best_output_psnr else ''))
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
