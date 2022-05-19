import os
import json
from math import ceil
import numpy as np
import imageio
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter
from matplotlib.cm import get_cmap, ScalarMappable
from matplotlib.axes import Axes
from matplotlib.path import Path
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
from evaluation.display_utils import experiment_title_dict
from deep_image_prior import PSNR, SSIM

IMAGES_PATH = os.path.join(os.path.dirname(__file__), 'images')

FIG_PATH = os.path.dirname(__file__)

save_fig = True

# data = 'ellipses_lotus_20'
# data = 'ellipses_limited_45'
# data = 'brain_walnut_120'
# data = 'ellipses_walnut_120'
# data = 'ellipsoids_walnut_3d'
data = 'ellipsoids_walnut_3d_60'

plot_name = 'images'
# plot_name = 'uncertainty'

is_3d = '_3d' in data

METRICS_PATH = os.path.join(
        IMAGES_PATH, '{}_metrics.json'.format(data))
MEDIAN_PSNR_REPS_PATH = os.path.join(
        IMAGES_PATH, '{}_median_psnr_reps.json'.format(data))

SAMPLE = 0  # there is only one test sample

SUBPLOT_TYPES = {
    'gt': 'image',
    'fbp': 'image',
    'init_reco': 'image',
    'best_reco': 'image',
    'iterate': 'image',
    'init_reco_std': 'std_image',
    'best_reco_std': 'std_image',
    'mean_reco_error': 'error_image',
    'uncertainty': 'plot',
}

CMAP = 'gray'
STD_CMAP = 'viridis'
ERROR_CMAP = 'PiYG'

runs_filename = None

if data in ['ellipses_lotus_20', 'ellipses_lotus_limited_45',
            'brain_walnut_120', 'ellipses_walnut_120']:

    if plot_name == 'images':
        images_to_plot = [
            {
                'type': 'best_reco',
                'experiment': 'pretrain_only_fbp',
                'repetition': 'median_psnr',
                'show_metrics': True,
            },
            {
                'type': 'best_reco',
                'experiment': 'no_pretrain',
                'repetition': 'median_psnr',
                'show_metrics': True,
            },
            {
                'type': 'gt',
            },
            {
                'type': 'init_reco',
                'experiment': 'pretrain_only_fbp',
                'repetition': 'median_psnr',
                'show_metrics': True,
            },
            ({
                'type': 'iterate',
                'experiment': 'pretrain_only_fbp',
                'name_filename': 'save_many_iterates',
                'iterate_iter': 4500,
                'repetition': 0,
                'show_metrics': True,
             } if data == 'ellipses_walnut_120' else ({
                'type': 'best_reco',
                'experiment': 'pretrain_only_fbp',
                'name_filename': 'repeated_epochs1',
                'name_title': '1 epoch',
                'repetition': 'median_psnr',
                'show_metrics': True,
             } if data == 'brain_walnut_120' else {
                'type': 'init_reco',
                'experiment': 'no_pretrain',
                'repetition': 'median_psnr',
             })
            ),
            {
                'type': 'fbp',
                'show_metrics': True,
            }
        ]

    elif plot_name == 'uncertainty':
        images_to_plot = [
            {
                'type': 'uncertainty',
                'experiment': 'pretrain_only_fbp',
            },
            {
                'type': 'best_reco_std',
                'experiment': 'pretrain_only_fbp',
            },
            {
                'type': 'mean_reco_error',
                'experiment': 'pretrain_only_fbp',
            },
            {
                'type': 'uncertainty',
                'experiment': 'no_pretrain',
            },
            {
                'type': 'best_reco_std',
                'experiment': 'no_pretrain',
            },
            {
                'type': 'mean_reco_error',
                'experiment': 'no_pretrain',
            },
        ]

elif data in ['ellipsoids_walnut_3d', 'ellipsoids_walnut_3d_60']:

    if plot_name == 'images':
        images_to_plot = [
            {
                'type': 'gt',
            },
            {
                'type': 'best_reco',
                'experiment': 'no_pretrain',
                'repetition': 'median_psnr',
                'show_metrics': True,
            },
            {
                'type': 'best_reco',
                'experiment': 'pretrain_only_fbp',
                'repetition': 'median_psnr',
                'show_metrics': True,
            },
            {
                'type': 'init_reco',
                'experiment': 'pretrain_only_fbp',
                'repetition': 'median_psnr',
                'show_metrics': True,
            },
            {
                'type': 'fbp',
                'show_metrics': True,
            }
        ]


if runs_filename is None:
    runs_filename = 'comparison'


plot_settings_dict = {
    'default': {
        'images': {
            'nrows': 2,
            'norm_group_inds': [
                    1 if (image_spec['type'] == 'init_reco' and
                            image_spec['experiment'] == 'no_pretrain')
                    else 0
                    for image_spec in images_to_plot],
            'norm_groups_use_vrange_from_images': {0: [2]},
            'colorbars_mode': 'off',
            'gridspec_kw': {'hspace': 0.3, 'wspace': -0.25},
            'pad_inches': 0.,
            'add_insets': ([
                # {
                #  'on_images': [0, 1, 2, 3, 4, 5],
                #  'rect': [262, 82, 70, 70],
                #  'add_metrics': {
                #      'gt_idx': 2,
                #      'pos': (-0.05, 0.95),
                #      'kwargs': {
                #          'ha': 'right',
                #          'va': 'top',
                #      }
                #  },
                #  'axes_rect': [0.69, 0.69, 0.31, 0.31],
                #  'frame_path': [[0., 1.], [0., 0.3], [0.45, 0.], [1., 0.]],
                #  'clip_path_closing': [[1., 1.]],
                # },
                {
                 'on_images': [0, 1, 2, 3, 4, 5],
                 'rect': [279, 81, 30, 60],
                 'add_metrics': {
                     'gt_idx': 2,
                     'pos': (-0.05, 0.95),
                     'kwargs': {
                         'ha': 'right',
                         'va': 'top',
                     },
                 },
                 'axes_rect': [0.82, 0.64, 0.18, 0.36],
                 'frame_path': [[0., 1.], [0., 0.], [1., 0.]],
                 'clip_path_closing': [[1., 1.]],
                },
                {
                 'on_images': [0, 1, 2, 3, 4, 5],
                 'rect': [200, 220, 65, 55],
                 'add_metrics': {
                     'gt_idx': 2,
                     'pos': (1.05, 0.05),
                     'kwargs': {
                         'ha': 'left',
                         'va': 'bottom',
                     }
                 },
                 'axes_rect': [0., 0., 0.39, 0.33],
                 'frame_path': [[0., 1.], [0.5, 1.], [1., 0.7], [1., 0.]],
                 'clip_path_closing': [[0., 0.]],
                },
            ] if data.endswith('walnut_120') and plot_name == 'images'
            else ([
                {
                 'on_images': [0, 1, 2, 3, 4],
                 'on_3d_slice_views': [0],
                 'rect': [71, 19, 26, 32],
                 'add_metrics': {
                     'gt_idx': 0,
                     'pos': (1.05, 0.95),
                     'kwargs': {
                         'ha': 'left',
                         'va': 'top',
                         'bbox': {'pad': 0.2, 'facecolor': 'black'},
                     },
                 },
                 'axes_rect': [0., 1.-26/(46/0.35), 32/(46/0.35), 26/(46/0.35)],
                 'frame_path': [[1., 1.], [1., 0.], [0., 0.]],
                 'clip_path_closing': [[0., 1.]],
                 'frame_color_overrides': {4: '#EEEEEE'},
                },
                {
                 'on_images': [0, 1, 2, 3, 4],
                 'on_3d_slice_views': [0],
                 'rect': [84, 108, 22, 48],
                 'add_metrics': {
                     'gt_idx': 0,
                     'pos': (-0.05, 0.05),
                     'kwargs': {
                         'ha': 'right',
                         'va': 'bottom',
                         'bbox': {'pad': 0.2, 'facecolor': 'black'},
                     },
                 },
                 'axes_rect': [1.-48/(46/0.35), 0., 48/(46/0.35), 22/(46/0.35)],
                 'frame_path': [[1., 1.], [0., 1.], [0., 0.]],
                 'clip_path_closing': [[1., 0.]],
                 'frame_color_overrides': {4: '#EEEEEE'},
                },
                {
                 'on_images': [0, 1, 2, 3, 4],
                 'on_3d_slice_views': [1],
                 'rect': [62, 17, 39, 26],
                 'add_metrics': {
                     'gt_idx': 0,
                     'pos': (1.05, 0.95),
                     'kwargs': {
                         'ha': 'left',
                         'va': 'top',
                         'bbox': {'pad': 0.2, 'facecolor': 'black'},
                     },
                 },
                 'axes_rect': [0., 1.-39/(46/0.35), 26/(46/0.35), 39/(46/0.35)],
                 'frame_path': [[1., 1.], [1., 0.35], [.7, 0.], [0., 0.]],
                 'clip_path_closing': [[0., 1.]],
                 'frame_color_overrides': {4: '#EEEEEE'},
                },
                {
                 'on_images': [0, 1, 2, 3, 4],
                 'on_3d_slice_views': [1],
                 'rect': [114, 32, 23, 46],
                 'add_metrics': {
                     'gt_idx': 0,
                     'pos': (1.05, 0.05),
                     'kwargs': {
                         'ha': 'left',
                         'va': 'bottom',
                         'bbox': {'pad': 0.2, 'facecolor': 'black'},
                     },
                 },
                 'axes_rect': [0., 0., 46/(46/0.35), 23/(46/0.35)],
                 'frame_path': [[0., 1.], [1., 1.], [1., 0.]],
                 'clip_path_closing': [[0., 0.]],
                 'frame_color_overrides': {4: '#EEEEEE'},
                },
                {
                 'on_images': [0, 1, 2, 3, 4],
                 'on_3d_slice_views': [2],
                 'rect': [26, 87, 44, 24],
                 'add_metrics': {
                     'gt_idx': 0,
                     'pos': (-0.05, 0.95),
                     'kwargs': {
                         'ha': 'right',
                         'va': 'top',
                         'bbox': {'pad': 0.2, 'facecolor': 'black'},
                     },
                 },
                 'axes_rect': [1.-24/(46/0.35), 1.-44/(46/0.35), 24/(46/0.35), 44/(46/0.35)],
                 'frame_path': [[0., 1.], [0., 0.], [1., 0.]],
                 'clip_path_closing': [[1., 1.]],
                 'frame_color_overrides': {4: '#EEEEEE'},
                },
                {
                 'on_images': [0, 1, 2, 3, 4],
                 'on_3d_slice_views': [2],
                 'rect': [63, 62, 36, 50],
                 'add_metrics': {
                     'gt_idx': 0,
                     'pos': (1.05, 0.05),
                     'kwargs': {
                         'ha': 'left',
                         'va': 'bottom',
                         'bbox': {'pad': 0.2, 'facecolor': 'black'},
                     },
                 },
                 'axes_rect': [0., 0., 50/(46/0.35), 36/(46/0.35)],
                 'frame_path': [[0., 1.], [0.5, 1.], [1., 0.7], [1., 0.]],
                 'clip_path_closing': [[0., 0.]],
                 'frame_color_overrides': {4: '#EEEEEE'},
                },
            ] if data in ['ellipsoids_walnut_3d', 'ellipsoids_walnut_3d_60'] and plot_name == 'images'
            else [])),
        },
        'uncertainty': {
            'nrows': 2,
            'norm_group_inds': [None, 0, 1, None, 0, 1],
            'colorbars_mode': 'norm_groups',
            'skip_xlabels': [0],
            'colorbar_location': 'bottom',
            'use_inset_positioned_colorbar': True,
            'scilimits': (-1, 1),
            'titlepad': 16,
            'figsize': (9, 7),
            'gridspec_kw': {'hspace': 0.45, 'wspace': 0.1},
            'pad_inches': 0.,
        },
    },
    # 'ellipses_lotus_20': {
    # },
    # 'ellipses_lotus_limited_45': {
    # },
    # 'brain_walnut_120': {
    # },
    # 'ellipses_walnut_120': {
    # },
    'ellipsoids_walnut_3d': {
        'images': {
            'nrows': 3,
            'norm_group_inds': [
                    1 if (image_spec['type'] == 'init_reco' and
                            image_spec['experiment'] == 'no_pretrain')
                    else 0
                    for image_spec in images_to_plot],
            'norm_groups_use_vrange_from_images': {0: [2]},
            'figsize': (13.5, 8.),
            'gridspec_kw': {'hspace': 0.025, 'wspace': 0.025},
        },
    },
    'ellipsoids_walnut_3d_60': {
        'images': {
            'nrows': 3,
            'norm_group_inds': [
                    1 if (image_spec['type'] == 'init_reco' and
                            image_spec['experiment'] == 'no_pretrain')
                    else 0
                    for image_spec in images_to_plot],
            'norm_groups_use_vrange_from_images': {0: [2]},
            'figsize': (13.5, 8.),
            'gridspec_kw': {'hspace': 0.025, 'wspace': 0.025},
        },
    },
}

try:
    with open(METRICS_PATH, 'r') as f:
        metrics = json.load(f)
except FileNotFoundError:
    metrics = None

try:
    with open(MEDIAN_PSNR_REPS_PATH, 'r') as f:
        median_psnr_reps = json.load(f)
except FileNotFoundError:
    median_psnr_reps = None

def get_run_name_for_filename(image_spec):
    assert image_spec.get('type') in (
            'init_reco', 'best_reco', 'iterate',
            'init_reco_std', 'best_reco_std', 'mean_reco_error')

    name_filename = image_spec.get('name_filename', image_spec.get('name'))

    run_name_for_filename = (
            image_spec['experiment'] if not name_filename
            else '{}_{}'.format(image_spec['experiment'], name_filename))

    return run_name_for_filename

def get_rep(image_spec):
    rep = image_spec.get('repetition')

    if rep is None or rep == 'median_psnr':
        num_name_for_filename = get_run_name_for_filename(image_spec)
        rep = median_psnr_reps[num_name_for_filename][
                'sample_{:d}'.format(SAMPLE)]

    return rep

def get_filename(image_spec):
    filename = image_spec.get('filename')

    if filename is None:
        image_type = image_spec['type']
        if image_type == 'gt':
            filename = '{}_gt_sample_{:d}'.format(data, SAMPLE)
        elif image_type == 'fbp':
            filename = '{}_fbp_sample_{:d}'.format(data, SAMPLE)
        elif image_type == 'init_reco':
            run_name_for_filename = get_run_name_for_filename(image_spec)
            rep = get_rep(image_spec)
            filename = '{}_{}_init_rep_{:d}_sample_{:d}'.format(
                    data, run_name_for_filename, rep, SAMPLE)
        elif image_type == 'best_reco':
            run_name_for_filename = get_run_name_for_filename(image_spec)
            rep = get_rep(image_spec)
            filename = '{}_{}_rep_{:d}_sample_{:d}'.format(
                    data, run_name_for_filename, rep, SAMPLE)
        elif image_type == 'iterate':
            run_name_for_filename = get_run_name_for_filename(image_spec)
            rep = get_rep(image_spec)
            iterate_iter = image_spec['iterate_iter']
            filename = '{}_{}_rep_{:d}_sample_{:d}_iter_{:d}'.format(
                    data, run_name_for_filename, rep, SAMPLE, iterate_iter)
        elif image_type == 'init_reco_std':
            run_name_for_filename = get_run_name_for_filename(image_spec)
            filename = '{}_{}_init_std_sample_{:d}'.format(
                    data, run_name_for_filename, SAMPLE)
        elif image_type == 'best_reco_std':
            run_name_for_filename = get_run_name_for_filename(image_spec)
            filename = '{}_{}_std_sample_{:d}'.format(
                    data, run_name_for_filename, SAMPLE)
        elif image_type == 'mean_reco_error':
            run_name_for_filename = get_run_name_for_filename(image_spec)
            filename = '{}_{}_mean_error_sample_{:d}'.format(
                    data, run_name_for_filename, SAMPLE)
        else:
            raise ValueError(
                    'Unknown "type" \'{}\' in image spec'.format(image_type))

    if not filename.endswith('.npy'):
        filename = filename + '.npy'

    return filename

images = [
    (np.load(os.path.join(IMAGES_PATH, get_filename(image_spec)))
     if SUBPLOT_TYPES[image_spec['type']] in (
            'image', 'std_image', 'error_image')
     else None)
    for image_spec in images_to_plot
]

plot_settings = plot_settings_dict['default'][plot_name]
plot_settings.update(plot_settings_dict.get(data, {}).get(plot_name, {}))

nrows = plot_settings.get('nrows', 3 if is_3d else 1)

norm_group_inds = plot_settings.get('norm_group_inds', 'global')
if norm_group_inds == 'global':
    norm_group_inds = [0 if im is not None else None
                       for im in images]
elif norm_group_inds == 'individual':
    norm_group_inds = [i if im is not None else None
                       for i, im in enumerate(images)]
else:
    norm_group_inds = [int(ind) if ind is not None else None
                       for ind in norm_group_inds]

unique_norm_group_inds = np.unique(
        [ind for ind in norm_group_inds if ind is not None])

norm_groups_use_vrange_from_images = plot_settings.get(
        'norm_groups_use_vrange_from_images', {})
for group_ind in unique_norm_group_inds:
    norm_groups_use_vrange_from_images.setdefault(
            group_ind,
            [i for i, g_ind in enumerate(norm_group_inds)
             if g_ind == group_ind])

colorbars_mode = plot_settings.get('colorbars_mode', 'individual')
colorbar_location = plot_settings.get('colorbar_location', 'right')
use_inset_positioned_colorbar = plot_settings.get(
        'use_inset_positioned_colorbar', False)
scilimits = plot_settings.get('scilimits')
metrics_fontsize = plot_settings.get('metrics_fontsize', 9)
titlepad = plot_settings.get('titlepad')
figsize = plot_settings.get('figsize', (9, 6))
gridspec_kw = plot_settings.get('gridspec_kw', {})
pad_inches = plot_settings.get('pad_inches', 0.1)
add_insets = plot_settings.get('add_insets')


def get_experiment_title(image_spec):
    experiment_title = experiment_title_dict[image_spec['experiment']]
    if image_spec.get('name_title'):
        experiment_title = '{} [{}]'.format(
                experiment_title, image_spec['name_title'])
    return experiment_title

def get_title(image_spec):
    image_type = image_spec['type']

    if image_type == 'gt':
        title = 'Reference' if 'lotus' in data else 'Ground truth'
    elif image_type == 'fbp':
        title = 'FBP'  # if 'lotus' in data else 'FDK'
    elif image_type == 'init_reco':
        experiment_title = get_experiment_title(image_spec)
        title =  '{} initial'.format(experiment_title)
    elif image_type == 'best_reco':
        experiment_title = get_experiment_title(image_spec)
        title = experiment_title
    elif image_type == 'iterate':
        experiment_title = get_experiment_title(image_spec)
        title = '{} iter. {:d}'.format(
                experiment_title, image_spec['iterate_iter'])
    elif image_type == 'init_reco_std':
        experiment_title = get_experiment_title(image_spec)
        title = 'Std. of {} initial'.format(experiment_title)
    elif image_type == 'best_reco_std':
        experiment_title = get_experiment_title(image_spec)
        title = 'Std. of {}'.format(experiment_title)
    elif image_type == 'mean_reco_error':
        experiment_title = get_experiment_title(image_spec)
        title = 'Mean error of {}'.format(experiment_title)
    elif image_type == 'uncertainty':
        experiment_title = get_experiment_title(image_spec)
        title = 'Calibration of {}'.format(experiment_title)
    else:
        raise ValueError(
                'Unknown "type" \'{}\' in image spec'.format(image_type))

    return title

def get_image_metrics(image_spec):
    image_type = image_spec['type']
    if image_type == 'fbp':
        image_metrics = metrics['fbp']['sample_{:d}'.format(SAMPLE)]
    else:
        run_name_for_filename = get_run_name_for_filename(image_spec)
        rep = get_rep(image_spec)
        if image_type == 'init_reco':
            type_key = 'init'
        elif image_type == 'best_reco':
            type_key = 'best'
        elif image_type == 'iterate':
            type_key = 'iterates'
        else:
            raise ValueError(
                    'No metrics available for "type" \'{}\''.format(image_type))
        image_metrics = metrics[run_name_for_filename]['rep_{:d}'.format(rep)][
                'sample_{:d}'.format(SAMPLE)][type_key]
        if image_type == 'iterate':
            image_metrics = image_metrics[str(image_spec['iterate_iter'])]

    return image_metrics

def get_uncertainty_infos(image_spec):
    err_image_spec = {
        'type': 'mean_reco_error',
        'experiment': image_spec['experiment'],
        'name': image_spec.get('name'),
    }
    std_image_spec = {
        'type': 'best_reco_std',
        'experiment': image_spec['experiment'],
        'name': image_spec.get('name'),
    }
    err_image_filename = get_filename(err_image_spec)
    std_image_filename = get_filename(std_image_spec)
    err_image = np.load(os.path.join(IMAGES_PATH, err_image_filename))
    std_image = np.load(os.path.join(IMAGES_PATH, std_image_filename))

    squared_err = (err_image ** 2).ravel()
    squared_std = (std_image ** 2).ravel()

    return squared_err, squared_std

vranges_per_norm_group = [
    (min(np.min(images[i])
         for i in norm_groups_use_vrange_from_images[group_ind]),
     max(np.max(images[i])
         for i in norm_groups_use_vrange_from_images[group_ind]))
    for group_ind in unique_norm_group_inds
]

vranges = [vranges_per_norm_group[g_ind] if g_ind is not None else None
           for g_ind in norm_group_inds]

ncols = ceil(len(images_to_plot) * (3 if is_3d else 1) / nrows)
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize,
                        gridspec_kw=gridspec_kw)

axs_list = list(axs.T.flat)
if is_3d:
    axs_list = [axs_list[i*3:(i+1)*3] for i in range(len(images_to_plot))]
    label_fontsize = plt.rcParams['axes.titlesize']
    axs_list[0][0].set_ylabel('yz-slice', fontsize=label_fontsize)
    axs_list[0][1].set_ylabel('xz-slice', fontsize=label_fontsize)
    axs_list[0][2].set_ylabel('xy-slice', fontsize=label_fontsize)

image_axes_list = []

for i, (image_spec, image, vrange, ax) in enumerate(zip(
        images_to_plot, images, vranges, axs_list)):

    image_type = image_spec['type']

    image_axes = None

    ax_or_first_ax = (ax[0] if is_3d else ax)
    ax_or_last_ax = (ax[-1] if is_3d else ax)

    title = get_title(image_spec)
    ax_or_first_ax.set_title(title, pad=titlepad)

    if SUBPLOT_TYPES[image_type] in ('image', 'std_image', 'error_image'):
        # plot image
        cmap = CMAP
        if SUBPLOT_TYPES[image_type] == 'std_image':
            cmap = STD_CMAP
        elif SUBPLOT_TYPES[image_type] == 'error_image':
            cmap = ERROR_CMAP

        vmin, vmax = vrange

        imshow_kwargs = {
                'cmap': cmap, 'vmin': vmin, 'vmax': vmax,
                'interpolation': 'none'}

        if image.ndim == 2:
            image_axes = ax.imshow(image.T, **imshow_kwargs)
        elif image.ndim == 3:
            image_3d_slice_view0 = image.T[image.shape[0] // 2, :, :]
            image_3d_slice_view1 = image.T[:, image.shape[1] // 2, :]
            image_3d_slice_view2 = image.T[:, :, image.shape[1] // 2]
            image_axes0 = ax[0].imshow(image_3d_slice_view0, **imshow_kwargs)
            image_axes1 = ax[1].imshow(image_3d_slice_view1, **imshow_kwargs)
            image_axes2 = ax[2].imshow(image_3d_slice_view2, **imshow_kwargs)
            image_axes = image_axes2  # only use one of the subplots (vrange is the same for all three)
            image_3d_slice_views = (image_3d_slice_view0, image_3d_slice_view1, image_3d_slice_view2)

        if image_spec.get('show_metrics'):
            image_metrics = get_image_metrics(image_spec)
            ax_or_last_ax.set_xlabel('PSNR: ${:.2f}\,$dB, SSIM: ${:.4f}$'.format(
                    image_metrics['psnr'], image_metrics['ssim']),
                    fontsize=metrics_fontsize)

        for ax_ in ax if is_3d else [ax]:
            ax_.set_xticks([])
            ax_.set_yticks([])
            for spine in ax_.spines.values():
                spine.set_visible(False)

        if add_insets:
            for add_inset in add_insets:
                if i in add_inset['on_images']:
                    for j in (add_inset['on_3d_slice_views'] if is_3d else (None,)):
                        add_metrics = add_inset.get('add_metrics')
                        if add_metrics:
                            gt_idx = add_metrics['gt_idx']
                            assert images_to_plot[gt_idx]['type'] == 'gt'
                        if is_3d:
                            ax_ = ax[j]
                            image_2d = image_3d_slice_views[j]
                            if add_metrics:
                                gt = images[gt_idx]
                                gt_2d = [
                                    gt.T[image.shape[0] // 2, :, :],
                                    gt.T[:, image.shape[0] // 2, :],
                                    gt.T[:, :, image.shape[0] // 2]][j]
                        else:
                            ax_ = ax
                            image_2d = image
                            if add_metrics:
                                gt_2d = images[gt_idx]
                        ip = InsetPosition(ax_, add_inset['axes_rect'])
                        axins = Axes(fig, [0., 0., 1., 1.])
                        axins.set_axes_locator(ip)
                        fig.add_axes(axins)
                        rect = add_inset['rect']
                        slice0 = slice(rect[0], rect[0]+rect[2])
                        slice1 = slice(rect[1], rect[1]+rect[3])
                        inset_image = image_2d[slice0, slice1]
                        inset_image_handle = axins.imshow(
                                inset_image, cmap=cmap, vmin=vmin, vmax=vmax,
                                interpolation='none')
                        add_metrics = add_inset.get('add_metrics')
                        if add_metrics:
                            if i != gt_idx:
                                inset_gt = gt_2d[slice0, slice1]
                                inset_psnr = PSNR(inset_image, inset_gt,
                                        data_range=np.max(gt_2d)-np.min(gt_2d))
                                inset_ssim = SSIM(inset_image, inset_gt,
                                        data_range=np.max(gt_2d)-np.min(gt_2d))
                                axins.text(
                                        *add_metrics.get('pos', [0., 0.]),
                                        'PSNR: {:.2f}$\,$dB\nSSIM: {:.4f}'.format(
                                                inset_psnr, inset_ssim),
                                        transform=axins.transAxes,
                                        fontsize=add_metrics.get('fontsize', 6),
                                        color=add_metrics.get('color', '#cccccc'),
                                        **add_metrics.get('kwargs', {}),
                                        )
                        axins.set_xticks([])
                        axins.set_yticks([])
                        axins.patch.set_visible(False)
                        for spine in axins.spines.values():
                            spine.set_visible(False)
                        frame_path = add_inset.get(
                                'frame_path', [[0., 0.], [1., 0.], [0., 1.], [1., 1]])
                        if frame_path:
                            axins.plot(
                                    *np.array(frame_path).T,
                                    transform=axins.transAxes,
                                    color=add_inset.get('frame_color_overrides', {}).get(i) or add_inset.get('frame_color', '#555555'),
                                    solid_capstyle='butt')
                            inset_image_handle.set_clip_path(Path(
                                    frame_path + add_inset.get(
                                            'clip_path_closing', [])),
                                    transform=axins.transAxes)
                            inset_image_handle.set_clip_on(True)

    elif SUBPLOT_TYPES[image_type] == 'plot':
        assert not is_3d
        if image_type == 'uncertainty':
            squared_err, squared_std = get_uncertainty_infos(image_spec)
            min_val = min([np.min(squared_std), np.min(squared_err)])
            max_val = max([np.max(squared_std), np.max(squared_err)])
            ax.plot([min_val, max_val], [min_val, max_val],
                    color='gray', linestyle='dashed')
            ax.plot(squared_std, squared_err, '.')
            ax.set_aspect('equal')
            if scilimits is not None:
                ax.ticklabel_format(scilimits=scilimits)
            # formatter = FormatStrFormatter('%.1e')
            # ax.xaxis.set_major_formatter(formatter)
            # ax.yaxis.set_major_formatter(formatter)
            if not i in plot_settings.get('skip_xlabels', []):
                ax.set_xlabel('uncertainty')
            if not i in plot_settings.get('skip_ylabels', []):
                ax.set_ylabel('error')
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError

    image_axes_list.append(image_axes)

def add_inset_positioned_colorbar(image_axes, ax, location='right'):
    if not isinstance(ax, Axes):
        ax = np.asarray(ax).ravel()
        ax = {'right': ax[-1],
              'bottom': ax[-1],
              'left': ax[0],
              'top': ax[0]}[location]
    ip = InsetPosition(
            ax,
            {'right': [1.05, 0., 0.05, 1.],
             'bottom': [0., -0.1, 1., 0.05],
             'left': [-0.1, 0., 0.05, 1.],
             'top': [0., 1.05, 1., 0.05]}[location])
    cax = Axes(fig, [0., 0., 1., 1.])
    cax.set_axes_locator(ip)
    fig.add_axes(cax)
    orientation = {'right': 'vertical',
                   'bottom': 'horizontal',
                   'left': 'vertical',
                   'top': 'horizontal'}[location]
    cb = fig.colorbar(image_axes, orientation=orientation, cax=cax)
    return cb

if colorbars_mode == 'global':
    assert len(unique_norm_group_inds) == 1  # only global normalization allowed
    image_axes = next(image_axes for image_axes in image_axes_list
                      if image_axes is not None)
    colorbar_fun = (add_inset_positioned_colorbar
                    if use_inset_positioned_colorbar else
                    fig.colorbar)
    cb = colorbar_fun(image_axes, ax=axs, location=colorbar_location)
    cb.ax.ticklabel_format(scilimits=scilimits)
elif colorbars_mode == 'individual':
    for image_axes, ax in zip(image_axes_list, axs.flat):
        if image_axes is not None:
            colorbar_fun = (add_inset_positioned_colorbar
                            if use_inset_positioned_colorbar else
                            fig.colorbar)
            cb = colorbar_fun(image_axes, ax=[ax], location=colorbar_location)
            cb.ax.ticklabel_format(scilimits=scilimits)
elif colorbars_mode == 'norm_groups':
    for group_ind in unique_norm_group_inds:
        image_axes = next(
                image_axes for image_axes, g_ind in zip(
                        image_axes_list, norm_group_inds)
                if g_ind == group_ind)
        axs_group = [
                ax for ax, g_ind in zip(axs.flat, norm_group_inds)
                if g_ind == group_ind]
        colorbar_fun = (add_inset_positioned_colorbar
                        if use_inset_positioned_colorbar else
                        fig.colorbar)
        cb = colorbar_fun(
                image_axes, ax=axs_group, location=colorbar_location)
        if scilimits is not None:
            cb.ax.ticklabel_format(scilimits=scilimits)
elif colorbars_mode == 'off':
    pass
else:
    raise NotImplementedError

if save_fig:
    filename = '{}_{}_{}.pdf'.format(data, plot_name, runs_filename)
    fig.savefig(os.path.join(FIG_PATH, filename),
                bbox_inches='tight', pad_inches=pad_inches)

plt.show()
