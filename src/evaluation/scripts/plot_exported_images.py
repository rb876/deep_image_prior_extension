import os
import json
from math import ceil
import numpy as np
import imageio
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap, ScalarMappable
from evaluation.display_utils import experiment_title_dict

IMAGES_PATH = os.path.join(os.path.dirname(__file__), 'images')

FIG_PATH = os.path.dirname(__file__)

save_fig = True

data = 'ellipses_lotus_20'

METRICS_PATH = os.path.join(
        IMAGES_PATH, '{}_metrics.json'.format(data))
MEDIAN_PSNR_REPS_PATH = os.path.join(
        IMAGES_PATH, '{}_median_psnr_reps.json'.format(data))

SAMPLE = 0  # there is only one test sample

CMAP = 'gray'
STD_CMAP = 'viridis'

# possible values for 'type' are
# 'gt', 'fbp', 'init_reco', 'best_reco', 'init_reco_std', 'best_reco_std'

# images_to_plot = [
#     {
#         'type': 'best_reco',
#         'experiment': 'pretrain_only_fbp',
#         'repetition': 'median_psnr',
#         'show_metrics': True,
#     },
#     {
#         'type': 'best_reco',
#         'experiment': 'no_pretrain',
#         'repetition': 'median_psnr',
#         'show_metrics': True,
#     },
#     {
#         'type': 'gt',
#     },
#     {
#         'type': 'init_reco',
#         'experiment': 'pretrain_only_fbp',
#         'repetition': 'median_psnr',
#     },
#     {
#         'type': 'init_reco',
#         'experiment': 'no_pretrain',
#         'repetition': 'median_psnr',
#     },
#     {
#         'type': 'fbp',
#     }
# ]

# nrows = 2

images_to_plot = [
    {
        'type': 'best_reco_std',
        'experiment': 'pretrain_only_fbp',
        'repetition': 'median_psnr',
    },
    {
        'type': 'best_reco_std',
        'experiment': 'no_pretrain',
        'repetition': 'median_psnr',
    },
]

nrows = 1

runs_filename = 'comparison'

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
            'init_reco', 'best_reco', 'init_reco_std', 'best_reco_std')

    run_name_for_filename = (
            image_spec['experiment'] if image_spec.get('name') is None
            else '{}_{}'.format(image_spec['experiment'], image_spec['name']))

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
        elif image_type == 'init_reco_std':
            run_name_for_filename = get_run_name_for_filename(image_spec)
            filename = '{}_{}_init_std_sample_{:d}'.format(
                    data, run_name_for_filename, SAMPLE)
        elif image_type == 'best_reco_std':
            run_name_for_filename = get_run_name_for_filename(image_spec)
            filename = '{}_{}_std_sample_{:d}'.format(
                    data, run_name_for_filename, SAMPLE)
        else:
            raise ValueError(
                    'Unknown "type" \'{}\' in image spec'.format(image_type))

    if not filename.endswith('.npy'):
        filename = filename + '.npy'

    return filename

images = [
    np.load(os.path.join(IMAGES_PATH, get_filename(image_spec)))
    for image_spec in images_to_plot
]


# def save_value_range(filename, vmin, vmax):
#     with open(filename, 'w') as f:
#         json.dump({
#             'min': float(vmin),
#             'max': float(vmax),
#         }, f, indent=4)

# def save_colorbar(filename, cmap, vmin, vmax):
#     fig = plt.figure(figsize=(1, 4))
#     cax = fig.add_axes([0.05, 0.05, 0.2, 0.9])
#     fig.colorbar(ScalarMappable(Normalize(vmin, vmax), get_cmap(cmap)), cax=cax)
#     fig.savefig(filename)  # , bbox_inches='tight'

# def save_image(filename, im, fmt='png', transpose=True, cmap='gray',
#                vmin=None, vmax=None, save_vrange=True, save_cbar=True):
#     if not filename.endswith('.{}'.format(fmt)):
#         filename = filename + '.{}'.format(fmt)

#     im = im.copy()

#     vmin = np.min(im) if vmin is None else vmin
#     vmax = np.max(im) if vmax is None else vmax

#     if save_vrange:
#         filename_vrange = filename[:-len('.{}'.format(fmt))] + '_vrange.json'
#         save_value_range(filename_vrange, vmin, vmax)

#     if save_cbar:
#         filename_cbar = filename[:-len('.{}'.format(fmt))] + '_cbar.pdf'
#         save_colorbar(filename_cbar, cmap, vmin, vmax)

#     if transpose:
#         im = im.T

#     cmap = get_cmap(cmap)
#     normalize = Normalize(vmin=vmin, vmax=vmax)

#     im = normalize(im)
#     im_rgba = cmap(im)
#     imageio.imwrite(filename, im_rgba, format=fmt)


def get_title(image_spec):
    image_type = image_spec['type']

    if image_type == 'gt':
        title = 'Ground truth'
    elif image_type == 'fbp':
        title = 'FBP' if 'lotus' in data else 'FDK'
    elif image_type == 'init_reco':
        experiment_title = experiment_title_dict[image_spec['experiment']]
        title =  '{} initial'.format(experiment_title)
    elif image_type == 'best_reco':
        experiment_title = experiment_title_dict[image_spec['experiment']]
        title = experiment_title
    elif image_type == 'init_reco_std':
        experiment_title = experiment_title_dict[image_spec['experiment']]
        title = 'Std. of {} initial'.format(experiment_title)
    elif image_type == 'best_reco_std':
        experiment_title = experiment_title_dict[image_spec['experiment']]
        title = 'Std. of {}'.format(experiment_title)
    else:
        raise ValueError(
                'Unknown "type" \'{}\' in image spec'.format(image_type))

    return title

def is_std_type(image_type):
    if not isinstance(image_type, str):
        image_type = image_type['type']

    is_std = image_type.endswith('_std')

    return is_std

def get_image_metrics(image_spec):
    run_name_for_filename = get_run_name_for_filename(image_spec)
    rep = get_rep(image_spec)
    image_type = image_spec['type']
    if image_type == 'init_reco':
        type_key = 'init'
    elif image_type == 'best_reco':
        type_key = 'best'
    else:
        raise ValueError(
                'No metrics available for "type" \'{}\''.format(image_type))

    image_metrics = metrics[run_name_for_filename]['rep_{:d}'.format(rep)][
            'sample_{:d}'.format(SAMPLE)][type_key]

    return image_metrics

ncols = ceil(len(images_to_plot) / nrows)
fig, axs = plt.subplots(nrows=nrows, ncols=ncols)

for image_spec, image, ax in zip(images_to_plot, images, axs.flat):
    cmap = STD_CMAP if is_std_type(image_spec) else CMAP
    vmin, vmax = None, None

    title = get_title(image_spec)
    ax.set_title(title)
    im = ax.imshow(image.T, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax)
    if image_spec.get('show_metrics'):
        image_metrics = get_image_metrics(image_spec)
        ax.set_xlabel('PSNR: ${:.2f}\,$dB\nSSIM: ${:.4f}\,$'.format(
                image_metrics['psnr'], image_metrics['ssim']))

if save_fig:
    filename = '{}_images_{}.pdf'.format(data, runs_filename)
    fig.savefig(os.path.join(FIG_PATH, filename), bbox_inches='tight')

plt.show()
