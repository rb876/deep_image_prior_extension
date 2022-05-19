import os
import json
from math import ceil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition

IMAGES_PATH = os.path.join(os.path.dirname(__file__), 'images')

FIG_PATH = os.path.dirname(__file__)

save_fig = True

# data = 'ellipses_lotus_20'
# data = 'ellipses_limited_45'
# data = 'brain_walnut_120'
# data = 'ellipses_walnut_120'
# data = 'ellipsoids_walnut_3d'
data = 'ellipsoids_walnut_3d_60'

is_3d = '_3d' in data

fold = 'train'

METRICS_PATH = os.path.join(
        IMAGES_PATH, '{}_pretraining_{}_metrics.json'.format(data, fold))

samples = range(1)

CMAP = 'gray'

# possible values for 'type' are
# 'gt', 'fbp', 'reco'

if data in ['ellipses_lotus_20', 'ellipses_lotus_limited_45',
            'brain_walnut_120', 'ellipses_walnut_120',
            'ellipsoids_walnut_3d', 'ellipsoids_walnut_3d_60']:

    images_to_plot = []
    for k in samples:
        images_to_plot += [
            {
                'type': 'fbp',
                'show_metrics': True,
                'sample': k,
                # 'ylabel': 'Sample {:d}'.format(k),
            },
            {
                'type': 'reco',
                'experiment': 'pretrain_only_fbp',
                'show_metrics': True,
                'sample': k,
            },
            {
                'type': 'gt',
                'sample': k,
            },
        ]


plot_settings_dict = {
    'default': {
        'nrows': len(samples) * (3 if is_3d else 1),
        'only_top_row_titles': True,
        'norm_group_inds': 'global',
        'norm_groups_use_vrange_from_images': {0: [len(images_to_plot)-1]},
        'colorbars_mode': 'off',
        'gridspec_kw': {'wspace': 0.025},
        'pad_inches': 0.,
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
        'figsize': (9, 9),
        'gridspec_kw': {'hspace': 0.02, 'wspace': 0.02},
    },
    'ellipsoids_walnut_3d_60': {
        'figsize': (9, 9),
        'gridspec_kw': {'hspace': 0.02, 'wspace': 0.02},
    }
}


try:
    with open(METRICS_PATH, 'r') as f:
        metrics = json.load(f)
except FileNotFoundError:
    metrics = None

def get_filename(image_spec):
    filename = image_spec.get('filename')

    if filename is None:
        sample = image_spec['sample']
        image_type = image_spec['type']
        if image_type == 'gt':
            filename = '{}_pretraining_{}_gt_sample_{:d}'.format(
                    data, fold, sample)
        elif image_type == 'fbp':
            filename = '{}_pretraining_{}_fbp_sample_{:d}'.format(
                    data, fold, sample)
        elif image_type == 'reco':
            filename = '{}_pretraining_{}_reco_sample_{:d}'.format(
                    data, fold, sample)
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

plot_settings = plot_settings_dict['default']
plot_settings.update(plot_settings_dict.get(data, {}))

nrows = plot_settings.get('nrows', len(samples) * (3 if is_3d else 1))
only_top_row_titles = plot_settings.get('only_top_row_titles', False)

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
figsize = plot_settings.get('figsize', (9, 3))
gridspec_kw = plot_settings.get('gridspec_kw', {})
pad_inches = plot_settings.get('pad_inches', 0.1)


def get_title(image_spec):
    image_type = image_spec['type']

    if image_type == 'gt':
        title = 'Ground truth'
    elif image_type == 'fbp':
        title = 'FBP'  # if 'lotus' in data else 'FDK'
    elif image_type == 'reco':
        title = 'U-Net'
    else:
        raise ValueError(
                'Unknown "type" \'{}\' in image spec'.format(image_type))

    return title

def get_image_metrics(image_spec):
    sample = image_spec['sample']
    image_type = image_spec['type']
    if image_type == 'fbp':
        image_metrics = metrics['fbp']['sample_{:d}'.format(sample)]
    elif image_type == 'reco':
        image_metrics = metrics['reco']['sample_{:d}'.format(sample)]
    else:
        raise NotImplementedError

    return image_metrics

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
    if not (only_top_row_titles and i // ncols > 0):
        ax_or_first_ax.set_title(title, pad=titlepad)

    vmin, vmax = vrange

    imshow_kwargs = {
            'cmap': CMAP, 'vmin': vmin, 'vmax': vmax,
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

    if image_spec.get('ylabel') and not is_3d:
        ax.set_ylabel(image_spec['ylabel'])

    if image_spec.get('show_metrics'):
        image_metrics = get_image_metrics(image_spec)
        ax_or_last_ax.set_xlabel('PSNR: ${:.2f}\,$dB, SSIM: ${:.4f}\,$'.format(
                image_metrics['psnr'], image_metrics['ssim']),
                fontsize=metrics_fontsize)

    for ax_ in ax if is_3d else [ax]:
        ax_.set_xticks([])
        ax_.set_yticks([])
        for spine in ax_.spines.values():
            spine.set_visible(False)

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
    filename = '{}_pretraining_{}_images.pdf'.format(data, fold)
    fig.savefig(os.path.join(FIG_PATH, filename),
                bbox_inches='tight', pad_inches=pad_inches)

plt.show()
