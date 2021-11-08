import os
import yaml
from warnings import warn
import numpy as np
from tqdm import tqdm
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
from matplotlib.ticker import FormatStrFormatter

try:
    import tensorflow as tf
    from tensorflow.core.util import event_pb2
    from tensorflow.python.lib.io import tf_record
    from tensorflow.errors import DataLossError
    TF_AVAILABLE = True
except ModuleNotFoundError:
    TF_AVAILABLE = False

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# PATH = '/media/chen/Res/deep_image_prior_extension/'
# PATH = '/localdata/jleuschn/experiments/deep_image_prior_extension/'
PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

NPZ_CACHE_PATH = os.path.join(os.path.dirname(__file__), 'trn_logs_npz_cache')

FIG_PATH = os.path.dirname(__file__)

save_fig = True
formats = ('pdf', 'png')

with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'runs_publish.yaml'),
        'r') as f:
    runs = yaml.load(f, Loader=yaml.FullLoader)

data = 'ellipses_lotus_20'
# data = 'ellipses_lotus_limited_45'
# data = 'brain_walnut_120'
# data = 'ellipses_walnut_120'


if data in ('ellipses_lotus_20', 'ellipses_lotus_limited_45',
            'brain_walnut_120', 'ellipses_walnut_120'):
    runs_to_compare = [
        {
        'experiment': 'pretrain',
        'name': 'no_stats_no_sigmoid_train_run0',
        'name_title': 'Run 0',
        'color': '#404099',
        },
        {
        'experiment': 'pretrain',
        'name': 'no_stats_no_sigmoid_train_run1',
        'name_title': 'Run 1',
        'color': '#994040',
        },
        {
        'experiment': 'pretrain',
        'name': 'no_stats_no_sigmoid_train_run2',
        'name_title': 'Run 2',
        'color': '#409940',
        },
    ]
elif data == 'ellipses_lotus_limited_45':
    runs_to_compare = [
        {
        'experiment': 'pretrain',
        'name': 'no_stats_no_sigmoid_train_run0_1',
        'name_title': 'Run 0',
        'color': '#404099',
        'sub_runs': [0],
        },
        {
        'experiment': 'pretrain',
        'name': 'no_stats_no_sigmoid_train_run0_1',
        'name_title': 'Run 1',
        'color': '#994040',
        'sub_runs': [1],
        },
        {
        'experiment': 'pretrain',
        'name': 'no_stats_no_sigmoid_train_run2',
        'name_title': 'Run 2',
        'color': '#409940',
        },
    ]

title = None
runs_title = 'Pretraining convergence'

plot_settings_dict = {
    'ellipses_lotus_20': {
        'ylim': (16., 25.),
        'zorders_per_run_idx': {0: {0: 2.3}, 1: {0: 2.2}, 2: {0: 2.1}},
        'lr_legend_loc': 'upper right',
    },
    'ellipses_lotus_limited_45': {
        'ylim': (16., None),
        'zorders_per_run_idx': {0: {0: 2.3}, 1: {0: 2.2}, 2: {0: 2.1}},
        'lr_legend_loc': 'upper right',
    },
    'brain_walnut_120': {
        'ylim': (20., 36.),
        'zorders_per_run_idx': {0: {0: 2.3}, 1: {0: 2.2}, 2: {0: 2.1}},
        'lr_legend_loc': 'upper right',
    },
    'ellipses_walnut_120': {
        'ylim': (25., 36.),
        'zorders_per_run_idx': {0: {0: 2.3}, 1: {0: 2.2}, 2: {0: 2.1}},
        'lr_legend_loc': 'upper right',
    },
}

data_title = data_title_dict[data]

num_steps_per_epoch = None

cfgs_list = []
experiment_names_list = []
trn_log_filepaths_list = []

for run_spec in runs_to_compare:
    experiment = run_spec['experiment']
    available_runs = runs['pretraining'][data][experiment]
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
    trn_log_paths = [
            os.path.join(run_path_multirun, '{:d}'.format(sub_runs[i] if sub_runs is not None else i),
                         cfg['trn']['log_path'])
            for i, cfg in enumerate(cfgs)]
    trn_log_filepaths = []
    for cfg, trn_log_path in zip(cfgs, trn_log_paths):
        dir_candidates = [
                f for f in os.listdir(trn_log_path)
                if f.endswith('trainer.train')]
        assert len(dir_candidates) == 1
        dirpath = os.path.join(trn_log_path, dir_candidates[0])
        file_candidates = [
                f for f in os.listdir(dirpath)
                if f.startswith('events.out')]
        assert len(file_candidates) == 1
        trn_log_filepaths.append(os.path.join(dirpath, file_candidates[0]))
    if len(cfgs) == 0:
        warn('No runs found at path "{}", skipping.'.format(run_path_multirun))
        continue
    assert all((cfg['data']['name'] == data for cfg in cfgs))
    if experiment in ('pretrain', 'pretrain_only_fbp'):
        assert all((en in ('pretrain', 'pretrain_only_fbp')
                   for en in experiment_names))
    else:
        assert all((en == experiment for en in experiment_names))
    if num_steps_per_epoch is None:
        num_steps_per_epoch = ceil(
                cfgs[0]['trn']['train_len'] / cfgs[0]['trn']['batch_size'])
    assert all(
            (ceil(cfg['trn']['train_len'] / cfg['trn']['batch_size']) ==
             num_steps_per_epoch for cfg in cfgs))

    num_runs = len(cfgs)
    print('Found {:d} runs at path "{}".'.format(num_runs, run_path_multirun))

    cfgs_list.append(cfgs)
    experiment_names_list.append(experiment_names)
    trn_log_filepaths_list.append(trn_log_filepaths)


def extract_tensorboard_scalars(log_file=None, save_as_npz=None, tags=None):
    if not TF_AVAILABLE:
        raise RuntimeError('Tensorflow could not be imported, which is '
                           'required by `extract_tensorboard_scalars`')

    def my_summary_iterator(path):
        try:
            for r in tf_record.tf_record_iterator(path):
                yield event_pb2.Event.FromString(r)
        except DataLossError:
            warn('DataLossError occured, terminated reading file')

    if tags is not None:
        tags = [t.replace('/', '_').lower() for t in tags]
    values = {}
    try:
        for event in tqdm(my_summary_iterator(log_file)):
            if event.WhichOneof('what') != 'summary':
                continue
            step = event.step
            for value in event.summary.value:
                use_value = True
                if hasattr(value, 'simple_value'):
                    v = value.simple_value
                elif value.tensor.ByteSize():
                    v = tf.make_ndarray(value.tensor)
                else:
                    use_value = False
                if use_value:
                    tag = value.tag.replace('/', '_').lower()
                    if tags is None or tag in tags:
                        values.setdefault(tag, []).append((step, v))
    except DataLossError as e:
        warn('stopping for log_file "{}" due to DataLossError: {}'.format(
            log_file, e))
    scalars = {}
    for k in values.keys():
        v = np.asarray(values[k])
        steps, steps_counts = np.unique(v[:, 0], return_counts=True)
        scalars[k + '_steps'] = steps
        scalars[k + '_scalars'] = v[np.cumsum(steps_counts)-1, 1]  # last of
        #                                                            each step

    if save_as_npz is not None:
        np.savez(save_as_npz, **scalars)

    return scalars


scalars_per_run_spec = []

for trn_log_filepaths in trn_log_filepaths_list:

    scalars_multirun = []

    for trn_log_filepath in trn_log_filepaths:
        npz_dirpath = os.path.join(
                NPZ_CACHE_PATH,
                os.path.dirname(os.path.relpath(trn_log_filepath, PATH)))
        os.makedirs(npz_dirpath, exist_ok=True)
        npz_filepath = os.path.join(
                npz_dirpath,
                '{}.npz'.format(os.path.basename(trn_log_filepath)))
        if os.path.isfile(npz_filepath):
            print('Loading from {}'.format(npz_filepath))
            scalars = np.load(npz_filepath)
        else:
            print('Extracting from {}'.format(trn_log_filepath))
            scalars = extract_tensorboard_scalars(
                    trn_log_filepath, save_as_npz=npz_filepath)

        scalars_multirun.append(scalars)

    scalars_per_run_spec.append(scalars_multirun)


min_common_steps = np.inf
min_common_steps_val = np.inf
for scalars_multirun in scalars_per_run_spec:
    for scalars in scalars_multirun:
        min_common_steps = min(min_common_steps, len(scalars['lr_steps']))
        min_common_steps = min(min_common_steps, len(scalars['lr_scalars']))
        min_common_steps = min(min_common_steps, len(scalars['psnr_steps']))
        min_common_steps = min(min_common_steps, len(scalars['psnr_scalars']))
        min_common_steps_val = min(min_common_steps_val, len(scalars['val_psnr_steps']))
        min_common_steps_val = min(min_common_steps_val, len(scalars['val_psnr_scalars']))


lr_steps = None
lr_scalars = None
for scalars_multirun in scalars_per_run_spec:
    for scalars in scalars_multirun:
        if lr_steps is None:
            lr_steps = scalars['lr_steps'][:min_common_steps]
            lr_scalars = scalars['lr_scalars'][:min_common_steps]
        else:
            assert np.array_equal(
                    scalars['lr_steps'][:min_common_steps], lr_steps)
            assert np.array_equal(
                    scalars['lr_scalars'][:min_common_steps], lr_scalars)


fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=plot_settings_dict[data].get('figsize', (8, 4)),
        gridspec_kw={'height_ratios': (3., 1.), 'hspace': 0.4})

val_handles = []
handles = []

for i, (run_spec, cfgs, experiment_names, scalars_multirun) in enumerate(zip(
        runs_to_compare, cfgs_list, experiment_names_list,
        scalars_per_run_spec)):
    for j, scalars in enumerate(scalars_multirun):
        label = run_spec.get('name_title')
        color = run_spec.get('color')
        zorder = plot_settings_dict[data].get('zorders_per_run_idx', {}).get(
                i, {}).get(j)
        h = ax0.plot(
                scalars['val_psnr_steps'] / num_steps_per_epoch,
                scalars['val_psnr_scalars'],
                label='{}, val.'.format(label),
                color=color, linestyle='--', zorder=zorder)
        val_handles += h
        h = ax0.plot(
                scalars['psnr_steps'] / num_steps_per_epoch,
                scalars['psnr_scalars'],
                label='{}, train.'.format(label),
                color=color, linestyle='-', zorder=zorder)
        handles += h


ax0.grid(True, linestyle='-')
ax0.set_xlim(plot_settings_dict[data].get('xlim', (None, None)))
ax0.set_xlabel('Epoch')
ax0.set_ylabel('PSNR [dB]', labelpad=plot_settings_dict[data].get('ylabel_pad'))
ax0.set_ylim(plot_settings_dict[data].get('ylim', (None, None)))
ax0.spines['right'].set_visible(False)
ax0.spines['top'].set_visible(False)
# ax0.set_xscale('log')

all_handles = []
for h, val_h in zip(handles, val_handles):
    all_handles += [h, val_h]

ax0.legend(
        handles=all_handles,
        loc='lower right',
        ncol=len(handles), framealpha=1.)

lr_handle = ax1.plot(
        lr_steps / num_steps_per_epoch, lr_scalars,
        color='k', label='Learning rate', linestyle='-')[0]

ax1.sharex(ax0)
ax1.grid(True, linestyle='-')
ax1.spines['right'].set_visible(False)
# ax1.spines['top'].set_visible(False)
# ax1.spines['left'].set_visible(False)
ax1.spines['bottom'].set_color(plt.rcParams['grid.color'])
ax1.set_ylim((0., None))
ax1.set_yticks([lr_scalars[0], np.max(lr_scalars)])
ax1.yaxis.set_major_formatter(FormatStrFormatter(
        plot_settings_dict[data].get('lr_yaxis_tick_fmt', '%.2e')))
ax1.xaxis.set_ticks_position('top')
ax1.xaxis.set_tick_params(labeltop=False)

ax1.legend(
        handles=[lr_handle],
        loc=plot_settings_dict[data].get('lr_legend_loc', 'upper right'),
        framealpha=1.)

if title is None:
    title = ('{} on {}'.format(runs_title, data_title) if runs_title else
             data_title)
ax0.set_title(title)

if save_fig:
    for fmt in formats:
        filename = 'pretraining_convergence_on_{}.{}'.format(data, fmt)
        fig.savefig(os.path.join(FIG_PATH, filename), bbox_inches='tight',
                    dpi=200)

plt.show()
