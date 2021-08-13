import os
from warnings import warn
import numpy as np
import yaml
from evaluation.utils import (
        get_run_cfg, get_run_experiment_name, get_multirun_iterates,
        uses_swa_weights)
from evaluation.display_utils import data_title_dict, get_title_from_run_spec
from dataset import get_test_data

import matplotlib.pyplot as plt

PATH = '/localdata/jleuschn/experiments/deep_image_prior_extension/'

FIG_PATH = '.'

save_fig = False

with open('../runs.yaml', 'r') as f:
    runs = yaml.load(f, Loader=yaml.FullLoader)

data = 'ellipses_lotus_limited_30'

runs_to_compare = [
    {
      'experiment': 'pretrain',
      'name': None,
      'sub_runs': [1],
      'title': 'Pretrained DIP (switch to noise)',
    },
    {
      'experiment': 'no_pretrain',
      'name': None,
      'sub_runs': [0],
      'title': 'DIP (noise)',
    },
    {
      'experiment': 'pretrain_only_fbp',
      'name': None,
      'sub_runs': [1],
      'title': 'Pretrained DIP (FBP)',
    },
    {
      'experiment': 'no_pretrain_fbp',
      'name': None,
      'sub_runs': [0],
      'title': 'DIP (FBP)',
    },
#     {
#       'experiment': 'pretrain_noise',
#       'name': None,
#       'sub_runs': [1],
#       'title': 'Pretrained DIP (FBP + noise)',
#     },
#     {
#       'experiment': 'no_pretrain_2inputs',
#       'name': None,
#       'sub_runs': [0],
#       'title': 'DIP (FBP + noise)',
#     },
]
iterates_iters_to_plot = [0, 10, 50, 100, 500, 1000, 5000, 10000]

data_title = data_title_dict[data]

seed_in_titles = False

def get_run_title(run_spec, cfg, include_seed=False):
    title_parts = [get_title_from_run_spec(run_spec)]

    if cfg['mdl']['load_pretrain_model']:
        if uses_swa_weights(cfg):
            title_parts.append('SWA weights')

    if include_seed:
        title_parts.append(
                'seed={:d}'.format(cfg['mdl']['torch_manual_seed']))

    title = ', '.join(title_parts)
    return title

iterates_to_plot_all_runs = []
run_titles = []

prev_ground_truth = None

for run_spec in runs_to_compare:
    assert len(run_spec['sub_runs']) == 1
    sub_run = run_spec['sub_runs'][0]

    experiment = run_spec['experiment']
    available_runs = runs['reconstruction_with_iterates'][data][experiment]
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

    run_path = os.path.join(run_path_multirun, '{:d}'.format(sub_run))

    cfg = get_run_cfg(run_path)
    experiment_name = get_run_experiment_name(run_path)

    _, _, ground_truth = get_test_data(data, cfg.data,
                                       return_torch_dataset=False)

    assert cfg['data']['name'] == data
    assert experiment_name == experiment

    if prev_ground_truth is not None:
        assert np.all(np.equal(ground_truth, prev_ground_truth))
    prev_ground_truth = ground_truth

    iterates_list, iterates_iters_list = get_multirun_iterates(
            run_path_multirun)
    iterates = iterates_list[sub_run]
    iterates_iters = iterates_iters_list[sub_run]

    iterates_to_plot = [it for it, it_iter in zip(iterates, iterates_iters)
                        if it_iter in iterates_iters_to_plot]

    iterates_to_plot_all_runs.append(iterates_to_plot)
    run_titles.append(get_run_title(run_spec, cfg,
                                    include_seed=seed_in_titles))

for i, iterate_iter in enumerate(iterates_iters_to_plot):

    nrows = 1
    ncols = len(runs_to_compare) + 1

    fig, axs = plt.subplots(nrows, ncols, figsize=(20, 4))

    iterates_all_runs = [iterates_to_plot[i]
                         for iterates_to_plot in iterates_to_plot_all_runs]

    for run_spec, run_title, iterate, ax in zip(
            runs_to_compare, run_titles, iterates_all_runs, axs):

        ax.imshow(iterate.T, cmap='gray')
        ax.set_title(run_title)

    axs[-1].imshow(ground_truth.T, cmap='gray')
    axs[-1].set_title('Ground truth')

    title = '{} after {:d} iterations'.format(data_title, iterate_iter)
    fig.suptitle(title)

    if save_fig:
        filename = '{}_on_{}_iter{:d}.pdf'.format(
                '_vs_'.join([(r['experiment'] if r.get('name') is None else
                              '{}_{}'.format(r['experiment'], r['name']))
                             for r in runs_to_compare]),
                data,
                iterate_iter)
        fig.savefig(os.path.join(FIG_PATH, filename), bbox_inches='tight')
