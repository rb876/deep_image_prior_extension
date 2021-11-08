import os
import json
from warnings import warn
import numpy as np
import yaml
import torch
import imageio
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap, ScalarMappable
from evaluation.utils import (
        get_multirun_cfgs, get_multirun_experiment_names,
        get_multirun_reconstructions, uses_swa_weights)
from dataset import get_standard_dataset, get_test_data
from deep_image_prior import DeepImagePriorReconstructor, PSNR, SSIM

PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CONFIG_PATH = os.path.join('..', '..', 'cfgs')

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'images')

os.makedirs(OUTPUT_PATH, exist_ok=True)

SAVE_ONLY_MEDIAN_REP = True

data = 'ellipses_lotus_20'
# data = 'ellipses_limited_45'
# data = 'brain_walnut_120'
# data = 'ellipses_walnut_120'

OUTPUT_METRICS_PATH = os.path.join(
        OUTPUT_PATH, '{}_metrics.json'.format(data))
OUTPUT_MEDIAN_PSNR_REPS_PATH = os.path.join(
        OUTPUT_PATH, '{}_median_psnr_reps.json'.format(data))

with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'runs.yaml'),
        'r') as f:
    runs = yaml.load(f, Loader=yaml.FullLoader)

NUM_REPEATS = 5

if data == 'ellipses_lotus_20':
    runs_to_export = [
        {
        'experiment': 'no_pretrain',
        'name': 'no_stats_no_sigmoid',
        'name_title': '',
        'name_filename': None,
        },
        {
        'experiment': 'no_pretrain_fbp',
        'name': 'no_stats_no_sigmoid',
        'name_title': '',
        'name_filename': None,
        },
        {
        'experiment': 'pretrain_only_fbp',
        'name': 'no_stats_no_sigmoid_train_run2_epochs100',
        'name_title': '',
        'name_filename': None,
        },
        {
        'experiment': 'pretrain',
        'name': 'no_stats_no_sigmoid_train_run2_epochs100',
        'name_title': '',
        'name_filename': None,
        },
    ]

elif data == 'ellipses_lotus_limited_45':
    runs_to_export = [
        {
        'experiment': 'no_pretrain',
        },
        {
        'experiment': 'no_pretrain_fbp',
        },
        {
        'experiment': 'pretrain_only_fbp',
        },
        {
        'experiment': 'pretrain',
        },
    ]

elif data == 'brain_walnut_120':
    runs_to_export = [
        {
        'experiment': 'no_pretrain',
        'name': 'no_stats_no_sigmoid',
        'name_title': '',
        'name_filename': None,
        },
        {
        'experiment': 'no_pretrain_fbp',
        'name': 'no_stats_no_sigmoid',
        'name_title': '',
        'name_filename': None,
        },
        {
        'experiment': 'pretrain_only_fbp',
        'name': 'no_stats_no_sigmoid_train_run1',
        'name_title': '',
        'name_filename': None,
        },
        {
        'experiment': 'pretrain',
        'name': 'no_stats_no_sigmoid_train_run1',
        'name_title': '',
        'name_filename': None,
        },
    ]

elif data == 'ellipses_walnut_120':
    runs_to_export = [
        {
        'experiment': 'no_pretrain',
        'name': 'no_stats_no_sigmoid',
        'name_title': '',
        'name_filename': None,
        },
        {
        'experiment': 'no_pretrain_fbp',
        'name': 'no_stats_no_sigmoid',
        'name_title': '',
        'name_filename': None,
        },
        {
        'experiment': 'pretrain_only_fbp',
        'name': 'no_stats_no_sigmoid_train_run1',
        'name_title': '',
        'name_filename': None,
        },
        {
        'experiment': 'pretrain',
        'name': 'no_stats_no_sigmoid_train_run1',
        'name_title': '',
        'name_filename': None,
        },
    ]

cfgs_list = []
experiment_names_list = []
reconstructions_list = []

for run_spec in runs_to_export:
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
    reconstructions = get_multirun_reconstructions(
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
    print('Found {:d} runs at path "{}".'.format(
            num_runs, run_path_multirun))
    assert num_runs == NUM_REPEATS

    cfgs_list.append(cfgs)
    experiment_names_list.append(experiment_names)
    reconstructions_list.append(reconstructions)


def get_init_reco(reconstructor, fbp):
    fbp = fbp[None]
    # imitate first part of DeepImagePriorReconstructor.reconstruct
    if reconstructor.cfg.torch_manual_seed:
        torch.random.manual_seed(reconstructor.cfg.torch_manual_seed)

    reconstructor.init_model()
    if reconstructor.cfg.load_pretrain_model:
        path = \
            reconstructor.cfg.learned_params_path if reconstructor.cfg.learned_params_path.endswith('.pt') \
                else reconstructor.cfg.learned_params_path + '.pt'
        reconstructor.model.load_state_dict(torch.load(path, map_location=reconstructor.device))
    else:
        reconstructor.model.to(reconstructor.device)

    reconstructor.model.train()

    if reconstructor.cfg.recon_from_randn:
        net_input = 0.1 * \
            torch.randn(1, *reconstructor.reco_space.shape)[None].to(reconstructor.device)
        if reconstructor.cfg.add_init_reco:
            net_input = \
                torch.cat([fbp.to(reconstructor.device), net_input], dim=1)
    else:
        net_input = fbp.to(reconstructor.device)

    output = reconstructor.apply_model_on_test_data(net_input)

    return output[0]

out_fbps = None
out_fbp_metrics = None
out_gts = None

out_init_recos_list = []
out_best_recos_list = []
out_init_reco_metrics_list = []
out_best_reco_metrics_list = []
out_init_reco_stds_list = []
out_best_reco_stds_list = []
out_mean_reco_errors_list = []



for run_spec, cfgs, experiment_names, reconstructions in zip(
        runs_to_export, cfgs_list, experiment_names_list, reconstructions_list):

    experiment_name = experiment_names[0]

    out_init_recos = []
    out_best_recos = []
    out_init_reco_metrics = []
    out_best_reco_metrics = []

    for j, (cfg, recos) in enumerate(zip(cfgs, reconstructions)):

        dataset, ray_trafos = get_standard_dataset(cfg.data.name, cfg.data)

        dataset_test = get_test_data(cfg.data.name, cfg.data)

        ray_trafo = {'ray_trafo_module': ray_trafos['ray_trafo_module'],
                     'reco_space': dataset.space[1],
                     'observation_space': dataset.space[0]
                    }

        reconstructor = DeepImagePriorReconstructor(**ray_trafo, cfg=cfg.mdl)

        cur_fbps = []
        cur_gts = []
        cur_init_recos = []
        cur_best_recos = []
        cur_init_reco_metrics = []
        cur_best_reco_metrics = []

        for k, (_, fbp, gt) in enumerate(dataset_test):
            init_reco = get_init_reco(reconstructor, fbp)

            fbp = fbp[0].detach().cpu().numpy()
            gt = gt[0].detach().cpu().numpy()
            init_reco = init_reco[0].detach().cpu().numpy()
            best_reco = recos[k]

            cur_fbps.append(fbp)
            cur_gts.append(gt)
            cur_init_recos.append(init_reco)
            cur_best_recos.append(best_reco)

            cur_init_reco_metrics.append(
                    {'psnr': PSNR(init_reco, gt),
                     'ssim': SSIM(init_reco, gt)})
            cur_best_reco_metrics.append(
                    {'psnr': PSNR(best_reco, gt),
                     'ssim': SSIM(best_reco, gt)})

        if out_gts is None:
            out_fbps = cur_fbps
            out_gts = cur_gts

            out_fbp_metrics = [
                    {'psnr': PSNR(fbp, gt),
                     'ssim': SSIM(fbp, gt)}
                    for fbp, gt in zip(out_fbps, out_gts)]
        else:
            pass
            # assert len(cur_gts) == len(out_gts)
            # assert all(np.array_equal(cur_fbp, out_fbp)
            #         for cur_fbp, out_fbp in zip(cur_fbps, out_fbps))
            # assert all(np.array_equal(cur_gt, out_gt)
            #         for cur_gt, out_gt in zip(cur_gts, out_gts))

        out_init_recos.append(cur_init_recos)
        out_best_recos.append(cur_best_recos)
        out_init_reco_metrics.append(cur_init_reco_metrics)
        out_best_reco_metrics.append(cur_best_reco_metrics)

    out_init_recos_list.append(out_init_recos)
    out_best_recos_list.append(out_best_recos)
    out_init_reco_metrics_list.append(out_init_reco_metrics)
    out_best_reco_metrics_list.append(out_best_reco_metrics)

    out_init_reco_stds = [
            np.std(np.stack([recos[k] for recos in out_init_recos]), axis=0)
            for k, _ in enumerate(out_gts)]
    out_best_reco_stds = [
            np.std(np.stack([recos[k] for recos in out_best_recos]), axis=0)
            for k, _ in enumerate(out_gts)]
    out_mean_reco_errors = [
            (np.mean(np.stack([recos[k] for recos in out_best_recos]), axis=0)
                    - gt)
            for k, gt in enumerate(out_gts)]
    out_init_reco_stds_list.append(out_init_reco_stds)
    out_best_reco_stds_list.append(out_best_reco_stds)
    out_mean_reco_errors_list.append(out_mean_reco_errors)


def get_run_name_for_filename(run_spec):
    name_filename = run_spec.get('name_filename', run_spec.get('name'))

    run_name_for_filename = (
            run_spec['experiment'] if not name_filename
            else '{}_{}'.format(run_spec['experiment'], name_filename))

    return run_name_for_filename

fbp_metrics = {
    'sample_{:d}'.format(k): m for k, m in enumerate(out_fbp_metrics)
}
metrics_list = [
    {
        'rep_{:d}'.format(j): {
            'sample_{:d}'.format(k): {
                'init': init_metrics,
                'best': best_metrics,
            } for k, (init_metrics, best_metrics) in enumerate(
                    zip(cur_init_reco_metrics, cur_best_reco_metrics))
        } for j, (cur_init_reco_metrics, cur_best_reco_metrics) in enumerate(
                zip(out_init_reco_metrics, out_best_reco_metrics))
    } for run_spec, out_init_reco_metrics, out_best_reco_metrics in zip(
            runs_to_export,
            out_init_reco_metrics_list, out_best_reco_metrics_list)
]
metrics = {
    get_run_name_for_filename(run_spec): m
    for m, run_spec in zip(metrics_list, runs_to_export)
}
metrics['fbp'] = fbp_metrics

print('metrics:\n{}'.format(metrics))

if OUTPUT_METRICS_PATH is not None:
    with open(OUTPUT_METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=4)


def argmedian(arr):
    arr = np.asarray(arr)
    idx = int(np.argmin(np.abs(arr - np.median(arr))))
    return idx

median_psnr_reps_list = [
    {
        'sample_{:d}'.format(k): argmedian(
                [rep_metrics['sample_{:d}'.format(k)]['best']['psnr']
                 for _, rep_metrics in sorted(run_metrics.items())])
        for k in range(len(out_gts))
    } for run_spec, run_metrics in zip(runs_to_export, metrics_list)
]
median_psnr_reps = {
    get_run_name_for_filename(run_spec): m
    for m, run_spec in zip(median_psnr_reps_list, runs_to_export)
}

print('median_psnr_reps:\n{}'.format(median_psnr_reps))

if OUTPUT_MEDIAN_PSNR_REPS_PATH is not None:
    with open(OUTPUT_MEDIAN_PSNR_REPS_PATH, 'w') as f:
        json.dump(median_psnr_reps, f, indent=4)


for k, (fbp, gt) in enumerate(zip(out_fbps, out_gts)):
    fbp_filename = '{}_fbp_sample_{:d}'.format(data, k)
    gt_filename = '{}_gt_sample_{:d}'.format(data, k)

    np.save(os.path.join(OUTPUT_PATH, fbp_filename), fbp)
    np.save(os.path.join(OUTPUT_PATH, gt_filename), gt)

for (run_spec, run_median_psnr_reps,
     out_init_recos, out_best_recos,
     out_init_reco_stds, out_best_reco_stds,
     out_mean_reco_errors) in zip(
        runs_to_export, median_psnr_reps_list,
        out_init_recos_list, out_best_recos_list,
        out_init_reco_stds_list, out_best_reco_stds_list,
        out_mean_reco_errors_list):
    run_name_for_filename = get_run_name_for_filename(run_spec)

    for j, (cur_init_recos, cur_best_recos) in enumerate(zip(
            out_init_recos, out_best_recos)):
        for k, (init_reco, best_reco) in enumerate(zip(
                cur_init_recos, cur_best_recos)):
            if (not SAVE_ONLY_MEDIAN_REP or
                    j == run_median_psnr_reps['sample_{:d}'.format(k)]):
                init_reco_filename = '{}_{}_init_rep_{:d}_sample_{:d}'.format(
                        data, run_name_for_filename, j, k)
                best_reco_filename = '{}_{}_rep_{:d}_sample_{:d}'.format(
                        data, run_name_for_filename, j, k)

                np.save(os.path.join(OUTPUT_PATH, init_reco_filename),
                        init_reco)
                np.save(os.path.join(OUTPUT_PATH, best_reco_filename),
                        best_reco)

    for k, (init_reco_std, best_reco_std, mean_reco_error) in enumerate(zip(
            out_init_reco_stds, out_best_reco_stds, out_mean_reco_errors)):
        init_reco_std_filename = '{}_{}_init_std_sample_{:d}'.format(
                data, run_name_for_filename, k)
        best_reco_std_filename = '{}_{}_std_sample_{:d}'.format(
                data, run_name_for_filename, k)
        mean_reco_error_filename = '{}_{}_mean_error_sample_{:d}'.format(
                data, run_name_for_filename, k)
        np.save(os.path.join(OUTPUT_PATH, init_reco_std_filename),
                init_reco_std)
        np.save(os.path.join(OUTPUT_PATH, best_reco_std_filename),
                best_reco_std)
        np.save(os.path.join(OUTPUT_PATH, mean_reco_error_filename),
                mean_reco_error)
