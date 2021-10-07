import os
from warnings import warn
import numpy as np
import yaml
import png
import torch
from evaluation.utils import (
        get_multirun_cfgs, get_multirun_experiment_names,
        get_multirun_reconstructions, uses_swa_weights)
from dataset import get_standard_dataset, get_test_data
from deep_image_prior import DeepImagePriorReconstructor

PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CONFIG_PATH = os.path.join('..', '..', 'cfgs')

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'reconstructions')

os.makedirs(OUTPUT_PATH, exist_ok=True)

with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'runs.yaml'),
        'r') as f:
    runs = yaml.load(f, Loader=yaml.FullLoader)

SUFFIX = ''
# SUFFIX = '_norm_global'

NORMALIZATION_MODE = 'individual'
# NORMALIZATION_MODE = 'global'

FORMATS = ['png']

data = 'ellipses_lotus_20'

if data == 'ellipses_lotus_20':
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

elif data == 'ellipses_lotus_limited_30':
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
    assert all((cfg['data']['name'] == data for cfg in cfgs))
    swa = cfgs[0]['mdl']['load_pretrain_model'] and uses_swa_weights(cfgs[0])
    assert all(((cfg['mdl']['load_pretrain_model'] and uses_swa_weights(cfg))
                == swa) for cfg in cfgs)
    assert all((en == experiment for en in experiment_names))

    num_runs = len(cfgs)
    print('Found {:d} runs at path "{}".'.format(
            num_runs, run_path_multirun))

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
out_gts = None

out_init_recos_list = []
out_best_recos_list = []

for run_spec, cfgs, experiment_names, reconstructions in zip(
        runs_to_export, cfgs_list, experiment_names_list, reconstructions_list):

    experiment_name = experiment_names[0]

    out_init_recos = []
    out_best_recos = []

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

        for k, (_, fbp, gt) in enumerate(dataset_test):
            init_reco = get_init_reco(reconstructor, fbp)
            best_reco = recos[k]

            cur_fbps.append(fbp[0].detach().cpu().numpy())
            cur_gts.append(gt[0].detach().cpu().numpy())
            cur_init_recos.append(init_reco[0].detach().cpu().numpy())
            cur_best_recos.append(best_reco)

        if out_gts is None:
            out_fbps = cur_fbps
            out_gts = cur_gts
        else:
            assert len(cur_gts) == len(out_gts)
            assert all(np.array_equal(cur_fbp, out_fbp)
                    for cur_fbp, out_fbp in zip(cur_fbps, out_fbps))
            assert all(np.array_equal(cur_gt, out_gt)
                    for cur_gt, out_gt in zip(cur_gts, out_gts))

        out_init_recos.append(cur_init_recos)
        out_best_recos.append(cur_best_recos)

    out_init_recos_list.append(out_init_recos)
    out_best_recos_list.append(out_best_recos)


all_images = []
all_images += out_fbps
all_images += out_gts
for out_init_recos, out_best_recos in zip(
        out_init_recos_list, out_best_recos_list):
    for cur_init_recos, cur_best_recos in zip(
            out_init_recos, out_best_recos):
        all_images += cur_init_recos
        all_images += cur_best_recos

# normalize in-place (elements in all_images are the original arrays)

if NORMALIZATION_MODE == 'global':
    global_min = min(np.min(im) for im in all_images)
    global_max = max(np.max(im) for im in all_images)
    for im in all_images:
        im -= global_min
        im /= (global_max - global_min)
elif NORMALIZATION_MODE == 'individual':
    for im in all_images:
        im_min, im_max = np.min(im), np.max(im)
        im -= im_min
        im /= (im_max - im_min)
else:
    raise NotImplementedError


def save_as_format(filename, reco, fmt):
    if fmt == 'png':
        filename = filename if filename.endswith('.png') else filename + '.png'
        reco_uint8 = np.array(reco * 255, dtype=np.uint8)
        png.from_array(reco_uint8, mode='L').save(filename)
    else:
        raise NotImplementedError

for k, (fbp, gt) in enumerate(zip(out_fbps, out_gts)):
    fbp_filename = 'fbp_{}_sample_{:d}{}'.format(
            data, k, SUFFIX)
    gt_filename = 'gt_{}_sample_{:d}{}'.format(
            data, k, SUFFIX)

    for fmt in FORMATS:
        save_as_format(os.path.join(OUTPUT_PATH, fbp_filename), fbp, fmt)
        save_as_format(os.path.join(OUTPUT_PATH, gt_filename), gt, fmt)

for run_spec, out_init_recos, out_best_recos in zip(
        runs_to_export, out_init_recos_list, out_best_recos_list):
    run_filename = (run_spec['experiment'] if run_spec.get('name') is None
            else '{}_{}'.format(run_spec['experiment'], run_spec['name']))

    filename_base = '{}_on_{}'.format(run_filename, data)

    for j, (cur_init_recos, cur_best_recos) in enumerate(zip(
            out_init_recos, out_best_recos)):
        for k, (init_reco, best_reco) in enumerate(zip(
                cur_init_recos, cur_best_recos)):
            init_reco_filename = '{}_init_rep_{:d}_sample_{:d}{}'.format(
                    filename_base, j, k, SUFFIX)
            best_reco_filename = '{}_rep_{:d}_sample_{:d}{}'.format(
                    filename_base, j, k, SUFFIX)

            for fmt in FORMATS:
                save_as_format(os.path.join(OUTPUT_PATH, init_reco_filename),
                               init_reco, fmt)
                save_as_format(os.path.join(OUTPUT_PATH, best_reco_filename),
                               best_reco, fmt)
