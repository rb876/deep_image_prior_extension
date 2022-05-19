import os
import json
from warnings import warn
import contextlib
from itertools import islice
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
from evaluation.utils import (
        get_multirun_cfgs, get_multirun_experiment_names, uses_swa_weights)
from dataset import get_standard_dataset
from deep_image_prior import DeepImagePriorReconstructor, PSNR, SSIM
from torch.cuda.amp import autocast

PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CONFIG_PATH = os.path.join('..', '..', 'cfgs')

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'images')

os.makedirs(OUTPUT_PATH, exist_ok=True)

# data = 'ellipses_lotus_20'
# data = 'ellipses_lotus_limited_45'
# data = 'brain_walnut_120'
# data = 'ellipses_walnut_120'
# data = 'ellipsoids_walnut_3d'
data = 'ellipsoids_walnut_3d_60'

fold = 'train'

NUM_SAMPLES = 4

OUTPUT_METRICS_PATH = os.path.join(
        OUTPUT_PATH, '{}_pretraining_{}_metrics.json'.format(data, fold))

with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'runs.yaml'),
        'r') as f:
    runs = yaml.load(f, Loader=yaml.FullLoader)

if data == 'ellipses_lotus_20':
    run_spec = {
        'experiment': 'pretrain_only_fbp',
        'name': 'no_stats_no_sigmoid_train_run2_epochs100',
        'name_title': '',
    }
elif data == 'ellipses_lotus_limited_45':
    run_spec = {
        'experiment': 'pretrain_only_fbp',
    }
elif data == 'brain_walnut_120':
    run_spec = {
        'experiment': 'pretrain_only_fbp',
        'name': 'no_stats_no_sigmoid_train_run1',
        'name_title': '',
    }
elif data == 'ellipses_walnut_120':
    run_spec = {
        'experiment': 'pretrain_only_fbp',
        'name': 'no_stats_no_sigmoid_train_run1',
        'name_title': '',
    }
elif data == 'ellipsoids_walnut_3d':
    run_spec = {
        'experiment': 'pretrain_only_fbp',
        'name': 'epochs0_steps8000',
        'name_title': '',
    }
elif data == 'ellipsoids_walnut_3d_60':
    run_spec = {
        'experiment': 'pretrain_only_fbp',
        'name': 'epochs0_steps8000',
        'name_title': '',
    }


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
if len(cfgs) == 0:
    raise RuntimeError('No runs found at path "{}", aborting.'.format(
            run_path_multirun))
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
assert all((cfg['mdl']['load_pretrain_model'] for cfg in cfgs))
swa = uses_swa_weights(cfgs[0])
assert all((uses_swa_weights(cfg) == swa for cfg in cfgs))
assert all((en == experiment for en in experiment_names))

cfg = cfgs[0]

dataset, ray_trafos = get_standard_dataset(cfg.data.name, cfg.data)

ray_trafo = {'ray_trafo_module': ray_trafos['ray_trafo_module'],
             'reco_space': dataset.space[1],
             'observation_space': dataset.space[0]
            }

reconstructor = DeepImagePriorReconstructor(**ray_trafo, cfg=cfg.mdl)
reconstructor.model.eval()

out_fbps = []
out_gts = []
out_recos = []
out_fbp_metrics = []
out_reco_metrics = []

torch_dataset = dataset.create_torch_dataset(
    fold=fold, reshape=((1,) + dataset.space[0].shape,
                        (1,) + dataset.space[1].shape,
                        (1,) + dataset.space[1].shape))

data_loader = DataLoader(torch_dataset, batch_size=1, num_workers=0,
                         shuffle=False)

for k, (_, fbp, gt) in enumerate(islice(data_loader, NUM_SAMPLES)):
    fbp_np = np.asarray(fbp[0, 0].detach().cpu().numpy())
    gt_np = np.asarray(gt[0, 0].detach().cpu().numpy())

    path = (cfg.mdl.learned_params_path
            if cfg.mdl.learned_params_path.endswith('.pt')
            else cfg.mdl.learned_params_path + '.pt')
    reconstructor.model.load_state_dict(
            torch.load(path, map_location=reconstructor.device))

    with torch.no_grad():
        with autocast() if reconstructor.cfg.use_mixed else contextlib.nullcontext():
            reco = reconstructor.model(fbp.to(reconstructor.device)).detach().cpu()

    reco_np = np.asarray(reco[0, 0].detach().cpu().numpy())

    if reconstructor.cfg.arch.use_relu_out == 'post':
        for r in [fbp_np, reco_np]:
            np.clip(r, 0., None, out=r)  # apply relu to reconstruction

    out_fbps.append(fbp_np)
    out_gts.append(gt_np)
    out_recos.append(reco_np)
    out_fbp_metrics.append({'psnr': PSNR(fbp_np, gt_np),
                            'ssim': SSIM(fbp_np, gt_np)})
    out_reco_metrics.append({'psnr': PSNR(reco_np, gt_np),
                             'ssim': SSIM(reco_np, gt_np)})

metrics = {
    'fbp': {
        'sample_{:d}'.format(k): m for k, m in enumerate(out_fbp_metrics)
    },
    'reco': {
        'sample_{:d}'.format(k): m for k, m in enumerate(out_reco_metrics)
    },
}

print('metrics:\n{}'.format(metrics))

if OUTPUT_METRICS_PATH is not None:
    with open(OUTPUT_METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=4)


for k, (fbp, gt, reco) in enumerate(zip(out_fbps, out_gts, out_recos)):
    fbp_filename = '{}_pretraining_{}_fbp_sample_{:d}'.format(data, fold, k)
    gt_filename = '{}_pretraining_{}_gt_sample_{:d}'.format(data, fold, k)
    reco_filename = '{}_pretraining_{}_reco_sample_{:d}'.format(data, fold, k)

    np.save(os.path.join(OUTPUT_PATH, fbp_filename), fbp)
    np.save(os.path.join(OUTPUT_PATH, gt_filename), gt)
    np.save(os.path.join(OUTPUT_PATH, reco_filename), reco)
