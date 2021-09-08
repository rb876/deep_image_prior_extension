# -*- coding: utf-8 -*-
import os
import numpy as np
from deep_image_prior import DeepImagePriorReconstructor
from copy import deepcopy

def val_sub_sub_path(i, i_sample):
    sub_sub_path_sample = os.path.join('rep_{:d}'.format(i),
                                       'sample_{:d}'.format(i_sample))
    return sub_sub_path_sample

# TODO docs
def reconstruct(noisy_obs, fbp, gt, ray_trafo, save_val_sub_path, cfg, cfg_mdl_val):
    reconstructor = DeepImagePriorReconstructor(**ray_trafo, cfg=cfg_mdl_val)
    reco, *optional_out = reconstructor.reconstruct(
            noisy_obs.float().unsqueeze(dim=0), fbp.unsqueeze(dim=0), gt.unsqueeze(dim=0),
            return_histories=True,
            return_iterates=cfg.save_iterates_path is not None)
    psnr_history = optional_out[0]['psnr']

    if cfg.save_histories_path is not None:
        histories = {k: np.array(v, dtype=np.float32)
                     for k, v in optional_out[0].items()}
        save_histories_path = os.path.join(
                cfg.save_histories_path, save_val_sub_path)
        os.makedirs(save_histories_path, exist_ok=True)
        np.savez(os.path.join(save_histories_path, 'histories.npz'),
                 **histories)
    if cfg.save_iterates_path is not None:
        iterates = optional_out[1]
        iterates_iters = optional_out[2]
        save_iterates_path = os.path.join(
                cfg.save_iterates_path, save_val_sub_path)
        os.makedirs(save_iterates_path, exist_ok=True)
        np.savez_compressed(
                os.path.join(save_iterates_path, 'iterates.npz'),
                iterates=np.asarray(iterates),
                iterates_iters=iterates_iters)
    return reco, psnr_history

# TODO docs
def validate_model(val_dataset, ray_trafo, seed, val_sub_path_mdl, baseline_psnr_steady, log_path_base, cfg, cfg_mdl_val):
    cfg_mdl_val = deepcopy(cfg_mdl_val)
    psnr_histories = []
    for i in range(cfg.val.num_repeats):
        for i_sample, (noisy_obs, fbp, *gt) in enumerate(val_dataset):
            gt = gt[0] if gt else None

            if cfg.val.load_histories_from_run_path is not None:
                load_histories_path = os.path.join(
                        cfg.val.load_histories_from_run_path,
                        cfg.save_histories_path,
                        val_sub_path_mdl,
                        val_sub_sub_path(i=i, i_sample=i_sample))
                psnr_history = np.load(os.path.join(load_histories_path, 'histories.npz'))['psnr'].tolist()
            else:
                cfg_mdl_val.torch_manual_seed = seed + i
                cfg_mdl_val.log_path = os.path.join(
                        log_path_base,
                        val_sub_path_mdl,
                        val_sub_sub_path(i=i, i_sample=i_sample))
                save_val_sub_path = os.path.join(
                        val_sub_path_mdl,
                        val_sub_sub_path(i=i, i_sample=i_sample))
                _, psnr_history = reconstruct(
                        noisy_obs=noisy_obs, fbp=fbp, gt=gt,
                        ray_trafo=ray_trafo,
                        save_val_sub_path=save_val_sub_path,
                        cfg=cfg, cfg_mdl_val=cfg_mdl_val)

            psnr_histories.append(psnr_history)

    median_psnr_output = np.median(psnr_histories, axis=0)
    psnr_steady = np.median(median_psnr_output[
            cfg.val.psnr_steady_start:cfg.val.psnr_steady_stop])
    rise_time = int(np.argwhere(
        median_psnr_output > psnr_steady - cfg.val.rise_time_remaining_psnr)[0][0])
    if baseline_psnr_steady == 'own_PSNR_steady':
        baseline_psnr_steady = psnr_steady
    rise_time_to_baseline = None if baseline_psnr_steady is None else int(np.argwhere(
        median_psnr_output > baseline_psnr_steady - cfg.val.rise_time_to_baseline_remaining_psnr)[0][0])

    info = {'rise_time': rise_time,
            'rise_time_to_baseline': rise_time_to_baseline,
            'PSNR_steady': psnr_steady,
            'PSNR_0': median_psnr_output[0]}

    return psnr_history, info
