# -*- coding: utf-8 -*-
import os
import numpy as np
from deep_image_prior import DeepImagePriorReconstructor
from copy import deepcopy

def val_sub_sub_path(i, i_sample):
    sub_sub_path_sample = os.path.join('rep_{:d}'.format(i),
                                       'sample_{:d}'.format(i_sample))
    return sub_sub_path_sample

def reconstruct(noisy_obs, fbp, gt, ray_trafo, save_val_sub_path, cfg, cfg_mdl_val):
    """
    Run a DIP validation reconstruction.

    Parameters
    ----------
    noisy_observation : :class:`torch.Tensor`
        Noisy observation.
    fbp : :class:`torch.Tensor`
        Input reconstruction (e.g. filtered backprojection).
    ground_truth : :class:`torch.Tensor`
        Ground truth image.
    ray_trafo : dict
        Dictionary with the following entries:

            `'ray_trafo_module'` : :class:`torch.nn.Module`
                Ray transform module.
            `'reco_space'` : :class:`odl.DiscretizedSpace`
                Image domain.
            `'observation_space'` : :class:`odl.DiscretizedSpace`
                Observation domain.

    save_val_sub_path : str
        Path to append to `cfg.save_histories_path` and
        `cfg.save_iterates_path` (if those are specified, respectively).
        This allows to store the results for multiple reconstructions within
        the validation run path.

    cfg : :class:`omegaconf.OmegaConf`
        Full configuration of the run. Note that ``cfg.mdl`` is ignored and
        instead `cfg_mdl_val` is used as the model configuration.
    cfg_mdl_val : :class:`omegaconf.OmegaConf`
        Configuration of the model.

    Returns
    -------
    reco :  :class:`numpy.ndarray`
        The reconstruction.
    psnr_history : list of scalar values
        PSNR history.
    """
    reconstructor = DeepImagePriorReconstructor(**ray_trafo, cfg=cfg_mdl_val)
    reco, *optional_out = reconstructor.reconstruct(
            noisy_obs, fbp, gt,
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

def validate_model(val_dataset, ray_trafo, seed, val_sub_path_mdl, baseline_psnr_steady, log_path_base, cfg, cfg_mdl_val):
    """
    Validate a model on a validation dataset with repeated runs.

    Parameters
    ----------
    val_dataset : :class:`torch.utils.data.TensorDataset`
        Validation dataset, as returned by :func:`dataset.get_validation_data`.
    ray_trafo : dict
        Dictionary with the following entries:

            `'ray_trafo_module'` : :class:`torch.nn.Module`
                Ray transform module.
            `'reco_space'` : :class:`odl.DiscretizedSpace`
                Image domain.
            `'observation_space'` : :class:`odl.DiscretizedSpace`
                Observation domain.

    seed : int
        Initial seed for the model.  In each repetition the seed is set by
        ``cfg_mdl_val.torch_manual_seed = seed + i``, where
        ``i in range(cfg.val.num_repeats)``.
    val_sub_path_mdl : str
        Path to append to `cfg.save_histories_path`,
        `cfg.save_iterates_path` (if those are specified, respectively) and
        `log_path_base` to identify the model. This allows to store the results
        for models within the validation run path. Inside the path specific to
        the model, the results of each individual run are saved in
        ``val_sub_sub_path(i, i_sample)``, where
        ``i in range(cfg.val.num_repeats)`` and
        ``i_sample in range(len(val_dataset))``.
    baseline_psnr_steady : float or `'own_PSNR_steady'`
        Steady PSNR of a baseline, by which ``info['rise_time_to_baseline']``
        is determined. If it is `'own_PSNR_steady'`, then
        ``info['PSNR_steady']`` is used. If it is `None`, also
        ``info['rise_time_to_baseline']`` is set to `None`.

    log_path_base : str
        Base path under which to save the tensorboard logs.
    cfg : :class:`omegaconf.OmegaConf`
        Full configuration of the run. Note that ``cfg.mdl`` is ignored and
        instead `cfg_mdl_val` is used as the model configuration.
    cfg_mdl_val : :class:`omegaconf.OmegaConf`
        Configuration of the model. This function will override
        ``cfg_mdl_val.torch_manual_seed`` and ``cfg_mdl_val.log_path`` (in a
        copy of the configuration).

    Returns
    -------
    psnr_histories : list of lists of lists of scalar values
        PSNR histories of all runs.
        The PSNR history of repetition `i` on validation sample `i_sample` is
        given by ``psnr_histories[i][i_sample]``.
    info : dict
        Validation info about the model. It is based on the median PSNR history
        that is the point-wise median w.r.t. all repetitions and validation
        samples, i.e. ``np.median(psnr_histories, axis=(0, 1))``.

            `'rise_time'` : int
                The first iteration where the median PSNR reaches
                ``info['PSNR_steady'] - cfg.val.rise_time_remaining_psnr``.
            `'rise_time_to_baseline'` : int or None
                The first iteration where the median PSNR reaches
                ``baseline_psnr_steady - cfg.val.rise_time_to_baseline_remaining_psnr``,
                or `None` if `baseline_psnr_steady` is `None`.
            `'PSNR_steady'` : scalar
                The steady PSNR, determined as the median of the median PSNR
                history in the configured interval, i.e.
                ``np.median(np.median(psnr_histories, axis=(0, 1))[cfg.val.psnr_steady_start:cfg.val.psnr_steady_stop])``.
            `'PSNR_0'`: scalar
                The initial median PSNR, i.e.
                ``np.median(psnr_histories, axis=(0, 1))[0]``.
    """
    cfg_mdl_val = deepcopy(cfg_mdl_val)
    psnr_histories = []
    for i in range(cfg.val.num_repeats):
        psnr_histories_i = []
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
                        noisy_obs=noisy_obs.float().unsqueeze(dim=0),
                        fbp=fbp.unsqueeze(dim=0), gt=gt.unsqueeze(dim=0),
                        ray_trafo=ray_trafo,
                        save_val_sub_path=save_val_sub_path,
                        cfg=cfg, cfg_mdl_val=cfg_mdl_val)

            psnr_histories_i.append(psnr_history)

        psnr_histories.append(psnr_histories_i)

    median_psnr_output = np.median(psnr_histories, axis=(0, 1))
    psnr_steady = np.median(median_psnr_output[
            cfg.val.psnr_steady_start:cfg.val.psnr_steady_stop])
    rise_time = int(np.argwhere(
        median_psnr_output > psnr_steady - cfg.val.rise_time_remaining_psnr)[0][0])
    if baseline_psnr_steady == 'own_PSNR_steady':
        baseline_psnr_steady = psnr_steady
    if baseline_psnr_steady is None:
        rise_time_to_baseline = None
    else:
        argwhere_close_enough_to_baseline = np.argwhere(
                median_psnr_output > baseline_psnr_steady - cfg.val.rise_time_to_baseline_remaining_psnr)
        rise_time_to_baseline = (
                int(argwhere_close_enough_to_baseline[0][0])
                if len(argwhere_close_enough_to_baseline) >= 1 else None)

    info = {'rise_time': rise_time,
            'rise_time_to_baseline': rise_time_to_baseline,
            'PSNR_steady': psnr_steady,
            'PSNR_0': median_psnr_output[0]}

    return psnr_histories, info
