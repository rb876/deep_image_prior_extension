import torch
import numpy as np
from skimage.metrics import structural_similarity

def tv_loss(x):
    """
    Anisotropic TV loss for 2D images.
    """
    dh = torch.abs(x[..., :, 1:] - x[..., :, :-1])
    dw = torch.abs(x[..., 1:, :] - x[..., :-1, :])
    return torch.sum(dh[..., :-1, :] + dw[..., :, :-1])

def tv_loss_3d(x):
    """
    Anisotropic TV loss for 3D images.

    Differences to 2D version (kept for backward comp., same as Baguer et al.):
      - mean instead of sum (more natural to combine with an MSE loss)
      - include last difference value in each dimension
    """
    dx = torch.abs(x[..., :, :, 1:] - x[..., :, :, :-1])
    dy = torch.abs(x[..., :, 1:, :] - x[..., :, :-1, :])
    dz = torch.abs(x[..., 1:, :, :] - x[..., :-1, :, :])
    return torch.mean(dx) + torch.mean(dy) + torch.mean(dz)

def poisson_loss(y_pred, y_true, photons_per_pixel, mu_max):
    """
    Loss corresponding to Poisson regression.
    References
    ----------
    .. [2] https://en.wikipedia.org/wiki/Poisson_regression
    """

    def get_photons(y):
        y = torch.exp(-y * mu_max) * photons_per_pixel
        return y

    def get_photons_log(y):
        y = -y * mu_max + np.log(photons_per_pixel)
        return y

    y_true_photons = get_photons(y_true)
    y_pred_photons = get_photons(y_pred)
    y_pred_photons_log = get_photons_log(y_pred)

    return torch.sum(y_pred_photons - y_true_photons * y_pred_photons_log)

def PSNR(reconstruction, ground_truth, data_range=None):
    gt = np.asarray(ground_truth)
    mse = np.mean((np.asarray(reconstruction) - gt)**2)
    if mse == 0.:
        return float('inf')
    if data_range is not None:
        return 20*np.log10(data_range) - 10*np.log10(mse)
    else:
        data_range = np.max(gt) - np.min(gt)
        return 20*np.log10(data_range) - 10*np.log10(mse)

def SSIM(reconstruction, ground_truth, data_range=None):
    gt = np.asarray(ground_truth)
    if data_range is not None:
        return structural_similarity(reconstruction, gt, data_range=data_range)
    else:
        data_range = np.max(gt) - np.min(gt)
        return structural_similarity(reconstruction, gt, data_range=data_range)

def normalize(x, inplace=False):
    if inplace:
        x -= x.min()
        x /= x.max()
    else:
        x = x - x.min()
        x = x / x.max()
    return x

def is_name_in_set(name, adms_params_set):
    for el in adms_params_set:
        if el in name:
            return True
    return False

def extract_learnable_params(model, adms_params_set):
    params_list = [param for (name, param) in model.named_parameters()
                   if is_name_in_set(name, adms_params_set)]

    return params_list
