import torch
import numpy as np
import torch.nn as nn
import tensorboardX
from torch.optim import Adam
from torch.nn import MSELoss
from warnings import warn
from tqdm import tqdm

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

def tv_loss(x):
    """
    Isotropic TV loss.
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

def show_image(outputname, data, cmap, clim=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10,5))
    im = ax.imshow(data, cmap=cmap)
    ax.axis('off')
    if clim is not None:
        im.set_clim(*clim)
    plt.savefig(outputname + '.png', bbox_inches='tight', pad_inches=0.0)

class TVAdamReconstructor:

    """
    Reconstructor minimizing a TV-functional with the Adam optimizer.
    """

    def __init__(self, ray_trafo_module, reco_space, observation_space, cfg):

        self.device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.forward_op_module = ray_trafo_module.to(self.device)
        self.cfg = cfg

    def reconstruct(self, observation, fbp, ground_truth=None, log=False, **kwargs):

        torch.random.manual_seed(10)
        self.output = fbp.clone().detach().to(self.device)
        self.output.requires_grad = True
        self.model = torch.nn.Identity()
        self.optimizer = Adam([self.output], lr=self.cfg.lr)
        y_delta = torch.tensor(observation).to(self.device)

        if self.cfg.loss_function == 'mse':
            criterion = MSELoss()
        else:
            warn('Unknown loss function, falling back to MSE')
            criterion = MSELoss()

        if log:
            self.writer = tensorboardX.SummaryWriter()

        tv_loss_fun = tv_loss if len(fbp.shape) == 4 else tv_loss_3d

        best_loss = np.infty
        best_output = self.model(self.output).detach().clone()
        with tqdm(range(self.cfg.iterations), desc='TV', disable=not self.cfg.show_pbar) as pbar:
            for i in pbar:
                self.optimizer.zero_grad()
                output = self.model(self.output)
                loss = criterion(self.forward_op_module(output),
                                 y_delta) + self.cfg.gamma * tv_loss_fun(output)
                loss.backward()
                self.optimizer.step()
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    pbar.set_postfix({'best_loss': best_loss})
                    best_output = torch.nn.functional.relu(output.detach()) if self.cfg.use_relu_out else output.detach()
                if i % 100 == 0: # display and save
                    if len(best_output.shape) == 4:
                        show_image('tv_reco_{}'.format(str(i)), best_output[0, 0, ...].cpu().numpy(), 'gray')
                    else:
                        show_image('tv_reco_mid_slice_{}'.format(str(i)), best_output[0, 0, best_output.shape[2] // 2, ...].cpu().numpy(), 'gray')
                if log and ground_truth is not None:
                    best_output_psnr = PSNR(best_output.detach().cpu(), ground_truth.cpu())
                    output_psnr = PSNR((torch.nn.functional.relu(output.detach()) if self.cfg.use_relu_out else output.detach()).cpu(), ground_truth.cpu())
                    pbar.set_postfix({'output_psnr': output_psnr})
                    self.writer.add_scalar('best_output_psnr', best_output_psnr, i)
                    self.writer.add_scalar('output_psnr', output_psnr, i)

        if log:
            self.writer.close()

        return best_output[0, 0, ...].cpu().numpy()
