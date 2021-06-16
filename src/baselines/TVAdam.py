import torch
import cv2
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch.nn import MSELoss
from warnings import warn
from tqdm import tqdm

def tv_loss(x):
    """
    Isotropic TV loss.
    """
    dh = torch.abs(x[..., :, 1:] - x[..., :, :-1])
    dw = torch.abs(x[..., 1:, :] - x[..., :-1, :])
    return torch.sum(dh[..., :-1, :] + dw[..., :, :-1])

def show_imageHU(window_name, data, cmin = 0.01, cmax = 0.45):
    data = (np.minimum(np.maximum(data, cmin), cmax) - cmin)/(cmax-cmin)
    data = np.array(np.ceil(data/np.max(data)*255), dtype = np.uint8)
    cv2.imshow(window_name, data)
    cv2.waitKey(0)
    cv2.imwrite(window_name+'.png', data)

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

    def reconstruct(self, observation, fbp, ground_truth=None, **kwargs):

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

        best_loss = np.infty
        best_output = self.model(self.output).detach().clone()
        with tqdm(range(self.cfg.iterations), desc='TV', disable=not self.cfg.show_pbar) as pbar:
            for i in pbar:
                self.optimizer.zero_grad()
                output = self.model(self.output)
                loss = criterion(self.forward_op_module(output),
                                 y_delta) + self.cfg.gamma * tv_loss(output)
                loss.backward()
                self.optimizer.step()
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    pbar.set_postfix({'best_loss': best_loss})
                    best_output = output.detach()
                if i % 100 == 0: # display and save
                    show_image('lotus_{}'.format(str(i)), best_output[0, 0, ...].cpu().numpy(), 'gray')

        return best_output[0, 0, ...].cpu().numpy()
