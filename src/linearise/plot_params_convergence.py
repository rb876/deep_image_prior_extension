import os
import glob
import hydra
import torch
import numpy as np
from deep_image_prior import DeepImagePriorReconstructor
from torch_utils import parameters_to_vector
from omegaconf import DictConfig

@hydra.main(config_path='../cfgs', config_name='config')
def coordinator(cfg : DictConfig) -> None:
  
    search_dir = cfg.spct.path_to_checkpoints
    files = list(filter(os.path.isfile, glob.glob(search_dir + "*")))
    files.sort(key=lambda x: os.path.getmtime(x))
    paths_to_checkpoints = []
    for file in files:
        if file.endswith(".pt"):
            paths_to_checkpoints.append(os.path.join(search_dir, file))
    
    ray_trafo = {'ray_trafo_module': torch.Tensor([1]), # placeholder 
                'reco_space': None,
                'observation_space': None
            }

    x_axes = [int(el.split("/params_iters")[-1].split(".pt")[0]) for el in paths_to_checkpoints]
    converged_model_path = paths_to_checkpoints[-1]
    state_dict = torch.load(converged_model_path)
    reconstructor = DeepImagePriorReconstructor(**ray_trafo, cfg=cfg.mdl)
    reconstructor.model.load_state_dict(state_dict)
    converged_params = \
        parameters_to_vector(reconstructor.model.named_parameters(),
        cfg.spct.skip_layers)
    se = []
    for i, path in enumerate(paths_to_checkpoints):
        state_dict = torch.load(path)
        reconstructor.model.load_state_dict(state_dict)
        params = \
            parameters_to_vector(reconstructor.model.named_parameters(),
            cfg.spct.skip_layers)
        if i == 0: 
            se0 = torch.sum((params - converged_params)**2)
        se.append(torch.sum((params - converged_params)**2 / se0).detach().cpu().numpy())


    import matplotlib.pyplot as plt         
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.loglog(x_axes, se, linewidth=2.5)
    ax.set_ylim([0, 1])

    ax.grid()
    fig.savefig('test.pdf', bbox_inches='tight')
    
if __name__ == '__main__':
    coordinator()

