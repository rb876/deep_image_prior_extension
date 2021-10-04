import os
import hydra
import torch
import numpy as np
from omegaconf import DictConfig
from dataset import get_standard_dataset, get_test_data, get_validation_data
from deep_image_prior import DeepImagePriorReconstructor
from utils import randomised_SVD_jacobian
from torch_utils import parameters_to_vector, list_norm_layers

def unpack_loader(testloader, store_device):

    unpck = [el[1] for el in testloader]
    fbp = unpck[0].unsqueeze(dim=0)
    return fbp.to(store_device)

def extract_params(reconstructor, cfg):

    state_dict = torch.load(os.path.join(cfg.path_to_checkpoints,
                    cfg.filename))
    reconstructor.model.load_state_dict(state_dict)
    params = \
        parameters_to_vector(reconstructor.model.named_parameters(),
        cfg.skip_layers)
    return params

@hydra.main(config_path='../cfgs', config_name='config')
def coordinator(cfg : DictConfig) -> None:

    dataset, ray_trafos = get_standard_dataset(cfg.data.name, cfg.data)

    if cfg.validation_run:
        if cfg.data.validation_data:
            dataset_test = get_validation_data(cfg.data.name, cfg.data)
        else:
            dataset_test = dataset.create_torch_dataset(
                fold='validation', reshape=((1,) + dataset.space[0].shape,
                                            (1,) + dataset.space[1].shape,
                                            (1,) + dataset.space[1].shape))
    else:
        if cfg.data.test_data:
            dataset_test = get_test_data(cfg.data.name, cfg.data)
        else:
            dataset_test = dataset.create_torch_dataset(
                fold='test', reshape=((1,) + dataset.space[0].shape,
                                      (1,) + dataset.space[1].shape,
                                      (1,) + dataset.space[1].shape))

    ray_trafo = {'ray_trafo_module': ray_trafos['ray_trafo_module'],
                 'reco_space': dataset.space[1],
                 'observation_space': dataset.space[0]
                }

    '''Jacobian approx. with pre-sigmoid network's output'''
    if cfg.mdl.arch.use_sigmoid:
        cfg.mdl.arch.use_sigmoid = False
        cfg.mdl.normalize_by_stats = False

    reconstructor = DeepImagePriorReconstructor(**ray_trafo, cfg=cfg.mdl)
    '''Exclude GroupNorm & Scale_in/_out params from spectral analsyis & their biases'''
    skip_layers = list_norm_layers(reconstructor.model)
    if skip_layers is not None:
        cfg.spct.skip_layers += skip_layers # adding group norms. layers 

    store_device = reconstructor.device

    if cfg.mdl.torch_manual_seed:
        torch.random.manual_seed(cfg.mdl.torch_manual_seed)

    input = unpack_loader(dataset_test, store_device)
    if cfg.mdl.recon_from_randn:
        reconstructor.init_model()  # to advance the random seed like in DeepImagePriorReconstructor.reconstruct
        input = 0.1 * torch.randn(*input.shape).to(store_device)
    
    params = extract_params(reconstructor, cfg.spct)
    params = params.detach().cpu().numpy()
    s, v = randomised_SVD_jacobian(input, reconstructor.model, ray_trafo['ray_trafo_module'], 
                                    cfg, return_on_cpu=True)
                                    
    spct_data = {'filename': cfg.spct.filename, 
                'values':  s.numpy(), 
                'vectors': v.numpy()[:cfg.spct.n_projs_to_be_stored],
                'params': params}
    
    np.savez('spct_data', **spct_data)
    
if __name__ == '__main__':
    coordinator()
