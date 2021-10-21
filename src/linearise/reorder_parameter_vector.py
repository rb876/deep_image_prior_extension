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

def get_named_parameters_sorted(reconstructor, cfg):
    named_parameters = [(k, v) for k, v in reconstructor.model.named_parameters()
                        if k not in cfg.skip_layers]
    named_parameters_sorted = sorted(named_parameters,
            key=lambda item: ['inc', 'down', 'up', 'outc'].index(
                    item[0].split('.')[0]))
    return named_parameters_sorted

def extract_params_sorted(reconstructor, cfg):

    state_dict = torch.load(os.path.join(cfg.path_to_checkpoints,
                    cfg.filename))
    reconstructor.model.load_state_dict(state_dict)
    named_parameters = dict(reconstructor.model.named_parameters())
    named_parameters_sorted = get_named_parameters_sorted(reconstructor, cfg)
    params = \
        parameters_to_vector(named_parameters_sorted,
        cfg.skip_layers)
    return params

def get_slices_for_params_in_vector(parameters, cfg):
    # Pointer for slicing the vector for each parameter
    slices = {}
    pointer = 0
    for name, param in parameters:
        if name not in cfg.skip_layers:
            num_param = param.numel()
            slices[name] = slice(pointer, pointer + num_param)

            # Increment the pointer
            pointer += num_param

    return slices

def reorder_params_vector(vec, parameters, parameters_sorted, cfg):
    parameters_list = list(parameters)
    slices_in_orig = get_slices_for_params_in_vector(parameters_list, cfg)
    slices_in_sorted = get_slices_for_params_in_vector(parameters_sorted, cfg)

    vec_sorted = np.zeros_like(vec)
    for name, _ in parameters_list:
        if name not in cfg.skip_layers:
            vec_sorted[slices_in_sorted[name]] = vec[slices_in_orig[name]]

    return vec_sorted

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

    params_sorted = extract_params_sorted(reconstructor, cfg.spct)
    params_sorted = params_sorted.detach().cpu().numpy()
    params_sorted2 = reorder_params_vector(
            params,
            reconstructor.model.named_parameters(),
            get_named_parameters_sorted(reconstructor, cfg.spct),
            cfg.spct)

    print(np.sum(np.abs(params_sorted2-params_sorted)))

    reorder_idx = reorder_params_vector(
            np.arange(len(params)),
            reconstructor.model.named_parameters(),
            get_named_parameters_sorted(reconstructor, cfg.spct),
            cfg.spct)
    print(reorder_idx.shape)
    np.save('reorder_idx.npy', reorder_idx)

if __name__ == '__main__':
    coordinator()
