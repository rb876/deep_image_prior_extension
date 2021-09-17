import os
import hydra
import torch
import glob
from os import walk
from omegaconf import DictConfig
from dataset import get_standard_dataset, get_test_data, get_validation_data
from deep_image_prior import DeepImagePriorReconstructor
from utils import randomised_SVD_jacobian, get_params, singular_values_comparison_plot, singular_vecors_comparison_plot, projected_diff_params_plot, get_fbp_from_loader

ckp_filename = 'params_iters4500.pt'

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

    skip_layers = ['scale_in.scale_layer.weight',
        'scale_in.scale_layer.bias', 'scale_out.scale_layer.weight',
        'scale_out.scale_layer.bias']

    reconstructor = DeepImagePriorReconstructor(**ray_trafo, cfg=cfg.mdl)
    store_device = reconstructor.device

    if cfg.mdl.torch_manual_seed:
        torch.random.manual_seed(cfg.mdl.torch_manual_seed)

    input = get_fbp_from_loader(dataset_test, store_device)
    if cfg.mdl.recon_from_randn:
        input = 0.1 * torch.randn(*input.shape).to(store_device)
    
    params_start = get_params(reconstructor, cfg, 'params_iters0.pt', skip_layers)
    s_approx_start, v_approx_start = randomised_SVD_jacobian(input,
            reconstructor.model, cfg.lin.n_projs, ray_trafo['ray_trafo_module'], skip_layers, cfg.mdl, 
            return_on_cpu=True)

    params_end = get_params(reconstructor, cfg, ckp_filename, skip_layers)
    s_approx_end, v_approx_end = randomised_SVD_jacobian(input,
            reconstructor.model, cfg.lin.n_projs, ray_trafo['ray_trafo_module'], skip_layers, cfg.mdl,
            return_on_cpu=True)

    singular_values_comparison_plot(s_approx_start.numpy(), 'pretrained', s_approx_end.numpy(), 'adapted')
    singular_vecors_comparison_plot(v_approx_start.numpy(), v_approx_end.numpy())
                        
    delta_params = (params_start - params_end).detach().cpu()
    acc_projected_params_error = torch.zeros_like(params_start).cpu()
    for i in range(cfg.lin.n_projs):
        acc_projected_params_error += torch.dot(delta_params, v_approx_start[i, :]) * v_approx_start[i, :]

    projected_diff_params_plot(delta_params, v_approx_start)

    print('L2 norm of diff params: {}'.format((delta_params**2).sum()))
    print('L2 norm of projected diff params: {}'.format((acc_projected_params_error**2).sum()))
    print('L2 norm ratio: {}'.format((acc_projected_params_error**2).sum()/(delta_params**2).sum()))
    print('n_projs / n_params: {}'.format(cfg.lin.n_projs/len(params_start)))
    
if __name__ == '__main__':
    coordinator()
