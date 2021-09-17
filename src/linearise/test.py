import hydra
import torch
from omegaconf import DictConfig
from dataset import get_standard_dataset, get_test_data, get_validation_data
from deep_image_prior import DeepImagePriorReconstructor
from utils import randomised_SVD_jacobian, compute_jacobian_single_batch, singular_values_comparison_plot, singular_vecors_comparison_plot

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

    # fix network to a small architecture
    cfg.mdl.arch.scales = 2
    cfg.mdl.arch.channels = [32, 32]
    cfg.mdl.arch.skip_channels = [0, 4]

    reconstructor = DeepImagePriorReconstructor(**ray_trafo, cfg=cfg.mdl)
    out_dim = 16**2
    dummy_input = torch.ones((1, 1, 16, 16)).to(reconstructor.device)

    s_approx, v_approxh = randomised_SVD_jacobian(dummy_input,
            reconstructor.model, 30, None, ['scale_in.scale_layer.weight',
            'scale_in.scale_layer.bias', 'scale_out.scale_layer.weight',
            'scale_out.scale_layer.bias'], cfg.mdl, return_on_cpu=True)
    v_approx = v_approxh
    jac = compute_jacobian_single_batch(dummy_input, reconstructor.model, out_dim, ['scale_in.scale_layer.weight',
            'scale_in.scale_layer.bias', 'scale_out.scale_layer.weight',
            'scale_out.scale_layer.bias'], cfg.mdl, return_on_cpu=True)
    _, s, vh = torch.svd(jac)
    v = vh.t()
    import numpy as np
    for i in range(v_approx.shape[0]):
        fct = torch.dot(v_approx[i, :], v[i, :]).sign()
        v_approx[i, :] *= fct

    singular_values_comparison_plot(s.numpy(), 'exact', s_approx.numpy(), 'approx')
    singular_vecors_comparison_plot(v_approx.numpy(), v.numpy())

if __name__ == '__main__':
    coordinator()
