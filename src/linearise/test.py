import hydra
import torch
from omegaconf import DictConfig
from dataset import get_standard_dataset, get_test_data, get_validation_data
from deep_image_prior import DeepImagePriorReconstructor
from utils import randomised_SVD_jacobian, compute_jacobian_single_batch
from plot_spectral_data import singular_values_plot, singular_vectors_plot
from torch_utils import list_norm_layers

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
    cfg.mdl.arch.use_sigmoid = False
    cfg.mdl.normalize_by_stats = False

    torch.random.manual_seed(10)
    reconstructor = DeepImagePriorReconstructor(**ray_trafo, cfg=cfg.mdl)
    
    '''Exclude GroupNorm & scale_in/_out params from spectral analsyis & their sbiases'''
    skip_layers = list_norm_layers(reconstructor.model)
    if skip_layers is not None:
        cfg.spct.skip_layers += skip_layers # adding group norms. layers 

    out_dim = 16**2
    dummy_input = torch.ones((1, 1, 16, 16)).to(reconstructor.device)

    s_approx, v_approxh = randomised_SVD_jacobian(dummy_input,
            reconstructor.model, None, cfg, return_on_cpu=True)
    v_approx = v_approxh
    jac = compute_jacobian_single_batch(dummy_input, reconstructor.model, out_dim,
            cfg, return_on_cpu=True)
    _, s, vh = torch.svd(jac)
    v = vh.t()

    for i in range(v_approx.shape[0]):
        fct = torch.dot(v_approx[i, :], v[i, :]).sign()
        v_approx[i, :] *= fct

    singular_values_plot([s.numpy(), s_approx.numpy()], ['Exact.','Approx.'],  ['#EC2215', '#3D78B2'], ['-', '-'])
    singular_vectors_plot(v_approx.numpy(), v.numpy(), plot_first_k=16, labels=['Exact.','Approx.'], filename='test_vectors.pdf')

if __name__ == '__main__':
    coordinator()
