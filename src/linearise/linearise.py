import hydra
import torch
from omegaconf import DictConfig
from dataset import get_standard_dataset, get_test_data, get_validation_data
from deep_image_prior import DeepImagePriorReconstructor
from utils import randomised_SVD_jacobian, compute_jacobian_single_batch

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

    if cfg.torch_manual_seed_pretrain_init_model:
        torch.random.manual_seed(cfg.torch_manual_seed_pretrain_init_model)

    reconstructor = DeepImagePriorReconstructor(**ray_trafo, cfg=cfg.mdl)
    # TODO: 
    input = [el[1] for el in dataset_test][0].unsqueeze(dim=0).to(reconstructor.device)
    s_approx, v_approx = randomised_SVD_jacobian(input,
            reconstructor.model, cfg.linearise.n_projs, ['scale_in.scale_layer.weight',
            'scale_in.scale_layer.bias', 'scale_out.scale_layer.weight',
            'scale_out.scale_layer.bias'], cfg.mdl)

if __name__ == '__main__':
    coordinator()
