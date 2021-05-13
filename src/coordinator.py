import hydra
import os
from omegaconf import DictConfig
from dataset import get_standard_dataset
from torch.utils.data import DataLoader
from deep_image_prior import DeepImagePriorReconstructor
from pre_training import Trainer
from copy import deepcopy

@hydra.main(config_path='standard_ellipses.yaml')
def coordinator(cfg : DictConfig) -> None:

    dataset, ray_trafo = get_standard_dataset('ellipses', cfg.dataset_specs)
    dataset_train = dataset.create_torch_dataset(
        fold='test', reshape=((1,) + dataset.space[0].shape,
                              (1,) + dataset.space[1].shape,
                              (1,) + dataset.space[1].shape))

    dataloader = DataLoader(dataset_train, batch_size=1, num_workers=0,
                            shuffle=True, pin_memory=True)

    ray_trafo = {'ray_trafo': ray_trafo,
                 'reco_space': ray_trafo.domain,
                 'observation_space': ray_trafo.range
                 }

    reconstructor = DeepImagePriorReconstructor(**ray_trafo, arch_cfg=cfg.recon_specs.arch)
    model = deepcopy(reconstructor.model)
    Trainer(model=model, configs=cfg.trainer_specs).train(dataset)

    for i, (noisy_obs, fbp, gt) in enumerate(dataloader):
        reco = reconstructor.reconstruct(cfg.recon_specs, noisy_obs.float(), fbp, gt)

if __name__ == '__main__':
    coordinator()
