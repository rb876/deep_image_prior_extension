import hydra
from omegaconf import DictConfig
from dataset import get_standard_dataset
from torch.utils.data import DataLoader
from deep_image_prior import DeepImagePriorReconstructor

@hydra.main(config_path='standard_ellipses.yaml')
def coordinator(cfg : DictConfig) -> None:

    dataset, ray_trafo = get_standard_dataset('ellipses', cfg)
    dataset_train = dataset.create_torch_dataset(
        fold='train', reshape=((1,) + dataset.space[0].shape,
                               (1,) + dataset.space[1].shape,
                               (1,) + dataset.space[1].shape))

    dataloader = DataLoader(
            dataset_train, batch_size=1,
            num_workers=0, shuffle=True,
            pin_memory=True)

    reconstructor = DeepImagePriorReconstructor(ray_trafo=ray_trafo)
    for i, (noisy_obs, fbp, gt) in enumerate(dataloader):
        reconstructor.reconstruct(cfg.reconstructor_specs, noisy_obs.float(), gt)

if __name__ == '__main__':
    coordinator()
