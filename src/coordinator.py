import hydra
import os
import h5py
import numpy as np
from omegaconf import DictConfig
from dataset import get_standard_dataset, get_lotus_data
from torch.utils.data import DataLoader
from deep_image_prior import DeepImagePriorReconstructor
from pre_training import Trainer
from copy import deepcopy

@hydra.main(config_path='cfgs', config_name='config')
def coordinator(cfg : DictConfig) -> None:

    dataset, ray_trafo = get_standard_dataset(cfg.data.name, cfg.data)
    if cfg.data.name == 'ellipses':
        dataset_test = dataset.create_torch_dataset(
            fold='test', reshape=((1,) + dataset.space[0].shape,
                                  (1,) + dataset.space[1].shape,
                                  (1,) + dataset.space[1].shape))

        dataloader = DataLoader(dataset_test, batch_size=1, num_workers=0,
                                shuffle=True, pin_memory=True)
    elif cfg.data.name == 'ellipses_lotus':
        dataloader, ray_trafo = get_lotus_data(cfg.data)
    else:
        raise KeyError

    ray_trafo = {'ray_trafo': ray_trafo,
                 'reco_space': dataset.space[1],
                 'observation_space': dataset.space[0]
                 }

    reconstructor = DeepImagePriorReconstructor(**ray_trafo, cfg=cfg.mdl,
                                                data_cfg=cfg.data)
    model = deepcopy(reconstructor.model)
    if cfg.pretraining:
        Trainer(model=model, cfg=cfg.trn).train(dataset)

    if not os.path.exists(cfg.save_reconstruction_path):
        os.makedirs(cfg.save_reconstruction_path)

    filename = os.path.join(cfg.save_reconstruction_path,'recos.hdf5')
    file = h5py.File(filename, 'w')
    dataset = file.create_dataset('recos', shape=(1, )
        + (128, 128), maxshape=(1, ) + (128, 128), dtype=np.float32, chunks=True)

    for i, (noisy_obs, fbp, *gt) in enumerate(dataloader):
        reco = reconstructor.reconstruct(noisy_obs.float(), fbp, gt)
        dataset[i] = reco

if __name__ == '__main__':
    coordinator()
