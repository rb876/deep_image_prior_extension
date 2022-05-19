import os
import h5py
import hydra
import numpy as np
from omegaconf import DictConfig
from dataset import get_standard_dataset, get_test_data
from TVAdam import TVAdamReconstructor
from torch.utils.data import DataLoader
from pre_training import Trainer
from copy import deepcopy
from scipy.io import savemat

@hydra.main(config_path='../', config_name='baselines/tvadamconfg')
def coordinator(cfg : DictConfig) -> None:

    dataset, ray_trafos = get_standard_dataset(cfg.data.name, cfg.data)
    dataset_test = get_test_data(cfg.data.name, cfg.data)
    ray_trafo = {'ray_trafo_module': ray_trafos['ray_trafo_module'],
                 'reco_space': dataset.space[1],
                 'observation_space': dataset.space[0]
                 }
    reconstructor = TVAdamReconstructor(**ray_trafo, cfg=cfg.baselines)

    if not os.path.exists(cfg.baselines.save_reconstruction_path):
        os.makedirs(cfg.baselines.save_reconstruction_path)

    filename = os.path.join(cfg.baselines.save_reconstruction_path,'recos.hdf5')
    file = h5py.File(filename, 'w')
    dataset = file.create_dataset('recos', shape=(1, )
        + dataset.space[1].shape, maxshape=(1, ) + dataset.space[1].shape, dtype=np.float32, chunks=True)

    dataloader = DataLoader(dataset_test, batch_size=1, num_workers=0,
                            shuffle=True, pin_memory=True)

    for i, (noisy_obs, fbp, *gt) in enumerate(dataloader):
        gt = gt[0] if gt else None
        reco = reconstructor.reconstruct(noisy_obs.float(), fbp, ground_truth=gt, log=True)
        dataset[i] = reco

    # filename_mat = os.path.join(cfg.baselines.save_reconstruction_path,'TVGroundTruthLotus128.mat')
    # dict_mat = {'recon': reco.T, 'label': 'ground_truth'}
    # savemat(filename_mat, dict_mat)
if __name__ == '__main__':
    coordinator()
