import hydra
import os
import numpy as np
from omegaconf import DictConfig, OmegaConf
from dataset import get_standard_dataset
import torch
from deep_image_prior import DeepImagePriorReconstructor
from pre_training import MetaTrainer
from copy import deepcopy


def get_maml_datasets(datasets_cfg_list):
    datasets = []
    for cfg in datasets_cfg_list:
        data_cfg = OmegaConf.load(os.path.join(hydra.utils.get_original_cwd(), cfg.base_cfg_filepath))
        for k, v in cfg.overrides.items():
            OmegaConf.update(data_cfg, k, v, merge=True)
        image_dataset_kwargs = OmegaConf.to_object(cfg.image_dataset_kwargs)
        dataset, _ = get_standard_dataset(data_cfg.name, data_cfg, **image_dataset_kwargs)
        datasets.append(dataset)
    return datasets


@hydra.main(config_path='cfgs', config_name='config')
def coordinator(cfg : DictConfig) -> None:

    dataset, ray_trafos = get_standard_dataset(cfg.data.name, cfg.data)

    assert cfg.trn.meta_trainer.datasets_cfg_list, "cfg.trn.meta_trainer.datasets_cfg_list appears to be empty"
    meta_datasets = get_maml_datasets(cfg.trn.meta_trainer.datasets_cfg_list)

    ray_trafo = {'ray_trafo_module': ray_trafos['ray_trafo_module'],
                 'reco_space': dataset.space[1],
                 'observation_space': dataset.space[0]
                 }

    if cfg.torch_manual_seed_pretrain_init_model:
        torch.random.manual_seed(cfg.torch_manual_seed_pretrain_init_model)

    reconstructor = DeepImagePriorReconstructor(**ray_trafo, cfg=cfg.mdl)
    model = deepcopy(reconstructor.model)
    if cfg.pretraining:
        assert not cfg.trn.add_randn_mask
        assert not cfg.trn.perform_swa
        assert not cfg.trn.use_adversarial_attacks
        assert not cfg.trn.use_mixed
        MetaTrainer(model=model,
                    cfg=cfg.trn).metatrain(meta_datasets)

if __name__ == '__main__':
    coordinator()
