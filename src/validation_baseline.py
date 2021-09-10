import hydra
import os
import os.path
import json
from omegaconf import DictConfig, OmegaConf
from dataset import get_validation_data, get_standard_dataset
from validation import validate_model
from copy import deepcopy

@hydra.main(config_path='cfgs', config_name='config')
def coordinator(cfg : DictConfig) -> None:

    assert not cfg.mdl.load_pretrain_model, \
    'load_pretrain_model is True, assertion failed'

    dataset, ray_trafos = get_standard_dataset(cfg.data.name, cfg.data)
    ray_trafo = {'ray_trafo_module': ray_trafos['ray_trafo_module'],
                 'reco_space': dataset.space[1],
                 'observation_space': dataset.space[0]
                 }
    val_dataset = get_validation_data(cfg.data.name, cfg.data)

    infos = {}
    log_path_base = cfg.mdl.log_path
    seed = cfg.mdl.torch_manual_seed
    cfg_mdl_val = deepcopy(cfg.mdl)

    for k, v in cfg.val.mdl_overrides.items():
        OmegaConf.update(cfg_mdl_val, k, v, merge=False)

    _, info = validate_model(
            val_dataset=val_dataset, ray_trafo=ray_trafo,
            val_sub_path_mdl='baseline',
            baseline_psnr_steady='own_PSNR_steady', seed=seed,
            log_path_base=log_path_base,
            cfg=cfg, cfg_mdl_val=cfg_mdl_val)

    infos['baseline'] = info

    os.makedirs(os.path.dirname(os.path.abspath(cfg.val.results_filename)), exist_ok=True)
    with open(cfg.val.results_filename, 'w') as f:
        json.dump(infos, f, indent=1)

if __name__ == '__main__':
    coordinator()
