import json
from itertools import islice
import hydra
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig
from dataset.standard import get_standard_dataset

def compute_dataset_stats_fbp_gt(dataset, max_samples=None):
    """
    Compute means and standard deviations of the FBP and ground truth images
    in the ``'train'`` fold of a dataset.
    """
    # Adapted from: https://github.com/ahendriksen/msd_pytorch/blob/162823c502701f5eedf1abcd56e137f8447a72ef/msd_pytorch/msd_model.py#L95
    mean_fbp = 0.
    mean_gt = 0.
    square_fbp = 0.
    square_gt = 0.
    n = dataset.get_len('train')
    if max_samples is not None:
        n = min(max_samples, n)
    for obs, fbp, gt in tqdm(islice(dataset.generator('train'), max_samples),
                             total=n, desc='computing dataset stats'):
        mean_fbp += np.mean(fbp)
        mean_gt += np.mean(gt)
        square_fbp += np.mean(np.square(fbp))
        square_gt += np.mean(np.square(gt))
    mean_fbp /= n
    mean_gt /= n
    square_fbp /= n
    square_gt /= n
    std_fbp = np.sqrt(square_fbp - mean_fbp**2)
    std_gt = np.sqrt(square_gt - mean_gt**2)
    stats = {'mean_fbp': mean_fbp,
             'std_fbp': std_fbp,
             'mean_gt': mean_gt,
             'std_gt': std_gt}
    return stats

@hydra.main(config_path='../cfgs', config_name='config')
def compute_mean(cfg : DictConfig) -> None:
    dataset, ray_trafo = get_standard_dataset(cfg.data.name, cfg.data, return_ray_trafo_torch_module=False)
    stats = compute_dataset_stats_fbp_gt(dataset)
    print(stats)
    with open('stats_standard_{}.json'.format(cfg.data.name), 'w') as f:
        json.dump(stats, f, indent=1)

if __name__ == '__main__':
    compute_mean()
