import os
import io
import numpy as np
from imageio import imwrite
import matplotlib.pyplot as plt
from tqdm import tqdm
from math import ceil
from PIL import Image

PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
FIG_PATH = os.path.dirname(__file__)

save_fig = True

# filename = 'iterates_ellipses_walnut_120.pdf'

# titles = [
#     'EDIP (FBP)',
#     'EDIP (noise)',  # [warm-up]',
#     'DIP (noise)',
# ]
# iterates_paths = [
#     'multirun/2021-10-28/04-32-17/0/results/pretrainingonlyfbp/iterates.npz',
#     'multirun/2021-11-05/16-47-01/0/results/pretraining/iterates.npz',
#     'multirun/2021-10-28/04-30-33/0/results/nopretrainingnoise/iterates.npz',
# ]

# iterates_iters_inds = [100, 150, 250]

# scaling_fct = 14.  # implicit_scaling_except_for_test_data

# SAMPLE = 0  # test sample (there is only one)

# IMAGE_SIZE = 501
# IMAGE_SEP = 10
# TITLE_HEIGHT = 35
# SUP_TITLE_HEIGHT = 40

# FIGSIZE = (6, 6)
# gridspec_kw = {'wspace': 0.01, 'hspace': 0.01}

# grid_orientation_iterations = 'vertical'

filename = 'edip_noise_vs_dip_noise_iterates_lotus_20_part0.pdf'
# filename = 'edip_noise_vs_dip_noise_iterates_lotus_20_part1.pdf'

titles = [
    'EDIP (noise)',
    'DIP (noise)',
]
iterates_paths = [
    'multirun/2021-10-22/14-40-42/0/results/pretraining/iterates.npz',
    'multirun/2021-10-22/14-32-46/0/results/nopretrainingnoise/iterates.npz',
]

iterates_iters_inds = [2, 25, 100]  # part0
# iterates_iters_inds = [200, 500, 1000]  # part1

scaling_fct = 41.  # implicit_scaling_except_for_test_data

SAMPLE = 0  # test sample (there is only one)

IMAGE_SIZE = 128
IMAGE_SEP = 10
TITLE_HEIGHT = 35
SUP_TITLE_HEIGHT = 40

FIGSIZE = (4, 6)
gridspec_kw = {'wspace': 0.01, 'hspace': 0.01}

grid_orientation_iterations = 'vertical'


os.makedirs(FIG_PATH, exist_ok=True)

num_images = len(iterates_paths)

iterates_npz_list = [np.load(os.path.join(PATH, p)) for p in iterates_paths]
iterates_list = [npz['iterates'] for npz in iterates_npz_list]
iterates_iters_list = [npz['iterates_iters'] for npz in iterates_npz_list]

iterates_iters = iterates_iters_list[0]
iterates_iters_selected = iterates_iters[iterates_iters_inds]
assert all(
        np.array_equal(it_iters[iterates_iters_inds], iterates_iters_selected)
        for it_iters in iterates_iters_list)

nrow, ncol = (
        (num_images, len(iterates_iters_inds))
        if grid_orientation_iterations == 'horizontal' else
        (len(iterates_iters_inds), num_images))

fig, ax = plt.subplots(nrow, ncol, figsize=FIGSIZE, gridspec_kw=gridspec_kw)

if grid_orientation_iterations == 'horizontal':
    for i, title in enumerate(titles):
        ax[i, 0].set_ylabel(title)
    for j, iterates_iter in enumerate(iterates_iters_selected):
        ax[0, j].set_title('Iteration {:d}'.format(iterates_iter))
else:
    for i, iterates_iter in enumerate(iterates_iters_selected):
        ax[i, 0].set_ylabel('Iteration {:d}'.format(iterates_iter))
    for j, title in enumerate(titles):
        ax[0, j].set_title(title)

for i_iter, iterates_iter in enumerate(iterates_iters_selected):
    for i_image, iterates in enumerate(iterates_list):
        ax_ = (ax[i_image, i_iter]
               if grid_orientation_iterations == 'horizontal' else
               ax[i_iter, i_image])
        iterate = iterates[iterates_iter][SAMPLE]
        ax_.imshow(iterate.T * scaling_fct, vmin=0., vmax=1., cmap='gray',
                   interpolation=None)
        ax_.set_xticks([])
        ax_.set_yticks([])
        for spine in ax_.spines.values():
            spine.set_visible(False)

if save_fig:
    fig.savefig(os.path.join(FIG_PATH, filename), bbox_inches='tight',
                pad_inches=0.)
