import os
import io
import numpy as np
from imageio import imwrite
import matplotlib.pyplot as plt
from tqdm import tqdm
from math import ceil
from PIL import Image

# after running this script, combine the images in OUT_PATH with:
# ffmpeg -r 30 -start_number 0 -i iter%05d.png -c:v libx264 -pix_fmt yuv420p -r 30 out_filename.mp4

PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# OUT_PATH = '/localdata/edip_noise_vs_dip_noise_iterates_walnut/'

# titles = [
#     'EDIP (noise)\nEllipses-Walnut',
#     'EDIP (noise)\nBrain-Walnut',
#     'DIP (noise)\nWalnut',
# ]
# iterates_paths = [
#     'multirun/2021-10-12/15-21-35/0/results/pretraining/iterates.npz',
#     'multirun/2021-10-12/22-48-15/0/results/pretraining/iterates.npz',
#     'multirun/2021-10-12/22-49-04/0/results/nopretrainingnoise/iterates.npz',
# ]

# iterates_iters_slice = slice(None)

# scaling_fct = 14.  # implicit_scaling_except_for_test_data

# SAMPLE = 0  # test sample (there is only one)

# IMAGE_SIZE = 501
# UPSCALE_IMAGE = 1
# IMAGE_SEP = 10
# TITLE_HEIGHT = 70
# SUP_TITLE_HEIGHT = 40

# DPI = 192

OUT_PATH = '/localdata/edip_noise_vs_dip_noise_iterates_lotus_20/'

titles = [
    'EDIP (noise)',
    'DIP (noise)',
]
iterates_paths = [
    'multirun/2021-10-13/23-41-41/0/results/pretraining/iterates.npz',
    'multirun/2021-10-14/21-05-40/0/results/nopretrainingnoise/iterates.npz',
]

iterates_iters_slice = slice(None)

scaling_fct = 41.  # implicit_scaling_except_for_test_data

SAMPLE = 0  # test sample (there is only one)

IMAGE_SIZE = 128
UPSCALE_IMAGE = 4
IMAGE_SEP = 10
TITLE_HEIGHT = 35
SUP_TITLE_HEIGHT = 40

DPI = 192


image_size = IMAGE_SIZE * UPSCALE_IMAGE

os.makedirs(OUT_PATH, exist_ok=True)

num_images = len(iterates_paths)

iterates_npz_list = [np.load(os.path.join(PATH, p)) for p in iterates_paths]
iterates_list = [npz['iterates'] for npz in iterates_npz_list]
iterates_iters_list = [npz['iterates_iters'] for npz in iterates_npz_list]

iterates_iters = iterates_iters_list[0]
iterates_iters_selected = iterates_iters[iterates_iters_slice]
assert all(
        np.array_equal(it_iters[iterates_iters_slice], iterates_iters_selected)
        for it_iters in iterates_iters_list)

SHAPE = (ceil((image_size + TITLE_HEIGHT + SUP_TITLE_HEIGHT) / 2) * 2,
         ceil((image_size * num_images + IMAGE_SEP * (num_images - 1)) / 2) * 2)

fig = plt.figure(figsize=(SHAPE[1] / DPI, SHAPE[0] / DPI), dpi=DPI)

for iterates_iter in tqdm(iterates_iters_selected):

    fig.clear()
    for i, title in enumerate(titles):
        plt.figtext(image_size / 2 + (image_size + IMAGE_SEP) * i,
                    SHAPE[0] - SUP_TITLE_HEIGHT - TITLE_HEIGHT / 2,
                    title,
                    size=plt.rcParams['axes.titlesize'],
                    weight=plt.rcParams['axes.titleweight'],
                    horizontalalignment='center', verticalalignment='center',
                    transform=None, figure=fig)
        plt.figtext(SHAPE[1] / 2, SHAPE[0],
                    'Iteration {:05d}'.format(iterates_iter),
                    size=plt.rcParams['figure.titlesize'],
                    weight=plt.rcParams['figure.titleweight'],
                    horizontalalignment='center', verticalalignment='top',
                    transform=None, figure=fig)

    with io.BytesIO() as io_buf:
        fig.savefig(io_buf, format='raw', dpi=DPI)
        io_buf.seek(0)
        img = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                         (int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
        img = np.array(Image.fromarray(img).convert('L'))
        if img.shape[0] < SHAPE[0]:
            img = np.pad(img, ((0, 1), (0, 0)), constant_values=1.)
        elif img.shape[0] > SHAPE[0]:
            img = img[:-1, :]
        if img.shape[1] < SHAPE[1]:
            img = np.pad(img, ((0, 0), (0, 1)), constant_values=1.)
        elif img.shape[1] > SHAPE[1]:
            img = img[:, :-1]

    for i, iterates in enumerate(iterates_list):
        i0 = SHAPE[0] - image_size
        i1 = ((image_size + IMAGE_SEP) * i
              if i < num_images - 1 else
              SHAPE[1] - image_size)
        iterate = (
                iterates[iterates_iter][SAMPLE]
                .repeat(UPSCALE_IMAGE, axis=0).repeat(UPSCALE_IMAGE, axis=1))
        img[i0:i0+image_size, i1:i1+image_size] = np.clip(
                iterate.T * scaling_fct * 255., 0., 255.)

    out_filepath = os.path.join(OUT_PATH, 'iter{:05d}.png'.format(iterates_iter))
    imwrite(out_filepath, np.array(img))
