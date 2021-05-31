import torch
import torch.nn as nn

N = 1; C = 32; D = 128; H = 128; W = 128
input2d = torch.ones((N, C, H, W))
conv2d = nn.Conv2d(32, 128, 3, stride=1, padding=1)
torch.nn.init.xavier_uniform(conv2d.weight)
conv2d.bias.data.fill_(0)

def init_weights(layer, source):
    if type(source) == nn.Conv2d:
        init = source.weight.unsqueeze(dim=-1).repeat(1, 1, 1, 1, 3)
        if type(layer) == nn.Conv3d:
            layer.weight.data = init
            layer.bias.data.fill_(0.001)
        else:
            raise KeyError
    else:
        raise KeyError

input3d = torch.ones((N, C, D, H, W))
conv3d = nn.Conv3d(32, 128, 3, stride=1, padding=1)
init_weights(conv3d, conv2d)
output = conv3d(input3d)

from deep_image_prior import get_unet_model
from deep_image_prior import get_unet_model_3D, load_learned_unet2d

silly_input = torch.rand((1, 1, 64, 64, 64))
model_3D = get_unet_model_3D()
model_2D = get_unet_model(scales=6, channels=[128, 128, 128, 128, 128, 128])
load_learned_unet2d(model_3D, model_2D)
silly_output = model_3D(silly_input)
