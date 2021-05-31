# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from inspect import Parameter

def get_unet_model_3D(in_ch=1, out_ch=1, scales=6,
                   channels=[128, 128, 128, 128, 128, 128], use_sigmoid=True,
                   use_norm=True):
    skip_channels = [0, 0, 0, 0, 4, 4]
    return UNet3D(in_ch=in_ch, out_ch=out_ch, channels=channels[:scales],
                skip_channels=skip_channels, use_sigmoid=use_sigmoid,
                use_norm=use_norm)

def load_learned_unet2d(model, source):

    own_state = model.state_dict()
    state_dict_source = source.state_dict()
    for name, param in state_dict_source.items():
        if isinstance(param, Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        if own_state[name].shape == param.data.shape:
            own_state[name].copy_(param)
        elif own_state[name].shape != param.data.shape:
            K = param.shape[-1]
            aug_param = param.unsqueeze(dim=-1).repeat(1, 1, 1, 1, K)
            own_state[name].copy_(aug_param)
        else:
            KeyError

class UNet3D(nn.Module):
    def __init__(self, in_ch, out_ch, channels, skip_channels,
                 use_sigmoid=True, use_norm=True):
        super(UNet3D, self).__init__()
        assert (len(channels) == len(skip_channels))
        self.scales = len(channels)
        self.use_sigmoid = use_sigmoid
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.inc = InBlock(in_ch, channels[0], use_norm=use_norm)
        for i in range(1, self.scales):
            self.down.append(DownBlock(in_ch=channels[i - 1],
                                       out_ch=channels[i],
                                       use_norm=use_norm))
        for i in range(1, self.scales):
            self.up.append(UpBlock(in_ch=channels[-i],
                                   out_ch=channels[-i - 1],
                                   skip_ch=skip_channels[-i],
                                   use_norm=use_norm))
        self.outc = OutBlock(in_ch=channels[0],
                             out_ch=out_ch)

    def forward(self, x0):
        xs = [self.inc(x0), ]
        for i in range(self.scales - 1):
            xs.append(self.down[i](xs[-1]))
        x = xs[-1]
        for i in range(self.scales - 1):
            x = self.up[i](x, xs[-2 - i])
        return torch.sigmoid(self.outc(x)) if self.use_sigmoid else self.outc(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, num_groups=4, use_norm=True):
        super(DownBlock, self).__init__()
        to_pad = int((kernel_size - 1) / 2)
        if use_norm:
            self.conv = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size,
                          stride=2, padding=to_pad),
                nn.GroupNorm(num_channels=out_ch, num_groups=num_groups),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv3d(out_ch, out_ch, kernel_size,
                          stride=1, padding=to_pad),
                nn.GroupNorm(num_channels=out_ch, num_groups=num_groups),
                nn.LeakyReLU(0.2, inplace=True))
        else:
            self.conv = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size,
                          stride=2, padding=to_pad),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv3d(out_ch, out_ch, kernel_size,
                          stride=1, padding=to_pad),
                nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class InBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, num_groups=2, use_norm=True):
        super(InBlock, self).__init__()
        to_pad = int((kernel_size - 1) / 2)
        if use_norm:
            self.conv = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size,
                          stride=1, padding=to_pad),
                nn.GroupNorm(num_channels=out_ch, num_groups=num_groups),
                nn.LeakyReLU(0.2, inplace=True))
        else:
            self.conv = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size,
                          stride=1, padding=to_pad),
                nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch=4, kernel_size=3, num_groups=2, use_norm=True):
        super(UpBlock, self).__init__()
        to_pad = int((kernel_size - 1) / 2)
        self.skip = skip_ch > 0
        if skip_ch == 0:
            skip_ch = 1
        if use_norm:
            self.conv = nn.Sequential(
                nn.GroupNorm(num_channels=in_ch + skip_ch, num_groups=1), #LayerNorm
                nn.Conv3d(in_ch + skip_ch, out_ch, kernel_size, stride=1,
                          padding=to_pad),
                nn.GroupNorm(num_channels=out_ch, num_groups=num_groups),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv3d(out_ch, out_ch, kernel_size,
                          stride=1, padding=to_pad),
                nn.GroupNorm(num_channels=out_ch, num_groups=num_groups),
                nn.LeakyReLU(0.2, inplace=True))
        else:
            self.conv = nn.Sequential(
                nn.Conv3d(in_ch + skip_ch, out_ch, kernel_size, stride=1,
                          padding=to_pad),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv3d(out_ch, out_ch, kernel_size,
                          stride=1, padding=to_pad),
                nn.LeakyReLU(0.2, inplace=True))

        if use_norm:
            self.skip_conv = nn.Sequential(
                nn.Conv3d(out_ch, skip_ch, kernel_size=1, stride=1),
                nn.GroupNorm(num_channels=skip_ch, num_groups=1), #LayerNorm
                nn.LeakyReLU(0.2, inplace=True))
        else:
            self.skip_conv = nn.Sequential(
                nn.Conv3d(out_ch, skip_ch, kernel_size=1, stride=1),
                nn.LeakyReLU(0.2, inplace=True))

        self.up = nn.Upsample(scale_factor=2, mode='trilinear',
                              align_corners=True)
        self.concat = Concat()

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x2 = self.skip_conv(x2)
        if not self.skip:
            x2 = x2 * 0
        x = self.concat(x1, x2)
        x = self.conv(x)
        return x


class Concat(nn.Module):
    def __init__(self):
        super(Concat, self).__init__()

    def forward(self, *inputs):
        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]

        if (np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and
                np.all(np.array(inputs_shapes3) == min(inputs_shapes3))):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)

            inputs_ = []
            for inp in inputs:
                diff2 = (inp.size(2) - target_shape2) // 2
                diff3 = (inp.size(3) - target_shape3) // 2
                inputs_.append(inp[:, :, diff2: diff2 + target_shape2,
                                   diff3:diff3 + target_shape3])
        return torch.cat(inputs_, dim=1)


class OutBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutBlock, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv(x)
        return x

    def __len__(self):
        return len(self._modules)
