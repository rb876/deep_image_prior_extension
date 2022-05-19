# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from inspect import Parameter
from .approx_3d_conv import ApproxConv3d

def get_unet_model_3D(in_ch=1, out_ch=1, scales=6,
                   channels=[128, 128, 128, 128, 128, 128], down_channel_overrides=(), down_single_conv=False, use_sigmoid=True,
                   use_norm=True, out_kernel_size=1, pre_out_channels=(), pre_out_kernel_size=3, insert_res_blocks_before=(),
                   use_relu_out=False, approx_conv3d_at_scales=[],approx_conv3d_low_rank_dim=1):
    skip_channels = [0, 0, 0, 0, 4, 4]
    return UNet3D(in_ch=in_ch, out_ch=out_ch, channels=channels[:scales],
                down_channel_overrides=down_channel_overrides, down_single_conv=down_single_conv,
                skip_channels=skip_channels, use_sigmoid=use_sigmoid,
                use_norm=use_norm, out_kernel_size=out_kernel_size,
                pre_out_channels=pre_out_channels, pre_out_kernel_size=pre_out_kernel_size,
                insert_res_blocks_before=insert_res_blocks_before, use_relu_out=use_relu_out, 
                approx_conv3d_at_scales=approx_conv3d_at_scales, approx_conv3d_low_rank_dim=approx_conv3d_low_rank_dim)

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

def get_norm_layer(num_features, kind='group', num_groups=None):
    if kind == 'group':
        assert num_groups is not None
        norm_layer = nn.GroupNorm(num_groups=num_groups, num_channels=num_features)
    elif kind == 'batch':
        norm_layer = nn.BatchNorm3d(num_features)
    else:
        raise NotImplementedError
    return norm_layer

class UNet3D(nn.Module):
    def __init__(self, in_ch, out_ch, channels, skip_channels, down_channel_overrides=(), down_single_conv=False,
                 use_sigmoid=True, use_norm=True, out_kernel_size=1, pre_out_channels=(), pre_out_kernel_size=3,
                 insert_res_blocks_before=(), use_relu_out=False, approx_conv3d_at_scales=[], approx_conv3d_low_rank_dim=1):
        super(UNet3D, self).__init__()
        assert (len(channels) == len(skip_channels))
        self.scales = len(channels)
        self.use_sigmoid = use_sigmoid
        self.use_relu_out = use_relu_out
        assert not (self.use_sigmoid and self.use_relu_out)
        self.approx_conv3d_at_scales = approx_conv3d_at_scales
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        down_channel_overrides = down_channel_overrides + (None,) * (len(channels) - len(down_channel_overrides))
        down_channels = [c if c_override is None else c_override for c, c_override in zip(channels, down_channel_overrides)]
        self.inc = InBlock(in_ch, down_channels[0], use_norm=use_norm)
        for i in range(1, self.scales):
            self.down.append(DownBlock(in_ch=down_channels[i - 1],
                                       out_ch=down_channels[i],
                                       use_norm=use_norm,
                                       single_conv=down_single_conv, 
                                       use_approx_conv3d=i in self.approx_conv3d_at_scales, 
                                       approx_conv3d_low_rank_dim=approx_conv3d_low_rank_dim))
        for i in range(1, self.scales):
            self.up.append(UpBlock(in_ch=down_channels[-i] if i == 1 else channels[-i],
                                   out_ch=channels[-i - 1],
                                   skip_ch=skip_channels[-i],
                                   skip_in_ch=down_channels[-i - 1],
                                   use_norm=use_norm, 
                                   use_approx_conv3d=i in self.approx_conv3d_at_scales, 
                                   approx_conv3d_low_rank_dim=approx_conv3d_low_rank_dim))
        self.outc = OutBlock(in_ch=channels[0],
                             out_ch=out_ch, out_kernel_size=out_kernel_size,
                             pre_out_channels=pre_out_channels, pre_out_kernel_size=pre_out_kernel_size,
                             insert_res_blocks_before=insert_res_blocks_before, use_norm=use_norm)

    def forward(self, x0):
        xs = [self.inc(x0), ]
        for i in range(self.scales - 1):
            xs.append(self.down[i](xs[-1]))
        x = xs[-1]
        for i in range(self.scales - 1):
            x = self.up[i](x, xs[-2 - i])
        return (torch.nn.functional.relu(self.outc(x)) if self.use_relu_out else (
                torch.sigmoid(self.outc(x)) if self.use_sigmoid else self.outc(x)))


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, num_groups=4, use_norm=True, single_conv=False, use_approx_conv3d=False, approx_conv3d_low_rank_dim=1):
        super(DownBlock, self).__init__()
        to_pad = int((kernel_size - 1) / 2)
        norm_kind = use_norm if use_norm and isinstance(use_norm, str) else 'group'
        if single_conv:
            if use_norm:
                self.conv = nn.Sequential(
                    nn.Conv3d(in_ch, out_ch, kernel_size,
                            stride=2, padding=to_pad) if not use_approx_conv3d  else ApproxConv3d(
                            in_ch, out_ch, approx_conv3d_low_rank_dim, kernel_size, stride=2
                            ),
                    get_norm_layer(out_ch, kind=norm_kind, num_groups=num_groups),
                    nn.LeakyReLU(0.2, inplace=True))
            else:
                self.conv = nn.Sequential(
                    nn.Conv3d(in_ch, out_ch, kernel_size,
                            stride=2, padding=to_pad) if not use_approx_conv3d  else ApproxConv3d(
                            in_ch, out_ch, approx_conv3d_low_rank_dim, kernel_size, stride=2,
                            ),
                    nn.LeakyReLU(0.2, inplace=True))
        else:
            if use_norm:
                self.conv = nn.Sequential(
                    nn.Conv3d(in_ch, out_ch, kernel_size,
                            stride=2, padding=to_pad) if not use_approx_conv3d  else ApproxConv3d(
                            in_ch, out_ch, approx_conv3d_low_rank_dim, kernel_size, stride=2,
                            ),
                    get_norm_layer(out_ch, kind=norm_kind, num_groups=num_groups),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv3d(out_ch, out_ch, kernel_size,
                            stride=1, padding=to_pad) if not use_approx_conv3d  else ApproxConv3d(
                            out_ch, out_ch, approx_conv3d_low_rank_dim, kernel_size, stride=1,
                            ),
                    get_norm_layer(out_ch, kind=norm_kind, num_groups=num_groups),
                    nn.LeakyReLU(0.2, inplace=True))
            else:
                self.conv = nn.Sequential(
                    nn.Conv3d(in_ch, out_ch, kernel_size,
                            stride=2, padding=to_pad) if not use_approx_conv3d  else ApproxConv3d(
                            in_ch, out_ch, approx_conv3d_low_rank_dim, kernel_size, stride=2,
                            ),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv3d(out_ch, out_ch, kernel_size,
                            stride=1, padding=to_pad) if not use_approx_conv3d  else ApproxConv3d(
                            out_ch, out_ch, approx_conv3d_low_rank_dim, kernel_size, stride=1,
                            ),
                    nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class InBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, num_groups=2, use_norm=True):
        super(InBlock, self).__init__()
        to_pad = int((kernel_size - 1) / 2)
        norm_kind = use_norm if use_norm and isinstance(use_norm, str) else 'group'
        if use_norm:
            self.conv = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size,
                          stride=1, padding=to_pad),
                get_norm_layer(out_ch, kind=norm_kind, num_groups=num_groups),
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
    def __init__(self, in_ch, out_ch, skip_ch=4, skip_in_ch=None, kernel_size=3, num_groups=2, use_norm=True, use_approx_conv3d=False, approx_conv3d_low_rank_dim=1):
        super(UpBlock, self).__init__()
        to_pad = int((kernel_size - 1) / 2)
        norm_kind = use_norm if use_norm and isinstance(use_norm, str) else 'group'
        self.skip = skip_ch > 0
        skip_in_ch = out_ch if skip_in_ch is None else skip_in_ch
        if skip_ch == 0:
            skip_ch = 1
        if use_norm:
            self.conv = nn.Sequential(
                get_norm_layer(in_ch + skip_ch, kind=norm_kind, num_groups=1),  # LayerNorm if kind='group'
                nn.Conv3d(in_ch + skip_ch, out_ch, kernel_size,
                    stride=1, padding=to_pad) if not use_approx_conv3d  else ApproxConv3d(
                    in_ch + skip_ch, out_ch, approx_conv3d_low_rank_dim, kernel_size, stride=1,
                    ),
                get_norm_layer(out_ch, kind=norm_kind, num_groups=num_groups),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv3d(out_ch, out_ch, kernel_size,
                    stride=1, padding=to_pad) if not use_approx_conv3d  else ApproxConv3d(
                    out_ch, out_ch, approx_conv3d_low_rank_dim, kernel_size, stride=1,
                    ),
                get_norm_layer(out_ch, kind=norm_kind, num_groups=num_groups),
                nn.LeakyReLU(0.2, inplace=True))
        else:
            self.conv = nn.Sequential(
                nn.Conv3d(in_ch + skip_ch, out_ch, kernel_size,
                    stride=1, padding=to_pad) if not use_approx_conv3d  else ApproxConv3d(
                    in_ch + skip_ch, out_ch, kernel_size, approx_conv3d_low_rank_dim, stride=1,
                    ),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv3d(out_ch, out_ch, kernel_size,
                    stride=1, padding=to_pad) if not use_approx_conv3d  else ApproxConv3d(
                    out_ch, out_ch, approx_conv3d_low_rank_dim, kernel_size, stride=1,
                    ),
                nn.LeakyReLU(0.2, inplace=True))

        if use_norm:
            self.skip_conv = nn.Sequential(
                nn.Conv3d(skip_in_ch, skip_ch, kernel_size=1, stride=1),
                get_norm_layer(skip_ch, kind=norm_kind, num_groups=1),  # LayerNorm if kind='group'
                nn.LeakyReLU(0.2, inplace=True))
        else:
            self.skip_conv = nn.Sequential(
                nn.Conv3d(skip_in_ch, skip_ch, kernel_size=1, stride=1),
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
        inputs_shapes4 = [x.shape[4] for x in inputs]

        if (    np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and
                np.all(np.array(inputs_shapes3) == min(inputs_shapes3)) and
                np.all(np.array(inputs_shapes4) == min(inputs_shapes4))):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)
            target_shape4 = min(inputs_shapes4)

            inputs_ = []
            for inp in inputs:
                diff2 = (inp.size(2) - target_shape2) // 2
                diff3 = (inp.size(3) - target_shape3) // 2
                diff4 = (inp.size(4) - target_shape4) // 2
                inputs_.append(inp[:, :,
                                   diff2:diff2 + target_shape2,
                                   diff3:diff3 + target_shape3,
                                   diff3:diff4 + target_shape4])
        return torch.cat(inputs_, dim=1)


class ResBlock(nn.Module):
    def __init__(self, ch, kernel_size=3, use_norm=True, num_groups=4):
        super(ResBlock, self).__init__()
        to_pad = int((kernel_size - 1) / 2)
        norm_kind = use_norm if use_norm and isinstance(use_norm, str) else 'group'
        if use_norm:
            self.conv = nn.Sequential(
                nn.Conv3d(ch, ch, kernel_size,
                        stride=1, padding=to_pad),
                get_norm_layer(ch, kind=norm_kind, num_groups=num_groups),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv3d(ch, ch, kernel_size,
                        stride=1, padding=to_pad),
                get_norm_layer(ch, kind=norm_kind, num_groups=num_groups))
        else:
            self.conv = nn.Sequential(
                nn.Conv3d(ch, ch, kernel_size,
                        stride=1, padding=to_pad),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv3d(ch, ch, kernel_size,
                        stride=1, padding=to_pad))

    def forward(self, x):
        out = self.conv(x)
        out += x  # identity
        out = torch.nn.functional.leaky_relu(out, 0.2, inplace=True)
        return out


class OutBlock(nn.Module):
    def __init__(self, in_ch, out_ch, out_kernel_size=1, pre_out_channels=(), pre_out_kernel_size=3, insert_res_blocks_before=(), use_norm=True, num_groups=4):
        super(OutBlock, self).__init__()
        _pre_out_channels = [in_ch] + list(pre_out_channels)
        pre_out_to_pad = int((pre_out_kernel_size - 1) / 2)
        norm_kind = use_norm if use_norm and isinstance(use_norm, str) else 'group'
        self.pre_out = nn.ModuleList()
        for i, (c_in, c_out) in enumerate(zip(_pre_out_channels[:-1], _pre_out_channels[1:])):
            if i in insert_res_blocks_before:
                self.pre_out.append(ResBlock(c_in, kernel_size=3, use_norm=use_norm))
            if use_norm:
                self.pre_out.append(nn.Sequential(
                    nn.Conv3d(c_in, c_out, pre_out_kernel_size,
                            stride=1, padding=pre_out_to_pad),
                    get_norm_layer(c_out, kind=norm_kind, num_groups=num_groups),
                    nn.LeakyReLU(0.2, inplace=True)))
            else:
                self.pre_out.append(nn.Sequential(
                    nn.Conv3d(c_in, c_out, pre_out_kernel_size,
                            stride=1, padding=pre_out_to_pad),
                    nn.LeakyReLU(0.2, inplace=True)))
        if -1 in insert_res_blocks_before or len(pre_out_channels) in insert_res_blocks_before:
            self.pre_out.append(ResBlock(_pre_out_channels[-1], kernel_size=3, use_norm=use_norm))
        out_to_pad = int((out_kernel_size - 1) / 2)
        self.conv = nn.Conv3d(_pre_out_channels[-1], out_ch, kernel_size=out_kernel_size, stride=1, padding=out_to_pad)

    def forward(self, x):
        for pre_out_conv in self.pre_out:
            x = pre_out_conv(x)
        x = self.conv(x)
        return x

    def __len__(self):
        return len(self._modules)
