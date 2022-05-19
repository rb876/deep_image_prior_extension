from builtins import breakpoint
import torch
import torch.nn as nn 
import torch.nn.functional as F
from opt_einsum import contract

class ApproxConv3d(nn.Module):

    def __init__(self,  in_channels, out_channels, low_rank_dim, kernel_size, stride=1):
        super().__init__()
        self.pad = (kernel_size - 1)//2
        self.conv_d11 = nn.parameter.Parameter(torch.zeros(
            (low_rank_dim, in_channels, kernel_size, 1, 1))
            )
        self.conv_1d1 = nn.parameter.Parameter(torch.zeros(
            (low_rank_dim, low_rank_dim, 1, kernel_size, 1))
            )
        self.conv_11d = nn.parameter.Parameter(torch.zeros(
            (out_channels, low_rank_dim, 1, 1, kernel_size))
            )
        self.stride = stride

    def forward(self, x):

        dkabc = contract('dnabc,nkabc->dkabc', self.conv_1d1, self.conv_d11)
        okabc = contract('onabc,nkabc->okabc', self.conv_11d, dkabc)

        return F.conv3d(x, okabc, padding=self.pad, stride=self.stride)


# class ApproxConv3d(nn.Module):

#     def __init__(self,  in_channels, out_channels, low_rank_dim, kernel_size, stride=1):
#         super().__init__()
#         pad = (kernel_size - 1)//2
#         self.conv_d11 = nn.Conv3d(in_channels, low_rank_dim,
#             (kernel_size, 1, 1), padding=(pad, 0, 0), stride=(stride, 1, 1),
#             )
#         self.conv_1d1 = nn.Conv3d(low_rank_dim, low_rank_dim,
#             (1, kernel_size, 1), padding=(0, pad, 0), stride=(1, stride, 1),
#             )
#         self.conv_11d = nn.Conv3d(low_rank_dim, out_channels,
#             (1, 1, kernel_size), padding=(0, 0, pad), stride=(1, 1, stride),
#             )
#     def forward(self, x):

#         return self.conv_11d( 
#                     self.conv_1d1(
#                         self.conv_d11(x)
#                     )
#                 )