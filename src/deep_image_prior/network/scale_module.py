import torch
import torch.nn as nn

class ScaleModule(nn.Module):

    """
    This class provides methods for normalizing the input data.

    Based on the scaling_module in
    https://github.com/ahendriksen/msd_pytorch/blob/master/msd_pytorch/msd_model.py
    """

    def __init__(self, num_channels, mean=0., std=1., conv3d=False):
        super().__init__()

        """
        Parameters
        ----------
        num_channels: int
            The number of channels.
        mean: float
            Mean of values.
        std: float
            Standard deviation of values.
        param conv3d: bool
            Indicates that the input data is 3D instead of 2D.

        * saved when the network is saved;
        * not updated by the gradient descent solvers.
        """
        self.mean = mean
        self.std = std

        if conv3d:
            self.scale_layer = nn.Conv3d(num_channels, num_channels, 1)
        else:
            self.scale_layer = nn.Conv2d(num_channels, num_channels, 1)

        self._scaling_module_set_scale(1 / self.std)
        self._scaling_module_set_bias(-self.mean / self.std)

    def _scaling_module_set_scale(self, scale):

        self.scale_layer.weight.requires_grad = False
        c_out, c_in = self.scale_layer.weight.shape[:2]
        assert c_out == c_in
        self.scale_layer.weight.data.zero_()
        for i in range(c_out):
            self.scale_layer.weight.data[i, i] = scale

    def _scaling_module_set_bias(self, bias):
        self.scale_layer.bias.requires_grad = False
        self.scale_layer.bias.data.fill_(bias)

    def forward(self, x):
        return self.scale_layer(x)

def get_scale_modules(ch_in, ch_out, mean_in=0., mean_out=0., std_in=1.,
                     std_out=1., conv3d=False):
    scale_in = ScaleModule(ch_in, mean_in, std_in, conv3d)
    scale_out = ScaleModule(ch_out, -mean_out, 1./std_out, conv3d)
    return scale_in, scale_out
