import torch.nn as nn
try:
    from msd_pytorch.msd_model import scaling_module
    from msd_pytorch.msd_module import MSDModule
except ImportError:
    MSD_PYTORCH_AVAILABLE = False
else:
    MSD_PYTORCH_AVAILABLE = True


class MSDNet(nn.Module):
    def __init__(self, in_ch, out_ch, depth, width, dilations,
                 mean_in=0., mean_out=0., std_in=1., std_out=1.):
        super(MSDNet, self).__init__()

        if not MSD_PYTORCH_AVAILABLE:
            raise RuntimeError('Missing `msd_pytorch`')

        self.scale_in = scaling_module(in_ch)
        self.scale_out = scaling_module(out_ch)
        self.msd = MSDModule(in_ch, out_ch, depth, width, dilations=dilations)

        self.set_normalization(mean_in, mean_out, std_in, std_out)

    def set_normalization(self, mean_in, mean_out, std_in, std_out):
        """
        Initialize scaling layers such that the actual MSD network only needs
        to handle approximately normal distributed input and output values.

        Parameters
        ----------
        mean_in : float
            Mean of input values.
        mean_out : float
            Mean of output values.
        std_in : float
            Standard deviation of input values.
        std_out : float
            Standard deviation of output values.
        """
        # Taken from: https://github.com/ahendriksen/msd_pytorch/blob/162823c502701f5eedf1abcd56e137f8447a72ef/msd_pytorch/msd_model.py#L95
        # The input data should be roughly normally distributed after
        # passing through scale_in. Note that the input is first
        # scaled and then recentered.
        self.scale_in.weight.data.fill_(1 / std_in)
        self.scale_in.bias.data.fill_(-mean_in / std_in)
        # The scale_out layer should rather 'denormalize' the network
        # output.
        self.scale_out.weight.data.fill_(std_out)
        self.scale_out.bias.data.fill_(mean_out)

    def forward(self, inp):
        x = self.scale_in(inp)
        x = self.msd(x)
        x = self.scale_out(x)
        return x
