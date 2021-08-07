import numpy as np
import odl
from odl import ResizingOperator
import torch
import torch.nn.functional as F
from util.odl_fourier_transform_torch import (
    FourierTransformModule, FourierTransformInverseModule)
from util.fbp import _fbp_filter


class FBPFilterModule(torch.nn.Module):
    def __init__(self, proj_space, padding=True,
                 filter_type='Ram-Lak', frequency_scaling=1.0,
                 scaling_factor=1., device='auto'):
        super().__init__()

        self.proj_space = proj_space
        if not isinstance(self.proj_space, odl.DiscretizedSpace):
            # Construct proj_space from shape specification
            # `min_pt` and `max_pt` are not relevant for the resulting values
            self.proj_space = odl.uniform_discr(
                    [0., -0.5], [2.*np.pi, 0.5], self.proj_space)

        self.padding = padding
        self.filter_type = filter_type
        self.frequency_scaling = frequency_scaling
        self.scaling_factor = scaling_factor

        # Define ramp filter
        def fourier_filter(x):
            abs_freq = np.abs(x[1])
            norm_freq = abs_freq / np.max(abs_freq)
            filt = _fbp_filter(
                    norm_freq, self.filter_type, self.frequency_scaling)
            return filt
            # scaling = 1. / (2. * alen)
            # return filt * np.max(abs_freq) * scaling

        # Define fourier transform
        if self.padding:
            # Define padding operator (only to infer padded space)
            ran_shp = (self.proj_space.shape[0],
                       self.proj_space.shape[1] * 2 - 1)
            resizing = ResizingOperator(self.proj_space, ran_shp=ran_shp)

            self.fourier_mod = FourierTransformModule(resizing.range)
            self.fourier_inverse_mod = FourierTransformInverseModule(
                    resizing.range)
        else:
            self.fourier_mod = FourierTransformModule(self.proj_space)
            self.fourier_inverse_mod = FourierTransformInverseModule(
                    self.proj_space)
        
        # Create ramp in the detector direction
        ramp_function = fourier_filter(
                (None, self.fourier_mod.fourier_domain.meshgrid[1]))
        ramp_function *= self.scaling_factor
        self.register_buffer('ramp_function',
                torch.from_numpy(ramp_function).float()[..., None],
                persistent=False)

    def forward(self, x):
        if self.padding:
            pad_offset = (self.proj_space.shape[1]-1)//2
            x = F.pad(x, (pad_offset, self.proj_space.shape[1]-1-pad_offset))

        x_f = self.fourier_mod(x)
        x_filtered_f = self.ramp_function * x_f
        x_filtered = self.fourier_inverse_mod(x_filtered_f)

        if self.padding:
            x_filtered = x_filtered[
                    :, :, :, pad_offset:pad_offset+self.proj_space.shape[1]]

        x_filtered = x_filtered.contiguous()
        return x_filtered
