import torch
from odl.contrib.torch import OperatorModule
from .fbp import get_fbp_filter_op

class FBPModule(torch.nn.Module):
    """
    Module for filtered back-projection applying a filter returned by
    :func:`get_fbp_filter_op`, followed by calling a specified adjoint.
    """
    def __init__(self, adjoint_func, *args, **kwargs):
        """
        Parameters
        ----------
        adjoint_func : callable
            Callable module calculating the adjoint of the ray transform.
            Receives a projection and returns an image.
        *args, **kwargs
            Arguments passed to :func:`get_fbp_filter_op`.
        """
        super().__init__()
        self.adjoint_func = adjoint_func
        self.filter_op = OperatorModule(get_fbp_filter_op(*args, **kwargs))

    def forward(self, y):
        """
        Apply the filtered back-projection.
        """
        y = self.filter_op(y)
        x = self.adjoint_func(y)
        return x


def get_matrix_fbp_module(adjoint_func, proj_space, **kwards):

    return FBPModule(adjoint_func, proj_space, **kwards)
