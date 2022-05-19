import torch
from torch import nn
import tomosipo as ts
from tomosipo.torch_support import to_autograd

# clone from tomosipo.torch_support, but with
# @torch.cuda.amp.custom_fwd and @torch.cuda.amp.custom_bwd in order to enforce
# float32 within amp autocast
class OperatorFunctionFloat32(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, operator):
        # if input.requires_grad:  # seems to be always False if custom_fwd is
        # used, even if grad of the original input is required
        ctx.operator = operator
        assert (
            input.ndim == 4
        ), "Autograd operator expects a 4-dimensional input (3+1 for Batch dimension). "

        B, C, H, W = input.shape
        out = input.new_empty(B, *operator.range_shape)

        for i in range(B):
            operator(input[i], out=out[i])

        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        operator = ctx.operator

        B, C, H, W = grad_output.shape
        grad_input = grad_output.new_empty(B, *operator.domain_shape)

        for i in range(B):
            operator.T(grad_output[i], out=grad_input[i])

        # do not return gradient for operator
        return grad_input, None

# clone from tomosipo.torch_support, but using OperatorFunctionFloat32
def to_autograd_float32(operator):
    def f(x):
        return OperatorFunctionFloat32.apply(x, operator)

    return f

class TorchLinkedRayTrafoModule(nn.Module):
    """
    Module applying ASTRA direct 3D forward- or back-projections via tomosipo.
    Gradients will be computed via the discretization of the analytical adjoint,
    which might deviate slightly from the adjoint of the discrete forward pass.
    """
    def __init__(self, vol_geom, proj_geom, adjoint=False):
        """
        Parameters
        ----------
        vol_geom : dict
            ASTRA 3D volume geometry
        proj_geom : dict
            ASTRA 3D projection geometry
        adjoint : bool, optional
            If `False` (the default), compute the forward-projection in
            :meth:`forward`; if `True`, compute the back-projection instead.
        """
        super().__init__()
        self.vol_geom = vol_geom
        self.proj_geom = proj_geom
        self.adjoint = adjoint

        ts_operator_fp = ts.operator(
                ts.from_astra(vol_geom), ts.from_astra(proj_geom))
        self.ts_operator = ts_operator_fp.T if self.adjoint else ts_operator_fp
        self.forward_fun = to_autograd_float32(self.ts_operator)

    def forward(self, inp):
        """
        Apply the forward- or back-projection.

        Parameters
        ----------
        inp : :class:`torch.Tensor`
            For forward-projection (:attr:`adjoint` is `False`):
                    shape ``... x Z x Y x X``;
            for backward-projection (:attr:`adjoint` is `True`):
                    shape ``... x det_rows x angles x det_cols``.
            Any leading dimensions are treated as batch dimensions.
        """
        orig_batch_dims = inp.shape[:-3]
        inp = inp.view(-1, *inp.shape[-3:])

        out = self.forward_fun(inp)

        out = out.view(*orig_batch_dims, *out.shape[-3:])
        return out
