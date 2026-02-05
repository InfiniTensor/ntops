# import torch

# import ntops
# from ntops.torch.utils import _cached_make

# def logical_xor(input, other, *, out=None):
#     if out is None:
#         out = torch.empty_like(input, dtype=torch.bool)

#     kernel = _cached_make(ntops.kernels.logical_xor.premake, input.ndim)
#     kernel(input, other, out)
#     return out

import torch

import ntops
from ntops.torch.utils import _cached_make


def logical_xor(input, other, *, out=None):
    kernel = _cached_make(ntops.kernels.logical_xor.premake, input.ndim)

    if out is None:
        out = torch.empty_like(input, dtype=torch.bool)
        kernel(input, other, out)
        return out

    if out is input or out is other:
        tmp = torch.empty_like(out)
        kernel(input, other, tmp)
        out.copy_(tmp)
    else:
        kernel(input, other, out)

    return out