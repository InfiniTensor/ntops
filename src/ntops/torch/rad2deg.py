import torch

import ntops
from ntops.torch.utils import _cached_make


def rad2deg(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.rad2deg.premake, input.ndim)

    kernel(input, out)

    return out
