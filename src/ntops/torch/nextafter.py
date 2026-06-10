import torch

import ntops
from ntops.torch.utils import _cached_make


def nextafter(input, other, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.nextafter.premake, input.ndim)

    kernel(input, other, out)

    return out
