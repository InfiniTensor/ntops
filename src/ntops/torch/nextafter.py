import torch

import ntops
from ntops.torch.utils import _cached_make


def nextafter(x, y, *, out=None):
    if out is None:
        out = torch.empty_like(x)

    kernel = _cached_make(ntops.kernels.nextafter.premake, x.ndim)

    kernel(x, y, out)

    return out
