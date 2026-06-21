import torch

import ntops
from ntops.torch.utils import _cached_make


def lcm(a, b, *, out=None):
    if out is None:
        out = torch.empty_like(a)

    kernel = _cached_make(ntops.kernels.lcm.premake, a.ndim)

    kernel(a, b, out)

    return out
