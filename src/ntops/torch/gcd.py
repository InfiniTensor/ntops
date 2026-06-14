import torch

import ntops
from ntops.torch.utils import _cached_make


def gcd(a, b, *, out=None):
    if out is None:
        out = torch.empty_like(a)

    kernel = _cached_make(ntops.kernels.gcd.premake, a.ndim)

    kernel(a, b, out)

    return out
