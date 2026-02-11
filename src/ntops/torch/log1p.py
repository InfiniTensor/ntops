import torch

import ntops
from ntops.torch.utils import _cached_make


def log1p(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    block_size = 1024
    kernel = _cached_make(ntops.kernels.log1p.premake, input.ndim, block_size)

    kernel(input, out)

    return out
