import torch

import ntops
from ntops.torch.utils import _cached_make


def minimum(input, other, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    block_size = 1024
    kernel = _cached_make(
        ntops.kernels.minimum.premake, input.ndim, input.dtype, block_size=block_size
    )

    kernel(input, other, out)

    return out
