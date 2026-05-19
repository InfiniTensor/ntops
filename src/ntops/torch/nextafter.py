import torch

import ntops
from ntops.torch.utils import _cached_make


_BLOCK_SIZE = 1024
_NUM_WARPS = 4
_NUM_STAGES = 2


def nextafter(input, other, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = _cached_make(
        ntops.kernels.nextafter.premake,
        input.ndim,
        block_size=_BLOCK_SIZE,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )

    kernel(input, other, out)

    return out
