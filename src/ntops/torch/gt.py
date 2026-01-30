import torch

import ntops
from ntops.torch.utils import _cached_make


def gt(input, other, *, out=None):
    if out is None:
        out = torch.empty(input.shape, dtype=torch.bool, device=input.device)

    block_size = 1024
    kernel = _cached_make(
        ntops.kernels.gt.premake, input.ndim, dtype=input.dtype, block_size=block_size
    )

    kernel(input, other, out)

    return out
