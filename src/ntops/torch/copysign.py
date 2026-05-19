import torch

import ntops
from ntops.torch.utils import _cached_make


def copysign(input, other, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = _cached_make(
        ntops.kernels.copysign.premake,
        input.ndim,
        block_size=1024,
        num_warps=4,
    )

    kernel(input, other, out)

    return out
