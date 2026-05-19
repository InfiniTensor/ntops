import torch

import ntops
from ntops.torch.utils import _cached_make


_LARGE_NUMEL_THRESHOLD = 2_000_000


def lgamma(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    if input.numel() >= _LARGE_NUMEL_THRESHOLD:
        kernel = _cached_make(
            ntops.kernels.lgamma.premake,
            input.ndim,
            block_size=1024,
            num_warps=4,
            num_stages=5,
        )
    else:
        kernel = _cached_make(ntops.kernels.lgamma.premake, input.ndim)

    kernel(input, out)

    return out
