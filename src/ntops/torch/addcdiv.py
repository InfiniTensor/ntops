import torch

import ntops
from ntops.torch.utils import _cached_make


def addcdiv(input, tensor1, tensor2, *, value=1.0, out=None):
    if out is None:
        out = torch.empty_like(input)

    block_size = 1024
    kernel = _cached_make(
        ntops.kernels.addcdiv.premake, input.ndim, input.dtype, block_size=block_size
    )

    kernel(input, tensor1, tensor2, value, out)

    return out
