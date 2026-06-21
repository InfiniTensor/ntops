import torch

import ntops
from ntops.torch.utils import _cached_make


def narrow(input, dim, start, length, *, out=None):
    dim = dim % input.ndim

    if out is None:
        shape = list(input.shape)
        shape[dim] = length
        out = torch.empty(shape, dtype=input.dtype, device=input.device)

    kernel = _cached_make(
        ntops.kernels.narrow.premake,
        input.ndim,
        dim,
        start,
        length,
    )

    kernel(input, out)

    return out