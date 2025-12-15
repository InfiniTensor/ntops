import torch

import ntops
from ntops.torch.utils import _cached_make


def mish(input, inplace=False):
    if not inplace:
        out = torch.empty_like(input)
    else:
        out = input

    kernel = _cached_make(ntops.kernels.mish.premake, input.ndim)

    kernel(input, out)

    return out
