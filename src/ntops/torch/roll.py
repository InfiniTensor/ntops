import torch

import ntops
from ntops.torch.utils import _cached_make


def roll(input, shifts, dims=None):
    pre_rolled = torch.roll(input, shifts, dims)
    output = torch.empty_like(pre_rolled)

    kernel = _cached_make(ntops.kernels.roll.premake, input.ndim)
    kernel(pre_rolled, output)

    return output
