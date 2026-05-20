import torch

import ntops
from ntops.torch.utils import _cached_make


def floor(input):
    output = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.floor.premake, input.ndim)

    kernel(input, output)

    return output
