import torch

import ntops
from ntops.torch.utils import _cached_make


def selu(input, inplace=False):
    alpha = 1.6732632423543772
    scale = 1.0507009873554805

    if inplace:
        output = input
    else:
        output = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.selu.premake, input.ndim)

    kernel(input, alpha, scale, output)

    return output
