import torch

import ntops
from ntops.torch.utils import _cached_make


def leaky_relu(input, negative_slope=0.01, inplace=False):
    if inplace:
        output = input
    else:
        output = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.leaky_relu.premake, input.ndim)

    kernel(input, negative_slope, output)

    return output
