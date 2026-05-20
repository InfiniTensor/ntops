import torch

import ntops
from ntops.torch.utils import _cached_make


def relu6(input, inplace=False):
    if inplace:
        output = input
    else:
        output = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.relu6.premake, input.ndim)

    kernel(input, output)

    return output
