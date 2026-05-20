import torch

import ntops
from ntops.torch.utils import _cached_make


def tanhshrink(input):
    output = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.tanhshrink.premake, input.ndim)

    kernel(input, output)

    return output
