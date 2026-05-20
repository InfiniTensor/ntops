import torch

import ntops
from ntops.torch.utils import _cached_make


def prelu(input, weight):
    output = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.prelu.premake, input.ndim)

    kernel(input, weight, output)

    return output
