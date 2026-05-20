import torch

import ntops
from ntops.torch.utils import _cached_make


def softplus(input, beta=1.0, threshold=20.0):
    output = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.softplus.premake, input.ndim)

    kernel(input, beta, threshold, output)

    return output
