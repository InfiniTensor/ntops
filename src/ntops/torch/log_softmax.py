import torch

import ntops
from ntops.torch.utils import _cached_make


def log_softmax(input, dim=-1):
    output = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.log_softmax.premake, input.ndim, dim=dim)

    kernel(input, output)

    return output
