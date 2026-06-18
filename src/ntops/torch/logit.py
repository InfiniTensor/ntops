import torch

import ntops
from ntops.torch.utils import _cached_make


def logit(input, eps=None):
    if eps is None:
        eps = 0.0

    output = torch.empty_like(input)
    eps_t = torch.full_like(input, eps)

    kernel = _cached_make(ntops.kernels.logit.premake, input.ndim)
    kernel(input, output, eps_t)

    return output
