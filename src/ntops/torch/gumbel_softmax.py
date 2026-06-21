import torch

import ntops
from ntops.torch.utils import _cached_make


def gumbel_softmax(input, tau=1.0, hard=False, eps=1e-10, dim=-1):
    output = torch.empty_like(input)

    kernel = _cached_make(
        ntops.kernels.gumbel_softmax.premake,
        input.ndim,
        dim,
    )

    hard_value = 1.0 if hard else 0.0

    kernel(input, float(tau), hard_value, output)

    return output