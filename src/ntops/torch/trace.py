import torch

import ntops
from ntops.torch.utils import _cached_make


def trace(input):
    diagonal = torch.diagonal(input)
    output = torch.zeros(1, dtype=diagonal.dtype, device=diagonal.device)

    kernel = _cached_make(ntops.kernels.trace.premake, diagonal.ndim)
    kernel(diagonal, output)

    return output.squeeze(0)
