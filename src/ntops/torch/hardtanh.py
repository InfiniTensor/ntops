import torch

import ntops
from ntops.torch.utils import _cached_make


def hardtanh(input, min_val=-1.0, max_val=1.0, inplace=False):
    if inplace:
        output = input
    else:
        output = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.hardtanh.premake, input.ndim)

    kernel(input, min_val, max_val, output)

    return output
