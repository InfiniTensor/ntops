import torch

import ntops
from ntops.torch.utils import _cached_make


def mode(input, dim=-1, keepdim=False):
    values, indices = torch.mode(input, dim, keepdim)

    if values.ndim == 0:
        return values, indices

    output_values = torch.empty_like(values)

    kernel = _cached_make(ntops.kernels.mode.premake, values.ndim)
    kernel(values, output_values)

    return output_values, indices
