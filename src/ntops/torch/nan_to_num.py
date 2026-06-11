import torch

import ntops
from ntops.torch.utils import _cached_make


def nan_to_num(input, nan=0.0, posinf=None, neginf=None):
    output = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.nan_to_num.premake, input.ndim)
    kernel(input, output)

    # Detect inf/nan from original input, since kernel already replaced them
    output = torch.where(torch.isnan(input), nan, output)
    if posinf is not None:
        output = torch.where(torch.isinf(input) & (input > 0), posinf, output)
    if neginf is not None:
        output = torch.where(torch.isinf(input) & (input < 0), neginf, output)

    return output
