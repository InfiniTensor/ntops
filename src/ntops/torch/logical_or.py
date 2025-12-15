# src/ntops/torch/logical_or.py
import torch

import ntops
from ntops.torch.utils import _cached_make


def logical_or(input, other, *, out=None):
    if out is None:
        out = torch.empty_like(input, dtype=torch.bool)

    kernel = _cached_make(ntops.kernels.logical_or.premake, input.ndim)

    kernel(input, other, out)

    return out