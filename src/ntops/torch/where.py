import torch

import ntops
from ntops.torch.utils import _cached_make


def where(condition, input, other, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    # 这里假设 input/other/out dtype 一致
    kernel = _cached_make(ntops.kernels.where.premake, input.ndim)
    kernel(condition, input, other, out)
    return out