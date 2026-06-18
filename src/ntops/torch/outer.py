import torch

import ntops
from ntops.torch.utils import _cached_make


def outer(input, other, *, out=None):
    m, n = input.size(0), other.size(0)

    if out is None:
        out = torch.empty(m, n, dtype=input.dtype, device=input.device)
    else:
        out = out.view(m, n)

    input_2d = input.unsqueeze(1)
    other_2d = other.unsqueeze(0)

    kernel = _cached_make(ntops.kernels.outer.premake)
    kernel(input_2d, other_2d, out)

    return out
