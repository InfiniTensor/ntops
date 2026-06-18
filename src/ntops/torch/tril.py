import torch

import ntops
from ntops.torch.utils import _cached_make


def tril(input, diagonal=0, *, out=None):
    n, m = input.shape[-2], input.shape[-1]

    rows = torch.arange(n, device=input.device).reshape(n, 1).expand(n, m)
    cols = torch.arange(m, device=input.device).reshape(1, m).expand(n, m)

    # Adjust column comparison for diagonal offset: cols <= rows + diagonal
    if diagonal != 0:
        cols = cols - diagonal

    # Expand rows and cols to match input's ndim
    rows = rows.reshape((1,) * (input.ndim - 2) + (n, m)).expand_as(input)
    cols = cols.reshape((1,) * (input.ndim - 2) + (n, m)).expand_as(input)

    if out is None:
        out = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.tril.premake, input.ndim)

    kernel(rows, cols, input, out)

    return out
