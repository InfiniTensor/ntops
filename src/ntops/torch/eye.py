import torch

import ntops
from ntops.torch.utils import _cached_make


def eye(n, m=None, *, dtype=None, device=None, out=None):
    if m is None:
        m = n

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dtype is None:
        dtype = torch.float32

    rows = torch.arange(n, device=device).reshape(n, 1).expand(n, m)
    cols = torch.arange(m, device=device).reshape(1, m).expand(n, m)

    if out is None:
        out = torch.empty(n, m, dtype=dtype, device=device)

    kernel = _cached_make(ntops.kernels.eye.premake, 2)

    kernel(rows, cols, out)

    return out
