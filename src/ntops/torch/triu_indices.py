import torch

import ntops
from ntops.torch.utils import _cached_make


def triu_indices(n, m=None, offset=0, *, device=None):
    if m is None:
        m = n

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = torch.arange(n, device=device).reshape(n, 1).expand(n, m)
    cols = torch.arange(m, device=device).reshape(1, m).expand(n, m)
    cols = cols - offset

    mask = torch.empty(n, m, dtype=torch.int32, device=device)

    kernel = _cached_make(ntops.kernels.triu_indices.premake, 2)
    kernel(rows, cols, mask)

    indices = torch.nonzero(mask)
    indices = indices.T.contiguous()

    return indices
