import torch

import ntops
from ntops.torch.utils import _cached_make


def repeat(input, *sizes):
    if len(sizes) == 1 and isinstance(sizes[0], (list, tuple)):
        sizes = tuple(sizes[0])

    repeated = input.repeat(*sizes)
    out = torch.empty_like(repeated)

    kernel = _cached_make(ntops.kernels.repeat.premake, repeated.ndim)

    kernel(repeated, out)

    return out
