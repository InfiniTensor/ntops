import torch

import ntops
from ntops.torch.utils import _cached_make


def index_add(input, dim, index, source, *, alpha=1, out=None):
    if index.dtype != torch.int64:
        raise AssertionError(
            "index_add is only applicable to index tensor of type LongTensor."
        )

    if dim != 0:
        raise AssertionError("Only dim=0 is supported.")

    if out is None:
        out = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.index_add.premake, input.ndim, dim)

    kernel(input, index, source, alpha, out)

    return out
