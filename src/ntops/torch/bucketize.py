import torch

import ntops
from ntops.torch.utils import _cached_make


def bucketize(input, boundaries, *, out=None, right=False):
    if out is None:
        out = torch.empty_like(input, dtype=torch.int64)

    if boundaries.ndim != 1:
        raise ValueError("boundaries must be 1 dimension")

    bound_len = boundaries.numel()
    if bound_len == 0:
        out.fill_(0)
        return out

    if (bound_len & (bound_len - 1)) == 0:
        padded_len = bound_len
    else:
        padded_len = 1 << bound_len.bit_length()

    padded_len = max(16, padded_len)

    block_size = 1024
    kernel = _cached_make(
        ntops.kernels.bucketize.premake,
        input.ndim,
        input.dtype,
        padded_len=padded_len,
        block_size=block_size,
    )

    kernel(input, boundaries, out, right, bound_len)

    return out
