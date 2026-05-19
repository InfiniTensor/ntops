import torch

import ntops
from ntops.torch.utils import _cached_make


_BLOCK_SIZE = 2048
_NUM_WARPS = 4
_NUM_STAGES = 1


def rad2deg(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    if input.ndim != 1 and input.is_contiguous() and out.is_contiguous():
        n = input.numel()
        in_view = input.view([n])
        out_view = out.view([n])
    else:
        in_view = input
        out_view = out

    kernel = _cached_make(
        ntops.kernels.rad2deg.premake,
        in_view.ndim,
        block_size=_BLOCK_SIZE,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )

    kernel(in_view, out_view)

    return out
