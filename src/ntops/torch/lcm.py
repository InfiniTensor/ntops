import torch

import ninetoothed
import ntops
from ntops.torch.utils import _cached_make

_NUM_STAGES = 1


def _block_size_for(torch_dtype):
    if torch_dtype == torch.int64:
        return 32
    return 512


def _num_warps_for(torch_dtype):
    if torch_dtype == torch.int64:
        return 1
    return 4


def _to_nt(torch_dtype):
    mapping = {
        torch.int8: ninetoothed.int8,
        torch.int16: ninetoothed.int16,
        torch.int32: ninetoothed.int32,
        torch.int64: ninetoothed.int64,
    }
    return mapping.get(torch_dtype)


def lcm(input, other, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    if input.ndim != 1 and input.is_contiguous() and other.is_contiguous() and out.is_contiguous():
        n = input.numel()
        in_view = input.view([n])
        other_view = other.view([n])
        out_view = out.view([n])
    else:
        in_view = input
        other_view = other
        out_view = out

    kernel = _cached_make(
        ntops.kernels.lcm.premake,
        in_view.ndim,
        dtype=_to_nt(input.dtype),
        block_size=_block_size_for(input.dtype),
        num_warps=_num_warps_for(input.dtype),
        num_stages=_NUM_STAGES,
    )

    kernel(in_view, other_view, out_view)

    return out
