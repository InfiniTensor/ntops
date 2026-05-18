import torch

import ntops
from ntops.torch.utils import _cached_make


def lcm(input, other, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.lcm.premake, input.ndim, dtype=_to_nt(input.dtype))

    kernel(input, other, out)

    return out


def _to_nt(torch_dtype):
    import ninetoothed
    mapping = {
        torch.int8: ninetoothed.int8,
        torch.int16: ninetoothed.int16,
        torch.int32: ninetoothed.int32,
        torch.int64: ninetoothed.int64,
    }
    return mapping.get(torch_dtype)
