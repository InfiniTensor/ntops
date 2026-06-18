import torch

import ninetoothed
import ntops
from ntops.torch.utils import _cached_make

_DTYPE_MAP = {
    torch.float16: ninetoothed.float16,
    torch.bfloat16: ninetoothed.bfloat16,
    torch.float32: ninetoothed.float32,
    torch.float64: ninetoothed.float64,
}


def copysign(input, other, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = _cached_make(
        ntops.kernels.copysign.premake,
        input.ndim,
        dtype=_DTYPE_MAP.get(input.dtype),
    )

    kernel(input, other, out)

    return out
