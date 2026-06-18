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

_INT_TYPES = {torch.int8, torch.int16, torch.int32, torch.int64}


def lgamma(input, *, out=None):
    if input.dtype in _INT_TYPES:
        # Pre-convert to float32 in Python before calling the kernel.
        # This reuses the existing float32 kernel for all four integer types,
        # avoiding four separate Triton JIT compilations (one per int dtype)
        # whose bodies would be identical (cast → float32 → lgamma).
        # torch.lgamma always returns float32 for integer inputs.
        if out is None:
            out = torch.empty_like(input, dtype=torch.float32)
        kernel_input = input.to(torch.float32)
        kernel = _cached_make(
            ntops.kernels.lgamma.premake,
            input.ndim,
            dtype=ninetoothed.float32,
        )
        kernel(kernel_input, out)
        return out

    if out is None:
        out = torch.empty_like(input)

    kernel = _cached_make(
        ntops.kernels.lgamma.premake,
        input.ndim,
        dtype=_DTYPE_MAP.get(input.dtype),
    )
    kernel(input, out)
    return out
