import torch

import ntops
from ntops.torch.utils import (
    _cached_make,
    _is_corex_compat_device,
    _torch_binary_fallback,
)


def copysign(input, other, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    if input.dtype in (torch.float16, torch.bfloat16) and _is_corex_compat_device(
        input.device
    ):
        _torch_binary_fallback("copysign", input, other, out)
        return out

    kernel = _cached_make(
        ntops.kernels.copysign.premake,
        input.ndim,
        dtype=_to_nt(input.dtype),
        block_size=1024,
        num_warps=4,
    )

    kernel(input, other, out)

    return out


def _to_nt(torch_dtype):
    import ninetoothed

    mapping = {
        torch.float16: ninetoothed.float16,
        torch.bfloat16: ninetoothed.bfloat16,
        torch.float32: ninetoothed.float32,
        torch.float64: ninetoothed.float64,
    }
    return mapping.get(torch_dtype)
