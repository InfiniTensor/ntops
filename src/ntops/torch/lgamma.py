import torch

import ntops
from ntops.torch.utils import _cached_make


_LARGE_NUMEL_THRESHOLD = 2_000_000


def lgamma(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    if input.numel() >= _LARGE_NUMEL_THRESHOLD:
        kernel = _cached_make(
            ntops.kernels.lgamma.premake,
            input.ndim,
            dtype=_to_nt(input.dtype),
            block_size=1024,
            num_warps=4,
            num_stages=5,
        )
    else:
        kernel = _cached_make(
            ntops.kernels.lgamma.premake,
            input.ndim,
            dtype=_to_nt(input.dtype),
        )

    kernel(input, out)

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
