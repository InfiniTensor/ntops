import torch

import ntops
from ntops.torch.utils import _cached_make, _flatten_kernel_tensors, _prepare_out


_kernel_1d = {}


def _get_kernel_1d(half):
    kernel = _kernel_1d.get(half)
    if kernel is None:
        kernel = _cached_make(
            ntops.kernels.lgamma.premake,
            1,
            half,
            block_size=ntops.kernels.lgamma.BLOCK_SIZE if half else 1024,
            num_warps=8 if half else 4,
            max_num_configs=1,
        )
        _kernel_1d[half] = kernel
    return kernel


def _promote_unary_input(input):
    if hasattr(torch, "is_floating_point") and not torch.is_floating_point(input):
        return input.to(torch.float32)
    return input


def lgamma(input, *, out=None):
    input = _promote_unary_input(input)
    if input.ndim == 1 and input.is_contiguous():
        half = input.dtype == torch.float16
        if out is None:
            out = torch.empty_like(input)
            _get_kernel_1d(half)(input, out)
            return out
        if tuple(out.shape) == tuple(input.shape) and out.dtype == input.dtype and out.is_contiguous():
            _get_kernel_1d(half)(input, out)
            return out

    out = _prepare_out(out, input.shape, input.dtype, input.device, like=input)

    kernel_input, kernel_out = _flatten_kernel_tensors(input, out)
    half = hasattr(torch, "float16") and input.dtype == torch.float16
    kernel = _cached_make(
        ntops.kernels.lgamma.premake,
        kernel_input.ndim,
        half,
        block_size=ntops.kernels.lgamma.BLOCK_SIZE if half else 1024,
        num_warps=8 if half else 4,
        max_num_configs=1,
    )
    kernel(kernel_input, kernel_out)

    return out
