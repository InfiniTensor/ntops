import torch

import ntops
from ntops.torch import _vendor_triton
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


def _metax_fast_path(input, out):
    if not _vendor_triton.is_metax_device(input):
        return False
    if input.dtype not in (torch.float16, torch.float32):
        return False
    if input.numel() < 1024:
        return False
    if input.ndim == 0:
        kernel_input = input.reshape((1,))
        kernel_out = out.reshape((1,))
    elif input.is_contiguous() and out.is_contiguous():
        kernel_input = input.reshape((input.numel(),))
        kernel_out = out.reshape((out.numel(),))
    elif input.ndim > 1 and tuple(input.shape) == tuple(out.shape):
        dims = tuple(sorted(range(input.ndim), key=lambda dim: input.stride()[dim], reverse=True))
        input_view = input.permute(dims)
        out_view = out.permute(dims)
        if not input_view.is_contiguous() or not out_view.is_contiguous():
            return False
        kernel_input = input_view.reshape((input.numel(),))
        kernel_out = out_view.reshape((out.numel(),))
    else:
        return False

    _vendor_triton.lgamma_metax_1d(kernel_input, kernel_out)
    return True


def lgamma(input, *, out=None):
    input = _promote_unary_input(input)
    if input.ndim == 1 and input.is_contiguous():
        half = input.dtype == torch.float16
        if out is None:
            out = torch.empty_like(input)
            if _metax_fast_path(input, out):
                return out
            _get_kernel_1d(half)(input, out)
            return out
        if tuple(out.shape) == tuple(input.shape) and out.dtype == input.dtype and out.is_contiguous():
            if _metax_fast_path(input, out):
                return out
            _get_kernel_1d(half)(input, out)
            return out

    out = _prepare_out(out, input.shape, input.dtype, input.device, like=input)

    if _metax_fast_path(input, out):
        return out

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
