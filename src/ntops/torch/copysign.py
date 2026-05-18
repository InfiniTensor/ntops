import functools

import torch

import ntops
from ntops.torch.utils import _cached_make, _flatten_kernel_tensors, _prepare_out


def _broadcast(input, other):
    if hasattr(torch, "broadcast_tensors"):
        return torch.broadcast_tensors(input, other)
    return input, other


def _prepare_inputs(input, other):
    if not hasattr(torch, "result_type"):
        return input, other, input.dtype

    result_dtype = torch.result_type(input, other)
    if not result_dtype.is_floating_point:
        result_dtype = torch.float32
    return input.to(result_dtype), other.to(result_dtype), result_dtype


@functools.cache
def _get_kernel_1d(half, double):
    return _cached_make(
        ntops.kernels.copysign.premake,
        1,
        half,
        double,
        block_size=ntops.kernels.copysign.BLOCK_SIZE,
        num_warps=4,
        max_num_configs=1,
    )


@functools.cache
def _get_broadcast_2d_kernel(half, double):
    return _cached_make(
        ntops.kernels.copysign.premake,
        2,
        half,
        double,
        True,
        block_size=4096,
        num_warps=8,
        max_num_configs=1,
    )


def copysign(input, other, *, out=None):
    if (
        input.ndim == 1
        and other.ndim == 1
        and tuple(input.shape) == tuple(other.shape)
        and input.dtype == other.dtype
        and input.dtype.is_floating_point
        and input.is_contiguous()
        and other.is_contiguous()
    ):
        half = input.dtype == torch.float16
        double = input.dtype == torch.float64
        if out is None:
            out = torch.empty_like(input)
            _get_kernel_1d(half, double)(input, other, out)
            return out
        if tuple(out.shape) == tuple(input.shape) and out.dtype == input.dtype and out.is_contiguous():
            _get_kernel_1d(half, double)(input, other, out)
            return out

    if (
        out is None
        and input.ndim == 2
        and other.ndim == 2
        and input.shape[1] == 1
        and other.shape[0] == 1
        and input.dtype == other.dtype
        and input.dtype.is_floating_point
        and input.is_contiguous()
        and other.is_contiguous()
    ):
        rows = input.shape[0]
        cols = other.shape[1]
        half = input.dtype == torch.float16
        double = input.dtype == torch.float64
        out = torch.empty((rows, cols), dtype=input.dtype, device=input.device)
        _get_broadcast_2d_kernel(half, double)(
            input,
            other,
            out,
        )
        return out

    input, other = _broadcast(input, other)
    input, other, result_dtype = _prepare_inputs(input, other)
    out = _prepare_out(out, input.shape, result_dtype, input.device, like=input)

    kernel_input, kernel_other, kernel_out = _flatten_kernel_tensors(input, other, out)
    kernel = _cached_make(
        ntops.kernels.copysign.premake,
        kernel_input.ndim,
        input.dtype == torch.float16,
        input.dtype == torch.float64,
        block_size=ntops.kernels.copysign.BLOCK_SIZE,
        num_warps=4,
        max_num_configs=1,
    )
    kernel(kernel_input, kernel_other, kernel_out)

    return out
