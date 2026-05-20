import functools

import torch

import ntops
from ntops.torch import _vendor_triton
from ntops.torch.utils import _cached_make, _flatten_kernel_tensors, _prepare_out


@functools.cache
def _is_iluvatar_device(index):
    if not hasattr(torch, "cuda"):
        return False
    if not torch.cuda.is_available():
        return False
    try:
        return "Iluvatar" in torch.cuda.get_device_name(index)
    except Exception:
        return False


def _use_iluvatar_double_kernel(tensor):
    if tensor.dtype != torch.float64 or tensor.device.type != "cuda":
        return False
    if not hasattr(torch, "cuda"):
        return False
    index = tensor.device.index
    if index is None:
        index = torch.cuda.current_device()
    return _is_iluvatar_device(index)


def _use_iluvatar_device(tensor):
    if tensor.device.type != "cuda":
        return False
    if not hasattr(torch, "cuda"):
        return False
    index = tensor.device.index
    if index is None:
        index = torch.cuda.current_device()
    return _is_iluvatar_device(index)


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


def _metax_fast_flatten(input, other, out):
    if input.ndim == 0:
        return input.reshape((1,)), other.reshape((1,)), out.reshape((1,))
    if input.is_contiguous() and other.is_contiguous() and out.is_contiguous():
        return (
            input.reshape((input.numel(),)),
            other.reshape((other.numel(),)),
            out.reshape((out.numel(),)),
        )
    if input.ndim <= 1 or tuple(input.shape) != tuple(other.shape) or tuple(input.shape) != tuple(out.shape):
        return None
    dims = tuple(sorted(range(input.ndim), key=lambda dim: input.stride()[dim], reverse=True))
    input_view = input.permute(dims)
    other_view = other.permute(dims)
    out_view = out.permute(dims)
    if input_view.is_contiguous() and other_view.is_contiguous() and out_view.is_contiguous():
        return (
            input_view.reshape((input.numel(),)),
            other_view.reshape((other.numel(),)),
            out_view.reshape((out.numel(),)),
        )
    return None


@functools.cache
def _get_kernel_1d(half, double, iluvatar_double=False, iluvatar_half=False):
    return _cached_make(
        ntops.kernels.copysign.premake,
        1,
        half,
        double,
        iluvatar_double,
        iluvatar_half,
        block_size=ntops.kernels.copysign.BLOCK_SIZE,
        num_warps=4,
        max_num_configs=1,
    )


@functools.cache
def _get_broadcast_2d_kernel(half, double, iluvatar_double=False, iluvatar_half=False):
    return _cached_make(
        ntops.kernels.copysign.premake,
        2,
        half,
        double,
        iluvatar_double,
        iluvatar_half,
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
        iluvatar_double = _use_iluvatar_double_kernel(input)
        iluvatar_half = half and _use_iluvatar_device(input)
        if out is None:
            out = torch.empty_like(input)
            if input.dtype == torch.float16 and _vendor_triton.is_metax_device(input):
                _vendor_triton.copysign_f16_1d(input, other, out)
                return out
            if input.dtype == torch.float64 and _vendor_triton.is_metax_device(input):
                _vendor_triton.copysign_f64_1d(input, other, out)
                return out
            if input.dtype == torch.float32 and _vendor_triton.is_corex_or_metax_device(input):
                _vendor_triton.copysign_f32_1d(input, other, out)
                return out
            _get_kernel_1d(half, double, iluvatar_double, iluvatar_half)(input, other, out)
            return out
        if tuple(out.shape) == tuple(input.shape) and out.dtype == input.dtype and out.is_contiguous():
            if input.dtype == torch.float16 and _vendor_triton.is_metax_device(input):
                _vendor_triton.copysign_f16_1d(input, other, out)
                return out
            if input.dtype == torch.float64 and _vendor_triton.is_metax_device(input):
                _vendor_triton.copysign_f64_1d(input, other, out)
                return out
            if input.dtype == torch.float32 and _vendor_triton.is_corex_or_metax_device(input):
                _vendor_triton.copysign_f32_1d(input, other, out)
                return out
            _get_kernel_1d(half, double, iluvatar_double, iluvatar_half)(input, other, out)
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
        iluvatar_double = _use_iluvatar_double_kernel(input)
        iluvatar_half = half and _use_iluvatar_device(input)
        out = torch.empty((rows, cols), dtype=input.dtype, device=input.device)
        if input.dtype == torch.float32 and _vendor_triton.is_corex_or_metax_device(input):
            _vendor_triton.copysign_f32_broadcast(input, other, out)
            return out
        _get_broadcast_2d_kernel(half, double, iluvatar_double, iluvatar_half)(
            input,
            other,
            out,
        )
        return out

    input, other = _broadcast(input, other)
    input, other, result_dtype = _prepare_inputs(input, other)
    out = _prepare_out(out, input.shape, result_dtype, input.device, like=input)

    if _vendor_triton.is_metax_device(input) and input.dtype.is_floating_point:
        fast_tensors = _metax_fast_flatten(input, other, out)
        if fast_tensors is not None:
            kernel_input, kernel_other, kernel_out = fast_tensors
            if input.dtype == torch.float16:
                _vendor_triton.copysign_f16_1d(kernel_input, kernel_other, kernel_out)
                return out
            if input.dtype == torch.float64:
                _vendor_triton.copysign_f64_1d(kernel_input, kernel_other, kernel_out)
                return out
            if input.dtype == torch.float32:
                _vendor_triton.copysign_f32_1d(kernel_input, kernel_other, kernel_out)
                return out

    kernel_input, kernel_other, kernel_out = _flatten_kernel_tensors(input, other, out)
    if (
        _vendor_triton.is_metax_device(input)
        and input.dtype == torch.float16
        and kernel_input.ndim == 1
        and kernel_input.is_contiguous()
        and kernel_other.ndim == 1
        and kernel_other.is_contiguous()
        and kernel_out.ndim == 1
        and kernel_out.is_contiguous()
    ):
        _vendor_triton.copysign_f16_1d(kernel_input, kernel_other, kernel_out)
        return out
    if (
        _vendor_triton.is_metax_device(input)
        and input.dtype == torch.float64
        and kernel_input.ndim == 1
        and kernel_input.is_contiguous()
        and kernel_other.ndim == 1
        and kernel_other.is_contiguous()
        and kernel_out.ndim == 1
        and kernel_out.is_contiguous()
    ):
        _vendor_triton.copysign_f64_1d(kernel_input, kernel_other, kernel_out)
        return out
    if (
        input.dtype == torch.float32
        and _vendor_triton.is_corex_or_metax_device(input)
        and kernel_input.ndim == 1
        and kernel_input.is_contiguous()
        and kernel_other.ndim == 1
        and kernel_other.is_contiguous()
        and kernel_out.ndim == 1
        and kernel_out.is_contiguous()
    ):
        _vendor_triton.copysign_f32_1d(kernel_input, kernel_other, kernel_out)
        return out
    kernel = _cached_make(
        ntops.kernels.copysign.premake,
        kernel_input.ndim,
        input.dtype == torch.float16,
        input.dtype == torch.float64,
        _use_iluvatar_double_kernel(input),
        input.dtype == torch.float16 and _use_iluvatar_device(input),
        block_size=ntops.kernels.copysign.BLOCK_SIZE,
        num_warps=4,
        max_num_configs=1,
    )
    kernel(kernel_input, kernel_other, kernel_out)

    return out
