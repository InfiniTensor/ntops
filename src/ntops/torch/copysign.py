import functools

import torch

import ntops
from ntops.torch import _iluvatar_triton
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
            _get_kernel_1d(half, double, iluvatar_double, iluvatar_half)(input, other, out)
            return out
        if tuple(out.shape) == tuple(input.shape) and out.dtype == input.dtype and out.is_contiguous():
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
        if input.dtype == torch.float32 and _iluvatar_triton.is_iluvatar_device(input):
            _iluvatar_triton.copysign_f32_broadcast(input, other, out)
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

    kernel_input, kernel_other, kernel_out = _flatten_kernel_tensors(input, other, out)
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
