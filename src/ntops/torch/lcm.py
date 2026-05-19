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
    if result_dtype == torch.bool or result_dtype.is_floating_point:
        raise NotImplementedError(f"lcm is not implemented for {result_dtype}")

    return input.to(result_dtype), other.to(result_dtype), result_dtype


def _iterations_for_dtype(dtype):
    if hasattr(torch, "int8") and dtype in (torch.int8, torch.uint8):
        return 16
    if hasattr(torch, "int16") and dtype == torch.int16:
        return 24
    if hasattr(torch, "int32") and dtype == torch.int32:
        return 48
    return 96


def _uses_absolute_overflow(dtype):
    return dtype in (torch.int32, torch.int64)


def _num_warps_for_dtype(dtype):
    return 1


def _block_size_for_dtype(dtype):
    return 32 if dtype == torch.int64 else ntops.kernels.lcm.BLOCK_SIZE


def _uses_small_integer_kernel(dtype):
    return dtype in (torch.int8, torch.uint8, torch.int16)


@functools.cache
def _get_kernel_1d(dtype):
    return _cached_make(
        ntops.kernels.lcm.premake,
        1,
        _iterations_for_dtype(dtype),
        _uses_absolute_overflow(dtype),
        dynamic_iterations=True,
        small_integer=_uses_small_integer_kernel(dtype),
        block_size=_block_size_for_dtype(dtype),
        num_warps=_num_warps_for_dtype(dtype),
        max_num_configs=1,
    )


@functools.cache
def _get_broadcast_2d_kernel(dtype):
    return _cached_make(
        ntops.kernels.lcm.premake,
        2,
        _iterations_for_dtype(dtype),
        _uses_absolute_overflow(dtype),
        dynamic_iterations=True,
        small_integer=_uses_small_integer_kernel(dtype),
        broadcast_2d=True,
        block_size=_block_size_for_dtype(dtype),
        num_warps=_num_warps_for_dtype(dtype),
        max_num_configs=1,
    )


def lcm(input, other, *, out=None):
    if (
        input.ndim == 1
        and other.ndim == 1
        and tuple(input.shape) == tuple(other.shape)
        and input.dtype == other.dtype
        and not input.dtype.is_floating_point
        and input.dtype != torch.bool
        and input.is_contiguous()
        and other.is_contiguous()
    ):
        if out is None:
            out = torch.empty_like(input)
            if _iluvatar_triton.is_iluvatar_device(input):
                if input.dtype == torch.uint8:
                    _iluvatar_triton.lcm_u8_1d(input, other, out)
                else:
                    _iluvatar_triton.lcm_1d(
                        input,
                        other,
                        out,
                        _iterations_for_dtype(input.dtype),
                        _uses_absolute_overflow(input.dtype),
                    )
                return out
            _get_kernel_1d(input.dtype)(input, other, out)
            return out
        if tuple(out.shape) == tuple(input.shape) and out.dtype == input.dtype and out.is_contiguous():
            if _iluvatar_triton.is_iluvatar_device(input):
                if input.dtype == torch.uint8:
                    _iluvatar_triton.lcm_u8_1d(input, other, out)
                else:
                    _iluvatar_triton.lcm_1d(
                        input,
                        other,
                        out,
                        _iterations_for_dtype(input.dtype),
                        _uses_absolute_overflow(input.dtype),
                    )
                return out
            _get_kernel_1d(input.dtype)(input, other, out)
            return out

    if (
        out is None
        and input.ndim == 2
        and other.ndim == 2
        and input.shape[1] == 1
        and other.shape[0] == 1
        and input.dtype == other.dtype
        and not input.dtype.is_floating_point
        and input.dtype != torch.bool
        and input.is_contiguous()
        and other.is_contiguous()
    ):
        rows = input.shape[0]
        cols = other.shape[1]
        out = torch.empty((rows, cols), dtype=input.dtype, device=input.device)
        if _iluvatar_triton.is_iluvatar_device(input):
            _iluvatar_triton.lcm_broadcast(
                input,
                other,
                out,
                _iterations_for_dtype(input.dtype),
                _uses_absolute_overflow(input.dtype),
            )
            return out
        _get_broadcast_2d_kernel(input.dtype)(input, other, out)
        return out

    input, other = _broadcast(input, other)
    input, other, result_dtype = _prepare_inputs(input, other)
    out = _prepare_out(out, input.shape, result_dtype, input.device, like=input)

    kernel_input, kernel_other, kernel_out = _flatten_kernel_tensors(input, other, out)
    if (
        _iluvatar_triton.is_iluvatar_device(input)
        and kernel_input.ndim == 1
        and kernel_other.ndim == 1
        and kernel_out.ndim == 1
        and kernel_input.is_contiguous()
        and kernel_other.is_contiguous()
        and kernel_out.is_contiguous()
    ):
        if input.dtype == torch.uint8:
            _iluvatar_triton.lcm_u8_1d(kernel_input, kernel_other, kernel_out)
        else:
            _iluvatar_triton.lcm_1d(
                kernel_input,
                kernel_other,
                kernel_out,
                _iterations_for_dtype(input.dtype),
                _uses_absolute_overflow(input.dtype),
            )
        return out

    kernel = _cached_make(
        ntops.kernels.lcm.premake,
        kernel_input.ndim,
        _iterations_for_dtype(input.dtype),
        _uses_absolute_overflow(input.dtype),
        dynamic_iterations=True,
        small_integer=_uses_small_integer_kernel(input.dtype),
        block_size=_block_size_for_dtype(input.dtype),
        num_warps=_num_warps_for_dtype(input.dtype),
        max_num_configs=1,
    )
    kernel(kernel_input, kernel_other, kernel_out)

    return out
