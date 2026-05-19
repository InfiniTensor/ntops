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
    if not result_dtype.is_floating_point:
        raise NotImplementedError("nextafter is only implemented for floating point inputs")

    return input.to(result_dtype), other.to(result_dtype), result_dtype


def _kernel_config(half, double, iluvatar):
    if iluvatar and not half and not double:
        return 256, 4
    return ntops.kernels.nextafter.BLOCK_SIZE, 1


@functools.cache
def _get_kernel_1d(half, double, iluvatar=False):
    block_size, num_warps = _kernel_config(half, double, iluvatar)
    return _cached_make(
        ntops.kernels.nextafter.premake,
        1,
        half,
        double,
        iluvatar=iluvatar and not half and not double,
        block_size=block_size,
        num_warps=num_warps,
        max_num_configs=1,
    )


@functools.cache
def _get_broadcast_2d_kernel(half, double, iluvatar=False):
    block_size, num_warps = _kernel_config(half, double, iluvatar)
    return _cached_make(
        ntops.kernels.nextafter.premake,
        2,
        half,
        double,
        True,
        iluvatar=iluvatar and not half and not double,
        block_size=block_size,
        num_warps=num_warps,
        max_num_configs=1,
    )


def nextafter(input, other, *, out=None):
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
        iluvatar = _use_iluvatar_device(input)
        if out is None:
            out = torch.empty_like(input)
            if half and iluvatar:
                _iluvatar_triton.nextafter_f16_1d(input, other, out)
                return out
            _get_kernel_1d(half, double, iluvatar)(input, other, out)
            return out
        if tuple(out.shape) == tuple(input.shape) and out.dtype == input.dtype and out.is_contiguous():
            if half and iluvatar:
                _iluvatar_triton.nextafter_f16_1d(input, other, out)
                return out
            _get_kernel_1d(half, double, iluvatar)(input, other, out)
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
        iluvatar = _use_iluvatar_device(input)
        out = torch.empty((rows, cols), dtype=input.dtype, device=input.device)
        if half and iluvatar:
            _iluvatar_triton.nextafter_f16_broadcast(input, other, out)
            return out
        if input.dtype == torch.float32 and iluvatar:
            _iluvatar_triton.nextafter_f32_broadcast(input, other, out)
            return out
        _get_broadcast_2d_kernel(half, double, iluvatar)(
            input,
            other,
            out,
        )
        return out

    input, other = _broadcast(input, other)
    input, other, result_dtype = _prepare_inputs(input, other)
    out = _prepare_out(out, input.shape, result_dtype, input.device, like=input)

    half = hasattr(torch, "float16") and input.dtype == torch.float16
    double = hasattr(torch, "float64") and input.dtype == torch.float64
    iluvatar = _use_iluvatar_device(input)

    kernel_input, kernel_other, kernel_out = _flatten_kernel_tensors(input, other, out)
    if (
        half
        and iluvatar
        and kernel_input.ndim == 1
        and kernel_input.is_contiguous()
        and kernel_other.ndim == 1
        and kernel_other.is_contiguous()
        and kernel_out.ndim == 1
        and kernel_out.is_contiguous()
    ):
        _iluvatar_triton.nextafter_f16_1d(kernel_input, kernel_other, kernel_out)
        return out
    if half and iluvatar and _iluvatar_triton.nextafter_f16_strided(input, other, out):
        return out
    kernel = _cached_make(
        ntops.kernels.nextafter.premake,
        kernel_input.ndim,
        half,
        double,
        iluvatar=iluvatar and not half and not double,
        block_size=_kernel_config(half, double, iluvatar)[0],
        num_warps=_kernel_config(half, double, iluvatar)[1],
        max_num_configs=1,
    )
    kernel(kernel_input, kernel_other, kernel_out)

    return out
