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


@functools.cache
def _get_kernel_1d(iluvatar_double=False):
    return _cached_make(
        ntops.kernels.rad2deg.premake,
        1,
        block_size=ntops.kernels.rad2deg.BLOCK_SIZE,
        iluvatar_double=iluvatar_double,
        num_warps=2,
        max_num_configs=1,
    )


def _promote_unary_input(input):
    if hasattr(torch, "is_floating_point") and not torch.is_floating_point(input):
        return input.to(torch.float32)
    return input


def rad2deg(input, *, out=None):
    input = _promote_unary_input(input)

    if input.ndim == 1 and input.is_contiguous():
        iluvatar_double = _use_iluvatar_double_kernel(input)
        if out is None:
            out = torch.empty_like(input)
            if input.dtype == torch.float32 and _iluvatar_triton.is_iluvatar_device(input):
                _iluvatar_triton.rad2deg_f32_1d(input, out)
                return out
            _get_kernel_1d(iluvatar_double)(input, out)
            return out
        if tuple(out.shape) == tuple(input.shape) and out.dtype == input.dtype and out.is_contiguous():
            if input.dtype == torch.float32 and _iluvatar_triton.is_iluvatar_device(input):
                _iluvatar_triton.rad2deg_f32_1d(input, out)
                return out
            _get_kernel_1d(iluvatar_double)(input, out)
            return out

    out = _prepare_out(out, input.shape, input.dtype, input.device, like=input)

    kernel_input, kernel_out = _flatten_kernel_tensors(input, out)
    if (
        input.dtype == torch.float32
        and _iluvatar_triton.is_iluvatar_device(input)
        and kernel_input.ndim == 1
        and kernel_out.ndim == 1
        and kernel_input.is_contiguous()
        and kernel_out.is_contiguous()
    ):
        _iluvatar_triton.rad2deg_f32_1d(kernel_input, kernel_out)
        return out
    kernel = _cached_make(
        ntops.kernels.rad2deg.premake,
        kernel_input.ndim,
        block_size=ntops.kernels.rad2deg.BLOCK_SIZE,
        iluvatar_double=_use_iluvatar_double_kernel(input),
        num_warps=2,
        max_num_configs=1,
    )
    kernel(kernel_input, kernel_out)

    return out
