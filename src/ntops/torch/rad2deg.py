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


def _metax_fast_flatten(input, out):
    if input.ndim == 0:
        return input.reshape((1,)), out.reshape((1,))
    if input.is_contiguous() and out.is_contiguous():
        return input.reshape((input.numel(),)), out.reshape((out.numel(),))
    if input.ndim <= 1 or tuple(input.shape) != tuple(out.shape):
        return None
    dims = tuple(sorted(range(input.ndim), key=lambda dim: input.stride()[dim], reverse=True))
    input_view = input.permute(dims)
    out_view = out.permute(dims)
    if input_view.is_contiguous() and out_view.is_contiguous():
        return input_view.reshape((input.numel(),)), out_view.reshape((out.numel(),))
    return None


def rad2deg(input, *, out=None):
    input = _promote_unary_input(input)

    if input.ndim == 1 and input.is_contiguous():
        iluvatar_double = _use_iluvatar_double_kernel(input)
        if out is None:
            out = torch.empty_like(input)
            if input.dtype == torch.float32 and _vendor_triton.is_corex_or_metax_device(input):
                _vendor_triton.rad2deg_f32_1d(input, out)
                return out
            _get_kernel_1d(iluvatar_double)(input, out)
            return out
        if tuple(out.shape) == tuple(input.shape) and out.dtype == input.dtype and out.is_contiguous():
            if input.dtype == torch.float32 and _vendor_triton.is_corex_or_metax_device(input):
                _vendor_triton.rad2deg_f32_1d(input, out)
                return out
            _get_kernel_1d(iluvatar_double)(input, out)
            return out

    out = _prepare_out(out, input.shape, input.dtype, input.device, like=input)

    if _vendor_triton.is_metax_device(input):
        fast_tensors = _metax_fast_flatten(input, out)
        if fast_tensors is not None:
            kernel_input, kernel_out = fast_tensors
            _vendor_triton.rad2deg_1d(kernel_input, kernel_out)
            return out

    kernel_input, kernel_out = _flatten_kernel_tensors(input, out)
    if (
        _vendor_triton.is_metax_device(input)
        and not input.is_contiguous()
        and kernel_input.ndim == 1
        and kernel_out.ndim == 1
        and kernel_input.is_contiguous()
        and kernel_out.is_contiguous()
    ):
        _vendor_triton.rad2deg_1d(kernel_input, kernel_out)
        return out
    if (
        input.dtype == torch.float32
        and _vendor_triton.is_corex_or_metax_device(input)
        and kernel_input.ndim == 1
        and kernel_out.ndim == 1
        and kernel_input.is_contiguous()
        and kernel_out.is_contiguous()
    ):
        _vendor_triton.rad2deg_f32_1d(kernel_input, kernel_out)
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
