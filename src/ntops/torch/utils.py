import functools
import warnings

import ninetoothed
import torch

import ntops


class _CachedMakeDefaultConfig:
    def __init__(self, num_warps=None, num_stages=None, max_num_configs=None):
        self.num_warps = num_warps

        self.num_stages = num_stages

        self.max_num_configs = max_num_configs


_cached_make_default_config = _CachedMakeDefaultConfig()


def get_default_num_warps():
    return _cached_make_default_config.num_warps


def set_default_num_warps(num_warps):
    _cached_make_default_config.num_warps = num_warps


def get_default_num_stages():
    return _cached_make_default_config.num_stages


def set_default_num_stages(num_stages):
    _cached_make_default_config.num_stages = num_stages


def get_default_max_num_configs():
    return _cached_make_default_config.max_num_configs


def set_default_max_num_configs(max_num_configs):
    _cached_make_default_config.max_num_configs = max_num_configs


@functools.cache
def _cached_make(
    premake, *args, num_warps=None, num_stages=None, max_num_configs=None, **keywords
):
    if num_warps is None:
        num_warps = _cached_make_default_config.num_warps

    if num_stages is None:
        num_stages = _cached_make_default_config.num_stages

    if max_num_configs is None:
        max_num_configs = _cached_make_default_config.max_num_configs

    return ninetoothed.make(
        *premake(*args, **keywords),
        num_warps=num_warps,
        num_stages=num_stages,
        max_num_configs=max_num_configs,
    )


def _reshape_tensor(tensor, shape):
    reshape = getattr(tensor, "reshape", None)
    if callable(reshape):
        return reshape(shape)
    return tensor.view(list(shape))


def _is_contiguous(tensor):
    is_contiguous = getattr(tensor, "is_contiguous", None)
    if callable(is_contiguous):
        return is_contiguous()
    return bool(is_contiguous)


def _strides(tensor):
    stride = getattr(tensor, "stride", None)
    if callable(stride):
        return tuple(stride())
    strides = getattr(tensor, "strides", None)
    if strides is not None:
        return tuple(strides)
    return None


def _permute_tensor(tensor, dims):
    permute = getattr(tensor, "permute", None)
    if callable(permute):
        return permute(dims)
    raise TypeError("tensor does not support permute")


def _physical_contiguous_permutation(tensors):
    if not tensors:
        return None

    ndim = tensors[0].ndim
    shape = tuple(tensors[0].shape)
    if ndim <= 1 or any(tensor.ndim != ndim or tuple(tensor.shape) != shape for tensor in tensors):
        return None

    strides = _strides(tensors[0])
    if strides is None or _is_contiguous(tensors[0]):
        return None

    dims = tuple(sorted(range(ndim), key=lambda dim: strides[dim], reverse=True))
    if dims == tuple(range(ndim)):
        return None

    try:
        if not _is_contiguous(_permute_tensor(tensors[0], dims)):
            return None
        if not all(_is_contiguous(_permute_tensor(tensor, dims)) for tensor in tensors[1:]):
            return None
    except TypeError:
        return None

    return dims


def _flatten_kernel_tensors(*tensors):
    kernel_tensors = tuple(
        _reshape_tensor(tensor, (1,)) if tensor.ndim == 0 else tensor
        for tensor in tensors
    )
    if all(tensor.ndim == 1 and _is_contiguous(tensor) for tensor in kernel_tensors):
        return kernel_tensors

    physical_order = _physical_contiguous_permutation(kernel_tensors)
    if physical_order is not None:
        kernel_tensors = tuple(_permute_tensor(tensor, physical_order) for tensor in kernel_tensors)

    if all(tensor.ndim > 0 and _is_contiguous(tensor) for tensor in kernel_tensors):
        return tuple(_reshape_tensor(tensor, (tensor.numel(),)) for tensor in kernel_tensors)
    return kernel_tensors


def _check_out_dtype(result_dtype, out):
    if out is None:
        return

    try:
        can_cast = torch.can_cast(result_dtype, out.dtype) if hasattr(torch, "can_cast") else True
    except TypeError:
        can_cast = result_dtype == out.dtype

    if not can_cast:
        raise RuntimeError(
            f"result type {result_dtype} can't be cast to the desired output type {out.dtype}"
        )


def _prepare_out(out, shape, dtype, device, like=None):
    _check_out_dtype(dtype, out)
    shape = tuple(shape)

    if out is None:
        if like is not None and tuple(like.shape) == shape and like.dtype == dtype:
            try:
                return torch.empty_like(like)
            except TypeError:
                import infinicore

                return infinicore.empty_like(like, dtype=dtype, device=device)
        try:
            return torch.empty(shape, dtype=dtype, device=device)
        except TypeError:
            import infinicore

            return infinicore.empty(list(shape), dtype=dtype, device=device)

    if tuple(out.shape) != tuple(shape):
        warnings.warn(
            (
                f"An output with one or more elements was resized since it had shape "
                f"{tuple(out.shape)}, which does not match the required output shape "
                f"{tuple(shape)}."
            ),
            UserWarning,
            stacklevel=2,
        )
        out.resize_(shape)

    return out


def _get_matmul_input_precision():
    if torch.get_float32_matmul_precision() == "highest":
        return ntops.kernels.mm.InputPrecisionVariant.IEEE

    return ntops.kernels.mm.InputPrecisionVariant.TF32
