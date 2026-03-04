import functools

import ninetoothed
import torch
import torch.nn.functional as F

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


def _get_matmul_input_precision():
    if torch.get_float32_matmul_precision() == "highest":
        return ntops.kernels.mm.InputPrecisionVariant.IEEE

    return ntops.kernels.mm.InputPrecisionVariant.TF32


# Current ninetoothed (0.23.0) does not support `ninetoothed.Tensor.pad` yet,
# so we use `torch.nn.functional.pad` for now.
# Todo: Switch to `ninetoothed.Tensor.pad` once it's supported.
def _pad_dims_to_next_power_of_2(tensor, dims, padding_right=True, value=0):
    if isinstance(dims, int):
        target_dims = [dims]
    elif isinstance(dims, (list, tuple)):
        target_dims = list(dims)
    else:
        raise ValueError("dims must be an int or a list/tuple of ints")

    for i, d in enumerate(target_dims):
        if d < 0:
            d += tensor.ndim

        if d < 0 or d >= tensor.ndim:
            raise ValueError(f"Invalid dims: {dims}")

        target_dims[i] = d

    padding = [0] * (tensor.ndim * 2)
    padding_flag = False

    for d in target_dims:
        current_len = tensor.size(d)

        if (current_len & (current_len - 1)) == 0:
            continue
        else:
            padding_flag = True
            exponent = current_len.bit_length()
            target_len = 1 << exponent

        pad_len = target_len - current_len

        pad_idx = (tensor.ndim - 1 - d) * 2 + (1 if padding_right else 0)
        padding[pad_idx] = pad_len

    if not padding_flag:
        return tensor

    padded_tensor = F.pad(tensor, padding, mode="constant", value=value)

    return padded_tensor
