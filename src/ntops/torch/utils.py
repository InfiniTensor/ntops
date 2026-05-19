import functools
import os

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


def _is_corex_compat_device(device=None):
    backend = os.getenv("NTOPS_BACKEND", "").strip().lower()
    if backend in {"corex", "iluvatar", "tian", "mr-v100"}:
        return True
    if backend in {"cuda", "nvidia"}:
        return False

    if not torch.cuda.is_available():
        return False

    if device is not None and getattr(device, "type", None) != "cuda":
        return False

    index = getattr(device, "index", None)
    if index is None:
        index = torch.cuda.current_device()

    try:
        name = torch.cuda.get_device_name(index).lower()
    except Exception:
        return False

    return "iluvatar" in name or "mr-v100" in name or "corex" in name


def _torch_binary_fallback(op_name, input, other, out):
    if not _is_infinicore_tensor(input):
        return getattr(torch, op_name)(input, other, out=out)

    input_torch = _infinicore_to_torch(input)
    other_torch = _infinicore_to_torch(other)
    result = getattr(torch, op_name)(input_torch, other_torch)
    _copy_torch_to_infinicore(result, out)
    return out


def _is_infinicore_tensor(value):
    return hasattr(value, "_underlying") and hasattr(value, "copy_")


def _infinicore_to_torch(value):
    if not _is_infinicore_tensor(value):
        return value

    result = torch.empty_strided(
        tuple(value.shape),
        tuple(value.stride()),
        dtype=_to_torch_dtype(value.dtype),
        device=str(value.device),
    )
    _infinicore_from_torch(result).copy_(value)
    return result


def _copy_torch_to_infinicore(value, out):
    if _is_infinicore_tensor(out):
        out.copy_(_infinicore_from_torch(value))
    else:
        out.copy_(value)


def _infinicore_from_torch(value):
    infinicore = __import__("infinicore")
    infini_device = infinicore.device(value.device.type, value.device.index or 0)
    kwargs = {"dtype": _to_infinicore_dtype(value.dtype), "device": infini_device}
    if value.is_contiguous():
        return infinicore.from_blob(value.data_ptr(), list(value.shape), **kwargs)
    return infinicore.strided_from_blob(
        value.data_ptr(), list(value.shape), list(value.stride()), **kwargs
    )


def _to_torch_dtype(dtype):
    infinicore = __import__("infinicore")
    mapping = {
        infinicore.float16: torch.float16,
        infinicore.bfloat16: torch.bfloat16,
        infinicore.float32: torch.float32,
        infinicore.float64: torch.float64,
    }
    return mapping.get(dtype, dtype)


def _to_infinicore_dtype(dtype):
    infinicore = __import__("infinicore")
    mapping = {
        torch.float16: infinicore.float16,
        torch.bfloat16: infinicore.bfloat16,
        torch.float32: infinicore.float32,
        torch.float64: infinicore.float64,
    }
    return mapping[dtype]


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
