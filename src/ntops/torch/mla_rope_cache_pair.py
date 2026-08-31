"""Internal pair-vectorized path for large T4 CUDA workloads."""

import functools
import os

import torch

import ntops.kernels.mla_rope_cache_pair as _kernel_module
from ntops.torch.utils import _cached_make

_LAST = None
_LAST_PUBLIC = None
_LAST_VIEWS = None
_PAIR_QUERY_ELEMENTS_THRESHOLD = 1 << 18


def should_use_pair_layout(q_pe, kv_c):
    """Choose pair vectorization from backend capability and workload size."""
    backend = os.environ.get("NINETOOTHED_BACKEND", "triton").lower()
    if backend != "cuda" or q_pe.ndim != 3:
        return False

    q_pairs_per_token = q_pe.shape[1] * q_pe.shape[2] // 2
    cache_width = kv_c.shape[-1] + q_pe.shape[-1]
    return (
        q_pairs_per_token >= cache_width
        and q_pe.numel() >= _PAIR_QUERY_ELEMENTS_THRESHOLD
    )


def mla_rope_cache_pair(
    q_pe,
    k_pe,
    kv_c,
    positions,
    cos_sin_cache,
    slot_mapping,
    kv_cache,
):
    """Run one-thread-per-RoPE-pair on contiguous T4 inputs."""
    global _LAST, _LAST_PUBLIC, _LAST_VIEWS
    public_objects = (
        q_pe,
        k_pe,
        kv_c,
        positions,
        cos_sin_cache,
        slot_mapping,
        kv_cache,
    )
    if _LAST_PUBLIC is not None and all(
        current is previous
        for current, previous in zip(public_objects, _LAST_PUBLIC[0])
    ):
        _LAST_PUBLIC[1]()
        return q_pe

    if q_pe.ndim != 3 or k_pe.shape[1] != 1:
        raise ValueError("q_pe and k_pe have unsupported shapes.")
    tokens, heads, rope_dim = q_pe.shape
    if rope_dim % 2 or q_pe.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("The pair candidate requires an even FP16/BF16 RoPE width.")
    if q_pe.device.type != "cuda":
        raise ValueError("The pair candidate requires a CUDA-compatible device.")
    if not all(
        tensor.is_contiguous()
        for tensor in (q_pe, k_pe, kv_c, positions, cos_sin_cache, slot_mapping, kv_cache)
    ):
        raise ValueError("T4 inputs must be contiguous.")

    view_inputs = (q_pe, k_pe, kv_cache)
    if _LAST_VIEWS is not None and all(
        current is previous
        for current, previous in zip(view_inputs, _LAST_VIEWS[:3])
    ):
        q_even, q_odd, k_even, k_odd, cache_flat = _LAST_VIEWS[3:]
    else:
        q_output = q_pe.view(tokens, heads * rope_dim)
        q_even = q_output[:, 0::2]
        q_odd = q_output[:, 1::2]
        k_flat = k_pe.view(tokens, rope_dim)
        k_even = k_flat[:, 0::2]
        k_odd = k_flat[:, 1::2]
        cache_flat = kv_cache.view(-1, kv_cache.shape[-1])
        _LAST_VIEWS = (
            q_pe,
            k_pe,
            kv_cache,
            q_even,
            q_odd,
            k_even,
            k_odd,
            cache_flat,
        )
    tensors = (
        q_even,
        q_odd,
        k_even,
        k_odd,
        kv_c,
        positions,
        cos_sin_cache,
        slot_mapping,
        cache_flat,
    )

    if _LAST is not None and all(a is b for a, b in zip(_LAST[0], tensors)):
        _LAST[1]()
        return q_pe

    q_pairs = q_even.shape[-1]
    cache_width = kv_c.shape[-1] + rope_dim
    if q_pairs < cache_width:
        raise ValueError("The pair candidate requires q_pairs >= cache width.")
    kernel = _cached_make(
        _kernel_module.premake,
        *(tuple(tensor.shape) for tensor in tensors),
        *(tensor.dtype for tensor in tensors),
        *(tuple(tensor.stride()) for tensor in tensors),
        num_warps=_kernel_module.NUM_WARPS,
        num_stages=_kernel_module.NUM_STAGES,
        max_num_configs=1,
    )
    launch = getattr(kernel, "_launch_prevalidated_noalias", kernel)
    launch(*tensors)
    bind = getattr(launch, "_ninetoothed_bind_prevalidated_noalias", None)
    invoke = bind(*tensors) if bind is not None else functools.partial(launch, *tensors)
    _LAST = (tensors, invoke)
    _LAST_PUBLIC = (public_objects, invoke)
    return q_pe


__all__ = ["mla_rope_cache_pair", "should_use_pair_layout"]
