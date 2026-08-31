"""Torch wrapper for fused MLA RoPE and compressed cache write."""

import functools

import torch

import ntops
from ntops.torch.mla_rope_cache_pair import (
    mla_rope_cache_pair,
    should_use_pair_layout,
)
from ntops.torch.utils import _cached_make

_LAST_VALIDATED = None
_LAST_VIEWS = None
_LAST_KERNEL = None
_LAST_CALL = None
_LAST_PAIR_OBJECTS = None
_GRAPH_UNAVAILABLE = object()


def _capture_or_launch(invoke, q_pe, slot_mapping, kv_cache):
    """Execute once while arming graph replay for stable inference buffers."""
    if not getattr(invoke, "_ninetoothed_external_graph_capture", True):
        invoke()
        return _GRAPH_UNAVAILABLE

    graph_type = getattr(torch.cuda, "CUDAGraph", None)
    graph_context = getattr(torch.cuda, "graph", None)

    if graph_type is None or graph_context is None:
        invoke()
        return _GRAPH_UNAVAILABLE

    q_backup = None
    cache_backup = None
    cache_flat = None
    slots = None
    capture_started = False

    try:
        # Capturing executes the kernel once.  Preserve only the locations the
        # in-place operator may update so a failed capture can safely fall back
        # without cloning the potentially very large paged KV cache.
        slots = slot_mapping.to(dtype=torch.long)
        cache_flat = kv_cache.view(-1, kv_cache.shape[-1])
        q_backup = q_pe.clone()
        cache_backup = cache_flat.index_select(0, slots)
        torch.cuda.synchronize()

        graph = graph_type()
        capture_started = True

        with graph_context(graph):
            invoke()

        # CUDA executes captured work while recording, whereas the BW graph
        # runtime records it without applying the in-place update.  Detect the
        # observed semantics once so this public invocation executes exactly
        # once on either backend.
        torch.cuda.synchronize()
        q_changed = not torch.equal(q_pe, q_backup)
        cache_changed = not torch.equal(
            cache_flat.index_select(0, slots),
            cache_backup,
        )

        if not q_changed and not cache_changed:
            graph.replay()

        return graph
    except (AttributeError, RuntimeError, TypeError):
        if capture_started and q_backup is not None and cache_backup is not None:
            # A backend may reject capture after partially launching work.
            # Restore the original outputs before executing the checked path.
            torch.cuda.synchronize()
            q_pe.copy_(q_backup)
            cache_flat.index_copy_(0, slots, cache_backup)

        invoke()
        return _GRAPH_UNAVAILABLE


def _validate(q_pe, k_pe, kv_c, positions, cos_sin_cache, slot_mapping, kv_cache):
    if q_pe.ndim != 3 or k_pe.ndim != 3 or k_pe.shape[1] != 1:
        raise ValueError("q_pe and k_pe must have shapes (T, H, R) and (T, 1, R).")
    tokens, _heads, rope_dim = q_pe.shape
    if k_pe.shape != (tokens, 1, rope_dim):
        raise ValueError("q_pe and k_pe must share token and RoPE dimensions.")
    if kv_c.ndim != 2 or kv_c.shape[0] != tokens:
        raise ValueError("kv_c must have shape (T, kv_lora_rank).")
    if positions.shape != (tokens,) or slot_mapping.shape != (tokens,):
        raise ValueError("positions and slot_mapping must have shape (T,).")
    if kv_cache.ndim != 3 or kv_cache.shape[-1] != kv_c.shape[1] + rope_dim:
        raise ValueError("kv_cache entry width must equal kv_lora_rank + rope_dim.")
    if cos_sin_cache.ndim != 2 or cos_sin_cache.shape[1] != rope_dim:
        raise ValueError("cos_sin_cache must have shape (max_position, rope_dim).")
    if rope_dim % 2:
        raise ValueError("rope_dim must be even.")
    if q_pe.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(
            "q_pe, k_pe, kv_c, cos_sin_cache and kv_cache must be FP16/BF16."
        )
    if any(
        tensor.dtype != q_pe.dtype
        for tensor in (k_pe, kv_c, cos_sin_cache, kv_cache)
    ):
        raise TypeError("All floating-point inputs must have the same dtype.")
    if positions.dtype not in (torch.int32, torch.int64):
        raise TypeError("positions must use int32 or int64.")
    if slot_mapping.dtype not in (torch.int32, torch.int64):
        raise TypeError("slot_mapping must use int32 or int64.")
    if any(
        tensor.device != q_pe.device
        for tensor in (k_pe, kv_c, positions, cos_sin_cache, slot_mapping, kv_cache)
    ):
        raise ValueError("All inputs must be on the same device.")
    if not all(
        tensor.is_contiguous()
        for tensor in (
            q_pe,
            k_pe,
            kv_c,
            positions,
            cos_sin_cache,
            slot_mapping,
            kv_cache,
        )
    ):
        raise ValueError("All T4 inputs must be contiguous.")


def mla_rope_cache(
    q_pe,
    k_pe,
    kv_c,
    positions,
    cos_sin_cache,
    slot_mapping,
    kv_cache,
):
    """Rotate q/k position channels and write ``kv_c + rotated(k_pe)``.

    ``q_pe`` and ``kv_cache`` are updated in place.  Rotated ``k_pe`` is kept
    inside the kernel and is never materialized in global memory.
    """
    global _LAST_CALL, _LAST_KERNEL, _LAST_PAIR_OBJECTS
    global _LAST_VALIDATED, _LAST_VIEWS

    objects = (q_pe, k_pe, kv_c, positions, cos_sin_cache, slot_mapping, kv_cache)

    if _LAST_PAIR_OBJECTS is not None and all(
        current is previous
        for current, previous in zip(objects, _LAST_PAIR_OBJECTS)
    ):
        return mla_rope_cache_pair(*objects)

    # The scalar lane layout has the lowest launch cost for decode and short
    # prefill.  Once the query domain is large enough, the alias-safe pair
    # layout halves CUDA threads and reuses each pair's cos/sin values.  The
    # decision depends only on backend capability and workload size.
    cached_call = _LAST_CALL

    if (
        cached_call is not None
        and q_pe is cached_call[0][0]
        and k_pe is cached_call[0][1]
        and kv_c is cached_call[0][2]
        and positions is cached_call[0][3]
        and cos_sin_cache is cached_call[0][4]
        and slot_mapping is cached_call[0][5]
        and kv_cache is cached_call[0][6]
    ):
        invoke, graph = cached_call[1:3]

        if graph is None:
            graph = _capture_or_launch(invoke, q_pe, slot_mapping, kv_cache)
            replay = invoke if graph is _GRAPH_UNAVAILABLE else graph.replay
            _LAST_CALL = (objects, replay, graph)
        else:
            invoke()

        return q_pe

    if should_use_pair_layout(q_pe, kv_c):
        _LAST_PAIR_OBJECTS = objects
        return mla_rope_cache_pair(*objects)

    same_objects = _LAST_VALIDATED is not None and all(
        current is previous for current, previous in zip(objects, _LAST_VALIDATED[0])
    )
    if not same_objects:
        _validate(q_pe, k_pe, kv_c, positions, cos_sin_cache, slot_mapping, kv_cache)
        _LAST_VALIDATED = (objects,)

    view_objects = (q_pe, k_pe, kv_cache)
    if _LAST_VIEWS is not None and all(
        current is previous
        for current, previous in zip(
            view_objects, _LAST_VIEWS[:3]
        )
    ):
        q_flat, k_flat, cache_flat = _LAST_VIEWS[3:6]
    else:
        q_flat = q_pe.view(q_pe.shape[0], -1)
        k_flat = k_pe.view(k_pe.shape[0], -1)
        cache_flat = kv_cache.view(-1, kv_cache.shape[-1])
        _LAST_VIEWS = (
            q_pe, k_pe, kv_cache, q_flat, k_flat, cache_flat
        )

    tensors = (q_flat, k_flat, kv_c, positions, cos_sin_cache, slot_mapping, cache_flat)
    if _LAST_KERNEL is not None and all(
        current is previous for current, previous in zip(tensors, _LAST_KERNEL[0])
    ):
        kernel = _LAST_KERNEL[1]
    else:
        kernel = _cached_make(
            ntops.kernels.mla_rope_cache.premake,
            *(tuple(tensor.shape) for tensor in tensors),
            *(tensor.dtype for tensor in tensors),
            *(tuple(tensor.stride()) for tensor in tensors),
            num_warps=ntops.kernels.mla_rope_cache.NUM_WARPS,
            num_stages=ntops.kernels.mla_rope_cache.NUM_STAGES,
            max_num_configs=1,
        )
        _LAST_KERNEL = (tensors, kernel)
    launch = getattr(kernel, "_launch_prevalidated_noalias", kernel)
    launch(*tensors)

    bind = getattr(launch, "_ninetoothed_bind_prevalidated_noalias", None)
    invoke_cached = bind(*tensors) if bind is not None else None

    if invoke_cached is None:
        invoke_cached = functools.partial(launch, *tensors)

    _LAST_CALL = (objects, invoke_cached, None)
    return q_pe


__all__ = ["mla_rope_cache"]
