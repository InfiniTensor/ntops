"""Pair-layout candidate for the fused MLA RoPE/cache kernel.

This module is intentionally kept separate while the layout is being
validated on vendor hardware.  It stores the even and odd RoPE channels in
two stride-2 views and assigns both views from one pair lane, so each CUDA
thread reuses the pair inputs and frequency values without an in-place race.
"""

import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor

NUM_WARPS = 2
NUM_STAGES = 1


def _row(tensor, width):
    arranged = tensor.tile((1, width))
    arranged.dtype = arranged.dtype.squeeze(0)
    return arranged


def _padded_row(tensor, width, padding):
    arranged = tensor.tile((1, width))
    arranged.dtype = arranged.dtype.squeeze(0).pad((padding,))
    return arranged


def _scalar(tensor):
    arranged = tensor.tile((1,))
    arranged.dtype = arranged.dtype.squeeze(0)
    return arranged


def arrangement(
    q_even,
    q_odd,
    k_even,
    k_odd,
    kv_c,
    positions,
    cos_sin_cache,
    slot_mapping,
    kv_cache,
    *,
    q_pairs,
    rope_dim,
    kv_lora_rank,
):
    cache_width = kv_lora_rank + rope_dim
    padding = q_pairs - cache_width
    k_padding = q_pairs - kv_lora_rank - rope_dim // 2
    if padding < 0:
        raise ValueError("q pair width must cover the MLA cache row.")
    if k_padding < 0:
        raise ValueError("q pair width must cover the padded RoPE pair row.")

    return (
        _row(q_even, q_pairs),
        _row(q_odd, q_pairs),
        _padded_row(
            k_even,
            rope_dim // 2,
            (kv_lora_rank, k_padding),
        ),
        _padded_row(
            k_odd,
            rope_dim // 2,
            (kv_lora_rank, k_padding),
        ),
        _padded_row(kv_c, kv_lora_rank, (0, q_pairs - kv_lora_rank)),
        _scalar(positions),
        _row(cos_sin_cache, rope_dim),
        _scalar(slot_mapping),
        _padded_row(kv_cache, cache_width, (0, padding)),
    )


def application(
    q_even,
    q_odd,
    k_even,
    k_odd,
    kv_c,
    positions,
    cos_sin_cache,
    slot_mapping,
    kv_cache,
):
    rope_dim = cos_sin_cache.shape[0]
    half_rope_dim = rope_dim // 2
    kv_lora_rank = kv_c.source.shape[-1]

    q_offsets = q_even.offsets(-1)
    q_frequency = q_offsets % half_rope_dim
    q_cos = cos_sin_cache.source[positions, q_frequency]
    q_sin = cos_sin_cache.source[positions, q_frequency + half_rope_dim]
    # Materialize both source values in SSA before either aliased stride-2
    # output view is written.  A plain Python alias would let the second store
    # reload the already updated even channel from memory.
    original_even = q_even + 0.0
    original_odd = q_odd + 0.0
    q_even = original_even * q_cos - original_odd * q_sin  # noqa: F841
    q_odd = original_odd * q_cos + original_even * q_sin  # noqa: F841

    cache_offsets = kv_cache.offsets(-1)
    cache_pair = ntl.maximum(
        (cache_offsets - kv_lora_rank) // 2,
        0,
    )
    cache_k_index = cache_pair + kv_lora_rank
    raw_even = ntl.gather(k_even, cache_k_index, 0)
    raw_odd = ntl.gather(k_odd, cache_k_index, 0)
    cache_frequency = cache_pair % half_rope_dim
    cache_cos = cos_sin_cache.source[positions, cache_frequency]
    cache_sin = cos_sin_cache.source[
        positions, cache_frequency + half_rope_dim
    ]
    rotated_even = raw_even * cache_cos - raw_odd * cache_sin
    rotated_odd = raw_odd * cache_cos + raw_even * cache_sin
    rotated = ntl.where(
        cache_offsets % 2 == 0,
        rotated_even,
        rotated_odd,
    )
    combined = ntl.where(cache_offsets < kv_lora_rank, kv_c, rotated)
    kv_cache.source[slot_mapping] = combined


def premake(
    q_even_shape,
    q_odd_shape,
    k_even_shape,
    k_odd_shape,
    kv_c_shape,
    positions_shape,
    cos_sin_shape,
    slots_shape,
    cache_shape,
    q_even_dtype,
    q_odd_dtype,
    k_even_dtype,
    k_odd_dtype,
    kv_c_dtype,
    positions_dtype,
    cos_sin_dtype,
    slots_dtype,
    cache_dtype,
    q_even_strides,
    q_odd_strides,
    k_even_strides,
    k_odd_strides,
    kv_c_strides,
    positions_strides,
    cos_sin_strides,
    slots_strides,
    cache_strides,
):
    q_pairs = q_even_shape[-1]
    rope_dim = k_even_shape[-1] * 2
    kv_lora_rank = kv_c_shape[-1]
    arrangement_ = functools.partial(
        arrangement,
        q_pairs=q_pairs,
        rope_dim=rope_dim,
        kv_lora_rank=kv_lora_rank,
    )

    def tensor(shape, strides, dtype, *, other=None):
        return Tensor(
            shape=tuple(shape),
            strides=tuple(strides),
            dtype=dtype,
            other=other,
        )

    tensors = (
        tensor(q_even_shape, q_even_strides, q_even_dtype, other=0),
        tensor(q_odd_shape, q_odd_strides, q_odd_dtype, other=0),
        tensor(k_even_shape, k_even_strides, k_even_dtype, other=0),
        tensor(k_odd_shape, k_odd_strides, k_odd_dtype, other=0),
        tensor(kv_c_shape, kv_c_strides, kv_c_dtype, other=0),
        tensor(positions_shape, positions_strides, positions_dtype),
        tensor(cos_sin_shape, cos_sin_strides, cos_sin_dtype, other=0),
        tensor(slots_shape, slots_strides, slots_dtype),
        tensor(cache_shape, cache_strides, cache_dtype),
    )
    return arrangement_, application, tensors


__all__ = ["NUM_WARPS", "NUM_STAGES", "premake"]
