"""Fused MLA RoPE and compressed paged-KV-cache write."""

import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor

# A runtime warp sweep is counterproductive for this launch-bound fused
# operator: all supported shapes use the same one-row-per-program schedule,
# while CoreX may spend minutes benchmarking otherwise equivalent candidates.
# Two warps is legal on both target backends and was the best or statistically
# tied configuration across the representative decode/prefill shapes on BW.
NUM_WARPS = 2
NUM_STAGES = 1


def arrangement(
    q_pe,
    k_pe,
    kv_c,
    positions,
    cos_sin_cache,
    slot_mapping,
    kv_cache,
    *,
    q_width,
    rope_dim,
    kv_lora_rank,
):
    def row(tensor, width):
        tiled = tensor.tile((1, width))
        tiled.dtype = tiled.dtype.squeeze(0)
        return tiled

    def padded_row(tensor, width, padding):
        """Pad the per-program value tile while preserving the token grid."""
        tiled = tensor.tile((1, width))
        tiled.dtype = tiled.dtype.squeeze(0).pad((padding,))
        return tiled

    def scalar(tensor):
        tiled = tensor.tile((1,))
        tiled.dtype = tiled.dtype.squeeze(0)
        return tiled

    # Align every value tile with the already existing q application domain.
    # Triton block tensors require power-of-two dimensions; keeping a separate
    # 576-wide cache value would otherwise force an illegal ``arange(0, 576)``
    # when the backend maps one complete token to one program.  The physical
    # cache is still 576 elements wide and its source-layout bounds mask drops
    # all right-padding lanes.
    cache_width = kv_lora_rank + rope_dim
    alignment_padding = q_width - cache_width

    if alignment_padding < 0:
        raise ValueError("q_width must cover the compressed MLA cache row.")

    k_row = padded_row(k_pe, rope_dim, (kv_lora_rank, alignment_padding))
    kv_c_row = padded_row(
        kv_c,
        kv_lora_rank,
        (0, rope_dim + alignment_padding),
    )
    cache_row = padded_row(kv_cache, cache_width, (0, alignment_padding))
    return (
        row(q_pe, q_width),
        k_row,
        kv_c_row,
        scalar(positions),
        row(cos_sin_cache, rope_dim),
        scalar(slot_mapping),
        cache_row,
    )


def application(
    q_pe,
    k_pe,
    kv_c,
    positions,
    cos_sin_cache,
    slot_mapping,
    kv_cache,
):
    rope_dim = cos_sin_cache.shape[0]
    half_rope_dim = rope_dim // 2
    # ``kv_cache`` is logically padded to the q width by the arrangement;
    # derive the real compressed rank from the physical kv_c source instead.
    kv_lora_rank = kv_c.source.shape[-1]

    # The application sees one value tile per token.  ``offsets(-1)`` selects
    # the original feature dimension; ``offsets(0)`` would select the outer
    # token dimension and incorrectly rotate adjacent tokens instead of
    # adjacent RoPE channels.
    q_offsets = q_pe.offsets(-1)
    q_pair_offsets = q_offsets ^ 1
    q_frequency = (q_offsets % rope_dim) // 2
    q_pair = ntl.gather(q_pe, q_pair_offsets, 0)
    q_cos = cos_sin_cache.source[positions, q_frequency]
    q_sin = cos_sin_cache.source[positions, q_frequency + half_rope_dim]
    q_sign = ntl.where(q_offsets % 2 == 0, -1.0, 1.0)
    q_pe = q_pe * q_cos + q_pair * q_sin * q_sign  # noqa: F841

    # The left-padded k view maps logical lanes [512, 576) to source offsets
    # [0, 64).  Its feature offset is therefore already local to k_pe.
    k_offsets = k_pe.offsets(-1)
    k_local_offsets = ntl.maximum(k_offsets, 0)
    k_pair_offsets = ntl.minimum(
        (k_local_offsets ^ 1) + kv_lora_rank,
        kv_cache.shape[0] - 1,
    )
    k_frequency = k_local_offsets // 2
    k_pair = ntl.gather(k_pe, k_pair_offsets, 0)
    k_cos = cos_sin_cache.source[positions, k_frequency]
    k_sin = cos_sin_cache.source[positions, k_frequency + half_rope_dim]
    k_sign = ntl.where(k_local_offsets % 2 == 0, -1.0, 1.0)
    rotated_k = k_pe * k_cos + k_pair * k_sin * k_sign

    combined = ntl.where(
        kv_cache.offsets(-1) < kv_lora_rank,
        kv_c,
        rotated_k,
    )
    kv_cache.source[slot_mapping] = combined


def premake(
    q_shape,
    k_shape,
    kv_c_shape,
    positions_shape,
    cos_sin_shape,
    slots_shape,
    cache_shape,
    q_dtype,
    k_dtype,
    kv_c_dtype,
    positions_dtype,
    cos_sin_dtype,
    slots_dtype,
    cache_dtype,
    q_strides,
    k_strides,
    kv_c_strides,
    positions_strides,
    cos_sin_strides,
    slots_strides,
    cache_strides,
):
    q_width = q_shape[1]
    rope_dim = k_shape[1]
    kv_lora_rank = kv_c_shape[1]
    arrangement_ = functools.partial(
        arrangement,
        q_width=q_width,
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
        tensor(q_shape, q_strides, q_dtype, other=0),
        tensor(k_shape, k_strides, k_dtype, other=0),
        tensor(kv_c_shape, kv_c_strides, kv_c_dtype, other=0),
        tensor(positions_shape, positions_strides, positions_dtype),
        tensor(cos_sin_shape, cos_sin_strides, cos_sin_dtype, other=0),
        tensor(slots_shape, slots_strides, slots_dtype),
        tensor(cache_shape, cache_strides, cache_dtype),
    )
    return arrangement_, application, tensors


__all__ = ["premake"]
