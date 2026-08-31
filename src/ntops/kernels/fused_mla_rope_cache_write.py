"""NineToothed fused MLA RoPE + compressed KV-cache write kernel.

The cache entry follows vLLM's MLA layout:

    [compressed kv_c | RoPE(k_pe)]

``kv_c`` is the shared low-rank K/V latent.  It is not expanded into
per-head K/V in this kernel.  ``k_pe`` is the shared rotary key component and
is rotated in-place in the write path.
"""

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


def _source_view(tensor):
    return tensor.tile((1,) * tensor.ndim)


def arrangement(
    output_anchor,
    kv_c,
    k_pe,
    kv_cache,
    slot_mapping,
    positions,
    cos_sin_cache,
    entry_dim,
    tile_size,
    kv_lora_rank,
    rope_dim,
    cache_block_size,
):
    # A real output anchor keeps the side-effecting cache store in the SSA graph
    # on backends that cannot lower a cache-only launch. The wrapper discards
    # this duplicate compressed entry after the kernel completes.
    output_anchor = output_anchor.tile((1, tile_size.value))
    output_anchor.dtype = output_anchor.dtype.squeeze(0)
    kv_c, k_pe, kv_cache, slot_mapping, positions, cos_sin_cache = (
        _source_view(tensor)
        for tensor in (
            kv_c,
            k_pe,
            kv_cache,
            slot_mapping,
            positions,
            cos_sin_cache,
        )
    )
    return (
        output_anchor,
        kv_c,
        k_pe,
        kv_cache,
        slot_mapping,
        positions,
        cos_sin_cache,
        entry_dim,
        tile_size,
        kv_lora_rank,
        rope_dim,
        cache_block_size,
    )


def application(
    output_anchor,
    kv_c,
    k_pe,
    kv_cache,
    slot_mapping,
    positions,
    cos_sin_cache,
    entry_dim,
    tile_size,
    kv_lora_rank,
    rope_dim,
    cache_block_size,
):
    token_idx = output_anchor.offsets(0)
    feature = output_anchor.offsets(1)
    feature_valid = feature < entry_dim
    nope_mask = feature_valid & (feature < kv_lora_rank)
    rope_mask = feature_valid & (feature >= kv_lora_rank)
    nope_feature = ntl.where(nope_mask, feature, 0)
    rope_feature = ntl.where(rope_mask, feature - kv_lora_rank, 0)
    pair = rope_feature // 2
    even = (rope_feature & 1) == 0

    latent_value = kv_c.source[token_idx, nope_feature]
    kpe0 = k_pe.source[token_idx, pair * 2]
    kpe1 = k_pe.source[token_idx, pair * 2 + 1]
    position = positions.source[token_idx].to(ntl.int64)
    cos = cos_sin_cache.source[position, pair]
    sin = cos_sin_cache.source[position, rope_dim // 2 + pair]
    rotated = ntl.where(
        even,
        kpe0 * cos - kpe1 * sin,
        kpe0 * sin + kpe1 * cos,
    )
    cache_value = ntl.where(nope_mask, latent_value, rotated).to(kv_c.dtype)

    output_anchor = cache_value  # noqa: F841

    # Negative slots are padding. Indexed stores include source-shape bounds
    # masks, so mapping padding to a negative block suppresses the write.
    slot = slot_mapping.source[token_idx].to(ntl.int64)
    write_slot = ntl.where(slot >= 0, slot, -1)
    block_idx = write_slot // cache_block_size
    block_offset = write_slot % cache_block_size
    kv_cache.source[block_idx, block_offset, feature] = cache_value


def premake(
    kv_lora_rank,
    rope_dim,
    dtype=None,
    block_size=128,
    cache_block_size=16,
    cos_dtype=None,
):
    entry_dim = kv_lora_rank + rope_dim
    dynamic = {"constexpr": True, "upper_bound": 2**20}
    tensors = (
        Tensor(
            shape=(None, entry_dim),
            dtype=dtype,
            shape_options=(dynamic, {"constexpr": True}),
        ),
        Tensor(
            shape=(None, kv_lora_rank),
            dtype=dtype,
            shape_options=(dynamic, {"constexpr": True}),
        ),
        Tensor(
            shape=(None, rope_dim),
            dtype=dtype,
            shape_options=(dynamic, {"constexpr": True}),
        ),
        Tensor(
            shape=(None, None, entry_dim),
            dtype=dtype,
            shape_options=(dynamic, dynamic, {"constexpr": True}),
        ),
        Tensor(shape=(None,), dtype="int64", shape_options=(dynamic,)),
        Tensor(shape=(None,), dtype="int64", shape_options=(dynamic,)),
        Tensor(
            shape=(None, rope_dim),
            dtype=cos_dtype or ninetoothed.float32,
            shape_options=(dynamic, {"constexpr": True}),
        ),
        Tensor(0, constexpr=True, value=entry_dim),
        Tensor(0, constexpr=True, value=block_size),
        Tensor(0, constexpr=True, value=kv_lora_rank),
        Tensor(0, constexpr=True, value=rope_dim),
        Tensor(0, constexpr=True, value=cache_block_size),
    )
    return arrangement, application, tensors
