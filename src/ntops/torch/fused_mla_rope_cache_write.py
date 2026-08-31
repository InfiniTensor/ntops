"""PyTorch interface for fused MLA RoPE + compressed cache writes."""

import torch

import ntops
from ntops.torch.utils import _cached_make


def _validate_inputs(
    kv_c,
    k_pe,
    kv_cache,
    slot_mapping,
    positions,
    cos_sin_cache,
):
    if kv_c.ndim != 2:
        raise ValueError("kv_c must have shape [num_tokens, kv_lora_rank]")
    if k_pe.ndim not in (2, 3):
        raise ValueError(
            "k_pe must have shape [num_tokens, rope_dim] or [num_tokens, 1, rope_dim]"
        )
    if kv_cache.ndim != 3:
        raise ValueError("kv_cache must have shape [num_blocks, block_size, entry_dim]")
    if slot_mapping.ndim != 1 or positions.ndim != 1:
        raise ValueError("slot_mapping and positions must be one-dimensional")
    if cos_sin_cache.ndim != 2:
        raise ValueError("cos_sin_cache must have shape [max_position, rope_dim]")

    rope_dim = k_pe.shape[-1]
    num_tokens = slot_mapping.shape[0]
    if rope_dim <= 0 or rope_dim % 2:
        raise ValueError("rope_dim must be a positive even number")
    if kv_c.shape[1] <= 0:
        raise ValueError("kv_c must have a positive latent width")
    if kv_c.shape[0] < num_tokens or k_pe.shape[0] < num_tokens:
        raise ValueError(
            "source tensors must contain at least slot_mapping.size(0) tokens"
        )
    if positions.shape[0] != num_tokens:
        raise ValueError("positions must have one value per slot_mapping entry")
    if k_pe.ndim == 3 and k_pe.shape[1] != 1:
        raise ValueError("rank-3 k_pe must have shape [num_tokens, 1, rope_dim]")
    if k_pe.numel() < num_tokens * rope_dim:
        raise ValueError("k_pe does not contain enough token values")
    if cos_sin_cache.shape[1] != rope_dim:
        raise ValueError(
            "cos_sin_cache width must equal packed rope_dim (cos[0:R/2] | sin[0:R/2])"
        )
    if kv_cache.shape[2] != kv_c.shape[1] + rope_dim:
        raise ValueError(
            "kv_cache entry dimension must equal latent width + rope width"
        )
    if kv_cache.shape[0] <= 0 or kv_cache.shape[1] <= 0:
        raise ValueError("kv_cache must have positive block and block-size dimensions")
    if kv_c.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("MLA inputs and cache must use float16, bfloat16, or float32")
    if slot_mapping.dtype != torch.int64:
        raise TypeError("slot_mapping must be torch.int64")
    if positions.dtype not in (torch.int32, torch.int64):
        raise TypeError("positions must be torch.int32 or torch.int64")
    tensors = (kv_c, k_pe, kv_cache, slot_mapping, positions, cos_sin_cache)
    if any(t.device != kv_c.device for t in tensors):
        raise ValueError("all tensors must be on the same device")
    if any(t.dtype != kv_c.dtype for t in (k_pe, kv_cache)):
        raise TypeError("kv_c, k_pe, and kv_cache must share dtype")
    if cos_sin_cache.dtype not in (kv_c.dtype, torch.float32):
        raise TypeError("cos_sin_cache must use the input dtype or torch.float32")


def fused_mla_rope_cache_write(
    kv_c,
    k_pe,
    kv_cache,
    slot_mapping,
    positions,
    cos_sin_cache,
    *,
    block_size=128,
    num_warps=(1, 2, 4, 8),
    num_stages=(1, 2),
    max_num_configs=8,
):
    """Fuse RoPE(k_pe) with an in-place compressed MLA KV-cache write.

    The cache entry written for each non-padding slot is
    ``[kv_c[token] | RoPE(k_pe[token])]``.  ``kv_c`` remains the shared
    low-rank K/V latent; it is never expanded to per-head K/V.  This operation
    returns ``None`` like vLLM's ``concat_and_cache_mla``.  By default Triton
    autotunes the launch configuration and caches the winner by input key.
    """
    _validate_inputs(kv_c, k_pe, kv_cache, slot_mapping, positions, cos_sin_cache)
    if not kv_c.is_cuda:
        raise RuntimeError("fused_mla_rope_cache_write requires a CUDA device")
    if (
        not isinstance(block_size, int)
        or isinstance(block_size, bool)
        or block_size <= 0
    ):
        raise ValueError("block_size must be a positive integer")

    if k_pe.ndim == 3:
        k_pe = k_pe.reshape(k_pe.shape[0], k_pe.shape[-1])
    if positions.dtype == torch.int32:
        positions = positions.to(torch.int64)

    num_tokens = slot_mapping.shape[0]
    if num_tokens == 0:
        return None
    # Keep indirect source accesses on one contiguous specialization. Some
    # Triton backends cannot compile the non-unit-stride indexed-store variant.
    cache_output = kv_cache
    kv_c, k_pe, slot_mapping, positions, cos_sin_cache = (
        tensor if tensor.is_contiguous() else tensor.contiguous()
        for tensor in (kv_c, k_pe, slot_mapping, positions, cos_sin_cache)
    )
    if not kv_cache.is_contiguous():
        kv_cache = kv_cache.contiguous()

    latent = kv_c.shape[1]
    rope_dim = k_pe.shape[-1]
    entry_dim = latent + rope_dim
    tile_size = max(
        1 << (block_size - 1).bit_length(),
        1 << (entry_dim - 1).bit_length(),
    )
    # A real output root is required by the NineToothed SSA path on DCU. It is
    # one compressed entry per token, not a per-head query tensor, and is
    # discarded after anchoring the cache side effect.
    output_anchor = torch.empty(
        (num_tokens, entry_dim), dtype=kv_c.dtype, device=kv_c.device
    )
    kernel = _cached_make(
        ntops.kernels.fused_mla_rope_cache_write.premake,
        latent,
        rope_dim,
        dtype=kv_c.dtype,
        block_size=tile_size,
        cache_block_size=kv_cache.shape[1],
        cos_dtype=cos_sin_cache.dtype,
        num_warps=num_warps,
        num_stages=num_stages,
        max_num_configs=max_num_configs,
    )
    kernel(
        output_anchor,
        kv_c,
        k_pe,
        kv_cache,
        slot_mapping,
        positions,
        cos_sin_cache,
        entry_dim,
        tile_size,
        latent,
        rope_dim,
        kv_cache.shape[1],
    )
    if kv_cache is not cache_output:
        cache_output.copy_(kv_cache)
    return None


def fused_mla_rope_cache_write_reference(
    kv_c,
    k_pe,
    kv_cache,
    slot_mapping,
    positions,
    cos_sin_cache,
):
    """PyTorch reference for the fused cache-only operation."""
    _validate_inputs(kv_c, k_pe, kv_cache, slot_mapping, positions, cos_sin_cache)
    if k_pe.ndim == 3:
        k_pe = k_pe.reshape(k_pe.shape[0], k_pe.shape[-1])
    num_tokens = slot_mapping.shape[0]
    if num_tokens == 0:
        return None
    rope_dim = k_pe.shape[-1]
    half = rope_dim // 2
    table = cos_sin_cache.index_select(0, positions.to(torch.long))
    cos = table[:, :half].to(torch.float32)
    sin = table[:, half:].to(torch.float32)
    x0 = k_pe[:num_tokens, 0::2].to(torch.float32)
    x1 = k_pe[:num_tokens, 1::2].to(torch.float32)
    rotated = torch.empty_like(k_pe[:num_tokens], dtype=torch.float32)
    rotated[:, 0::2] = x0 * cos - x1 * sin
    rotated[:, 1::2] = x0 * sin + x1 * cos
    rotated = rotated.to(kv_c.dtype)

    for token_idx, slot in enumerate(slot_mapping.tolist()):
        if slot < 0:
            continue
        block_idx, block_offset = divmod(slot, kv_cache.shape[1])
        cache_base = kv_cache[block_idx, block_offset]
        cache_base[: kv_c.shape[1]] = kv_c[token_idx]
        cache_base[kv_c.shape[1] :] = rotated[token_idx]
    return None


__all__ = [
    "fused_mla_rope_cache_write",
    "fused_mla_rope_cache_write_reference",
]
