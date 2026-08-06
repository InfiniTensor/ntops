"""MLA RoPE and compressed KV Cache torch wrapper."""

import torch

import ntops
from ntops.torch.utils import _cached_make


def mla_rope_kv_cache(
    hidden_states,
    w_dkv,
    w_pe,
    kv_cache,
    cache_slots,
    positions,
    cos_table,
    sin_table,
    num_heads,
    kv_lora_rank,
    qk_rope_head_dim,
):
    """MLA RoPE with fused compressed KV Cache write.

    Performs Multi-head Latent Attention projection, applies RoPE, and
    writes compressed KV to paged cache in a single fused kernel.

    Args:
        hidden_states: Input hidden states (seq_len, hidden_dim).
        w_dkv: Weight for compressed KV projection (kv_lora_rank, hidden_dim).
        w_pe: Weight for positional encoding part (qk_rope_head_dim, hidden_dim).
        kv_cache: Paged KV cache buffer (num_pages, head_dim).
        cache_slots: Slot indices in paged cache (seq_len,).
        positions: Token positions for RoPE (seq_len,).
        cos_table: Precomputed cos values (max_seq_len, head_dim // 2).
        sin_table: Precomputed sin values (max_seq_len, head_dim // 2).
        num_heads: Number of attention heads.
        kv_lora_rank: Rank of compressed KV.
        qk_rope_head_dim: Dimension of RoPE'd query/key parts.

    Returns:
        output: Projected output for attention computation.
    """
    seq_len = hidden_states.shape[0]
    output_dim = num_heads * (kv_lora_rank + qk_rope_head_dim)

    output = torch.empty(
        (seq_len, output_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    kernel = _cached_make(
        ntops.kernels.mla_rope_kv_cache.premake,
        hidden_dim=hidden_states.shape[-1],
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        num_heads=num_heads,
        dtype=hidden_states.dtype,
    )

    kernel(
        hidden_states,
        w_dkv,
        w_pe,
        kv_cache,
        cache_slots,
        positions,
        cos_table,
        sin_table,
        output,
        num_heads,
        kv_lora_rank,
        qk_rope_head_dim,
    )

    return output
