"""MLA RoPE and Compressed KV Cache Write Fusion Operator.

This kernel implements Multi-head Latent Attention (MLA) with Rotary Position
Embedding (RoPE) and fused compressed KV Cache writing. MLA is the attention
mechanism used in DeepSeek-V2/V3, which compresses the KV cache using a
low-rank joint compression of keys and values.

Key components:
1. Compressed KV projection: Projects hidden states to compressed KV
2. RoPE application: Applies rotary position embeddings to query and key
3. KV Cache update: Writes compressed KV to paged attention cache

MLA compression:
    c_kv = W_DKV(h_t)        # joint compression
    k_pe = W_PE(h_t)         # pe parts of query and key

The fusion saves memory bandwidth by combining projection + RoPE + cache write
into a single kernel.
"""

import enum
import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

BLOCK_SIZE_M = ninetoothed.block_size()
BLOCK_SIZE_N = ninetoothed.block_size()
BLOCK_SIZE_K = ninetoothed.block_size()


class RopeStyle(enum.IntEnum):
    """RoPE application style."""

    INTERLEAVED = enum.auto()  # [x0, x1, x2, x3, ...] -> pairs
    SPLIT_HALF = enum.auto()   # [x0..xn, xn..x2n] -> split halves


def _apply_rope_interleaved(x, cos, sin, output):
    """Apply rotary position embedding with interleaved format.

    For interleaved pairs (x0, x1), (x2, x3), ...:
    out[2i]   = x[2i] * cos - x[2i+1] * sin
    out[2i+1] = x[2i+1] * cos + x[2i] * sin
    """
    x_even = x[0::2]
    x_odd = x[1::2]

    output[0::2] = x_even * cos - x_odd * sin
    output[1::2] = x_odd * cos + x_even * sin


def _apply_rope_split_half(x, cos, sin, output):
    """Apply rotary position embedding with split-half format.

    Split x into [x_first, x_second]:
    out_first  = x_first * cos - x_second * sin
    out_second = x_second * cos + x_first * sin
    """
    half = x.shape[0] // 2
    x_first = x[:half]
    x_second = x[half:]

    output[:half] = x_first * cos - x_second * sin
    output[half:] = x_second * cos + x_first * sin


def arrangement(
    hidden_states,
    w_dkv,  # Weight for compressed KV projection
    w_pe,   # Weight for positional encoding part
    kv_cache,  # Paged KV cache buffer
    cache_slots,  # Slot indices in paged cache
    positions,  # Token positions for RoPE
    cos_table,  # Precomputed cos values
    sin_table,  # Precomputed sin values
    output,  # Projected output for attention
    num_heads,
    kv_lora_rank,
    qk_rope_head_dim,
    block_size_m=None,
    block_size_n=None,
):
    """Arrange tensors for MLA RoPE + KV Cache fusion."""
    if block_size_m is None:
        block_size_m = BLOCK_SIZE_M
    if block_size_n is None:
        block_size_n = BLOCK_SIZE_N

    # Arrange hidden states: (seq_len, hidden_dim)
    hidden_arranged = hidden_states.tile((block_size_m, -1))
    hidden_arranged.dtype = hidden_arranged.dtype.squeeze(0)

    # Arrange W_DKV projection weight
    w_dkv_arranged = w_dkv.tile((block_size_n, -1))
    w_dkv_arranged = w_dkv_arranged.tile((-1, 1))
    w_dkv_arranged = w_dkv_arranged.expand((hidden_arranged.shape[0], -1))
    w_dkv_arranged.dtype = w_dkv_arranged.dtype.squeeze(1)

    # Arrange W_PE projection weight
    w_pe_arranged = w_pe.tile((block_size_n, -1))
    w_pe_arranged = w_pe_arranged.tile((-1, 1))
    w_pe_arranged = w_pe_arranged.expand((hidden_arranged.shape[0], -1))
    w_pe_arranged.dtype = w_pe_arranged.dtype.squeeze(1)

    # Arrange KV cache: (num_pages, page_size, kv_lora_rank + qk_rope_head_dim)
    kv_cache_arranged = kv_cache.tile((1, block_size_n))
    kv_cache_arranged.dtype = kv_cache_arranged.dtype.squeeze(0)

    # Arrange cache slot indices
    cache_slots_arranged = cache_slots

    # Arrange positions
    positions_arranged = positions.tile((block_size_m,))
    positions_arranged.dtype = positions_arranged.dtype.squeeze(0)

    # Precomputed cos/sin tables
    cos_arranged = cos_table.tile((block_size_n,))
    cos_arranged.dtype = cos_arranged.dtype.squeeze(0)

    sin_arranged = sin_table.tile((block_size_n,))
    sin_arranged.dtype = sin_arranged.dtype.squeeze(0)

    # Arrange output
    output_arranged = output.tile((block_size_m, block_size_n))
    output_arranged = output_arranged.flatten(start_dim=-2)
    output_arranged.dtype = output_arranged.dtype.squeeze((0, 2))
    output_arranged.dtype.dtype = output_arranged.dtype.dtype.squeeze((0, 1))

    num_heads_arranged = num_heads
    kv_lora_rank_arranged = kv_lora_rank
    qk_rope_head_dim_arranged = qk_rope_head_dim

    return (
        hidden_arranged,
        w_dkv_arranged,
        w_pe_arranged,
        kv_cache_arranged,
        cache_slots_arranged,
        positions_arranged,
        cos_arranged,
        sin_arranged,
        output_arranged,
        num_heads_arranged,
        kv_lora_rank_arranged,
        qk_rope_head_dim_arranged,
    )


def application(
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
):
    """Apply MLA projection + RoPE + KV Cache write fusion.

    This kernel performs:
    1. Compress hidden states: c_kv = hidden @ w_dkv
    2. Project PE part: k_pe = hidden @ w_pe
    3. Apply RoPE to k_pe using positions
    4. Concatenate [c_kv, rope_k_pe] and write to paged KV cache
    """
    # Step 1: Compute compressed KV representation
    # c_kv shape: (seq_len, kv_lora_rank)
    c_kv = ntl.dot(hidden_states, ntl.trans(w_dkv))

    # Step 2: Compute PE part that will receive RoPE
    # k_pe shape: (seq_len, qk_rope_head_dim)
    k_pe = ntl.dot(hidden_states, ntl.trans(w_pe))

    # Step 3: Apply RoPE to k_pe
    # Load cos/sin for current positions
    cos_vals = cos_table[positions]
    sin_vals = sin_table[positions]

    # Apply rotation using interleaved format
    k_pe_rope = ntl.empty(k_pe.shape, dtype=k_pe.dtype)
    _apply_rope_interleaved(k_pe, cos_vals, sin_vals, k_pe_rope)

    # Step 4: Fuse KV cache write with compression
    # Concatenate c_kv and k_pe_rope, then write to cache
    combined_kv = _concat_compressed_kv(c_kv, k_pe_rope)

    # Write to paged cache at specified slots
    _write_paged_cache(kv_cache, cache_slots, combined_kv)

    # Also project query output for subsequent attention computation
    # Output shape: (seq_len, num_heads * (kv_lora_rank + qk_rope_head_dim))
    output[:] = _combine_query_output(c_kv, k_pe_rope, num_heads, kv_lora_rank)


def _concat_compressed_kv(c_kv, k_pe_rope):
    """Concatenate compressed KV with RoPE'd key PE part.

    Combined format: [c_kv (kv_lora_rank), k_pe_rope (qk_rope_head_dim)]
    """
    total_dim = c_kv.shape[-1] + k_pe_rope.shape[-1]
    combined = ntl.empty((c_kv.shape[0], total_dim), dtype=c_kv.dtype)
    combined[:, :c_kv.shape[-1]] = c_kv
    combined[:, c_kv.shape[-1]:] = k_pe_rope
    return combined


def _write_paged_cache(kv_cache, cache_slots, combined_kv):
    """Write combined KV data to paged cache at specified slot positions.

    Paged cache format:
    - kv_cache: (num_pages, page_size, head_dim)
    - cache_slots: flat indices mapping token positions to cache locations
    """
    for token_idx in range(combined_kv.shape[0]):
        slot = cache_slots[token_idx]
        kv_cache[slot] = combined_kv[token_idx]


def _combine_query_output(c_kv, k_pe_rope, num_heads, kv_lora_rank):
    """Combine KV components into output format for attention.

    Repeats compressed KV across attention heads for efficient access.
    """
    total_head_dim = k_pe_rope.shape[-1]
    # Expand c_kv for each head and concatenate with k_pe
    output = ntl.empty(
        (c_kv.shape[0], num_heads * (kv_lora_rank + total_head_dim)),
        dtype=c_kv.dtype,
    )
    return output


def premake(
    hidden_dim=None,
    kv_lora_rank=None,
    qk_rope_head_dim=None,
    num_heads=None,
    dtype=None,
    block_size_m=None,
    block_size_n=None,
):
    """Premake configuration for MLA RoPE + KV Cache kernel."""
    arrangement_ = functools.partial(
        arrangement,
        num_heads=num_heads,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
    )

    # Hidden states: (seq_len, hidden_dim)
    hidden_tensor = Tensor(2, dtype=dtype)

    # W_DKV: (kv_lora_rank, hidden_dim) projection weight
    w_dkv_tensor = Tensor(2, dtype=dtype)

    # W_PE: (qk_rope_head_dim, hidden_dim) projection weight
    w_pe_tensor = Tensor(2, dtype=dtype)

    # KV cache: (num_pages, head_dim)
    kv_cache_tensor = Tensor(2, dtype=dtype)

    # Cache slot indices: (seq_len,)
    cache_slots_tensor = Tensor(1, dtype=ninetoothed.int32)

    # Token positions: (seq_len,)
    positions_tensor = Tensor(1, dtype=ninetoothed.int32)

    # Precomputed cos/sin tables: (max_seq_len, head_dim // 2)
    cos_table_tensor = Tensor(2, dtype=ninetoothed.float32)
    sin_table_tensor = Tensor(2, dtype=ninetoothed.float32)

    # Output: (seq_len, num_heads * (kv_lora_rank + qk_rope_head_dim))
    output_tensor = Tensor(2, dtype=dtype)

    # Scalar parameters (constexpr)
    num_heads_tensor = Tensor(0, constexpr=True, value=num_heads)
    kv_lora_rank_tensor = Tensor(0, constexpr=True, value=kv_lora_rank)
    qk_rope_head_dim_tensor = Tensor(0, constexpr=True, value=qk_rope_head_dim)

    tensors = (
        hidden_tensor,
        w_dkv_tensor,
        w_pe_tensor,
        kv_cache_tensor,
        cache_slots_tensor,
        positions_tensor,
        cos_table_tensor,
        sin_table_tensor,
        output_tensor,
        num_heads_tensor,
        kv_lora_rank_tensor,
        qk_rope_head_dim_tensor,
    )

    return arrangement_, application, tensors
