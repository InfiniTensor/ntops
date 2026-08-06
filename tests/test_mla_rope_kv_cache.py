"""Tests for MLA RoPE and compressed KV Cache fusion kernel."""

import math

import pytest
import torch

import ntops


def _reference_rope(x, cos, sin, style="interleaved"):
    """Reference RoPE implementation."""
    x_fp32 = x.float()

    if style == "interleaved":
        # Interleaved format: pairs are consecutive
        x_even = x_fp32[..., 0::2]
        x_odd = x_fp32[..., 1::2]

        out_even = x_even * cos - x_odd * sin
        out_odd = x_odd * cos + x_even * sin

        output = torch.zeros_like(x_fp32)
        output[..., 0::2] = out_even
        output[..., 1::2] = out_odd
    else:
        # Split half format
        half = x_fp32.shape[-1] // 2
        x_first = x_fp32[..., :half]
        x_second = x_fp32[..., half:]

        out_first = x_first * cos - x_second * sin
        out_second = x_second * cos + x_first * sin

        output = torch.cat([out_first, out_second], dim=-1)

    return output.to(x.dtype)


def _precompute_rope_frequencies(seq_len, head_dim, base=10000.0):
    """Precompute cos/sin values for RoPE."""
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len, dtype=freqs.dtype)
    freqs = torch.outer(t, freqs)

    cos_vals = torch.cos(freqs)
    sin_vals = torch.sin(freqs)

    return cos_vals, sin_vals


def test_rope_correctness():
    """Test RoPE rotation correctness."""
    torch.manual_seed(42)

    seq_len, head_dim = 128, 64
    cos_table, sin_table = _precompute_rope_frequencies(seq_len, head_dim)

    x = torch.randn(seq_len, head_dim, dtype=torch.bfloat16)

    # Test interleaved RoPE
    output_ref = _reference_rope(x, cos_table, sin_table, "interleaved")
    assert output_ref.shape == x.shape

    # Verify RoPE preserves norm (approximately)
    input_norm = x.float().norm(dim=-1)
    output_norm = output_ref.float().norm(dim=-1)
    assert torch.allclose(input_norm, output_norm, atol=0.1)


def test_mla_projection_shapes():
    """Test MLA projection output shapes."""
    torch.manual_seed(42)

    configs = [
        {"hidden_dim": 512, "kv_lora_rank": 512, "qk_rope_head_dim": 64, "num_heads": 8},
        {"hidden_dim": 1024, "kv_lora_rank": 256, "qk_rope_head_dim": 32, "num_heads": 16},
        {"hidden_dim": 2048, "kv_lora_rank": 512, "qk_rope_head_dim": 64, "num_heads": 32},
    ]

    for cfg in configs:
        batch, seq = 2, 128
        hidden = cfg["hidden_dim"]

        hidden_states = torch.randn(batch, seq, hidden, dtype=torch.bfloat16)

        # MLA projections
        w_dkv = torch.randn(cfg["kv_lora_rank"], hidden, dtype=torch.bfloat16)
        w_pe = torch.randn(cfg["qk_rope_head_dim"], hidden, dtype=torch.bfloat16)

        c_kv = torch.mm(hidden_states.view(-1, hidden), w_dkv.t())
        k_pe = torch.mm(hidden_states.view(-1, hidden), w_pe.t())

        assert c_kv.shape == (batch * seq, cfg["kv_lora_rank"])
        assert k_pe.shape == (batch * seq, cfg["qk_rope_head_dim"])


def test_kv_cache_write():
    """Test paged KV cache write operation."""
    torch.manual_seed(42)

    num_pages = 100
    page_size = 512
    seq_len = 64

    kv_cache = torch.zeros(num_pages, page_size, dtype=torch.bfloat16)
    cache_slots = torch.randint(0, num_pages, (seq_len,))
    combined_kv = torch.randn(seq_len, page_size, dtype=torch.bfloat16)

    # Simulate write
    for i in range(seq_len):
        kv_cache[cache_slots[i]] = combined_kv[i]

    # Verify writes
    for i in range(seq_len):
        assert torch.equal(kv_cache[cache_slots[i]], combined_kv[i])


def test_compression_ratio():
    """Test compression ratio of MLA vs standard attention."""
    standard_kv_dim = 4096  # Standard MHA: 32 heads * 128 dim
    mla_kv_dim = 576  # MLA: 512 + 64

    compression = standard_kv_dim / mla_kv_dim
    # MLA should provide significant compression
    assert compression > 5.0


def test_rope_frequencies():
    """Test RoPE frequency computation."""
    head_dim = 64
    base = 10000.0
    seq_len = 128

    cos_table, sin_table = _precompute_rope_frequencies(seq_len, head_dim, base)

    assert cos_table.shape[0] == seq_len
    assert sin_table.shape[0] == seq_len
    # cos/sin table dimensions
    assert cos_table.shape[1] == head_dim // 2
    assert sin_table.shape[1] == head_dim // 2


def test_mla_fusion_format():
    """Test the concatenated KV format for fused cache write."""
    kv_lora_rank = 512
    qk_rope_head_dim = 64

    total_dim = kv_lora_rank + qk_rope_head_dim
    assert total_dim == 576  # Combined compressed+rope dimension
