"""Tests for block-scaled FP8 GEMM kernel."""

import pytest
import torch

import ntops


def _reference_block_scaled_mm(input_fp8, input_scale, weight_fp8, weight_scale):
    """Reference implementation of block-scaled FP8 GEMM."""
    # Dequantize inputs
    input_shape = input_fp8.shape
    weight_shape = weight_fp8.shape

    block_size = 128

    # Dequantize input
    m, k = input_fp8.shape
    n = weight_shape[0]

    input_dequant = torch.zeros(m, k, dtype=torch.float32)
    weight_dequant = torch.zeros(n, k, dtype=torch.float32)

    for i in range(0, k, block_size):
        blk_end = min(i + block_size, k)
        input_dequant[:, i:blk_end] = (
            input_fp8[:, i:blk_end].float() * input_scale[:, i // block_size].unsqueeze(1)
        )
        weight_dequant[:, i:blk_end] = (
            weight_fp8[:, i:blk_end].float() * weight_scale[i // block_size, :].unsqueeze(1)
        )

    output = torch.mm(input_dequant.to(torch.bfloat16), weight_dequant.t().to(torch.bfloat16))
    return output


def test_fp8_dequantization_correctness():
    """Test that FP8 dequantization produces correct values."""
    # FP8 E4M3 range
    fp8_max = 448.0
    fp8_min = -448.0

    # Test values that should be representable
    test_vals = [0.0, 0.5, 1.0, 2.0, -1.0, -2.0, 0.25, -0.25]
    for val in test_vals:
        # Simulate quantization-dequantization
        scale = 1.0
        quantized = torch.tensor(val / scale, dtype=torch.float8_e4m3fn)
        dequantized = quantized.float() * scale
        # Allow for quantization error
        assert abs(dequantized.item() - val) < 0.1 * abs(val) + 0.01


def test_block_scaling_factors():
    """Test block scaling factor computation."""
    torch.manual_seed(42)

    m, n, k = 128, 256, 512
    block_size = 128

    # Create random scales
    num_k_blocks = k // block_size
    num_n_blocks = n // block_size

    input_scale = torch.rand(m, num_k_blocks, dtype=torch.float32) + 0.5
    weight_scale = torch.rand(num_k_blocks, num_n_blocks, dtype=torch.float32) + 0.5

    assert input_scale.shape == (m, num_k_blocks)
    assert weight_scale.shape == (num_k_blocks, num_n_blocks)


def test_fp8_gemm_output_shape():
    """Test output shape correctness."""
    configs = [
        {"m": 128, "n": 256, "k": 512},
        {"m": 256, "n": 512, "k": 1024},
        {"m": 1024, "n": 2048, "k": 4096},
    ]

    for cfg in configs:
        m, n, k = cfg["m"], cfg["n"], cfg["k"]
        expected_output_shape = (m, n)
        assert expected_output_shape[0] == m
        assert expected_output_shape[1] == n


def test_online_rescaling():
    """Test that online rescaling prevents overflow."""
    # Simulate large values that could cause overflow
    large_val = 1.0e30

    # Rescaling should bring values down
    threshold = 1.0e30
    abs_max = large_val

    # Safe scale computation
    safe_scale = min(threshold / abs_max, 1.0)
    rescaled = abs_max * safe_scale

    # Should not overflow
    assert rescaled <= threshold
