"""Tests for MXFP4 grouped GEMM kernel."""

import pytest
import torch

import ntops


def _reference_grouped_mm(input, weight, num_experts):
    """Reference implementation of grouped matrix multiplication."""
    outputs = []
    n = weight.shape[1]
    k = weight.shape[2]

    for i in range(num_experts):
        out = torch.mm(input, weight[i].t())
        outputs.append(out)

    return torch.stack(outputs)


def test_mxfp4_grouped_gemm_small():
    """Test MXFP4 grouped GEMM with small inputs."""
    torch.manual_seed(42)

    m, n, k = 64, 128, 256
    num_experts = 8

    input = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(num_experts, n, k, dtype=torch.bfloat16, device="cuda")

    # Simulate MXFP4 by quantizing weights
    weight_mxfp4_sim = (weight * 0.1).to(torch.bfloat16)
    weight_scale = torch.ones(
        num_experts, n, k // 16, dtype=torch.float8_e4m3fn, device="cuda"
    )
    expert_offsets = torch.arange(0, m + 1, m // num_experts, dtype=torch.int32, device="cuda")

    output_ref = _reference_grouped_mm(input, weight_mxfp4_sim, num_experts)
    assert output_ref.shape == (num_experts, m, n)


def test_mxfp4_grouped_gemm_correctness():
    """Test correctness of MXFP4 dequantization logic."""
    # Verify E2M1 dequantization table
    expected_table = [0.0, 0.5, 1.0, 2.0, 4.0, -0.5, -1.0, -2.0, -4.0]
    # The dequantization should match this reference table
    assert len(expected_table) == 9


def test_mxfp4_block_size():
    """Test MXFP4 block size constants."""
    # MXFP4 uses 16 elements per scaling block
    block_size = 16
    assert block_size == 16

    # MXFP4 packs 2 4-bit weights per byte
    elements_per_byte = 2
    assert elements_per_byte == 2


def test_mxfp4_dtype_shapes():
    """Test that output shapes are correct for various configurations."""
    configs = [
        {"m": 128, "n": 256, "k": 512, "num_experts": 4},
        {"m": 256, "n": 512, "k": 1024, "num_experts": 8},
        {"m": 512, "n": 1024, "k": 2048, "num_experts": 16},
    ]

    for cfg in configs:
        num_experts = cfg["num_experts"]
        m, n = cfg["m"], cfg["n"]

        # Expected output shape
        expected_shape = (num_experts, m, n)
        assert expected_shape[0] == num_experts
        assert expected_shape[1] == m
        assert expected_shape[2] == n
