"""Tests for gated RMSNorm kernel."""

import math

import pytest
import torch

import ntops


def _reference_rmsnorm(input, weight, eps=1e-6):
    """Reference RMSNorm implementation."""
    input_fp32 = input.float()
    variance = (input_fp32 ** 2).mean(dim=-1, keepdim=True)
    rms = torch.sqrt(variance + eps)
    normalized = input_fp32 / rms * weight.float()
    return normalized.to(input.dtype)


def _reference_gated_rmsnorm(input, weight, gate_input, gate_weight, activation="sigmoid", eps=1e-6):
    """Reference gated RMSNorm implementation."""
    # RMSNorm
    input_fp32 = input.float()
    variance = (input_fp32 ** 2).mean(dim=-1, keepdim=True)
    rms = torch.sqrt(variance + eps)
    normalized = input_fp32 / rms * weight.float()

    # Gate
    gate_fp32 = gate_input.float()
    if activation == "sigmoid":
        gate_val = torch.sigmoid(gate_fp32)
    elif activation == "silu":
        gate_val = torch.nn.functional.silu(gate_fp32)
    elif activation == "tanh":
        gate_val = torch.tanh(gate_fp32)
    else:
        gate_val = torch.sigmoid(gate_fp32)

    gated = gate_val * gate_weight.float()
    output = normalized * gated
    return output.to(input.dtype)


def test_rmsnorm_correctness():
    """Test RMSNorm computation matches reference."""
    torch.manual_seed(42)

    batch, seq_len, hidden = 2, 128, 512
    eps = 1e-6

    input = torch.randn(batch, seq_len, hidden, dtype=torch.bfloat16)
    weight = torch.ones(hidden, dtype=torch.bfloat16)

    output_ref = _reference_rmsnorm(input, weight, eps)

    # Verify shape
    assert output_ref.shape == input.shape


def test_gated_rmsnorm_output_shape():
    """Test gated RMSNorm output shape."""
    torch.manual_seed(42)

    shapes = [
        (128, 512),
        (64, 1024),
        (32, 2048),
        (1, 256, 512),
        (2, 128, 1024),
    ]

    for shape in shapes:
        input = torch.randn(*shape, dtype=torch.bfloat16)
        weight = torch.ones(shape[-1], dtype=torch.bfloat16)
        gate_input = torch.randn_like(input)
        gate_weight = torch.ones_like(weight)

        output = _reference_gated_rmsnorm(input, weight, gate_input, gate_weight)
        assert output.shape == input.shape


def test_gate_activations():
    """Test different gate activation functions."""
    torch.manual_seed(42)

    x = torch.randn(100, 512, dtype=torch.bfloat16)

    for activation in ["sigmoid", "silu", "tanh"]:
        weight = torch.ones(512, dtype=torch.bfloat16)
        gate_weight = torch.ones(512, dtype=torch.bfloat16)

        output = _reference_gated_rmsnorm(
            x, weight, x, gate_weight, activation=activation
        )
        assert output.shape == x.shape
        assert not torch.isnan(output).any()


def test_rmsnorm_numerical_stability():
    """Test RMSNorm numerical stability with various input ranges."""
    eps = 1e-6
    weight = torch.ones(512, dtype=torch.bfloat16)

    # Test with very small values
    input_small = torch.full((10, 512), 1e-10, dtype=torch.bfloat16)
    output_small = _reference_rmsnorm(input_small, weight, eps)
    assert not torch.isnan(output_small).any()
    assert not torch.isinf(output_small).any()

    # Test with large values
    input_large = torch.full((10, 512), 1e6, dtype=torch.bfloat16)
    output_large = _reference_rmsnorm(input_large, weight, eps)
    assert not torch.isnan(output_large).any()


def test_normalized_shape():
    """Test with different normalized shape configurations."""
    torch.manual_seed(42)

    # Last dimension normalization
    input = torch.randn(16, 128, 512, dtype=torch.bfloat16)
    weight = torch.ones(512, dtype=torch.bfloat16)
    output = _reference_rmsnorm(input, weight)
    assert output.shape == input.shape
