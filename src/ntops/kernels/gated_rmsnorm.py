"""Gated RMSNorm Fusion Operator.

This kernel implements a fused RMSNorm with gating mechanism commonly used
in modern LLM architectures such as KDA (Key-gated Dot-product Attention)
and gated linear attention models.

The gating mechanism applies a sigmoid or SiLU gate to the normalized output:
output = RMSNorm(input) * gate_activation(gate_input)

Key fusion benefits:
1. Fuses RMSNorm computation with elementwise gating (reduces memory round-trips)
2. Single-pass normalization with online statistics computation
3. Supports multiple gate activations (sigmoid, swish/silu, tanh)

RMSNorm formula:
    RMS(x) = sqrt(mean(x^2) + eps)
    output = (x / RMS(x) * weight) * gate(x_gate)
"""

import enum
import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

BLOCK_SIZE = ninetoothed.block_size()


class GateActivation(enum.IntEnum):
    """Gate activation function selection."""

    SIGMOID = enum.auto()
    SILU = enum.auto()
    GELU = enum.auto()
    TANH = enum.auto()


def _apply_gate(x, activation_type):
    """Apply the selected gate activation function."""
    if activation_type == GateActivation.SIGMOID:
        return ntl.sigmoid(x)
    elif activation_type == GateActivation.SILU:
        return ntl.silu(x)
    elif activation_type == GateActivation.GELU:
        return ntl.gelu(x)
    elif activation_type == GateActivation.TANH:
        return ntl.tanh(x)
    else:
        return ntl.sigmoid(x)  # Default fallback


def arrangement(
    input,
    weight,
    gate_input,
    gate_weight,
    eps,
    output,
    gate_activation_type,
    num_normalized_dims,
    block_size=None,
):
    """Arrange tensors for gated RMSNorm.

    The input tensor is normalized along the last `num_normalized_dims`
    dimensions, then element-wise multiplied with a gated signal.
    """
    if block_size is None:
        block_size = BLOCK_SIZE

    # Normalize dimensions to handle negative indexing
    ndim = input.ndim
    if num_normalized_dims < 0:
        num_normalized_dims += ndim

    non_normalized_dims = ndim - num_normalized_dims

    def _arrange_non_normalized(tensor, has_extra_dims=False):
        """Arrange a tensor that spans all dimensions."""
        if has_extra_dims:
            # Input has extra batch dimensions
            arranged = tensor.permute(tuple(range(non_normalized_dims)) + tuple(
                -(i + 1) for i in range(num_normalized_dims)
            ))
            arranged = arranged.flatten(start_dim=-num_normalized_dims)
            inner_block = tuple(1 for _ in range(non_normalized_dims)) + (block_size,)
            outer_block = tuple(1 for _ in range(non_normalized_dims)) + (-1,)
            arranged = arranged.tile(inner_block)
            arranged = arranged.tile(outer_block)
            squeeze_dims = tuple(range(non_normalized_dims))
            arranged.dtype = arranged.dtype.squeeze(squeeze_dims)
            arranged.dtype.dtype = arranged.dtype.dtype.squeeze(squeeze_dims)
        else:
            arranged = tensor.tile(block_size)
            arranged = arranged.tile(-1)
            arranged.dtype = arranged.dtype.squeeze(0)
        return arranged

    # Arrange input to normalize
    input_arranged = _arrange_non_normalized(input, has_extra_dims=(input.ndim > num_normalized_dims))

    # Arrange weight (same shape as normalized dimensions)
    weight_arranged = weight.tile(block_size)
    weight_arranged = weight_arranged.tile(-1)
    weight_arranged.dtype = weight_arranged.dtype.squeeze(0)

    # Arrange gate input (same shape as input)
    gate_input_arranged = _arrange_non_normalized(gate_input, has_extra_dims=(gate_input.ndim > num_normalized_dims))

    # Arrange gate weight (same shape as weight)
    gate_weight_arranged = gate_weight.tile(block_size)
    gate_weight_arranged = gate_weight_arranged.tile(-1)
    gate_weight_arranged.dtype = gate_weight_arranged.dtype.squeeze(0)

    # Arrange output
    output_arranged = _arrange_non_normalized(output, has_extra_dims=(output.ndim > num_normalized_dims))

    # Epsilon (scalar)
    eps_arranged = eps

    # Gate activation type (constexpr)
    gate_activation_arranged = gate_activation_type

    return (
        input_arranged,
        weight_arranged,
        gate_input_arranged,
        gate_weight_arranged,
        eps_arranged,
        output_arranged,
        gate_activation_arranged,
    )


def application(
    input,
    weight,
    gate_input,
    gate_weight,
    eps,
    output,
    gate_activation_type,
    num_normalized_elements,
):
    """Apply gated RMSNorm computation.

    Computes:
        rms = sqrt(mean(input^2) + eps)
        normalized = input / rms * weight
        gate = gate_activation(gate_input) * gate_weight
        output = normalized * gate
    """
    # Compute sum of squares for RMSNorm
    input_sq = ntl.cast(input, ntl.float32) * ntl.cast(input, ntl.float32)
    sum_sq = ntl.sum(input_sq)

    # Compute RMS
    rms = ntl.sqrt(sum_sq / num_normalized_elements + eps)

    # Apply RMSNorm: normalized = input / rms * weight
    input_fp32 = ntl.cast(input, ntl.float32)
    weight_fp32 = ntl.cast(weight, ntl.float32)
    normalized = input_fp32 / rms * weight_fp32

    # Apply gate: gate = gate_activation(gate_input) * gate_weight
    gate_input_fp32 = ntl.cast(gate_input, ntl.float32)
    gate_weight_fp32 = ntl.cast(gate_weight, ntl.float32)
    gate_signal = _apply_gate(gate_input_fp32, gate_activation_type) * gate_weight_fp32

    # Fused multiply: output = normalized * gate
    output[:] = normalized * gate_signal


def premake(
    ndim=None,
    num_normalized_dims=None,
    input_dtype=None,
    weight_dtype=None,
    gate_dtype=None,
    output_dtype=None,
    gate_activation_type=None,
    block_size=None,
):
    """Premake configuration for gated RMSNorm kernel."""
    if block_size is None:
        block_size = BLOCK_SIZE

    num_norm = num_normalized_dims if num_normalized_dims is not None else 1

    def arrangement_(*tensors, dim=None):
        return arrangement(
            *tensors,
            gate_activation_type=gate_activation_type,
            num_normalized_dims=num_norm,
            block_size=block_size,
        )

    # Input tensor
    input_tensor = Tensor(ndim, other=0, dtype=input_dtype)

    # RMSNorm weight (learnable scale)
    weight_tensor = Tensor(ndim, dtype=weight_dtype)

    # Gate input (can be same or different from input)
    gate_input_tensor = Tensor(ndim, other=0, dtype=gate_dtype)

    # Gate weight (learnable gate scale)
    gate_weight_tensor = Tensor(ndim, dtype=gate_dtype)

    # Epsilon (scalar)
    eps_tensor = Tensor(0, dtype=ninetoothed.float64)

    # Output tensor
    output_tensor = Tensor(ndim, dtype=output_dtype)

    # Gate activation type (constexpr)
    gate_activation_tensor = Tensor(0, constexpr=True, value=gate_activation_type)

    tensors = (
        input_tensor,
        weight_tensor,
        gate_input_tensor,
        gate_weight_tensor,
        eps_tensor,
        output_tensor,
        gate_activation_tensor,
    )

    return arrangement_, application, tensors
