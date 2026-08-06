"""Gated RMSNorm torch wrapper."""

import torch

import ntops
from ntops.torch.utils import _cached_make


def gated_rmsnorm(
    input,
    normalized_shape,
    weight=None,
    gate_input=None,
    gate_weight=None,
    gate_activation="sigmoid",
    eps=None,
):
    """Gated RMSNorm with fused computation.

    Applies RMSNorm followed by elementwise gating:
    output = RMSNorm(input) * gate_activation(gate)

    Args:
        input: Input tensor.
        normalized_shape: Shape of the normalized dimensions.
        weight: Learnable scale for RMSNorm.
        gate_input: Gate signal input (defaults to input if None).
        gate_weight: Learnable scale for gate.
        gate_activation: Gate activation type ('sigmoid', 'silu', 'gelu', 'tanh').
        eps: Small value for numerical stability.

    Returns:
        output: Gated RMSNorm output.
    """
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)

    normalized_shape = tuple(normalized_shape)
    num_normalized_dims = len(normalized_shape)

    if weight is None:
        weight = torch.ones(normalized_shape, dtype=input.dtype, device=input.device)

    if gate_input is None:
        gate_input = input

    if gate_weight is None:
        gate_weight = torch.ones(normalized_shape, dtype=input.dtype, device=input.device)

    if eps is None:
        eps = torch.finfo(input.dtype).eps

    activation_map = {
        "sigmoid": 1,
        "silu": 2,
        "gelu": 3,
        "tanh": 4,
    }
    gate_activation_type = activation_map.get(gate_activation, 1)

    output = torch.empty_like(input)
    num_normalized_elements = 1
    for s in normalized_shape:
        num_normalized_elements *= s

    kernel = _cached_make(
        ntops.kernels.gated_rmsnorm.premake,
        ndim=input.ndim,
        num_normalized_dims=num_normalized_dims,
        input_dtype=input.dtype,
        weight_dtype=weight.dtype,
        gate_dtype=gate_input.dtype,
        output_dtype=output.dtype,
        gate_activation_type=gate_activation_type,
    )

    kernel(
        input,
        weight,
        gate_input,
        gate_weight,
        eps,
        output,
        gate_activation_type,
        num_normalized_elements,
    )

    return output
