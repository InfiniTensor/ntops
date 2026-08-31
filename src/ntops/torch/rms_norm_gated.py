import torch

import ntops
from ntops.torch.utils import _cached_make


def rms_norm_gated(
    input,
    z=None,
    weight=None,
    eps=1e-5,
    group_size=None,
    norm_before_gate=False,
    activation="swish",
    block_size=128,
    num_warps=(1, 2, 4, 8),
    num_stages=(1, 2),
    max_num_configs=8,
):
    """RMSNormGated with a specialized layout and shape-keyed autotuning.

    The default candidates target the vLLM GDN hot path and cache the best
    launch configuration for each input shape, dtype, and stride signature.
    """
    if input.ndim == 0:
        raise ValueError("input must have at least one dimension")

    if activation not in ("silu", "sigmoid", "swish"):
        raise ValueError("activation must be one of 'silu', 'sigmoid', or 'swish'")

    hidden_size = input.shape[-1]
    if group_size is not None:
        if not isinstance(group_size, int) or isinstance(group_size, bool):
            raise TypeError("group_size must be a positive integer or None")
        if group_size <= 0:
            raise ValueError("group_size must be a positive integer")
        if hidden_size % group_size != 0:
            raise ValueError("hidden size must be divisible by group_size")
        num_normalized_elements = group_size
    else:
        num_normalized_elements = hidden_size

    if weight is None:
        weight = torch.ones(
            hidden_size,
            device=input.device,
            dtype=input.dtype,
        )
    weight = weight.expand_as(input)

    has_gate = z is not None
    gate_dtype = input.dtype
    if has_gate:
        z = z.expand_as(input)
        gate_dtype = z.dtype

    output = torch.empty_like(input)
    kernel = _cached_make(
        ntops.kernels.rms_norm_gated.premake,
        input.ndim,
        hidden_size,
        group_size,
        norm_before_gate,
        activation,
        input_dtype=input.dtype,
        gate_dtype=gate_dtype,
        weight_dtype=weight.dtype,
        output_dtype=output.dtype,
        block_size=block_size,
        has_gate=has_gate,
        num_warps=num_warps,
        num_stages=num_stages,
        max_num_configs=max_num_configs,
    )

    if has_gate:
        kernel(input, z, weight, eps, output, num_normalized_elements)
    else:
        kernel(input, weight, eps, output, num_normalized_elements)

    return output
