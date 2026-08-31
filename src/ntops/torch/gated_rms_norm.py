import torch

import ntops
from ntops.kernels.gated_rms_norm import ActivationVariant
from ntops.torch.utils import _cached_make


def gated_rms_norm(
    input,
    gate=None,
    weight=None,
    eps=1e-5,
    group_size=None,
    norm_before_gate=False,
    activation="swish",
):
    if weight is None:
        raise ValueError("`weight` must be provided.")

    if gate is not None and input.shape != gate.shape:
        raise ValueError("`input` and `gate` must have the same shape.")

    hidden_size = input.shape[-1]

    if weight.shape != (hidden_size,):
        raise ValueError("`weight` must have shape `(input.shape[-1],)`.")

    if group_size is None:
        group_size = hidden_size

    if group_size <= 0 or hidden_size % group_size != 0:
        raise ValueError("`group_size` must be a positive divisor of the hidden size.")

    activation_variants = {
        "silu": ActivationVariant.SILU,
        "swish": ActivationVariant.SILU,
        "sigmoid": ActivationVariant.SIGMOID,
    }

    try:
        activation_variant = activation_variants[activation]
    except KeyError as error:
        raise ValueError(
            "`activation` must be one of `silu`, `swish`, or `sigmoid`."
        ) from error

    grouped_shape = input.shape[:-1] + (hidden_size // group_size, group_size)
    grouped_input = input.reshape(grouped_shape)
    grouped_gate = None if gate is None else gate.reshape(grouped_shape)
    prefix = input.shape[:-1]
    num_groups = hidden_size // group_size
    grouped_weight = weight.reshape((1,) * len(prefix) + (num_groups, group_size))
    grouped_weight = grouped_weight.expand(grouped_shape)
    grouped_output = torch.empty_like(grouped_input)

    kernel = _cached_make(
        ntops.kernels.gated_rms_norm.premake,
        grouped_input.ndim,
        has_gate=gate is not None,
        norm_before_gate=norm_before_gate,
        activation=activation_variant,
        block_size=1 << (group_size - 1).bit_length(),
    )
    if gate is None:
        kernel(grouped_input, grouped_weight, eps, grouped_output, group_size)
    else:
        kernel(
            grouped_input,
            grouped_gate,
            grouped_weight,
            eps,
            grouped_output,
            group_size,
        )

    return grouped_output.reshape(input.shape)
