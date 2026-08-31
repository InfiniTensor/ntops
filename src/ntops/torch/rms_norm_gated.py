import torch

import ntops
from ntops.torch.utils import _cached_make

_ACTIVATIONS = {
    "silu": ntops.kernels.rms_norm_gated.ActivationVariant.SILU,
    "swish": ntops.kernels.rms_norm_gated.ActivationVariant.SILU,
    "sigmoid": ntops.kernels.rms_norm_gated.ActivationVariant.SIGMOID,
}


def _next_power_of_two(value):
    return 1 << (value - 1).bit_length()


def _validate(input, gate, weight, group_size, activation):
    if input.shape != gate.shape:
        raise ValueError(
            "input and gate must have the same shape, but got "
            f"{tuple(input.shape)} and {tuple(gate.shape)}."
        )
    if input.ndim == 0:
        raise ValueError("input and gate must have at least one dimension.")
    if input.device != gate.device:
        raise ValueError("input and gate must be on the same device.")
    if input.dtype != gate.dtype:
        raise TypeError("input and gate must have the same dtype.")
    if input.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("input and gate must use float16, bfloat16, or float32.")
    if input.stride(-1) != 1 or gate.stride(-1) != 1:
        raise ValueError("The last dimension of input and gate must be contiguous.")

    hidden_size = input.shape[-1]
    if group_size is None:
        group_size = hidden_size
    if group_size <= 0 or hidden_size % group_size:
        raise ValueError(
            f"group_size={group_size} must divide hidden_size={hidden_size}."
        )

    if weight is not None:
        if weight.shape != (hidden_size,):
            raise ValueError(
                f"weight must have shape {(hidden_size,)}, got {tuple(weight.shape)}."
            )
        if weight.device != input.device or weight.dtype != input.dtype:
            raise ValueError("weight must share input's device and dtype.")
        if not weight.is_contiguous():
            raise ValueError("weight must be contiguous.")

    try:
        activation_variant = _ACTIVATIONS[activation.lower()]
    except (AttributeError, KeyError) as error:
        raise ValueError(
            "activation must be one of 'silu', 'swish', or 'sigmoid'."
        ) from error

    return group_size, activation_variant


def rms_norm_gated(
    input,
    gate,
    weight=None,
    eps=1e-5,
    group_size=None,
    norm_before_gate=False,
    activation="swish",
):
    group_size, activation_variant = _validate(
        input, gate, weight, group_size, activation
    )
    hidden_size = input.shape[-1]
    num_groups = hidden_size // group_size
    output = torch.empty_like(input)
    if input.ndim == 2 and num_groups == 1:
        grouped_input = input
        grouped_gate = gate
        grouped_output = output
    else:
        grouped_input = input.reshape(-1, group_size)
        grouped_gate = gate.reshape(-1, group_size)
        grouped_output = output.reshape(-1, group_size)

    grouped_weight = None
    if weight is not None:
        if num_groups == 1:
            # Keep the affine vector one-dimensional.  The SSA application
            # broadcasts it over rows, avoiding a stride-zero expanded view
            # and reducing the runtime launch ABI.
            grouped_weight = weight
        else:
            grouped_weight = weight.reshape(num_groups, group_size)
            expand_shape = (*input.shape[:-1], num_groups, group_size)
            grouped_weight = grouped_weight.reshape(
                *((1,) * (input.ndim - 1)), num_groups, group_size
            ).expand(expand_shape)
            grouped_weight = grouped_weight.reshape(-1, group_size)

    kernel = _cached_make(
        ntops.kernels.rms_norm_gated.premake,
        grouped_input.ndim,
        group_size,
        activation_variant,
        bool(norm_before_gate),
        input_dtype=input.dtype,
        gate_dtype=gate.dtype,
        weight_dtype=None if weight is None else weight.dtype,
        output_dtype=output.dtype,
        block_size=group_size,
        input_shape=tuple(grouped_input.shape),
        gate_shape=tuple(grouped_gate.shape),
        weight_shape=(
            None if grouped_weight is None else tuple(grouped_weight.shape)
        ),
        output_shape=tuple(grouped_output.shape),
        input_strides=tuple(grouped_input.stride()),
        gate_strides=tuple(grouped_gate.stride()),
        weight_strides=(
            None if grouped_weight is None else tuple(grouped_weight.stride())
        ),
        output_strides=tuple(grouped_output.stride()),
        num_warps=ntops.kernels.rms_norm_gated.NUM_WARPS,
        num_stages=ntops.kernels.rms_norm_gated.NUM_STAGES,
    )

    arguments = [grouped_input, grouped_gate]
    if grouped_weight is not None:
        arguments.append(grouped_weight)
    arguments.extend((eps, grouped_output))
    # `_validate` plus the freshly allocated output establish the static tensor
    # contracts and non-aliasing required by the compiler's private fast path.
    # Fall back to the fully verified public handle for other backends.
    launch = getattr(kernel, "_launch_prevalidated_noalias", kernel)
    launch(*arguments)

    return output


__all__ = ["rms_norm_gated"]
