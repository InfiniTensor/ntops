import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


def _silu(gate):
    return gate / (1 + ntl.exp(-gate))


def _sigmoid(gate):
    return 1 / (1 + ntl.exp(-gate))


def _arrange_last_dim(tensor, block_size):
    # Keep each row in one program so the RMS reduction is local.
    tile_shape = (1,) * (tensor.ndim - 1) + (block_size,)
    return tensor.tile(tile_shape)


def _arrange_group(tensor, group_size):
    return tensor.flatten().tile((group_size,))


def arrangement(
    input,
    gate,
    weight,
    eps,
    output,
    num_normalized_elements,
    group_size=None,
    block_size=128,
):
    if group_size is None:
        input, gate, weight, output = (
            _arrange_last_dim(tensor, block_size)
            for tensor in (input, gate, weight, output)
        )
    else:
        input, gate, weight, output = (
            _arrange_group(tensor, group_size)
            for tensor in (input, gate, weight, output)
        )

    return input, gate, weight, eps, output, num_normalized_elements


def arrangement_no_gate(
    input,
    weight,
    eps,
    output,
    num_normalized_elements,
    group_size=None,
    block_size=128,
):
    if group_size is None:
        input, weight, output = (
            _arrange_last_dim(tensor, block_size) for tensor in (input, weight, output)
        )
    else:
        input, weight, output = (
            _arrange_group(tensor, group_size) for tensor in (input, weight, output)
        )

    return input, weight, eps, output, num_normalized_elements


def application_silu_before(input, gate, weight, eps, output, num_normalized_elements):
    input_f32 = (input + 0).to(ntl.float32)
    gate_f32 = (gate + 0).to(ntl.float32)
    weight_f32 = (weight + 0).to(ntl.float32)
    rms_value = ntl.sqrt(ntl.sum(input_f32 * input_f32) / num_normalized_elements + eps)
    output = input_f32 / rms_value * weight_f32 * _silu(gate_f32)  # noqa: F841


def application_silu_after(input, gate, weight, eps, output, num_normalized_elements):
    input_f32 = (input + 0).to(ntl.float32)
    gate_f32 = (gate + 0).to(ntl.float32)
    weight_f32 = (weight + 0).to(ntl.float32)
    gated = input_f32 * _silu(gate_f32)
    rms_value = ntl.sqrt(ntl.sum(gated * gated) / num_normalized_elements + eps)
    output = gated / rms_value * weight_f32  # noqa: F841


def application_sigmoid_before(
    input, gate, weight, eps, output, num_normalized_elements
):
    input_f32 = (input + 0).to(ntl.float32)
    gate_f32 = (gate + 0).to(ntl.float32)
    weight_f32 = (weight + 0).to(ntl.float32)
    rms_value = ntl.sqrt(ntl.sum(input_f32 * input_f32) / num_normalized_elements + eps)
    output = (  # noqa: F841
        input_f32 / rms_value * weight_f32 * _sigmoid(gate_f32)
    )


def application_sigmoid_after(
    input, gate, weight, eps, output, num_normalized_elements
):
    input_f32 = (input + 0).to(ntl.float32)
    gate_f32 = (gate + 0).to(ntl.float32)
    weight_f32 = (weight + 0).to(ntl.float32)
    gated = input_f32 * _sigmoid(gate_f32)
    rms_value = ntl.sqrt(ntl.sum(gated * gated) / num_normalized_elements + eps)
    output = gated / rms_value * weight_f32  # noqa: F841


def application_no_gate(input, weight, eps, output, num_normalized_elements):
    input_f32 = (input + 0).to(ntl.float32)
    weight_f32 = (weight + 0).to(ntl.float32)
    rms_value = ntl.sqrt(ntl.sum(input_f32 * input_f32) / num_normalized_elements + eps)
    output = input_f32 / rms_value * weight_f32  # noqa: F841


def premake(
    ndim,
    hidden_size,
    group_size=None,
    norm_before_gate=True,
    activation="silu",
    dtype=None,
    input_dtype=None,
    gate_dtype=None,
    weight_dtype=None,
    output_dtype=None,
    block_size=128,
    has_gate=True,
):
    if activation not in ("silu", "sigmoid", "swish"):
        raise ValueError("activation must be one of 'silu', 'sigmoid', or 'swish'")
    if dtype is not None:
        input_dtype = input_dtype or dtype
        gate_dtype = gate_dtype or dtype
        weight_dtype = weight_dtype or dtype
        output_dtype = output_dtype or dtype
    if block_size is None:
        block_size = 128

    tensor_shape = (None,) * (ndim - 1) + (hidden_size,)

    if has_gate:
        arrangement_ = functools.partial(
            arrangement,
            group_size=group_size,
            block_size=max(block_size, hidden_size),
        )
        tensors = (
            Tensor(shape=tensor_shape, other=0, dtype=input_dtype),
            Tensor(shape=tensor_shape, other=0, dtype=gate_dtype),
            Tensor(shape=tensor_shape, dtype=weight_dtype),
            Tensor(0, dtype=ninetoothed.float64),
            Tensor(shape=tensor_shape, dtype=output_dtype),
            Tensor(0, dtype=ninetoothed.float64),
        )
    else:
        arrangement_ = functools.partial(
            arrangement_no_gate,
            group_size=group_size,
            block_size=max(block_size, hidden_size),
        )
        tensors = (
            Tensor(shape=tensor_shape, other=0, dtype=input_dtype),
            Tensor(shape=tensor_shape, dtype=weight_dtype),
            Tensor(0, dtype=ninetoothed.float64),
            Tensor(shape=tensor_shape, dtype=output_dtype),
            Tensor(0, dtype=ninetoothed.float64),
        )

    if not has_gate:
        application = application_no_gate
    elif activation == "sigmoid":
        application = (
            application_sigmoid_before
            if norm_before_gate
            else application_sigmoid_after
        )
    else:
        application = (
            application_silu_before if norm_before_gate else application_silu_after
        )

    return arrangement_, application, tensors
