import enum
import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

NUM_WARPS = (1, 2, 4, 8)
NUM_STAGES = 1


class ActivationVariant(enum.IntEnum):
    SILU = enum.auto()
    SIGMOID = enum.auto()


def arrangement(*tensors, block_size):
    return tuple(
        tensor.tile((1, block_size)) if tensor.ndim == 2 else tensor
        for tensor in tensors
    )


def silu_after_gate_application(
    input, gate, weight, eps, output
):
    input_value = input.to(ntl.float32)
    gate_value = gate.to(ntl.float32)
    value = input_value * gate_value / (1 + ntl.exp(-gate_value))
    variance = ntl.sum(value * value, axis=1) / input.shape[1]
    output = (  # noqa: F841
        value
        * ntl.rsqrt(variance[:, None] + eps)
        * weight.to(ntl.float32)
    )


def silu_before_gate_application(
    input, gate, weight, eps, output
):
    input_value = input.to(ntl.float32)
    gate_value = gate.to(ntl.float32)
    activated_gate = gate_value / (1 + ntl.exp(-gate_value))
    variance = ntl.sum(input_value * input_value, axis=1) / input.shape[1]
    output = (  # noqa: F841
        input_value
        * ntl.rsqrt(variance[:, None] + eps)
        * activated_gate
        * weight.to(ntl.float32)
    )


def sigmoid_after_gate_application(
    input, gate, weight, eps, output
):
    input_value = input.to(ntl.float32)
    gate_value = gate.to(ntl.float32)
    activated_gate = 1 / (1 + ntl.exp(-gate_value))
    value = input_value * activated_gate
    variance = ntl.sum(value * value, axis=1) / input.shape[1]
    output = (  # noqa: F841
        value
        * ntl.rsqrt(variance[:, None] + eps)
        * weight.to(ntl.float32)
    )


def sigmoid_before_gate_application(
    input, gate, weight, eps, output
):
    input_value = input.to(ntl.float32)
    gate_value = gate.to(ntl.float32)
    activated_gate = 1 / (1 + ntl.exp(-gate_value))
    variance = ntl.sum(input_value * input_value, axis=1) / input.shape[1]
    output = (  # noqa: F841
        input_value
        * ntl.rsqrt(variance[:, None] + eps)
        * activated_gate
        * weight.to(ntl.float32)
    )


def silu_after_gate_no_weight_application(
    input, gate, eps, output
):
    input_value = input.to(ntl.float32)
    gate_value = gate.to(ntl.float32)
    value = input_value * gate_value / (1 + ntl.exp(-gate_value))
    variance = ntl.sum(value * value, axis=1) / input.shape[1]
    output = value * ntl.rsqrt(variance[:, None] + eps)  # noqa: F841


def silu_before_gate_no_weight_application(
    input, gate, eps, output
):
    input_value = input.to(ntl.float32)
    gate_value = gate.to(ntl.float32)
    activated_gate = gate_value / (1 + ntl.exp(-gate_value))
    variance = ntl.sum(input_value * input_value, axis=1) / input.shape[1]
    output = (  # noqa: F841
        input_value * ntl.rsqrt(variance[:, None] + eps) * activated_gate
    )


def sigmoid_after_gate_no_weight_application(
    input, gate, eps, output
):
    input_value = input.to(ntl.float32)
    gate_value = gate.to(ntl.float32)
    activated_gate = 1 / (1 + ntl.exp(-gate_value))
    value = input_value * activated_gate
    variance = ntl.sum(value * value, axis=1) / input.shape[1]
    output = value * ntl.rsqrt(variance[:, None] + eps)  # noqa: F841


def sigmoid_before_gate_no_weight_application(
    input, gate, eps, output
):
    input_value = input.to(ntl.float32)
    gate_value = gate.to(ntl.float32)
    activated_gate = 1 / (1 + ntl.exp(-gate_value))
    variance = ntl.sum(input_value * input_value, axis=1) / input.shape[1]
    output = (  # noqa: F841
        input_value * ntl.rsqrt(variance[:, None] + eps) * activated_gate
    )


_APPLICATIONS = {
    (ActivationVariant.SILU, False, True): silu_after_gate_application,
    (ActivationVariant.SILU, True, True): silu_before_gate_application,
    (ActivationVariant.SIGMOID, False, True): sigmoid_after_gate_application,
    (ActivationVariant.SIGMOID, True, True): sigmoid_before_gate_application,
    (ActivationVariant.SILU, False, False): silu_after_gate_no_weight_application,
    (ActivationVariant.SILU, True, False): silu_before_gate_no_weight_application,
    (
        ActivationVariant.SIGMOID,
        False,
        False,
    ): sigmoid_after_gate_no_weight_application,
    (
        ActivationVariant.SIGMOID,
        True,
        False,
    ): sigmoid_before_gate_no_weight_application,
}


def premake(
    ndim,
    group_size,
    activation_variant,
    norm_before_gate,
    input_dtype=None,
    gate_dtype=None,
    weight_dtype=None,
    output_dtype=None,
    block_size=None,
    input_shape=None,
    gate_shape=None,
    weight_shape=None,
    output_shape=None,
    input_strides=None,
    gate_strides=None,
    weight_strides=None,
    output_strides=None,
):
    arrangement_ = functools.partial(arrangement, block_size=block_size)
    has_weight = weight_dtype is not None
    application = _APPLICATIONS[
        (activation_variant, norm_before_gate, has_weight)
    ]

    def tensor(dtype, shape, strides, *, other=None):
        if shape is None:
            return Tensor(ndim, other=other, dtype=dtype)
        return Tensor(
            shape=tuple(shape),
            strides=None if strides is None else tuple(strides),
            other=other,
            dtype=dtype,
        )

    tensors = [
        tensor(input_dtype, input_shape, input_strides, other=0),
        tensor(gate_dtype, gate_shape, gate_strides, other=0),
    ]
    if has_weight:
        tensors.append(tensor(weight_dtype, weight_shape, weight_strides))
    tensors.extend(
        (
            Tensor(0, dtype=ninetoothed.float64),
            tensor(output_dtype, output_shape, output_strides),
        )
    )

    return arrangement_, application, tuple(tensors)


__all__ = ["ActivationVariant", "premake"]
