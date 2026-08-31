import enum
import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.reduction import arrangement


class ActivationVariant(enum.IntEnum):
    SILU = enum.auto()
    SIGMOID = enum.auto()


def application_norm_before_silu(
    input, gate, weight, eps, output, num_normalized_elements
):
    value = input[0].to(ntl.float32)
    rms = ntl.sqrt(ntl.sum(value * value, axis=0) / num_normalized_elements + eps)
    gate_float = gate[0].to(ntl.float32)
    activated_gate = gate_float / (1 + ntl.exp(-gate_float))
    output[0] = value / rms * weight[0].to(ntl.float32) * activated_gate


def application_norm_before_sigmoid(
    input, gate, weight, eps, output, num_normalized_elements
):
    value = input[0].to(ntl.float32)
    rms = ntl.sqrt(ntl.sum(value * value, axis=0) / num_normalized_elements + eps)
    gate_float = gate[0].to(ntl.float32)
    activated_gate = 1 / (1 + ntl.exp(-gate_float))
    output[0] = value / rms * weight[0].to(ntl.float32) * activated_gate


def application_norm_after_silu(
    input, gate, weight, eps, output, num_normalized_elements
):
    value = input[0].to(ntl.float32)
    gate_float = gate[0].to(ntl.float32)
    value *= gate_float / (1 + ntl.exp(-gate_float))
    rms = ntl.sqrt(ntl.sum(value * value, axis=0) / num_normalized_elements + eps)
    output[0] = value / rms * weight[0].to(ntl.float32)


def application_norm_after_sigmoid(
    input, gate, weight, eps, output, num_normalized_elements
):
    value = input[0].to(ntl.float32)
    gate_float = gate[0].to(ntl.float32)
    value *= 1 / (1 + ntl.exp(-gate_float))
    rms = ntl.sqrt(ntl.sum(value * value, axis=0) / num_normalized_elements + eps)
    output[0] = value / rms * weight[0].to(ntl.float32)


def application_norm_only(input, weight, eps, output, num_normalized_elements):
    value = input[0].to(ntl.float32)
    rms = ntl.sqrt(ntl.sum(value * value, axis=0) / num_normalized_elements + eps)
    output[0] = value / rms * weight[0].to(ntl.float32)


def premake(
    ndim,
    norm_before_gate=False,
    activation=ActivationVariant.SILU,
    input_dtype=None,
    gate_dtype=None,
    weight_dtype=None,
    output_dtype=None,
    block_size=None,
    has_gate=True,
):
    arrangement_ = functools.partial(arrangement, dim=-1, block_size=block_size)

    if not has_gate:
        application = application_norm_only
    elif norm_before_gate:
        if activation == ActivationVariant.SIGMOID:
            application = application_norm_before_sigmoid
        else:
            application = application_norm_before_silu
    elif activation == ActivationVariant.SIGMOID:
        application = application_norm_after_sigmoid
    else:
        application = application_norm_after_silu

    if has_gate:
        tensors = (
            Tensor(ndim, other=0, dtype=input_dtype),
            Tensor(ndim, other=0, dtype=gate_dtype),
            Tensor(ndim, dtype=weight_dtype),
            Tensor(0, dtype=ninetoothed.float64),
            Tensor(ndim, dtype=output_dtype),
            Tensor(0, constexpr=True),
        )
    else:
        tensors = (
            Tensor(ndim, other=0, dtype=input_dtype),
            Tensor(ndim, dtype=weight_dtype),
            Tensor(0, dtype=ninetoothed.float64),
            Tensor(ndim, dtype=output_dtype),
            Tensor(0, constexpr=True),
        )

    return arrangement_, application, tensors
