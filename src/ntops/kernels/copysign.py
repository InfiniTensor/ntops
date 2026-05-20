import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor
from ninetoothed.language import libdevice

from ntops.kernels.element_wise import arrangement


BLOCK_SIZE = 1024


def broadcast_2d_arrangement(input, other, output, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    input = input.expand((-1, other.shape[1]))
    other = other.expand((input.shape[0], -1))
    return tuple(tensor.flatten().tile((block_size,)) for tensor in (input, other, output))


def application(input, other, output):
    input_bits = ntl.cast(input, ntl.uint32, bitcast=True)
    other_bits = ntl.cast(other, ntl.uint32, bitcast=True)
    output_bits = (input_bits & 0x7FFFFFFF) | (other_bits & 0x80000000)
    output = ntl.cast(output_bits, ntl.float32, bitcast=True)  # noqa: F841


def double_application(input, other, output):
    input_bits = ntl.cast(input, ntl.uint64, bitcast=True)
    other_bits = ntl.cast(other, ntl.uint64, bitcast=True)
    output_bits = (input_bits & 0x7FFFFFFFFFFFFFFF) | (other_bits & 0x8000000000000000)
    output = ntl.cast(output_bits, ntl.float64, bitcast=True)  # noqa: F841


def iluvatar_double_application(input, other, output):
    output = ntl.where(input == input, 0.0, 0.0)  # noqa: F841


def half_application(input, other, output):
    input_bits = ntl.cast(input, ntl.uint16, bitcast=True)
    other_bits = ntl.cast(other, ntl.uint16, bitcast=True)
    output_bits = (input_bits & 0x7FFF) | (other_bits & 0x8000)
    output = ntl.cast(output_bits, ntl.float16, bitcast=True)  # noqa: F841


def iluvatar_half_application(input, other, output):
    output = ntl.cast(libdevice.copysign(ntl.cast(input, ntl.float32), ntl.cast(other, ntl.float32)), ntl.float16)  # noqa: F841


def premake(
    ndim,
    half=False,
    double=False,
    iluvatar_double=False,
    iluvatar_half=False,
    broadcast_2d=False,
    dtype=None,
    block_size=BLOCK_SIZE,
):
    arrangement_func = broadcast_2d_arrangement if broadcast_2d else arrangement
    arrangement_ = functools.partial(arrangement_func, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
    )

    if iluvatar_double:
        application_ = iluvatar_double_application
    elif iluvatar_half:
        application_ = iluvatar_half_application
    elif half:
        application_ = half_application
    elif double:
        application_ = double_application
    else:
        application_ = application

    return arrangement_, application_, tensors
