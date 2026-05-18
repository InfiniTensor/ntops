import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor
from ninetoothed.language import libdevice

from ntops.kernels.element_wise import arrangement


BLOCK_SIZE = 128


def broadcast_2d_arrangement(input, other, output, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    input = input.expand((-1, other.shape[1]))
    other = other.expand((input.shape[0], -1))
    return tuple(tensor.flatten().tile((block_size,)) for tensor in (input, other, output))


def application(input, other, output):
    value = libdevice.nextafter(input, other)
    zero_value = ntl.where(other < 0, -1.401298464324817e-45, 1.401298464324817e-45)
    zero_value = ntl.where(other == 0, other, zero_value)
    value = ntl.where(input == 0, zero_value, value)
    output = ntl.where(other != other, other, value)  # noqa: F841


def double_application(input, other, output):
    value = libdevice.nextafter(input, other)
    zero_value = ntl.where(other < 0, -4.9406564584124654e-324, 4.9406564584124654e-324)
    zero_value = ntl.where(other == 0, other, zero_value)
    value = ntl.where(input == 0, zero_value, value)
    output = ntl.where(other != other, other, value)  # noqa: F841


def half_application(input, other, output):
    bits = ntl.cast(input, ntl.uint16, bitcast=True)
    next_bits = ntl.where(
        ntl.where(input > 0, other > input, other < input),
        bits + 1,
        bits - 1,
    )
    next_value = ntl.cast(next_bits, ntl.float16, bitcast=True)
    zero_value = ntl.where(other < 0, -5.960464477539063e-08, 5.960464477539063e-08)
    zero_value = ntl.where(other == 0, other, zero_value)
    value = ntl.where(input == 0, zero_value, next_value)
    value = ntl.where(input == other, other, value)
    value = ntl.where(input != input, input, value)
    output = ntl.where(other != other, other, value)  # noqa: F841


def premake(ndim, half=False, double=False, broadcast_2d=False, dtype=None, block_size=BLOCK_SIZE):
    arrangement_func = broadcast_2d_arrangement if broadcast_2d else arrangement
    arrangement_ = functools.partial(arrangement_func, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
    )

    if half:
        application_ = half_application
    elif double:
        application_ = double_application
    else:
        application_ = application

    return arrangement_, application_, tensors
