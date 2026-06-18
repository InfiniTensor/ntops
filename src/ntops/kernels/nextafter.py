import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor
from ninetoothed.language import libdevice

from ntops.kernels.element_wise import arrangement


# nextafter for integers: the nearest integer toward other is simply ±1 away.
def application_int(input, other, output):
    output = ntl.where(  # noqa: F841
        input == other,
        other,
        ntl.where(input < other, input + 1, input - 1),
    )


# nextafter for float16 / bfloat16.
# libdevice.nextafter does not accept narrow float types; we use bitcast to int16.
#
# Observation from bit patterns (int16 = bitcast of float16):
#   positive floats: int16 order == float order  (+3.0 → 16896 > +1.0 → 15360)
#   negative floats: int16 order is REVERSED     (-3.0 → −15872, -2.998 → −15873)
#
# So to advance one ULP toward other:
#   a > 0, a < b  →  a_i + 1   (moving up in both orderings)
#   a > 0, a > b  →  a_i - 1
#   a < 0, a < b  →  a_i - 1   (moving up in float but down in int16)
#   a < 0, a > b  →  a_i + 1
#
# Special cases (checked before the general step):
#   NaN  : propagate whichever input is NaN (a takes priority)
#   equal: return b's bit pattern (handles +0 == -0 correctly)
#   zero : return ±min_subnormal with b's sign (bitcast b < 0 detects -0.0)
def application_f16(input, other, output):
    i16 = ntl.int16
    one = ntl.cast(1, i16)
    zero_i = ntl.cast(0, i16)

    a_i = ntl.cast(input, i16, bitcast=True)
    b_i = ntl.cast(other, i16, bitcast=True)

    # NaN propagation: keep NaN bits of the offending operand
    a_nan = input != input
    b_nan = other != other
    nan_bits = ntl.where(a_nan, a_i, b_i)

    # Zero → smallest subnormal; sign of b determined by its sign bit
    from_zero = ntl.where(b_i < zero_i, ntl.cast(-32767, i16), one)

    # General case: positive int16 means positive float (same order),
    # negative int16 means negative float (reversed order) → flip step sign.
    going_up = input < other
    a_i_positive = a_i >= zero_i
    step = ntl.where(
        a_i_positive,
        ntl.where(going_up, one, -one),
        ntl.where(going_up, -one, one),
    )

    result_i = ntl.where(
        a_nan | b_nan,
        nan_bits,
        ntl.where(
            input == other,
            b_i,
            ntl.where(input == ntl.cast(0, ntl.float16), from_zero, a_i + step),
        ),
    )
    output = ntl.cast(result_i, output.dtype, bitcast=True)  # noqa: F841


def application_f32(input, other, output):
    output = libdevice.nextafter(input, other)  # noqa: F841


def application_f64(input, other, output):
    output = libdevice.nextafter(input, other)  # noqa: F841


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    int_types = (ninetoothed.int8, ninetoothed.int16, ninetoothed.int32, ninetoothed.int64)
    if dtype in int_types:
        application = application_int
    elif dtype in (ninetoothed.float16, ninetoothed.bfloat16):
        application = application_f16
    elif dtype == ninetoothed.float32:
        application = application_f32
    else:
        application = application_f64

    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
    )

    return arrangement_, application, tensors
