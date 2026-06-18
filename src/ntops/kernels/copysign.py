import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


# copysign(input, other) = magnitude of input + sign of other.
#
# Magnitude: ntl.abs(input) handles all float types directly.
# Sign detection: cast other to a same-width signed integer and check < 0.
#   - Signed int comparison checks the MSB, which is exactly the IEEE 754
#     sign bit for all standard float widths (16 / 32 / 64 bit).
#   - This correctly identifies -0.0 as negative (0x8000... as signed int
#     is INT_MIN, which is < 0), unlike the float comparison `other < 0`
#     which treats -0.0 as 0.
#
# Three functions because the bitcast target type depends on float width:
#   float16 / bfloat16  (16-bit)  ->  int16
#   float32             (32-bit)  ->  int32
#   float64             (64-bit)  ->  int64


def application_f16(input, other, output):
    other_sign_negative = ntl.cast(other, ntl.int16, bitcast=True) < 0
    abs_val = ntl.abs(input)
    output = ntl.where(other_sign_negative, -abs_val, abs_val)  # noqa: F841


def application_f32(input, other, output):
    other_sign_negative = ntl.cast(other, ntl.int32, bitcast=True) < 0
    abs_val = ntl.abs(input)
    output = ntl.where(other_sign_negative, -abs_val, abs_val)  # noqa: F841


def application_f64(input, other, output):
    other_sign_negative = ntl.cast(other, ntl.int64, bitcast=True) < 0
    abs_val = ntl.abs(input)
    output = ntl.where(other_sign_negative, -abs_val, abs_val)  # noqa: F841


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    if dtype in (ninetoothed.float16, ninetoothed.bfloat16):
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
