import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def application(input, other, output):
    # PyTorch nextafter spec, implemented via IEEE bit manipulation:
    #   if either is NaN: result is NaN
    #   if a == b: result is b (preserves sign of zero)
    #   if a == 0: result is smallest subnormal with sign of b
    #   otherwise: walk one ULP toward b in IEEE bit space
    dtype = output.dtype
    if dtype == ntl.float16 or dtype == ntl.bfloat16:
        int_dtype = ntl.int16
    elif dtype == ntl.float32:
        int_dtype = ntl.int32
    else:
        int_dtype = ntl.int64

    a = input
    b = other
    a_i = ntl.cast(a, int_dtype, bitcast=True)
    b_i = ntl.cast(b, int_dtype, bitcast=True)

    one = ntl.cast(1, int_dtype)
    zero = ntl.cast(0, int_dtype)
    sign_bit = one << (a_i.dtype.primitive_bitwidth - 1)

    is_nan = (a != a) | (b != b)
    eq = a == b
    is_zero = a == ntl.cast(0, dtype)

    b_sign = b_i & sign_bit
    zero_result = b_sign | one

    a_neg = a_i < zero
    a_lt_b = a < b
    step_up = a_neg ^ a_lt_b
    step = ntl.where(step_up, one, -one)
    general = a_i + step

    nan_bits = ntl.cast(ntl.cast(float("nan"), dtype), int_dtype, bitcast=True)
    result_i = ntl.where(
        is_nan,
        nan_bits,
        ntl.where(eq, b_i, ntl.where(is_zero, zero_result, general)),
    )
    output = ntl.cast(result_i, dtype, bitcast=True)  # noqa: F841


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
    )

    return arrangement_, application, tensors
