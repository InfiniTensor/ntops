import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


# Euclidean GCD: gcd(a, b) = gcd(b, a % b) until b == 0.
#
# All dtypes compute in int32 (for int8/int16/int32) or int64 (for int64).
# This avoids two problems in the original single-function implementation:
#   1. Computing in native dtype causes overflow then wrong abs
#      e.g. int8: 11*17=187 wraps to -69, ntl.abs gives 69 != torch.lcm's -69
#   2. 32 iterations is insufficient for int64 whose Fibonacci worst case
#      needs ~91 steps; int8/int16/int32 each need fewer.
#
# Iteration count per dtype (Fibonacci adversarial bound, a sorted >= b):
#   int8:  abs ≤ 128,     max ~10 steps  -> 12
#   int16: abs ≤ 32768,   max ~23 steps  -> 24
#   int32: abs ≤ 2^31,    max ~45 steps  -> 48
#   int64: abs ≤ 2^63,    max ~92 steps  -> 96
#
# LCM = (|a| / gcd) * |b|, computed in wide type, then cast to output dtype.
# The cast wraps on overflow, matching torch.lcm's behaviour for all dtypes.


def application_i8(input, other, output):
    w = ntl.int32
    abs_a = ntl.abs(ntl.cast(input, w))
    abs_b = ntl.abs(ntl.cast(other, w))
    or_ab = abs_a | abs_b
    a = ntl.where(abs_a >= abs_b, abs_a, abs_b)
    b = ntl.where(abs_a >= abs_b, abs_b, abs_a)
    zero = ntl.cast(0, w)
    for _ in range(12):
        b_safe = ntl.where(b != 0, b, zero + 1)
        r = a % b_safe
        a = ntl.where(b != 0, b, a)
        b = r
    gcd = ntl.where(a == 0, zero + 1, a)
    output = ntl.cast(  # noqa: F841
        ntl.where(or_ab == 0, zero, (abs_a // gcd) * abs_b), output.dtype
    )


def application_i16(input, other, output):
    w = ntl.int32
    abs_a = ntl.abs(ntl.cast(input, w))
    abs_b = ntl.abs(ntl.cast(other, w))
    or_ab = abs_a | abs_b
    a = ntl.where(abs_a >= abs_b, abs_a, abs_b)
    b = ntl.where(abs_a >= abs_b, abs_b, abs_a)
    zero = ntl.cast(0, w)
    for _ in range(24):
        b_safe = ntl.where(b != 0, b, zero + 1)
        r = a % b_safe
        a = ntl.where(b != 0, b, a)
        b = r
    gcd = ntl.where(a == 0, zero + 1, a)
    output = ntl.cast(  # noqa: F841
        ntl.where(or_ab == 0, zero, (abs_a // gcd) * abs_b), output.dtype
    )


def application_i32(input, other, output):
    w = ntl.int32
    abs_a = ntl.abs(ntl.cast(input, w))
    abs_b = ntl.abs(ntl.cast(other, w))
    or_ab = abs_a | abs_b
    a = ntl.where(abs_a >= abs_b, abs_a, abs_b)
    b = ntl.where(abs_a >= abs_b, abs_b, abs_a)
    zero = ntl.cast(0, w)
    for _ in range(48):
        b_safe = ntl.where(b != 0, b, zero + 1)
        r = a % b_safe
        a = ntl.where(b != 0, b, a)
        b = r
    gcd = ntl.where(a == 0, zero + 1, a)
    output = ntl.cast(  # noqa: F841
        ntl.where(or_ab == 0, zero, (abs_a // gcd) * abs_b), output.dtype
    )


def application_i64(input, other, output):
    w = ntl.int64
    abs_a = ntl.abs(ntl.cast(input, w))
    abs_b = ntl.abs(ntl.cast(other, w))
    or_ab = abs_a | abs_b
    a = ntl.where(abs_a >= abs_b, abs_a, abs_b)
    b = ntl.where(abs_a >= abs_b, abs_b, abs_a)
    zero = ntl.cast(0, w)
    for _ in range(96):
        b_safe = ntl.where(b != 0, b, zero + 1)
        r = a % b_safe
        a = ntl.where(b != 0, b, a)
        b = r
    gcd = ntl.where(a == 0, zero + 1, a)
    output = ntl.cast(  # noqa: F841
        ntl.where(or_ab == 0, zero, (abs_a // gcd) * abs_b), output.dtype
    )


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)
    if dtype == ninetoothed.int64:
        application = application_i64
    elif dtype == ninetoothed.int32:
        application = application_i32
    elif dtype == ninetoothed.int16:
        application = application_i16
    else:
        application = application_i8
    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
    )
    return arrangement_, application, tensors
