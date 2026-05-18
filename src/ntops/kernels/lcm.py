import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor
from ninetoothed.language import libdevice

from ntops.kernels.element_wise import arrangement


# Stein binary GCD with unroll counts derived from empirical worst-case
# outer-iteration counts (iter04):
#
#   int8  (value range <= 127):                            max  5 -> use  8
#   int16 (value range <= 32767):                          max 13 -> use 16
#   int32 (value range <= 2^31):                           max 31 -> use 36
#   int64 (value range <= 2^63):                           max 63 -> use 72
#
# Earlier iterations used 32/64/128 which were safe but ~2x over-engineered:
# my outer loop already absorbs all consecutive trailing-zero shifts in one
# iteration via "b >> ctz(b)", so the theoretical max outer iter count is
# bit_width (not 2 * bit_width). Empirical Fibonacci adversarial inputs
# confirm < bit_width.
def application_8(input, other, output):
    dtype = output.dtype
    compute_dtype = (
        ntl.int32 if dtype == ntl.int8 or dtype == ntl.int16 else dtype
    )
    abs_a = ntl.abs(ntl.cast(input, compute_dtype))
    abs_b = ntl.abs(ntl.cast(other, compute_dtype))
    or_ab = abs_a | abs_b
    safe_or = ntl.where(or_ab != 0, or_ab, 1)
    k = ntl.cast(libdevice.ffs(safe_or) - 1, compute_dtype)
    a0 = abs_a >> k
    b0 = abs_b >> k
    nonzero_a0 = a0 != 0
    safe_a0 = ntl.where(nonzero_a0, a0, 1)
    ctz_a0 = ntl.cast(libdevice.ffs(safe_a0) - 1, compute_dtype)
    a = ntl.where(nonzero_a0, a0 >> ctz_a0, b0)
    b = ntl.where(nonzero_a0, b0, ntl.cast(0, compute_dtype))
    for _ in range(8):
        nonzero_b = b != 0
        safe_b = ntl.where(nonzero_b, b, 1)
        ctz_b = ntl.cast(libdevice.ffs(safe_b) - 1, compute_dtype)
        b_odd = b >> ctz_b
        diff = b_odd - a
        a = ntl.where(nonzero_b, ntl.minimum(a, b_odd), a)
        b = ntl.where(nonzero_b, ntl.abs(diff), b)
    gcd = a << k
    safe_gcd = ntl.where(gcd == 0, 1, gcd)
    output = ntl.cast(  # noqa: F841
        ntl.where(gcd == 0, 0, ntl.abs((abs_a // safe_gcd) * abs_b)), dtype
    )


def application_16(input, other, output):
    dtype = output.dtype
    compute_dtype = (
        ntl.int32 if dtype == ntl.int8 or dtype == ntl.int16 else dtype
    )
    abs_a = ntl.abs(ntl.cast(input, compute_dtype))
    abs_b = ntl.abs(ntl.cast(other, compute_dtype))
    or_ab = abs_a | abs_b
    safe_or = ntl.where(or_ab != 0, or_ab, 1)
    k = ntl.cast(libdevice.ffs(safe_or) - 1, compute_dtype)
    a0 = abs_a >> k
    b0 = abs_b >> k
    nonzero_a0 = a0 != 0
    safe_a0 = ntl.where(nonzero_a0, a0, 1)
    ctz_a0 = ntl.cast(libdevice.ffs(safe_a0) - 1, compute_dtype)
    a = ntl.where(nonzero_a0, a0 >> ctz_a0, b0)
    b = ntl.where(nonzero_a0, b0, ntl.cast(0, compute_dtype))
    for _ in range(16):
        nonzero_b = b != 0
        safe_b = ntl.where(nonzero_b, b, 1)
        ctz_b = ntl.cast(libdevice.ffs(safe_b) - 1, compute_dtype)
        b_odd = b >> ctz_b
        diff = b_odd - a
        a = ntl.where(nonzero_b, ntl.minimum(a, b_odd), a)
        b = ntl.where(nonzero_b, ntl.abs(diff), b)
    gcd = a << k
    safe_gcd = ntl.where(gcd == 0, 1, gcd)
    output = ntl.cast(  # noqa: F841
        ntl.where(gcd == 0, 0, ntl.abs((abs_a // safe_gcd) * abs_b)), dtype
    )


def application_36(input, other, output):
    dtype = output.dtype
    compute_dtype = (
        ntl.int32 if dtype == ntl.int8 or dtype == ntl.int16 else dtype
    )
    abs_a = ntl.abs(ntl.cast(input, compute_dtype))
    abs_b = ntl.abs(ntl.cast(other, compute_dtype))
    or_ab = abs_a | abs_b
    safe_or = ntl.where(or_ab != 0, or_ab, 1)
    k = ntl.cast(libdevice.ffs(safe_or) - 1, compute_dtype)
    a0 = abs_a >> k
    b0 = abs_b >> k
    nonzero_a0 = a0 != 0
    safe_a0 = ntl.where(nonzero_a0, a0, 1)
    ctz_a0 = ntl.cast(libdevice.ffs(safe_a0) - 1, compute_dtype)
    a = ntl.where(nonzero_a0, a0 >> ctz_a0, b0)
    b = ntl.where(nonzero_a0, b0, ntl.cast(0, compute_dtype))
    for _ in range(36):
        nonzero_b = b != 0
        safe_b = ntl.where(nonzero_b, b, 1)
        ctz_b = ntl.cast(libdevice.ffs(safe_b) - 1, compute_dtype)
        b_odd = b >> ctz_b
        diff = b_odd - a
        a = ntl.where(nonzero_b, ntl.minimum(a, b_odd), a)
        b = ntl.where(nonzero_b, ntl.abs(diff), b)
    gcd = a << k
    safe_gcd = ntl.where(gcd == 0, 1, gcd)
    output = ntl.cast(  # noqa: F841
        ntl.where(gcd == 0, 0, ntl.abs((abs_a // safe_gcd) * abs_b)), dtype
    )


def application_72(input, other, output):
    dtype = output.dtype
    compute_dtype = (
        ntl.int32 if dtype == ntl.int8 or dtype == ntl.int16 else dtype
    )
    abs_a = ntl.abs(ntl.cast(input, compute_dtype))
    abs_b = ntl.abs(ntl.cast(other, compute_dtype))
    or_ab = abs_a | abs_b
    safe_or = ntl.where(or_ab != 0, or_ab, 1)
    k = ntl.cast(libdevice.ffs(safe_or) - 1, compute_dtype)
    a0 = abs_a >> k
    b0 = abs_b >> k
    nonzero_a0 = a0 != 0
    safe_a0 = ntl.where(nonzero_a0, a0, 1)
    ctz_a0 = ntl.cast(libdevice.ffs(safe_a0) - 1, compute_dtype)
    a = ntl.where(nonzero_a0, a0 >> ctz_a0, b0)
    b = ntl.where(nonzero_a0, b0, ntl.cast(0, compute_dtype))
    for _ in range(72):
        nonzero_b = b != 0
        safe_b = ntl.where(nonzero_b, b, 1)
        ctz_b = ntl.cast(libdevice.ffs(safe_b) - 1, compute_dtype)
        b_odd = b >> ctz_b
        diff = b_odd - a
        a = ntl.where(nonzero_b, ntl.minimum(a, b_odd), a)
        b = ntl.where(nonzero_b, ntl.abs(diff), b)
    gcd = a << k
    safe_gcd = ntl.where(gcd == 0, 1, gcd)
    output = ntl.cast(  # noqa: F841
        ntl.where(gcd == 0, 0, ntl.abs((abs_a // safe_gcd) * abs_b)), dtype
    )


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)
    if dtype == ninetoothed.int64:
        application = application_72
    elif dtype == ninetoothed.int32:
        application = application_36
    elif dtype == ninetoothed.int16:
        application = application_16
    else:
        application = application_8
    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
    )
    return arrangement_, application, tensors
