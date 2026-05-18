import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor
from ninetoothed.language import libdevice

from ntops.kernels.element_wise import arrangement


# Binary GCD (Stein's algorithm), inlined per dtype-bucketed unroll count.
# PyTorch CUDA lcm semantics:
#   lcm(a, b) = abs((|a| / gcd) * |b|) with two's-complement wrap.
#   Narrow ints (int8/int16) are promoted to int32 for arithmetic, then
#   truncated back, so wrap-around matches PyTorch.
#
# Algorithm (every step keeps a and b odd):
#   shift b right by ctz(b) so b is odd
#   diff = b_odd - a   (signed)
#   new_a = min(a, b_odd)
#   new_b = abs(diff)
# After enough steps b == 0; gcd = a << k where k = ctz(|input| | |other|).
#
# `libdevice.ffs(x)` returns 1 + ctz(x) for x != 0, and 0 for x == 0.
#
# NineToothed AST does not cross Python function boundaries cleanly
# (tuple returns from helpers raise compilation errors), so the prelude /
# step / finish blocks are duplicated inline in each application_<N>.
#
# Worst-case Stein iterations: each step removes at least one bit of
# information from b, so 2 * bit_width is a safe upper bound:
#   int8/int16 (promoted to int32, but value range still ≤ 32767):  ~32
#   int32:                                                            64
#   int64:                                                           128
def application_32(input, other, output):
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

    for _ in range(32):
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


def application_64(input, other, output):
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

    for _ in range(64):
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


def application_128(input, other, output):
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

    for _ in range(128):
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
        application = application_128
    elif dtype == ninetoothed.int32:
        application = application_64
    else:
        application = application_32

    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
    )

    return arrangement_, application, tensors
