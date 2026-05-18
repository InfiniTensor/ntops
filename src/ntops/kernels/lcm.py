import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


# Match PyTorch's CUDA C++ integer promotion: narrow ints (int8/int16)
# are promoted to int32 for arithmetic and abs, then truncated back.
# Worst-case Euclidean iteration count is bounded by adjacent Fibonacci
# F(N) > 2^bits, so we unroll just enough per dtype to avoid wasted work.
def application_28(input, other, output):
    dtype = output.dtype
    compute_dtype = (
        ntl.int32 if dtype == ntl.int8 or dtype == ntl.int16 else dtype
    )
    abs_a = ntl.abs(ntl.cast(input, compute_dtype))
    abs_b = ntl.abs(ntl.cast(other, compute_dtype))
    a, b = abs_a, abs_b
    for _ in range(28):
        nonzero = b != 0
        safe_b = ntl.where(nonzero, b, 1)
        new_a = ntl.where(nonzero, b, a)
        new_b = ntl.where(nonzero, a % safe_b, b)
        a = new_a
        b = new_b
    gcd = a
    safe_gcd = ntl.where(gcd == 0, 1, gcd)
    output = ntl.cast(  # noqa: F841
        ntl.where(gcd == 0, 0, ntl.abs((abs_a // safe_gcd) * abs_b)), dtype
    )


def application_56(input, other, output):
    dtype = output.dtype
    compute_dtype = (
        ntl.int32 if dtype == ntl.int8 or dtype == ntl.int16 else dtype
    )
    abs_a = ntl.abs(ntl.cast(input, compute_dtype))
    abs_b = ntl.abs(ntl.cast(other, compute_dtype))
    a, b = abs_a, abs_b
    for _ in range(56):
        nonzero = b != 0
        safe_b = ntl.where(nonzero, b, 1)
        new_a = ntl.where(nonzero, b, a)
        new_b = ntl.where(nonzero, a % safe_b, b)
        a = new_a
        b = new_b
    gcd = a
    safe_gcd = ntl.where(gcd == 0, 1, gcd)
    output = ntl.cast(  # noqa: F841
        ntl.where(gcd == 0, 0, ntl.abs((abs_a // safe_gcd) * abs_b)), dtype
    )


def application_104(input, other, output):
    dtype = output.dtype
    compute_dtype = (
        ntl.int32 if dtype == ntl.int8 or dtype == ntl.int16 else dtype
    )
    abs_a = ntl.abs(ntl.cast(input, compute_dtype))
    abs_b = ntl.abs(ntl.cast(other, compute_dtype))
    a, b = abs_a, abs_b
    for _ in range(104):
        nonzero = b != 0
        safe_b = ntl.where(nonzero, b, 1)
        new_a = ntl.where(nonzero, b, a)
        new_b = ntl.where(nonzero, a % safe_b, b)
        a = new_a
        b = new_b
    gcd = a
    safe_gcd = ntl.where(gcd == 0, 1, gcd)
    output = ntl.cast(  # noqa: F841
        ntl.where(gcd == 0, 0, ntl.abs((abs_a // safe_gcd) * abs_b)), dtype
    )


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    if dtype == ninetoothed.int64:
        application = application_104
    elif dtype == ninetoothed.int32:
        application = application_56
    else:
        application = application_28

    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
    )

    return arrangement_, application, tensors
