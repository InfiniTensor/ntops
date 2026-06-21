import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def application(a, b, output):
    # LCM formula: lcm(a, b) = |a * b| / gcd(a, b)
    # Handle zero case: lcm(a, 0) = 0

    a_abs = ntl.abs(a)
    b_abs = ntl.abs(b)

    # Check if either input is zero
    is_zero = (a == 0) | (b == 0)

    # Compute GCD using Euclidean algorithm (inlined)
    x = a_abs
    y = b_abs

    for _ in range(64):
        # Safe modulo: make y at least 1 to avoid division by zero
        y_safe = ntl.where(y == 0, ntl.cast(1, y.dtype), y)
        mod = x % y_safe

        # Update: x <- y, y <- x % y
        new_x = y
        new_y = mod

        # Convergence check
        converged = (y == 0)

        # Update with convergence protection
        x = ntl.where(converged, x, new_x)
        y = ntl.where(converged, ntl.cast(0, y.dtype), new_y)

    gcd_val = x

    # Compute LCM: (a / gcd) * b to avoid overflow
    # Use float64 for intermediate calculation to maintain precision
    gcd_float = ntl.cast(gcd_val, ntl.float64)
    a_float = ntl.cast(a_abs, ntl.float64)

    # Safe division (avoid division by zero)
    gcd_safe_float = ntl.where(gcd_float == 0, ntl.cast(1, ntl.float64), gcd_float)
    quotient_float = a_float / gcd_safe_float

    # Cast back to integer and multiply
    quotient = ntl.cast(quotient_float, a_abs.dtype)
    lcm_result = quotient * b_abs

    # Return 0 if either input was 0, otherwise return LCM
    zero_val = ntl.cast(0, output.dtype)
    output = ntl.where(is_zero, zero_val, lcm_result)  # noqa: F841


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
    )

    return arrangement_, application, tensors
