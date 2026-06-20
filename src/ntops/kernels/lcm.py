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

    # Compute LCM: (a_abs // gcd) * b_abs using pure integer arithmetic
    # Safe division (avoid division by zero)
    gcd_safe = ntl.where(gcd_val == 0, ntl.cast(1, a_abs.dtype), gcd_val)
    lcm_result = (a_abs // gcd_safe) * b_abs

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
