import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def application(a, b, output):
    # Euclidean algorithm with fixed iteration count
    # Uses 64 iterations which is sufficient for 64-bit integers

    # Work with absolute values
    a_abs = ntl.abs(a)
    b_abs = ntl.abs(b)

    # Initialize
    x = a_abs
    y = b_abs

    # Euclidean algorithm: gcd(a, b) = gcd(b, a % b)
    # Fixed loop unrolling for GPU (no data-dependent loops)
    for _ in range(64):
        # Make y safe for modulo (avoid division by zero)
        y_safe = ntl.where(y == 0, 1, y)

        # Compute modulo safely
        mod = x % y_safe

        # Update: x <- y, y <- x % y
        # But if y was 0 (converged), keep x as is and set y to 0
        new_x = y
        new_y = mod

        # Convergence check
        converged = (y == 0)

        # Update with convergence protection
        x = ntl.where(converged, x, new_x)
        y = ntl.where(converged, 0, new_y)

    output = x  # noqa: F841


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
    )

    return arrangement_, application, tensors
