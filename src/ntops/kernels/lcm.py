import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def application(input, other, output):
    a = ntl.abs(input)
    b = ntl.abs(other)
    for _ in range(32):
        b_is_zero = b == 0
        b_safe = ntl.where(b_is_zero, 1, b)
        a_mod_b = a % b_safe
        new_a = ntl.where(b_is_zero, a, b)
        new_b = ntl.where(b_is_zero, b, a_mod_b)
        a, b = new_a, new_b
    a_is_zero = a == 0
    a_safe = ntl.where(a_is_zero, 1, a)
    output = ntl.abs(input // a_safe * other)  # noqa: F841


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
    )

    return arrangement_, application, tensors
