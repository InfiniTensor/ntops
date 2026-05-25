import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def application(input, values, output):
    zero = input * 0
    one = zero + 1

    tmp = ntl.where(input < zero, zero, one)
    output = ntl.where(input == zero, values, tmp)  # noqa: F841


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype),  # input
        Tensor(ndim, dtype=dtype),  # values
        Tensor(ndim, dtype=dtype),  # output
    )

    return arrangement_, application, tensors