import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.reduction import arrangement


def application(input, output):
    accumulator = 0.0

    for i in range(input.shape[0]):
        block_sum = ntl.sum(input[i], axis=0)
        accumulator += block_sum

    output[0] = ntl.cast(accumulator, output.dtype.dtype)


def premake(ndim, dim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, dim=dim, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
    )

    return arrangement_, application, tensors


def arrangement_all_elements(input, output, block_size=None):
    input = input.flatten().tile((block_size,))
    output = output.tile((1,))
    return input, output


def application_all_elements(input, output):
    output[0] = ntl.sum(input, 0)


def premake_all_elements(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement_all_elements, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(1, dtype=dtype),
    )

    return arrangement_, application_all_elements, tensors
