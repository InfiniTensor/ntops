import functools

import ninetoothed
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def application(input, output):
    output = max(0.0, input)  # noqa: F841


def premake(ndim, dtype, block_size):
    tensors = (Tensor(ndim, dtype=dtype), Tensor(ndim, dtype=dtype))

    return functools.partial(arrangement, block_size=block_size), application, tensors


@functools.cache
def make(ndim, dtype=None, block_size=None):
    return ninetoothed.make(*premake(ndim, dtype, block_size))
