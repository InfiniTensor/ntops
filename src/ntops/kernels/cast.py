import functools

import ninetoothed
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def application(input, output):
    output = input  # noqa: F841


def premake(ndim, input_dtype=None, output_dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=input_dtype),
        Tensor(ndim, dtype=output_dtype),
    )

    return arrangement_, application, tensors
