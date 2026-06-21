import functools

import ninetoothed
from ninetoothed import Tensor


def arrangement(input, output, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    input = input.flatten().tile((block_size,))
    output = output.flatten().tile((block_size,))

    return input, output


def application(input, output):
    output = input


def premake(input_ndim, output_ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (
        Tensor(input_ndim, dtype=dtype),
        Tensor(output_ndim, dtype=dtype),
    )

    return arrangement_, application, tensors