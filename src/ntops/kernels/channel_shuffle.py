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


def premake(dtype=None, block_size=None):
    arrangement_ = functools.partial(
        arrangement,
        block_size=block_size,
    )

    tensors = (
        Tensor(5, dtype=dtype),
        Tensor(4, dtype=dtype),
    )

    return arrangement_, application, tensors