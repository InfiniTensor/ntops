import functools

import ninetoothed
from ninetoothed import Tensor


def arrangement(input, output, permutation, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    assert input.ndim == output.ndim

    input_arranged = input.permute(permutation)
    input_arranged = input_arranged.flatten().tile((block_size,))

    output_arranged = output.flatten().tile((block_size,))

    return input_arranged, output_arranged


def application(input, output):
    output = input


def premake(ndim, permutation, dtype=None, block_size=None):
    arrangement_ = functools.partial(
        arrangement,
        permutation=permutation,
        block_size=block_size,
    )

    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
    )

    return arrangement_, application, tensors