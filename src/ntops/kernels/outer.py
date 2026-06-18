import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


def arrangement(input, other, output, block_size_m=None, block_size_n=None):
    if block_size_m is None:
        block_size_m = ninetoothed.block_size()

    if block_size_n is None:
        block_size_n = ninetoothed.block_size()

    output_arranged = output.tile((block_size_m, block_size_n))

    input_arranged = input.tile((block_size_m, 1))
    input_arranged = input_arranged.expand((-1, output_arranged.shape[1]))

    other_arranged = other.tile((1, block_size_n))
    other_arranged = other_arranged.expand((output_arranged.shape[0], -1))

    return input_arranged, other_arranged, output_arranged


def application(input, other, output):
    output = input * other  # noqa: F841


def premake(dtype=None, block_size_m=None, block_size_n=None):
    arrangement_ = functools.partial(
        arrangement,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
    )

    tensors = (
        Tensor(2, dtype=dtype),
        Tensor(2, dtype=dtype),
        Tensor(2, dtype=dtype),
    )

    return arrangement_, application, tensors
