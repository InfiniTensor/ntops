import functools

import ninetoothed
from ninetoothed import Tensor

from ntops.kernels.mm import BLOCK_SIZE_K, BLOCK_SIZE_M, BLOCK_SIZE_N, application


def arrangement(
    input, other, output, block_size_m=None, block_size_n=None, block_size_k=None
):
    if block_size_m is None:
        block_size_m = BLOCK_SIZE_M

    if block_size_n is None:
        block_size_n = BLOCK_SIZE_N

    if block_size_k is None:
        block_size_k = BLOCK_SIZE_K

    output_arranged = output.tile((1, block_size_m, block_size_n))
    output_arranged.dtype = output_arranged.dtype.squeeze(0)

    input_arranged = input.tile((1, block_size_m, block_size_k))
    input_arranged = input_arranged.tile((1, 1, -1))
    input_arranged = input_arranged.expand((-1, -1, output_arranged.shape[-1]))
    input_arranged.dtype = input_arranged.dtype.squeeze((0, 1))
    input_arranged.dtype.dtype = input_arranged.dtype.dtype.squeeze(0)

    other_arranged = other.tile((1, block_size_k, block_size_n))
    other_arranged = other_arranged.tile((1, -1, 1))
    other_arranged = other_arranged.expand((-1, output_arranged.shape[-2], -1))
    other_arranged.dtype = other_arranged.dtype.squeeze((0, 2))
    other_arranged.dtype.dtype = other_arranged.dtype.dtype.squeeze(0)

    return input_arranged, other_arranged, output_arranged


@functools.cache
def make():
    return ninetoothed.make(arrangement, application, (Tensor(3), Tensor(3), Tensor(3)))
