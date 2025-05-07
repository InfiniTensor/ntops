import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

BLOCK_SIZE_M = ninetoothed.block_size()
BLOCK_SIZE_N = ninetoothed.block_size()
BLOCK_SIZE_K = ninetoothed.block_size()


def arrangement(input, other, output):
    output_arranged = output.tile((BLOCK_SIZE_M, BLOCK_SIZE_N))

    input_arranged = input.tile((BLOCK_SIZE_M, BLOCK_SIZE_K))
    input_arranged = input_arranged.tile((1, -1))
    input_arranged = input_arranged.expand((-1, output_arranged.shape[1]))
    input_arranged.dtype = input_arranged.dtype.squeeze(0)

    other_arranged = other.tile((BLOCK_SIZE_K, BLOCK_SIZE_N))
    other_arranged = other_arranged.tile((-1, 1))
    other_arranged = other_arranged.expand((output_arranged.shape[0], -1))
    other_arranged.dtype = other_arranged.dtype.squeeze(1)

    return input_arranged, other_arranged, output_arranged


def application(input, other, output):
    accumulator = ntl.zeros(output.shape, dtype=ntl.float32)

    for k in range(input.shape[0]):
        accumulator += ntl.dot(input[k], other[k])

    output = accumulator


@functools.cache
def make():
    return ninetoothed.make(arrangement, application, (Tensor(2), Tensor(2), Tensor(2)))
