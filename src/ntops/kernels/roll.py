import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


def arrangement(input, size, shift, output, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    input_arranged = input.tile((1, block_size))
    input_arranged.dtype = input_arranged.dtype.squeeze(0)

    output_arranged = output.tile((1, block_size))
    output_arranged.dtype = output_arranged.dtype.squeeze(0)

    size_arranged = size.tile((1,))
    shift_arranged = shift.tile((1,))

    return input_arranged, size_arranged, shift_arranged, output_arranged


def application(input, size, shift, output):
    n = ntl.cast(size[0], ntl.int32)
    s = ntl.cast(shift[0], ntl.int32)
    for i in range(output.shape[0]):
        src_idx = (i + n - s) % n
        output[i] = input[src_idx]


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(1, dtype=dtype),
        Tensor(1, dtype=dtype),
        Tensor(ndim, dtype=dtype),
    )

    return arrangement_, application, tensors
