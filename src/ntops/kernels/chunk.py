import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


def arrangement(input, output, dim, chunk_start, chunk_size, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    ndim = input.ndim

    slices = tuple(
        slice(chunk_start, chunk_start + chunk_size) if d == dim else slice(None)
        for d in range(ndim)
    )
    input_chunk = input[slices]

    input_arranged = input_chunk.flatten().tile((block_size,))
    output_arranged = output.flatten().tile((block_size,))

    return input_arranged, output_arranged


def application(input, output):
    output = input  # noqa: F841


def premake(ndim, dim, chunk_start, chunk_size, dtype=None, block_size=None):
    arrangement_ = functools.partial(
        arrangement,
        dim=dim,
        chunk_start=chunk_start,
        chunk_size=chunk_size,
        block_size=block_size,
    )

    tensors = (Tensor(ndim, dtype=dtype), Tensor(ndim, dtype=dtype))

    return arrangement_, application, tensors
