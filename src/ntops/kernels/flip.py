import functools

import ninetoothed
from ninetoothed import Tensor


def arrangement(input, output, dims, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    # Reverse the input's *logical* index along every flipped dim with a
    # step `-1` slice. ``_slice_dim`` turns logical index ``i`` into source
    # index ``size - 1 - i``, so the physical offset stays non-negative (no
    # negative strides, which PyTorch does not support). The output is left in
    # natural order; copying ``output = input`` therefore materializes the flip:
    #   output[..., i, ...] = input[..., size - 1 - i, ...]
    index = [slice(None)] * input.ndim
    for dim in dims:
        index[dim] = slice(None, None, -1)
    input_reversed = input[tuple(index)]

    input_arranged = input_reversed.flatten().tile((block_size,))
    output_arranged = output.flatten().tile((block_size,))

    return input_arranged, output_arranged


def application(input, output):
    output = input  # noqa: F841


def premake(ndim, dims, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, dims=dims, block_size=block_size)

    tensors = (Tensor(ndim, dtype=dtype), Tensor(ndim, dtype=dtype))

    return arrangement_, application, tensors
