# src/ntops/kernels/hsplit.py

import functools

from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement as element_wise_arrangement


def hsplit_arrangement(
    input,
    output,
    dim=None,
    start=None,
    end=None,
    block_size=None,
):
    if dim is None:
        dim = 0 if input.ndim == 1 else 1

    slices = [slice(None)] * input.ndim
    slices[dim] = slice(start, end)

    input_slice = input[tuple(slices)]

    return element_wise_arrangement(input_slice, output, block_size=block_size)


def hsplit_application(input, output):
    output = input  # noqa: F841


def premake(ndim, dim, start, end, dtype=None, block_size=None):
    arrangement_ = functools.partial(
        hsplit_arrangement,
        dim=dim,
        start=start,
        end=end,
        block_size=block_size,
    )

    tensors = (
        Tensor(ndim, dtype=dtype),  # input
        Tensor(ndim, dtype=dtype),  # output
    )

    return arrangement_, hsplit_application, tensors