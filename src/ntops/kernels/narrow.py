import functools

from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement as element_wise_arrangement


def arrangement(input, output, dim=None, start=None, length=None, block_size=None):
    slices = [slice(None)] * input.ndim
    slices[dim] = slice(start, start + length)

    input = input[tuple(slices)]

    return element_wise_arrangement(input, output, block_size=block_size)


def application(input, output):
    output = input  # noqa: F841


def premake(ndim, dim, start, length, dtype=None, block_size=None):
    arrangement_ = functools.partial(
        arrangement,
        dim=dim,
        start=start,
        length=length,
        block_size=block_size,
    )

    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
    )

    return arrangement_, application, tensors