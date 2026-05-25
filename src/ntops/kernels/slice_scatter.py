import functools

from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement as element_wise_arrangement


def copy_arrangement(input, output, block_size=None):
    return element_wise_arrangement(input, output, block_size=block_size)


def copy_application(input, output):
    output = input  # noqa: F841


def premake_copy(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(
        copy_arrangement,
        block_size=block_size,
    )

    tensors = (
        Tensor(ndim, dtype=dtype),  # input
        Tensor(ndim, dtype=dtype),  # output
    )

    return arrangement_, copy_application, tensors


def scatter_arrangement(
    source,
    output,
    dim=None,
    start=None,
    end=None,
    step=None,
    block_size=None,
):
    if step is None:
        step = 1

    slices = [slice(None)] * output.ndim
    slices[dim] = slice(start, end, step)

    output_slice = output[tuple(slices)]

    return element_wise_arrangement(source, output_slice, block_size=block_size)


def scatter_application(source, output):
    output = source  # noqa: F841


def premake_scatter(ndim, dim, start, end, step, dtype=None, block_size=None):
    arrangement_ = functools.partial(
        scatter_arrangement,
        dim=dim,
        start=start,
        end=end,
        step=step,
        block_size=block_size,
    )

    tensors = (
        Tensor(ndim, dtype=dtype),  # source
        Tensor(ndim, dtype=dtype),  # output
    )

    return arrangement_, scatter_application, tensors