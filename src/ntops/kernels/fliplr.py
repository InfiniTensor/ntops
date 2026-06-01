import functools

import ninetoothed
from ninetoothed import Tensor


def application(input, output):
    output = input  # noqa: F841


def arrangement(
    input,
    output,
    block_size=None,
):
    if block_size is None:
        block_size = ninetoothed.block_size()

    ndim = input.ndim
    assert ndim >= 2, "`fliplr` requires input with ndim >= 2."
    slices = tuple(
        slice(None, None, -1) if dim == 1 else slice(None)
        for dim in range(ndim)
    )

    input_arranged = input[slices]
    input_arranged = input_arranged.flatten().tile((block_size,))

    output_arranged = output.flatten().tile((block_size,))

    return input_arranged, output_arranged


def premake(
    ndim,
    dtype=None,
    block_size=None,
):
    assert ndim >= 2, "`fliplr` requires input with ndim >= 2."

    arrangement_ = functools.partial(
        arrangement,
        block_size=block_size,
    )

    tensors = (
        Tensor(
            ndim,
            dtype=dtype,
            shape_options={"constexpr": True},
        ),
        Tensor(
            ndim,
            dtype=dtype,
            shape_options={"constexpr": True},
        ),
    )

    return arrangement_, application, tensors