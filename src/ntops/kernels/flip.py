import functools

import ninetoothed
from ninetoothed import Tensor


def application(input, output):
    output = input  # noqa: F841


def _normalize_dims(dims, ndim):
    if isinstance(dims, int):
        dims = (dims,)

    dims = tuple(dim if dim >= 0 else dim + ndim for dim in dims)

    assert all(0 <= dim < ndim for dim in dims), "`dims` out of range."

    result = []
    for dim in dims:
        if dim not in result:
            result.append(dim)

    return tuple(result)


def arrangement(
    input,
    output,
    dims,
    block_size=None,
):
    if block_size is None:
        block_size = ninetoothed.block_size()

    ndim = input.ndim
    dims = _normalize_dims(dims, ndim)
    slices = tuple(
        slice(None, None, -1) if dim in dims else slice(None)
        for dim in range(ndim)
    )

    input_arranged = input[slices]
    input_arranged = input_arranged.flatten().tile((block_size,))

    output_arranged = output.flatten().tile((block_size,))

    return input_arranged, output_arranged


def premake(
    ndim,
    dims,
    dtype=None,
    block_size=None,
):
    dims = _normalize_dims(dims, ndim)

    arrangement_ = functools.partial(
        arrangement,
        dims=dims,
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