import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


def arrangement(input, output, k, dims, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    ndim = input.ndim
    dims = tuple(dim if dim >= 0 else dim + ndim for dim in dims)
    non_target_dims = tuple(i for i in range(ndim) if i not in dims)

    def _arrange_0(tensor):
        arranged = tensor.flatten()
        arranged = arranged.tile((block_size,))

        return arranged

    def _arrange_1_or_3(tensor, dims):
        arranged = tensor.permute(non_target_dims + dims)
        arranged = arranged.flatten(end_dim=-1)
        arranged = arranged.tile((1, -1))
        arranged.dtype = arranged.dtype.squeeze(0)

        return arranged

    def _arrange_2(tensor, dims):
        arranged = tensor.permute(non_target_dims + dims)

        if ndim == 2:
            arranged = arranged.unsqueeze(0)

        arranged = arranged.flatten(end_dim=-2)
        arranged = arranged.tile((1, -1, -1))
        arranged.dtype = arranged.dtype.squeeze(0)

        return arranged

    if k % 4 == 0:
        input_arranged = _arrange_0(input)
        output_arranged = _arrange_0(output)
    elif k % 4 == 1:
        input_arranged = _arrange_1_or_3(input, dims)
        output_arranged = _arrange_1_or_3(output, tuple(reversed(dims)))
    elif k % 4 == 3:
        input_arranged = _arrange_1_or_3(input, tuple(reversed(dims)))
        output_arranged = _arrange_1_or_3(output, dims)
    else:  # k % 4 == 2
        input_arranged = _arrange_2(input, dims)
        output_arranged = _arrange_2(output, dims)

    return input_arranged, output_arranged


def application_0(input, output):
    output = input  # noqa: F841


def application_1_or_3(input, output):
    if input.shape[0] == 1:
        output = input  # noqa: F841
    else:
        output = ntl.flip(input, 0)  # noqa: F841


def application_2(input, output):
    output = ntl.flip(ntl.flip(input, 0), 1)  # noqa: F841


def premake(ndim, k, dims, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, k=k, dims=dims, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype, shape_options={"constexpr": True}),
        Tensor(ndim, dtype=dtype, shape_options={"constexpr": True}),
    )

    if k % 4 == 0:
        application = application_0
    elif k % 4 == 2:
        application = application_2
    else:  # k % 4 == 1 or 3
        application = application_1_or_3

    return arrangement_, application, tensors
