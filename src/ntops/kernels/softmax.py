import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

BLOCK_SIZE = ninetoothed.block_size()


def arrangement(input, output, dim):
    assert input.ndim == output.ndim

    def create_axis_tile_shape(dim, dim_block):
        return (
            tuple(1 for _ in range(dim))
            + (dim_block,)
            + tuple(1 for _ in range(input.ndim - dim - 1))
        )

    def arrange(input):
        input_arranged = input.tile(inner_block_shape).tile(outer_block_shape)

        input_arranged.dtype = input_arranged.dtype.squeeze(
            tuple(d for d in range(input.ndim) if d != dim)
        )
        input_arranged.dtype.dtype = input_arranged.dtype.dtype.squeeze(
            tuple(d for d in range(input.ndim) if d != dim)
        )
        return input_arranged

    inner_block_shape = create_axis_tile_shape(dim, BLOCK_SIZE)
    outer_block_shape = create_axis_tile_shape(dim, -1)

    return arrange(input), arrange(output)


def _exp(x, dtype):
    exp_dtype = dtype if dtype != ntl.float16 else ntl.float32
    return ntl.cast(ntl.exp(ntl.cast(x, exp_dtype)), dtype)


def application(input, output):
    dtype = output.dtype.dtype
    prev_max = ntl.cast(float("-inf"), dtype)
    denominator = ntl.cast(0, dtype)

    for i in range(input.shape[0]):
        input_i = ntl.cast(input[i], dtype)
        curr_max = ntl.cast(ntl.maximum(prev_max, ntl.max(input_i)), dtype)
        input_max_diff_exp = _exp(input_i - curr_max, dtype)
        prev_curr_max_diff_exp = _exp(prev_max - curr_max, dtype)
        denominator = denominator * prev_curr_max_diff_exp + ntl.sum(input_max_diff_exp)
        prev_max = curr_max

    for i in range(input.shape[0]):
        numerator = _exp(input[i] - prev_max, dtype)
        output[i] = numerator / denominator


@functools.cache
def make(ndim, dim):
    return ninetoothed.make(
        functools.partial(arrangement, dim=dim),
        application,
        (
            Tensor(ndim, other=float("-inf"), shape_options={"constexpr": True}),
            Tensor(ndim),
        ),
    )
