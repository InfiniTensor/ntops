import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


def arrangement(input, output, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    ndim = input.ndim
    if ndim < 2:
        raise ValueError("nonzero_sum_gt_last2 requires ndim >= 2")

    non_target_dims = tuple(range(ndim - 2))
    reduce_dims = (ndim - 2, ndim - 1)

    input_arranged = input.permute(non_target_dims + reduce_dims)
    input_arranged = input_arranged.flatten(start_dim=-2)

    inner_block_shape = tuple(1 for _ in non_target_dims) + (block_size,)
    outer_block_shape = tuple(1 for _ in non_target_dims) + (-1,)

    input_arranged = input_arranged.tile(inner_block_shape)
    input_arranged = input_arranged.tile(outer_block_shape)
    input_arranged.dtype = input_arranged.dtype.squeeze(tuple(range(len(non_target_dims))))
    input_arranged.dtype.dtype = input_arranged.dtype.dtype.squeeze(
        tuple(range(len(non_target_dims)))
    )

    output_arranged = output.permute(non_target_dims + reduce_dims)
    output_arranged = output_arranged.flatten(start_dim=-2)
    output_arranged = output_arranged.tile(tuple(1 for _ in non_target_dims) + (1,))
    output_arranged.dtype = output_arranged.dtype.squeeze(tuple(range(len(non_target_dims))))

    return input_arranged, output_arranged


def application(input, output):
    acc = ntl.cast(0, ntl.float32)

    for i in range(input.shape[0]):
        acc += ntl.sum(ntl.cast(input[i], ntl.float32))

    output_dtype = output.dtype.dtype
    is_positive = acc > ntl.cast(0, ntl.float32)
    output[0] = ntl.where(
        is_positive, ntl.cast(1, output_dtype), ntl.cast(0, output_dtype)
    )


def premake(ndim, dtype=None, block_size=None):
    if ndim < 2:
        raise ValueError("nonzero_sum_gt_last2 requires ndim >= 2")

    arrangement_ = functools.partial(arrangement, block_size=block_size)

    input = Tensor(ndim, dtype=dtype, other=0)
    output = Tensor(ndim, dtype=dtype)
    output.shape = input.shape[:-2] + (1, 1)

    tensors = (input, output)

    return arrangement_, application, tensors
