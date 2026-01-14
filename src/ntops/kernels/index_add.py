import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.reduction import arrangement as reduction_arrangement


def arrangement(input, index, source, alpha, output, dim, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    if isinstance(dim, Tensor):
        dim = dim.value

    if dim < 0:
        dim += input.ndim

    index_expanded = index
    for _ in range(input.ndim - 1):
        index_expanded = index_expanded.unsqueeze(0)

    if dim != input.ndim - 1:
        permute_order = list(range(index_expanded.ndim))
        last = permute_order.pop(-1)
        permute_order.insert(dim, last)
        index_expanded = index_expanded.permute(tuple(permute_order))

    expand_shape = list(source.shape)
    expand_shape[dim] = -1
    index_expanded = index_expanded.expand(tuple(expand_shape))

    input_arranged, index_arranged, source_arranged, output_arranged = (
        reduction_arrangement(
            input, index_expanded, source, output, dim=dim, block_size=block_size
        )
    )

    return input_arranged, index_arranged, source_arranged, alpha, output_arranged


def _application_dim0(input, index, source, alpha, output):
    index_dtype = ntl.int64
    output_dtype = output.dtype.dtype
    alpha_cast = ntl.cast(alpha, output_dtype)

    zero_index = ntl.cast(0, index_dtype)
    zero_out = ntl.cast(0, output_dtype)
    dim_size = ntl.cast(output.source.shape[0], index_dtype)

    for out_block in range(output.shape[0]):
        out_vals = ntl.cast(input[out_block], output_dtype)
        out_positions = ntl.cast(output[out_block].offsets(0), index_dtype)
        valid_out = (out_positions >= zero_index) & (out_positions < dim_size)

        for src_block in range(source.shape[0]):
            idx_block = ntl.cast(index[src_block], index_dtype)
            src_vals = ntl.cast(source[src_block], output_dtype)
            matches = out_positions[:, None] == idx_block[None, :]
            contrib = ntl.sum(ntl.where(matches, src_vals[None, :], zero_out), 1)
            out_vals += alpha_cast * contrib

        output[out_block] = ntl.where(valid_out, out_vals, zero_out)


def premake(ndim, dim, dtype=None, block_size=None):
    if dim != 0:
        raise ValueError("Only dim=0 is supported for index_add.")

    arrangement_ = functools.partial(arrangement, dim=0, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype, other=0),
        Tensor(1, dtype=ninetoothed.int64, other=-1),
        Tensor(ndim, dtype=dtype, other=0),
        Tensor(0, dtype=dtype),
        Tensor(ndim, dtype=dtype),
    )

    return arrangement_, _application_dim0, tensors
