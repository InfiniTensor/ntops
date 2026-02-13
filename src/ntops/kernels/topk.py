import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.reduction import arrangement as reduction_arrangement


def _next_power_of_2(value):
    if value < 1:
        raise ValueError("`value` must be positive.")
    return 1 << (value - 1).bit_length()


def arrangement(
    input,
    dim_size,
    k_constexpr,
    values,
    indices,
    dim,
    block_size=None,
    output_block_size=None,
):
    if block_size is None:
        block_size = ninetoothed.block_size()

    if dim != -1:
        raise ValueError("Only dim=-1 is supported for topk.")

    dim = input.ndim - 1

    input_arranged = reduction_arrangement(input, dim=dim, block_size=block_size)[0]

    if output_block_size is None:
        output_block_size = values.shape[dim]
    values_arranged = reduction_arrangement(
        values, dim=dim, block_size=output_block_size
    )[0]
    indices_arranged = reduction_arrangement(
        indices, dim=dim, block_size=output_block_size
    )[0]

    return input_arranged, dim_size, k_constexpr, values_arranged, indices_arranged


def _application_last(input, dim_size, k_constexpr, values, indices):
    value_dtype = values.dtype.dtype
    index_dtype = ntl.int64

    dim_size_ = ntl.cast(dim_size, index_dtype)

    k = k_constexpr

    neg_inf = ntl.cast(float("-inf"), value_dtype)
    neg_one = ntl.cast(-1, index_dtype)

    top_vals = ntl.full(values.dtype.shape, float("-inf"), dtype=value_dtype)
    top_indices = ntl.full(indices.dtype.shape, -1, dtype=index_dtype)
    positions = ntl.cast(values[0].offsets(-1), index_dtype)

    for t in range(k):
        best_val = neg_inf
        best_idx = neg_one

        for block in range(input.shape[0]):
            block_vals = input[block]
            block_indices = ntl.cast(input[block].offsets(-1), index_dtype)

            valid = block_indices < dim_size_
            for prev in range(t):
                prev_idx = ntl.max(ntl.where(positions == prev, top_indices, neg_one))
                valid = valid & (block_indices != prev_idx)

            masked_vals = ntl.where(valid, block_vals, neg_inf)
            block_best_val = ntl.cast(ntl.max(masked_vals), value_dtype)
            block_best_idx = ntl.max(
                ntl.where(valid & (masked_vals == block_best_val), block_indices, neg_one)
            )

            better = block_best_val > best_val
            best_val = ntl.where(better, block_best_val, best_val)
            best_idx = ntl.where(better, block_best_idx, best_idx)

        write_mask = positions == t
        top_vals = ntl.where(write_mask, best_val, top_vals)
        top_indices = ntl.where(write_mask, best_idx, top_indices)

    values[0] = top_vals
    indices[0] = top_indices


def premake(ndim, dim, k, dtype=None, block_size=None):
    if dim != -1:
        raise ValueError("Only dim=-1 is supported for topk.")

    input = Tensor(ndim, dtype=dtype, other=float("-inf"))
    dim_size = Tensor(0, dtype=ninetoothed.int64)
    k_constexpr = Tensor(0, dtype=ninetoothed.int64, constexpr=True, value=k)
    values = Tensor(ndim, dtype=dtype)
    indices = Tensor(ndim, dtype=ninetoothed.int64)

    dim = ndim - 1

    output_block_size = _next_power_of_2(k)
    arrangement_ = functools.partial(
        arrangement,
        dim=-1,
        block_size=block_size,
        output_block_size=output_block_size,
    )

    values.shape = values.shape[:dim] + (k,) + values.shape[dim + 1 :]
    indices.shape = indices.shape[:dim] + (k,) + indices.shape[dim + 1 :]

    tensors = (input, dim_size, k_constexpr, values, indices)

    return arrangement_, _application_last, tensors
