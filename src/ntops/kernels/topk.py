import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.reduction import arrangement


def application(input, values, indices, k, largest):
    val_block = input[0]

    idx_block = ntl.arange(0, val_block.shape[0])

    res_vals = ntl.zeros(val_block.shape, dtype=val_block.dtype)
    res_idxs = ntl.zeros(val_block.shape, dtype=indices.dtype.dtype)
    output_range = ntl.arange(0, val_block.shape[0])

    if largest:
        working_val = val_block
    else:
        working_val = -val_block

    sentinel = float("-inf")

    for i in range(k):
        current_max_val = ntl.max(working_val, axis=0)
        current_max_idx = ntl.argmax(working_val, axis=0)

        real_val = -current_max_val if not largest else current_max_val
        real_val = ntl.cast(real_val, res_vals.dtype)

        target_mask = output_range == i
        res_vals = ntl.where(target_mask, real_val, res_vals)
        res_idxs = ntl.where(target_mask, current_max_idx, res_idxs)

        mask_selected = idx_block == current_max_idx
        updated_working_val = ntl.where(mask_selected, sentinel, working_val)
        working_val = ntl.cast(updated_working_val, working_val.dtype)

    values[0] = res_vals
    indices[0] = res_idxs


def premake(
    ndim, dim, k, largest, sorted=True, dtype=None, indices_dtype=None, block_size=None
):
    arrangement_ = functools.partial(arrangement, dim=dim, block_size=block_size)

    pad_val = float("-inf") if largest else float("inf")

    tensors = (
        Tensor(ndim, dtype=dtype, other=pad_val),
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=indices_dtype),
        Tensor(0, constexpr=True, value=k),
        Tensor(0, constexpr=True, value=largest),
    )

    return arrangement_, application, tensors


premake_all_elements = premake
