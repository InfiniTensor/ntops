"""Mode kernel: vectorized row comparison with ntl.sum and mask.

Arrangement: one program per row, K_tile elements per block.
Application: loops K_orig candidates, masks padding lanes.
"""

import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


def arrangement(*tensors, K_tile, block_size=None):
    """One program per row.  K_tile = next_power_of_2(K_orig)."""
    if block_size is None:
        block_size = ninetoothed.block_size()

    input_t, values_t, indices_t, K_orig_t, K_tile_t = tensors
    return (
        input_t.tile((1, K_tile)),
        # values_t.tile((1, 1)),
        # indices_t.tile((1, 1)),
        values_t.squeeze(1).tile((1, 1)),
        indices_t.squeeze(1).tile((1, 1)),
        K_orig_t,
        K_tile_t,
    )


def application(input, values, indices, K_orig, K_tile):
    """Compute mode value and index for one row.

    Padding lanes (offset >= K_orig) are masked out via
    `input.offsets(1) < K_orig`, so they never contribute to counts.
    """
    row = values.offsets(0)
    base = input.source.data_ptr() + row * input.source.stride(0)
    stride_k = input.source.stride(1)

    valid = ntl.cast(input.offsets(1) < K_orig, ntl.int32)

    best_value = ntl.load(base)
    best_count = ntl.cast(best_value * 0 - 1, ntl.int32)
    best_index = ntl.cast(best_value * 0, ntl.int32)

    for j in range(K_orig):                        # only real elements
        candidate = ntl.load(base + j * stride_k)

        matches = ntl.cast(input == candidate, ntl.int32) * valid
        count = ntl.sum(matches)

        better = (count > best_count) | (
            (count == best_count)
            & (
                (candidate < best_value)
                | ((candidate == best_value) & (j > best_index))
            )
        )

        best_count = ntl.where(better, count, best_count)
        best_value = ntl.where(better, candidate, best_value)
        best_index = ntl.where(better, j, best_index)

    values = best_value  # noqa: F841
    indices = ntl.cast(best_index, ntl.int64)  # noqa: F841


def premake(K_orig, K_tile, dtype=None, block_size=None):
    arrangement_fn = functools.partial(arrangement, K_tile=K_tile, block_size=block_size)
    tensors = (
        Tensor(2, dtype=dtype),                # input: (num_rows, K_tile)
        Tensor(2, dtype=dtype),                # values: (num_rows, 1)
        Tensor(2, dtype=ninetoothed.int64),    # indices: (num_rows, 1)
        Tensor(0, constexpr=True, value=K_orig),
        Tensor(0, constexpr=True, value=K_tile),
    )
    return arrangement_fn, application, tensors
