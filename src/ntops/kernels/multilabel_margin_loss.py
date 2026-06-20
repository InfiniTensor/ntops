import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


def arrangement(input, target, output, _C, BLOCK_C=None):
    """One program per batch row.  Class dim tiled by BLOCK_C.

    input:  (N, C)  → tiled to (N, ceil(C/BLOCK_C)) outer, (1, BLOCK_C) inner
    target: (N, C)  → same
    output: (N,)    → one scalar per batch row
    _C:     constexpr, ignored in arrangement
    """
    if BLOCK_C is None:
        BLOCK_C = ninetoothed.block_size()

    input_arranged = input.tile((1, BLOCK_C))
    target_arranged = target.tile((1, BLOCK_C))
    output_arranged = output.tile((1,))
    output_arranged = output_arranged.ravel()

    return input_arranged, target_arranged, output_arranged, _C


def application(input, target, output, C):
    """Compute per-sample multilabel margin loss.

    Uses input.offsets(1) to obtain a class_id vector of length BLOCK_C
    without needing ntl.arange or ntl.full.  The O(C^2) design:
      1. First pass: walk the target prefix, build positive_mask.
      2. Second pass: for each valid target entry, accumulate margin
         against all negative classes.
    """
    class_id = input.offsets(1)                     # vector: [0, 1, ..., BLOCK_C-1]
    class_valid = ntl.cast(class_id < C, ntl.int1)  # True for 0..C-1

    zero_vec = input * ntl.cast(0, ntl.float32)
    zero = ntl.sum(zero_vec)

    # --- Pass 1: build positive_mask from target prefix ---
    positive_mask = class_id < ntl.cast(0, ntl.int32)  # all-False vector
    still_valid = ntl.cast(1, ntl.int1)

    for j in range(C):
        # Extract target[b, j] without dynamic indexing:
        # t_j = sum over class_id dimension: if class_id == j, pick target, else 0
        t_j = ntl.sum(
            ntl.where(
                ntl.cast(class_id == j, ntl.int1),
                ntl.cast(target, ntl.int32),
                ntl.cast(target * ntl.cast(0, ntl.int32), ntl.int32),
            )
        )
        valid_j = still_valid & (t_j >= 0)
        positive_mask = positive_mask | (
            valid_j & ntl.cast(class_id == t_j, ntl.int1)
        )
        still_valid = still_valid & (t_j >= 0)

    # --- Pass 2: compute loss per valid target entry ---
    loss = zero

    still_valid_2 = ntl.cast(1, ntl.int1)

    for j in range(C):
        t_j = ntl.sum(
            ntl.where(
                ntl.cast(class_id == j, ntl.int1),
                ntl.cast(target, ntl.int32),
                ntl.cast(target * ntl.cast(0, ntl.int32), ntl.int32),
            )
        )
        valid_j = still_valid_2 & (t_j >= 0)

        # pos_value: gather input at class t_j
        pos_value = ntl.sum(
            ntl.where(
                class_valid & ntl.cast(class_id == t_j, ntl.int1),
                input,
                zero_vec,
            )
        )

        # margin = max(0, 1 - pos_value + input[k]) for each negative class k
        margin = ntl.cast(1, ntl.float32) - pos_value + input
        negative = class_valid & ntl.cast(ntl.cast(positive_mask, ntl.int32) == ntl.cast(0, ntl.int32), ntl.int1)

        contribution = ntl.where(
            valid_j & negative & ntl.cast(margin > zero, ntl.int1),
            margin,
            zero_vec,
        )
        loss = loss + ntl.sum(contribution)

        still_valid_2 = still_valid_2 & (t_j >= 0)

    output = loss / ntl.cast(C, ntl.float32)  # noqa: F841


def premake(C, BLOCK_C=None, dtype=None):
    """Create kernel factory.

    Args:
        C: Python int — number of classes (concrete, avoids Symbol arithmetic).
        BLOCK_C: next power of 2 >= C.  If None, uses ninetoothed.block_size().
        dtype: element type (None = default float32).
    """
    if BLOCK_C is None:
        BLOCK_C = ninetoothed.block_size()

    arrangement_ = functools.partial(arrangement, BLOCK_C=BLOCK_C)

    # N (batch) dimension needs upper_bound <= max_num_elements for autotuning.
    # C dimension is concrete (Python int in shape) to avoid Symbol arithmetic.
    shape_opts = ({"upper_bound": 2**14}, {})

    tensors = (
        Tensor(2, shape=(None, C), dtype=dtype, shape_options=shape_opts, other=0),
        Tensor(2, shape=(None, C), dtype=ninetoothed.int64, shape_options=shape_opts, other=-1),
        Tensor(1, dtype=dtype, shape_options=({"upper_bound": 2**14},)),
        Tensor(0, constexpr=True, value=C),
    )

    return arrangement_, application, tensors
