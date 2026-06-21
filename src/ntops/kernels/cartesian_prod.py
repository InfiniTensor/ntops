"""Cartesian product kernel: ntl.load with manual index computation.

Output-driven: for each output row, compute the source index in the input
using integer division and modulo, then ntl.load.
"""

import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


def arrangement(*tensors, block_size=None):
    """Flatten + tile for data tensors (skip 0-dim constexpr)."""
    if block_size is None:
        block_size = ninetoothed.block_size()
    return tuple(
        t.flatten().tile((block_size,)) if t.ndim != 0 else t
        for t in tensors
    )


def application(input, output, repeat_after, size):
    """Compute one column of the cartesian product.

    input:  1D tensor, size = L_col
    output: 1D column vector, size = total_rows
    repeat_after: product of sizes of columns to the right
    size: L_col (the size of this input)

    source_index = (row // repeat_after) % size
    """
    row = output.offsets(0)
    src_idx = (row // repeat_after) % size
    ptr = input.source.data_ptr() + src_idx * input.source.stride(0)
    output = ntl.load(ptr)  # noqa: F841


def premake(repeat_after, size, dtype=None, block_size=None):
    """Create cartesian_prod kernel for one input column.

    Args:
        repeat_after: product(sizes[col+1:]) — how many rows before repeating.
        size: L_col — the length of this input tensor.
    """
    arrangement_fn = functools.partial(arrangement, block_size=block_size)
    tensors = (
        Tensor(1, dtype=dtype),
        Tensor(1, dtype=dtype),
        Tensor(0, constexpr=True, value=repeat_after),
        Tensor(0, constexpr=True, value=size),
    )
    return arrangement_fn, application, tensors
