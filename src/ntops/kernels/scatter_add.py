import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


def arrangement(flat_index, src, output, block_size=None):
    """Arrange for scatter-add kernel.

    flat_index: 1D tensor of flat destination indices in output.
    src: 1D tensor of source values (same shape as flat_index).
    output: source tensor (passed through so .data_ptr() works).

    Each program handles a block of (flat_index, src) elements and
    atomically adds src values to the output at the given flat indices.
    """
    if block_size is None:
        block_size = ninetoothed.block_size()

    # Tile index and src for parallel processing.
    index_arranged = flat_index.flatten().tile((block_size,))
    src_arranged = src.flatten().tile((block_size,))

    # Pass output through unchanged so it remains a source tensor.
    # This is required for .data_ptr() to work in the application.
    return index_arranged, src_arranged, output


def application(flat_index, src, output):
    """Vectorized scatter-add: each lane does one atomic_add in parallel."""
    out_ptr = output.data_ptr()

    # src and output share the same dtype (ensured by the torch wrapper).
    # Filter out padding lanes where flat_index is negative (other=-1).
    valid = ntl.cast(flat_index >= 0, ntl.int1)

    ntl.atomic_add(out_ptr + flat_index, src, mask=valid)


def premake(dtype=None, block_size=None):
    """Create kernel factory for scatter_add.

    All tensors are 1D after the torch wrapper computes flat indices.

    Note: block_size must be a fixed integer (not a meta symbol) to
    prevent autotuning warmup from corrupting the atomic output.
    The torch wrapper passes block_size=128 and max_num_configs=1.
    """
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    # Use upper_bound=2**15 on the 1D tensors so autotuning can
    # reason about symbol bounds for the un-tiled output tensor.
    # upper_bound must be <= max_num_elements (typically 32768 on CUDA).
    shape_options = ({"upper_bound": 2**15},)

    tensors = (
        Tensor(1, dtype=ninetoothed.int64, shape_options=shape_options, other=-1),
        Tensor(1, dtype=dtype, shape_options=shape_options, other=0),
        Tensor(1, dtype=dtype, shape_options=shape_options),
    )

    return arrangement_, application, tensors
