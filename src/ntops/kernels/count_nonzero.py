import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

# ---------------------------------------------------------------------------
# count_nonzero = sum(input != 0), reduced either globally (dim=None) or over a
# set of dims. Both paths are single-pass partial-sum kernels: each program
# counts the nonzeros in one block and writes an int64 partial; the host sums
# the partials. int64 (not float32) partials keep the count exact for large
# inputs. ``other=0`` pads the trailing block -- 0 is counted as zero, so
# padding never inflates the count.
# ---------------------------------------------------------------------------


def _global_arrangement(input, output, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    input_arranged = input.flatten().tile((block_size,))
    output_arranged = output.flatten().tile((1,))

    return input_arranged, output_arranged


def global_application(input, output):
    output = ntl.sum(ntl.where(input != 0, 1, 0))  # noqa: F841


def global_premake(input_dtype=None, block_size=None):
    arrangement_ = functools.partial(_global_arrangement, block_size=block_size)

    tensors = (
        Tensor(1, other=0, dtype=input_dtype),
        Tensor(1, dtype=ninetoothed.int64),
    )

    return arrangement_, global_application, tensors


# Dim path: the host reshapes to (M, N) with the reduced dims trailing. Each
# (1, block_size) tile becomes one program writing a partial into the
# (M, num_blocks) buffer; the host then sums along the blocks per row.
def _dim_arrangement(input, output, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    input_arranged = input.tile((1, block_size))
    output_arranged = output.tile((1, 1))

    return input_arranged, output_arranged


def dim_application(input, output):
    output = ntl.sum(ntl.where(input != 0, 1, 0))  # noqa: F841


def dim_premake(input_dtype=None, block_size=None):
    arrangement_ = functools.partial(_dim_arrangement, block_size=block_size)

    tensors = (
        Tensor(2, other=0, dtype=input_dtype),
        Tensor(2, dtype=ninetoothed.int64),
    )

    return arrangement_, dim_application, tensors


# Leading path: reduce a contiguous block of *leading* dims, viewed host-side as
# ``(R, inner)`` with ``inner`` the contiguous trailing dims. Reducing axis 0
# directly (instead of permuting it to the back, which would materialize a
# transpose) is done with a ``(reduce_block, block_size)`` tile: the ``block_size``
# columns are read coalesced while ``ntl.sum(..., axis=0)`` reduces the rows. One
# partial per ``(row-block, column)`` is written; the host sums the row-blocks.
# This mirrors avg_pool2d's ``ntl.sum(axis=-1)`` + output squeeze, transposed to
# axis 0. ``other=0`` pads both ragged edges and is counted as zero.
def _leading_arrangement(input, output, block_size=None, reduce_block=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    if reduce_block is None:
        reduce_block = 32

    input_arranged = input.tile((reduce_block, block_size))

    output_arranged = output.tile((1, block_size))
    output_arranged.dtype = output_arranged.dtype.squeeze(0)

    return input_arranged, output_arranged


def leading_application(input, output):
    output = ntl.sum(ntl.where(input != 0, 1, 0), axis=0)  # noqa: F841


def leading_premake(input_dtype=None, block_size=None, reduce_block=None):
    arrangement_ = functools.partial(
        _leading_arrangement, block_size=block_size, reduce_block=reduce_block
    )

    tensors = (
        Tensor(2, other=0, dtype=input_dtype),
        Tensor(2, dtype=ninetoothed.int64),
    )

    return arrangement_, leading_application, tensors
