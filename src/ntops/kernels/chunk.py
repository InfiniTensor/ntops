import functools

from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


# The slicing is done by the torch wrapper (input.narrow) before calling this
# kernel, so this kernel only needs to copy an already-sliced (but possibly
# non-contiguous) tensor into a contiguous output.  The arrangement and
# application are identical to a plain element-wise copy.
#
# Cache key is (premake, ndim, dtype) — shared across all chunks of the same
# tensor dtype and ndim, regardless of which dim or position is being chunked.
# Before this change the key included dim / chunk_start / chunk_size, causing
# one separate Triton compilation per chunk.


def application(input, output):
    output = input  # noqa: F841


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)
    tensors = (Tensor(ndim, dtype=dtype), Tensor(ndim, dtype=dtype))
    return arrangement_, application, tensors
