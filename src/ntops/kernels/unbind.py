import functools

from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


# Plain element-wise copy: input → output.
# The torch wrapper calls this with `moved = input.movedim(dim, 0)` as the
# source and a fresh contiguous tensor as the destination.  By the time this
# kernel runs, the "unbind axis" has already been moved to dim-0 via a
# zero-cost view, so a single kernel invocation copies all slices in parallel
# instead of launching one kernel per slice.
def application(input, output):
    output = input  # noqa: F841


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)
    tensors = (Tensor(ndim, dtype=dtype), Tensor(ndim, dtype=dtype))
    return arrangement_, application, tensors
