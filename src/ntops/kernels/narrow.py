import functools

from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


# narrow returns the slice ``input[..., start:start+length, ...]``. The torch
# wrapper builds that as a strided view (a regular ``offset + stride`` pattern,
# unlike a data-dependent gather), and this trivial copy kernel materializes it
# into a contiguous output -- the same "strided view + copy" approach as
# pixel_unshuffle.
def application(input, output):
    output = input  # noqa: F841


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (Tensor(ndim, dtype=dtype), Tensor(ndim, dtype=dtype))

    return arrangement_, application, tensors
