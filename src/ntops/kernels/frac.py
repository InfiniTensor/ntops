import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def application(input, output):
    # frac(x) = x - trunc(x)
    # trunc(x) = floor(x) for x >= 0, ceil(x) for x < 0
    # No tl.trunc available, so implement manually.
    truncated = ntl.where(input >= 0, ntl.floor(input), ntl.ceil(input))
    output = input - truncated  # noqa: F841


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (Tensor(ndim, dtype=dtype), Tensor(ndim, dtype=dtype))

    return arrangement_, application, tensors
