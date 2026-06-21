import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def application(input, noise, scale, output):
    output = ntl.where(noise > 0, input * scale, -1.7580993408473766)  # noqa: F841


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=ninetoothed.float32),  # bernoulli noise mask
        Tensor(0, dtype=ninetoothed.float64),     # scale = 1 / (1 - p)
        Tensor(ndim, dtype=dtype),
    )

    return arrangement_, application, tensors
