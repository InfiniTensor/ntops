import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def application(input, p, seed, a, b, output):
    alpha_prime = -1.7580993408473766

    keep = ntl.rand(seed, input.offsets()) > p

    dropped = ntl.where(
        keep,
        input,
        alpha_prime,
    )

    output = dropped * a + b  # noqa: F841


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype),                  # input
        Tensor(0, dtype=ninetoothed.float64),       # p
        Tensor(0, dtype=ninetoothed.int64),         # seed
        Tensor(0, dtype=ninetoothed.float64),       # a
        Tensor(0, dtype=ninetoothed.float64),       # b
        Tensor(ndim, dtype=dtype),                  # output
    )

    return arrangement_, application, tensors