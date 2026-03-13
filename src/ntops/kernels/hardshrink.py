import enum
import functools
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def application(input, lambd, output):
    output = ntl.where(ntl.abs(input) > lambd, input, 0.0)


def hardshrink_premake(
    ndim,
    dtype=None,
    block_size=None,
):
    arrangement_ = functools.partial(
        arrangement,
        block_size=block_size,
    )

    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(0, dtype=ninetoothed.float32),
        Tensor(ndim, dtype=dtype),
    )

    return arrangement_, application, tensors
