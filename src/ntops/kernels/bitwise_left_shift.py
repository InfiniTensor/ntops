import functools

from ninetoothed import Tensor
import ninetoothed.language as ntl

from ntops.kernels.element_wise import arrangement


def application(input, other, output):
    if input.dtype == ntl.int32:
        mask = (other > 31) | (other < 0)
    elif input.dtype == ntl.int64:
        mask = (other > 63) | (other < 0)
    elif input.dtype == ntl.uint8:
        mask = (other > 7) | (other < 0)
    else:
        mask = ntl.zeros_like(other, dtype=ntl.bool)

    shift = ntl.where(mask, ntl.zeros_like(other), other)
    input = ntl.where(mask, ntl.zeros_like(input), input)
    output = input << shift


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
    )

    return arrangement_, application, tensors
