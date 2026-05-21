import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement as _element_wise_arrangement


def application(input, target, output):
    diff = input - target
    output = diff * diff  # noqa: F841


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(_element_wise_arrangement, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
    )

    return arrangement_, application, tensors


def _reduce_arrangement(input, target, output, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    input_arranged = input.flatten().tile((block_size,))
    target_arranged = target.flatten().tile((block_size,))
    output_arranged = output.flatten().tile((1,))

    return input_arranged, target_arranged, output_arranged


def reduce_application(input, target, output):
    diff = ntl.cast(input, ntl.float32) - ntl.cast(target, ntl.float32)
    output = ntl.sum(diff * diff)  # noqa: F841


def reduce_premake(input_dtype=None, block_size=None):
    arrangement_ = functools.partial(_reduce_arrangement, block_size=block_size)

    tensors = (
        Tensor(1, other=0, dtype=input_dtype),
        Tensor(1, other=0, dtype=input_dtype),
        Tensor(1, dtype=ninetoothed.float32),
    )

    return arrangement_, reduce_application, tensors
