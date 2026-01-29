import enum
import functools
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def application_default(input, output):
    output = ntl.where(input > -3 and input < 3, input / 6 + 0.5, input)
    output = ntl.where(output <= -3, 0, output)
    output = ntl.where(output >= 3, 1, output)


def application_inplace(input):
    input = ntl.where(input > -3 and input < 3, input / 6 + 0.5, input)
    input = ntl.where(input <= -3, 0, input)
    input = ntl.where(input >= 3, 1, input)


def premake(
    ndim,
    inplace=False,
    dtype=None,
    block_size=None,
):
    arrangement_ = functools.partial(
        arrangement,
        block_size=block_size,
    )

    if inplace:
        tensors = (Tensor(ndim, dtype=dtype),)
    else:
        tensors = (
            Tensor(ndim, dtype=dtype),
            Tensor(ndim, dtype=dtype),
        )

    application = application_inplace if inplace else application_default

    return arrangement_, application, tensors
