import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def application(input, output, eps):
    one = ntl.cast(1, ntl.float32)
    eps_f32 = ntl.cast(eps, ntl.float32)
    input_f32 = ntl.cast(input, ntl.float32)

    clamped = ntl.clamp(input_f32, eps_f32, one - eps_f32)

    output = ntl.log(clamped / (one - clamped))  # noqa: F841


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
    )

    return arrangement_, application, tensors
