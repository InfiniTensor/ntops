import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def application(input, output):
    input_f32 = ntl.cast(input, ntl.float32)
    max_finite = ntl.cast(3.4028234663852886e+38, ntl.float32)
    min_finite = ntl.cast(-3.4028234663852886e+38, ntl.float32)

    is_nan = input_f32 != input_f32
    is_posinf = input_f32 > max_finite
    is_neginf = input_f32 < min_finite

    result = ntl.where(is_nan, ntl.cast(0.0, ntl.float32), input_f32)
    result = ntl.where(is_posinf, max_finite, result)
    result = ntl.where(is_neginf, min_finite, result)

    output = result  # noqa: F841


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (Tensor(ndim, dtype=dtype), Tensor(ndim, dtype=dtype))

    return arrangement_, application, tensors
