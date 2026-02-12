import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def _softplus(x):
    return ntl.log(ntl.exp(-ntl.abs(x)) + 1.0) + ntl.maximum(x, 0.0)


def _tanh(x):
    return (ntl.exp(2 * x) - 1) / (ntl.exp(2 * x) + 1)


def application(input, output):
    dtype = input.dtype
    if dtype == ntl.float16:
        mish_dtype = ntl.float32
    elif dtype == ntl.bfloat16:
        mish_dtype = ntl.float32
    else:
        mish_dtype = dtype

    input_f32 = ntl.cast(input, mish_dtype)
    output_softplus_f32 = _softplus(input_f32)
    output_f32 = _tanh(output_softplus_f32)
    output = ntl.cast(output_f32 * input_f32, dtype)


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (Tensor(ndim, dtype=dtype), Tensor(ndim, dtype=dtype))

    return arrangement_, application, tensors
