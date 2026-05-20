import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def application(input, output):
    input_f32 = ntl.cast(input, ntl.float32)
    exp_input = ntl.exp(input_f32)
    exp_neg_input = ntl.exp(-input_f32)
    tanh_val = (exp_input - exp_neg_input) / (exp_input + exp_neg_input)
    output = ntl.cast(input_f32 - tanh_val, input.dtype)  # noqa: F841


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (Tensor(ndim, dtype=dtype), Tensor(ndim, dtype=dtype))

    return arrangement_, application, tensors
