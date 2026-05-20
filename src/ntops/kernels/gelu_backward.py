import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def application(grad_output, input, grad_input):
    input_f32 = ntl.cast(input, ntl.float32)
    cdf = 0.5 * (1.0 + ntl.erf(input_f32 * 0.7071067811865476))
    pdf = ntl.exp(-0.5 * input_f32 * input_f32) * 0.3989422804014327
    grad_input = grad_output * ntl.cast(cdf + input_f32 * pdf, grad_output.dtype)  # noqa: F841


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
    )

    return arrangement_, application, tensors
