import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def application(input, output, eps):
    # Clip input to [eps, 1-eps] to avoid log(0) and division by zero
    clipped = ntl.minimum(ntl.maximum(input, eps), 1.0 - eps)
    # Compute logit in float32 for numerical stability, then cast to output dtype
    x = ntl.cast(clipped, ntl.float32)
    output = ntl.cast(ntl.log(x / (1.0 - x)), output.dtype)  # noqa: F841


def premake(ndim, eps=1e-6, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
        Tensor(0, constexpr=True, value=eps),
    )

    return arrangement_, application, tensors
