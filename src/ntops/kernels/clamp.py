import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def application(input, min_val, max_val, output):
    output = ntl.clamp(input, min_val, max_val)  # noqa: F841


@functools.cache
def make(ndim):
    tensors = (Tensor(ndim), Tensor(ndim), Tensor(ndim), Tensor(ndim))

    return ninetoothed.make(arrangement, application, tensors)
