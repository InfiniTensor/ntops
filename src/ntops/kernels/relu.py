import functools

import ninetoothed
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def application(input, output):
    output = max(0.0, input)  # noqa: F841


@functools.cache
def make(ndim):
    return ninetoothed.make(arrangement, application,(Tensor(ndim), Tensor(ndim)))
