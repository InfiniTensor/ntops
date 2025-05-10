import functools

import ninetoothed
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def application(input, output):
    pos_result = input == float("+inf")
    neg_result = input == float("-inf")
    output = pos_result or neg_result  # noqa: F841


@functools.cache
def make(ndim):
    return ninetoothed.make(arrangement, application, (Tensor(ndim), Tensor(ndim)))
