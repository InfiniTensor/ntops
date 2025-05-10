import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def application(input, output):
    exp_input = ntl.exp(input)
    exp_neg_input = ntl.exp(-input)
    output = (exp_input - exp_neg_input) / (exp_input + exp_neg_input)  # noqa: F841


@functools.cache
def make(ndim):
    return ninetoothed.make(arrangement, application, (Tensor(ndim), Tensor(ndim)))
