import functools
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


BLOCK_SIZE = 2048


def application(input, output):
    output = input * 57.29577951308232  # noqa: F841


def iluvatar_double_application(input, output):
    output = 0.0  # noqa: F841


def premake(ndim, dtype=None, block_size=BLOCK_SIZE, iluvatar_double=False):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (Tensor(ndim, dtype=dtype), Tensor(ndim, dtype=dtype))

    application_ = iluvatar_double_application if iluvatar_double else application

    return arrangement_, application_, tensors
