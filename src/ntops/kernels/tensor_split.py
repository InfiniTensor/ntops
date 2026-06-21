import ninetoothed
from ninetoothed import Tensor


def arrangement(input, output, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    assert input.ndim == output.ndim

    input = input.flatten().tile((block_size,))
    output = output.flatten().tile((block_size,))

    return input, output


def application(input, output):
    output = input


def premake(ndim):
    return (
        arrangement,
        application,
        (
            Tensor(ndim),
            Tensor(ndim),
        ),
    )