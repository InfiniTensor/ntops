"""combinations gather kernel: output[m] = input[index[m]].

combinations is a combinatorial gather. The wrapper enumerates the flat gather
indices on the host (cheaply, e.g. triu_indices for r=2 — avoiding torch's
n**r meshgrid), and this kernel does the data-dependent gather via
``ntl.load(input.data_ptr() + index)`` (the read-side of scatter_add's atomic
trick). Affine addressing throughout; the index value comparison is allowed.
"""

import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


def arrangement(index, output, input, m, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    index_arranged = index.flatten().tile((block_size,))
    output_arranged = output.flatten().tile((block_size,))

    # index first → grid; input un-arranged source for data_ptr().
    return index_arranged, output_arranged, input, m


def application(index, output, input, m):
    p = index.offsets()
    mask = p < m

    output = ntl.load(input.data_ptr() + index, mask=mask, other=0)  # noqa: F841


def premake(block_size=None):
    tensors = (
        Tensor(1, dtype=int),  # index (M,) flat gather indices
        Tensor(1),  # output (M,)
        Tensor(1, shape_options={"constexpr": True}),  # input flat (source)
        Tensor(0, dtype=int, constexpr=True),  # m
    )

    arrangement_ = functools.partial(arrangement, block_size=block_size)

    return arrangement_, application, tensors
