import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def application(input, other, output):
    output = ntl.inline_asm_elementwise(  # noqa: F841
        "or.b32 $0, $1, $2;",
        "=r,r,r",
        args=[input, other],
        dtype=input.dtype,
        is_pure=True,
        pack=1,
    )


@functools.cache
def make(ndim):
    tensors = (Tensor(ndim), Tensor(ndim), Tensor(ndim))

    return ninetoothed.make(arrangement, application, tensors)
