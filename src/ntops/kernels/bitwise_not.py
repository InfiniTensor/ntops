import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def application(input, output):
    output = ntl.inline_asm_elementwise(  # noqa: F841
        "not.b32 $0, $1;",
        "=r,r",
        args=[input],
        dtype=input.dtype,
        is_pure=True,
        pack=1,
    )


@functools.cache
def make(ndim):
    tensors = (Tensor(ndim), Tensor(ndim))

    return ninetoothed.make(arrangement, application, tensors)
