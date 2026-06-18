import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor
from ninetoothed.language import libdevice

from ntops.kernels.element_wise import arrangement


# libdevice.lgamma supports float32 and float64 only.
#
# Integer types: handled entirely in the torch wrapper by pre-converting
#   the input tensor to float32 (torch.Tensor.to), then running the
#   float32 kernel. This reuses one kernel instead of compiling four
#   identical int→float32→lgamma kernels (one per int dtype).
#
# float16 / bfloat16: promote to float32 for the lgamma call, downcast result.
#
# float32 / float64: call libdevice.lgamma directly.


def application_narrow(input, output):
    promoted = ntl.cast(input, ntl.float32)
    output = ntl.cast(libdevice.lgamma(promoted), output.dtype)  # noqa: F841


def application_wide(input, output):
    output = libdevice.lgamma(input)  # noqa: F841


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    if dtype in (ninetoothed.float16, ninetoothed.bfloat16):
        application = application_narrow
    else:
        application = application_wide

    tensors = (Tensor(ndim, dtype=dtype), Tensor(ndim, dtype=dtype))

    return arrangement_, application, tensors
