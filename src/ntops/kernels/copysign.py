import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def application(input, other, output):
    # Pure bit manipulation: take magnitude bits of input, sign bit of other.
    # Avoids the fp16/bf16 -> fp32 -> fp16/bf16 round-trip required by
    # libdevice.copysign, which doesn't support narrow floats.
    dtype = output.dtype
    if dtype == ntl.float16 or dtype == ntl.bfloat16:
        int_dtype = ntl.int16
    elif dtype == ntl.float32:
        int_dtype = ntl.int32
    else:
        int_dtype = ntl.int64

    input_bits = ntl.cast(input, int_dtype, bitcast=True)
    other_bits = ntl.cast(other, int_dtype, bitcast=True)
    sign_bit = ntl.cast(1, int_dtype) << (ntl.cast(input, int_dtype, bitcast=True).dtype.primitive_bitwidth - 1)
    magn_mask = sign_bit - ntl.cast(1, int_dtype)
    output = ntl.cast(  # noqa: F841
        (input_bits & magn_mask) | (other_bits & sign_bit), dtype, bitcast=True
    )


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
    )

    return arrangement_, application, tensors
