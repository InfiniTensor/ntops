import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor


def arrangement(
    input, boundaries, output, right, bound_len, padded_len=None, block_size=None
):
    input_arranged = input.flatten().tile((block_size,))
    output_arranged = output.flatten().tile((block_size,))

    bound_arranged = boundaries.flatten().tile((padded_len,))
    bound_arranged = bound_arranged.expand((input_arranged.shape[0],))

    return input_arranged, bound_arranged, output_arranged, right, bound_len, padded_len


def application(input, boundaries, output, right, bound_len):
    val_in = ntl.cast(input, ntl.float32)
    val_bound = ntl.cast(boundaries, ntl.float32)

    bound_idx = ntl.arange(0, boundaries.shape[0])
    valid_mask = bound_idx < bound_len

    in_bc = ntl.expand_dims(val_in, 1)
    bound_bc = ntl.expand_dims(val_bound, 0)
    mask_bc = ntl.expand_dims(valid_mask, 0)

    if right:
        # count(b <= x)
        cond = bound_bc <= in_bc
    else:
        # count(b < x)
        cond = bound_bc < in_bc

    final_cond = cond & mask_bc

    bucket_idx = ntl.sum(ntl.cast(final_cond, ntl.int32), 1)

    output = ntl.cast(bucket_idx, output.dtype)


def premake(ndim, dtype=None, padded_len=None, block_size=None):
    arrangement_ = functools.partial(
        arrangement, padded_len=padded_len, block_size=block_size
    )

    tensors = (
        Tensor(ndim, dtype=dtype, shape_options={"constexpr": True}),  # input
        Tensor(1, dtype=dtype, shape_options={"constexpr": True}),  # boundaries
        Tensor(ndim, dtype=dtype, shape_options={"constexpr": True}),  # output
        Tensor(0, dtype=dtype),
        Tensor(0, dtype=dtype),
    )

    return arrangement_, application, tensors
