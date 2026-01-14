import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


def _next_power_of_2(value):
    if value < 1:
        raise ValueError("`value` must be positive.")
    return 1 << (value - 1).bit_length()


def arrangement(input, output, class_block_size, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    input_flat = input.flatten()
    input_flat = input_flat.unsqueeze(1)

    output_flat = output.flatten(end_dim=-1)

    input_arranged = input_flat.tile((block_size, 1))
    output_arranged = output_flat.tile((block_size, class_block_size))

    return input_arranged, output_arranged


def application(input, output):
    index_dtype = ntl.int64
    output_dtype = output.dtype

    input_values = ntl.cast(input, index_dtype)
    class_indices = ntl.cast(output.offsets(-1), index_dtype)

    output = ntl.where(
        input_values == class_indices,
        ntl.cast(1, output_dtype),
        ntl.cast(0, output_dtype),
    )


def premake(ndim, num_classes, block_size=None):
    class_block_size = _next_power_of_2(num_classes)
    arrangement_ = functools.partial(
        arrangement, block_size=block_size, class_block_size=class_block_size
    )

    input = Tensor(ndim, dtype=ninetoothed.int64, other=-1)
    output = Tensor(ndim + 1, dtype=ninetoothed.int64)

    output.shape = input.shape + (num_classes,)

    tensors = (input, output)

    return arrangement_, application, tensors
