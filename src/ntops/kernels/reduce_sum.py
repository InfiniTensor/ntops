import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


def arrangement(input, result, _scale, block_size=None):
    """Tile input for parallel reduction into a scalar result.

    input:  1D tensor of values to sum.
    result: scalar tensor (0-d), passed through for .data_ptr().
    _scale: constexpr scale factor passed through.
    """
    if block_size is None:
        block_size = ninetoothed.block_size()

    input_arranged = input.flatten().tile((block_size,))
    return input_arranged, result, _scale


def application(input, result, scale):
    """Atomic-add input * scale into result scalar.

    Each block processes a segment of input and atomically adds
    its sum to the shared scalar result.
    """
    out_ptr = result.data_ptr()
    valid = ntl.cast(input >= input * ntl.cast(0, ntl.float32), ntl.int1)  # always true mask
    # Sum the block and scale, then atomic-add to result
    block_sum = ntl.sum(input) * scale
    ntl.atomic_add(out_ptr, block_sum, mask=ntl.cast(1, ntl.int1))


def premake(dtype=None, block_size=None):
    """Create kernel factory for scalar reduction via atomic_add.

    Fixed block_size=128 and max_num_configs=1 required to prevent
    autotuning warmup from corrupting the atomic output.
    """
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    shape_options = ({"upper_bound": 2**15},)

    tensors = (
        Tensor(1, dtype=dtype, shape_options=shape_options, other=0),
        Tensor(1, dtype=dtype, shape_options=({'upper_bound': 1},)),
        Tensor(0, dtype=ninetoothed.float64, constexpr=True, value=1.0),
    )

    return arrangement_, application, tensors
