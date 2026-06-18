import torch

import ninetoothed
import ntops
from ntops.torch.utils import _cached_make

_DTYPE_MAP = {
    torch.float16: ninetoothed.float16,
    torch.bfloat16: ninetoothed.bfloat16,
    torch.float32: ninetoothed.float32,
    torch.float64: ninetoothed.float64,
    torch.int8: ninetoothed.int8,
    torch.int16: ninetoothed.int16,
    torch.int32: ninetoothed.int32,
    torch.int64: ninetoothed.int64,
}


def unbind(input, dim=0):
    if dim < 0:
        dim = input.ndim + dim

    # movedim is a zero-cost view: (d0,..,dim,..,dk) → (dim_size, d0,..,dk).
    # After this, every "slice" is simply moved[i], and the copy problem
    # reduces to a single contiguous-output kernel regardless of which dim
    # was originally requested.
    moved = input.movedim(dim, 0)
    n_slices = moved.shape[0]

    # Fast path: moved is already contiguous (happens when dim=0 and input is
    # contiguous).  Return views directly — zero kernel launches.
    if moved.is_contiguous():
        return tuple(moved[i] for i in range(n_slices))

    # General path: ONE kernel launch copies the entire non-contiguous `moved`
    # into a contiguous output buffer.  Previously this was n_slices separate
    # launches (one per slice), each suffering its own launch overhead.
    output = torch.empty_like(moved, memory_format=torch.contiguous_format)
    kernel = _cached_make(
        ntops.kernels.unbind.premake,
        moved.ndim,
        dtype=_DTYPE_MAP.get(input.dtype),
    )
    kernel(moved, output)

    # output[i] is a contiguous view into the output buffer.
    return tuple(output[i] for i in range(n_slices))
