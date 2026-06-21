"""Column stack kernel: identity copy into sliced output region.

Arrangement: flatten+tile both input and output.
Application: identity — the column offset is handled by output's data pointer
(given by the sliced view from the wrapper).
"""

import functools

import ninetoothed
from ninetoothed import Tensor


def arrangement(*tensors, block_size=None):
    """Flatten + tile for data tensors (skip 0-dim constexpr)."""
    if block_size is None:
        block_size = ninetoothed.block_size()
    return tuple(
        t.flatten().tile((block_size,)) if t.ndim != 0 else t
        for t in tensors
    )


def application(input, output):
    """Identity copy — output already points to correct column range."""
    output = input  # noqa: F841


def premake(ndim, dtype=None, block_size=None):
    """Create column_stack kernel for one input.

    Args:
        ndim: ndim of both input and output (2 for column_stack).
    """
    arrangement_fn = functools.partial(arrangement, block_size=block_size)
    tensors = (Tensor(ndim, dtype=dtype), Tensor(ndim, dtype=dtype))
    return arrangement_fn, application, tensors
