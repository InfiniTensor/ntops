"""Roll kernel: manual pointer arithmetic with ntl.load.

Arrangement: flatten + tile for data tensors (skip 0-dim constexpr).
Application: ndim-specialized manual address computation with safe modulo.
"""

import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


def arrangement(*tensors, block_size=None):
    """Flatten + tile for data tensors, pass-through for 0-dim constexpr."""
    if block_size is None:
        block_size = ninetoothed.block_size()
    return tuple(
        t.flatten().tile((block_size,)) if t.ndim != 0 else t
        for t in tensors
    )


# --- NDIM-specialized application functions ---
# Use (idx - shift + size) % size to avoid negative modulo in Triton.

def application_1d(input, output, shift):
    ptr = input.source.data_ptr()
    size = input.source.shape[0]
    i0 = (output.offsets(0) - shift + size) % size
    ptr = ptr + i0 * input.source.stride(0)
    output = ntl.load(ptr)  # noqa: F841


def application_2d(input, output, shift):
    ptr = input.source.data_ptr()
    size0 = input.source.shape[0]
    i0 = (output.offsets(0) - shift + size0) % size0
    i1 = output.offsets(1)
    ptr = ptr + i0 * input.source.stride(0) + i1 * input.source.stride(1)
    output = ntl.load(ptr)  # noqa: F841


def application_3d(input, output, shift):
    ptr = input.source.data_ptr()
    size0 = input.source.shape[0]
    i0 = (output.offsets(0) - shift + size0) % size0
    i1 = output.offsets(1)
    i2 = output.offsets(2)
    ptr = ptr + i0 * input.source.stride(0) + i1 * input.source.stride(1) + i2 * input.source.stride(2)
    output = ntl.load(ptr)  # noqa: F841


def application_4d(input, output, shift):
    ptr = input.source.data_ptr()
    size0 = input.source.shape[0]
    i0 = (output.offsets(0) - shift + size0) % size0
    i1 = output.offsets(1)
    i2 = output.offsets(2)
    i3 = output.offsets(3)
    ptr = (ptr + i0 * input.source.stride(0) + i1 * input.source.stride(1)
           + i2 * input.source.stride(2) + i3 * input.source.stride(3))
    output = ntl.load(ptr)  # noqa: F841


def application_5d(input, output, shift):
    ptr = input.source.data_ptr()
    size0 = input.source.shape[0]
    i0 = (output.offsets(0) - shift + size0) % size0
    i1 = output.offsets(1)
    i2 = output.offsets(2)
    i3 = output.offsets(3)
    i4 = output.offsets(4)
    ptr = (ptr + i0 * input.source.stride(0) + i1 * input.source.stride(1)
           + i2 * input.source.stride(2) + i3 * input.source.stride(3)
           + i4 * input.source.stride(4))
    output = ntl.load(ptr)  # noqa: F841


_APPLICATIONS = {1: application_1d, 2: application_2d, 3: application_3d,
                 4: application_4d, 5: application_5d}


def premake(ndim, shift, dtype=None, block_size=None):
    """Create roll kernel specialized for given ndim and shift.

    The calling wrapper must permute the tensor so the target dim
    is at position 0 before calling this kernel.
    """
    arrangement_fn = functools.partial(arrangement, block_size=block_size)
    application = _APPLICATIONS[ndim]
    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
        Tensor(0, constexpr=True, value=shift),
    )
    return arrangement_fn, application, tensors
