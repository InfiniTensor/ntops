import functools
import math

import torch

import ntops
from ntops.torch.utils import _cached_make


@functools.lru_cache(maxsize=None)
def _launch_config():
    """Pick ``(num_warps, block_size)`` for the count reduction on this GPU.

    Performance evaluation disables auto-tuning (``max_num_configs=1``), so
    explicit values are required; the block size also sets the partials buffer
    length and so must be known host-side. Tuned with
    ``bench/tune_count_nonzero.py``: the global (dim=None) path strongly prefers
    the largest 8192 block, which the dim path also accepts, so both share it.
    MetaX wants 4 warps, Iluvatar 8, NVIDIA 16. ``num_stages`` is a no-op here
    and stays 1. Keys on the hardware name only, never on input shapes.
    """
    name = torch.cuda.get_device_name().lower() if torch.cuda.is_available() else ""

    if "metax" in name:
        return 4, 8192
    if "iluvatar" in name:
        return 8, 8192
    return 16, 8192


def _normalize_dims(dim, ndim):
    dims = (dim,) if isinstance(dim, int) else tuple(dim)
    dims = tuple(d if d >= 0 else d + ndim for d in dims)

    for d in dims:
        if not (0 <= d < ndim):
            raise IndexError(
                f"Dimension out of range (expected to be in range of "
                f"[{-ndim}, {ndim - 1}])"
            )

    if len(set(dims)) != len(dims):
        raise RuntimeError(f"dim {dim} appears multiple times")

    return dims


def count_nonzero(input, dim=None):
    num_warps, block_size = _launch_config()

    if dim is None:
        flat = input.reshape(-1)
        numel = flat.numel()

        if numel == 0:
            return torch.zeros((), dtype=torch.int64, device=input.device)

        num_partials = max(1, math.ceil(numel / block_size))
        partials = torch.empty(num_partials, dtype=torch.int64, device=input.device)

        kernel = _cached_make(
            ntops.kernels.count_nonzero.global_premake,
            block_size=block_size,
            num_warps=num_warps,
            num_stages=1,
        )
        kernel(flat, partials)

        return partials.sum()

    dims = _normalize_dims(dim, input.dim())

    kept = tuple(i for i in range(input.dim()) if i not in dims)
    kept_shape = tuple(input.shape[i] for i in kept)

    # Fast path: reducing a consecutive block of *leading* dims (with trailing
    # dims kept) is a coalesced column reduction on the contiguous ``(R, inner)``
    # view -- no transpose. Trailing-dim reductions already need no real
    # transpose (the permute below is identity) and stay on the general path.
    sorted_dims = sorted(dims)
    is_consecutive = list(sorted_dims) == list(
        range(sorted_dims[0], sorted_dims[-1] + 1)
    )
    if is_consecutive and sorted_dims[0] == 0 and sorted_dims[-1] < input.dim() - 1:
        b = sorted_dims[-1]
        r = math.prod(input.shape[: b + 1])
        inner = math.prod(input.shape[b + 1 :])

        if r == 0 or inner == 0:
            return torch.zeros(kept_shape, dtype=torch.int64, device=input.device)

        x2d = input.reshape(r, inner)

        # Small column block keeps the (reduce_block, col_block) tile sized
        # sanely while staying coalesced; not as heavily tuned as the global
        # block size (which would make the 2-D tile far too large).
        col_block = 256
        reduce_block = 32
        num_row_blocks = max(1, math.ceil(r / reduce_block))
        partials = torch.empty(
            (num_row_blocks, inner), dtype=torch.int64, device=input.device
        )

        kernel = _cached_make(
            ntops.kernels.count_nonzero.leading_premake,
            block_size=col_block,
            reduce_block=reduce_block,
            num_warps=num_warps,
            num_stages=1,
        )
        kernel(x2d, partials)

        return partials.sum(dim=0).reshape(kept_shape)

    # General path: move reduced dims to the back and collapse to (M, N).
    permuted = input.permute(kept + dims).contiguous()
    m = math.prod(kept_shape) if kept_shape else 1
    n = permuted.numel() // m if m else 0
    x2d = permuted.reshape(m, n)

    if n == 0:
        return torch.zeros(kept_shape, dtype=torch.int64, device=input.device)

    num_blocks = max(1, math.ceil(n / block_size))
    partials = torch.empty((m, num_blocks), dtype=torch.int64, device=input.device)

    kernel = _cached_make(
        ntops.kernels.count_nonzero.dim_premake,
        block_size=block_size,
        num_warps=num_warps,
        num_stages=1,
    )
    kernel(x2d, partials)

    return partials.sum(dim=1).reshape(kept_shape)
