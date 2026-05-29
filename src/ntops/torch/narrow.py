import functools

import torch

import ntops
from ntops.torch.utils import _cached_make


@functools.lru_cache(maxsize=None)
def _launch_config():
    """Pick ``(block_size, num_warps)`` for the strided-copy kernel on this GPU.

    Performance evaluation disables auto-tuning (``max_num_configs=1``), so
    explicit values are required. Tuned with ``bench/tune_narrow.py`` on the
    asymptotic (large, HBM-bound) cases where configs differ: NVIDIA peaks at a
    small block with many warps, while the domestic cards prefer larger blocks
    with 4 warps. ``num_stages`` is a no-op (one block per program, no loop).
    Keys on the hardware name only, never on input shapes; unmeasured devices
    (e.g. Moore) fall through to the domestic-style default.
    """
    name = torch.cuda.get_device_name().lower() if torch.cuda.is_available() else ""

    if "metax" in name:
        return 4096, 4
    if "iluvatar" in name:
        return 2048, 4
    if "nvidia" in name:
        return 512, 16
    return 2048, 4


def narrow(input, dim, start, length):
    if input.dim() == 0:
        raise RuntimeError("narrow() cannot be applied to a 0-dim tensor.")

    ndim = input.dim()

    if not (-ndim <= dim < ndim):
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[{-ndim}, {ndim - 1}], but got {dim})"
        )

    dim = dim if dim >= 0 else dim + ndim

    size = input.shape[dim]

    # torch accepts a 0-dim tensor start; normalize to a Python int.
    if torch.is_tensor(start):
        start = int(start.item())

    if start < 0:
        start += size

    if start < 0 or start + length > size:
        raise RuntimeError(
            f"start ({start}) + length ({length}) exceeds dimension size ({size})."
        )

    # A strided view (no copy) of the requested slice; the kernel materializes it.
    src = input.narrow(dim, start, length)

    output = torch.empty(src.shape, dtype=input.dtype, device=input.device)

    if output.numel() == 0:
        return output

    block_size, num_warps = _launch_config()

    kernel = _cached_make(
        ntops.kernels.narrow.premake,
        src.ndim,
        block_size=block_size,
        num_warps=num_warps,
        num_stages=1,
    )
    kernel(src, output)

    return output
