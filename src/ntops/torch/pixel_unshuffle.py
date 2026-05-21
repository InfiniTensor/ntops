import functools

import torch

import ntops
from ntops.torch.utils import _cached_make


@functools.lru_cache(maxsize=None)
def _launch_config():
    """Pick ``(block_size, num_warps)`` for the strided-copy kernel on this GPU.

    Performance evaluation disables auto-tuning (``max_num_configs=1``), so
    explicit values are required. Tuned with ``bench/tune_pixel_unshuffle.py``:
    ``num_warps=4`` is best on all three platforms; NVIDIA peaks at a small 256
    block (the large, HBM-bound cases and all fp16 cases) while MetaX / Iluvatar
    prefer 2048. ``num_stages`` is a no-op (one block per program, no loop).
    Keys on the hardware only, never on input shapes.
    """
    name = torch.cuda.get_device_name().lower() if torch.cuda.is_available() else ""

    if "nvidia" in name:
        return 256, 4

    return 2048, 4


def pixel_unshuffle(input, downscale_factor):
    r = downscale_factor

    *batch, c, h, w = input.shape
    assert h % r == 0 and w % r == 0, (
        f"spatial dims ({h}, {w}) must be divisible by downscale_factor {r}"
    )

    h_, w_ = h // r, w // r

    # Split each r*r spatial window into (r, r), move those axes ahead of the
    # spatial dims, then merge them into the channel dim. The permute yields a
    # strided view that maps element-by-element onto the contiguous output; the
    # copy kernel materializes it.
    #   in:  (..., C, h_*r, w_*r)
    #   src: (..., C, r, r, h_, w_)  (strided view)
    src = input.reshape(*batch, c, h_, r, w_, r).movedim((-3, -1), (-4, -3))

    output = torch.empty(
        (*batch, c, r, r, h_, w_), dtype=input.dtype, device=input.device
    )

    block_size, num_warps = _launch_config()

    kernel = _cached_make(
        ntops.kernels.pixel_unshuffle.premake,
        src.ndim,
        block_size=block_size,
        num_warps=num_warps,
        num_stages=1,
    )
    kernel(src, output)

    return output.reshape(*batch, c * r * r, h_, w_)
