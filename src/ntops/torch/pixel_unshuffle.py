import torch

import ntops
from ntops.torch.utils import _cached_make


def pixel_unshuffle(input, downscale_factor):
    assert input.ndim == 4, "`pixel_unshuffle` only supports 4D NCHW input."

    n, c, h, w = input.shape
    r = downscale_factor

    assert isinstance(r, int), "`downscale_factor` must be int."
    assert r > 0, "`downscale_factor` must be positive."
    assert h % r == 0, "input height must be divisible by downscale_factor."
    assert w % r == 0, "input width must be divisible by downscale_factor."

    output = torch.empty(
        (n, c * r * r, h // r, w // r),
        dtype=input.dtype,
        device=input.device,
    )

    kernel = _cached_make(
        ntops.kernels.pixel_unshuffle.premake,
    )

    kernel(
        input,
        output,
        downscale_factor=r,
    )

    return output