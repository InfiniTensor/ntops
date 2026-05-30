import torch

import ntops
from ntops.torch.utils import _cached_make, _device_key

# (num_warps, block_size) tuned per platform at [8192, 8192]; see
# bench/tune_frac.py. Both fp32 and fp16 are within ~4% of their respective
# peaks with these configs.
_CONFIGS = {
    "nvidia": (8, 512),
    "iluvatar": (4, 2048),
    "metax": (4, 1024),
    "default": (4, 2048),
}


def frac(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    num_warps, block_size = _CONFIGS[_device_key()]

    kernel = _cached_make(
        ntops.kernels.frac.premake,
        input.ndim,
        block_size=block_size,
        num_warps=num_warps,
        num_stages=1,
    )

    kernel(input, out)

    return out
