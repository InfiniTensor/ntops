import torch

import ntops
from ntops.torch.utils import _cached_make, _device_key

# (num_warps, block_size) tuned per platform at [8192, 8192]; see
# bench/tune_heaviside.py.
_CONFIGS = {
    "nvidia": (8, 1024),
    "iluvatar": (4, 2048),
    "metax": (4, 8192),
    "default": (4, 2048),
}


def heaviside(input, values, *, out=None):
    # `torch.heaviside` requires `input` and `values` to share a dtype and
    # only broadcasts `values` against `input`.
    assert input.dtype == values.dtype, (
        "`heaviside` requires `input` and `values` to have the same dtype."
    )

    if out is None:
        out = torch.empty_like(input)

    # A stride-0 broadcast view is read correctly by ninetoothed (the offset is
    # `index * stride`, and `stride == 0` repeats the element), so a scalar
    # `values` is not materialized to `input`'s size.
    values = values.broadcast_to(input.shape)

    num_warps, block_size = _CONFIGS[_device_key()]

    kernel = _cached_make(
        ntops.kernels.heaviside.premake,
        input.ndim,
        block_size=block_size,
        num_warps=num_warps,
        num_stages=1,
    )

    kernel(input, values, out)

    return out
