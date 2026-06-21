import ntops
from ntops.torch.utils import _cached_make, _device_key

# (num_warps, block_size) tuned per platform at [8192, 8192]; see bench/tune_copy.py.
_CONFIGS = {
    "nvidia": (8, 1024),
    "iluvatar": (4, 2048),
    "metax": (4, 8192),
    "default": (4, 2048),
}


def _copy(input, output):
    """Materialize (possibly strided) ``input`` into ``output`` via the copy
    kernel. Internal helper shared by ``slice_scatter`` and ``hsplit``; not a
    public op."""
    num_warps, block_size = _CONFIGS[_device_key()]

    kernel = _cached_make(
        ntops.kernels.copy.premake,
        input.ndim,
        block_size=block_size,
        num_warps=num_warps,
        num_stages=1,
    )

    kernel(input, output)

    return output
