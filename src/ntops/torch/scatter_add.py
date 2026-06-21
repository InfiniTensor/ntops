import torch

import ntops
from ntops.torch.utils import _cached_make, _device_key

# (num_warps, block_size) tuned per platform; see bench/tune_scatter_add.py.
_CONFIGS = {
    "nvidia": (8, 256),
    "iluvatar": (2, 1024),
    "metax": (2, 1024),
    "default": (8, 256),
}


def scatter_add(input, dim, index, src):
    # Move `dim` to the last axis and flatten to output (R, T) / index,src (R, K).
    # output starts as a clone of input (the accumulation base); the kernel
    # atomic-adds each src element into output[r, index]. Supports the common
    # case where index/src match input in the non-`dim` dims (asserted).
    dim = dim % input.ndim

    inp = input.movedim(dim, -1).contiguous()
    permuted_shape = inp.shape
    t_size = permuted_shape[-1]
    rows = inp.numel() // t_size

    idx = index.movedim(dim, -1).contiguous()
    k_size = idx.shape[-1]
    assert idx.numel() // k_size == rows, (
        "scatter_add currently requires index/src to match input in non-`dim` dims."
    )

    output = inp.reshape(rows, t_size).clone()
    idx = idx.reshape(rows, k_size).to(torch.int64)
    src = src.movedim(dim, -1).contiguous().reshape(rows, k_size)

    # Force a single config (eval-aligned): the un-arranged `output` source has
    # no autotunable size bounds, so the autotuner can't run over it.
    num_warps, block_size = _CONFIGS[_device_key()]
    kernel = _cached_make(
        ntops.kernels.scatter_add.premake,
        block_size=block_size,
        num_warps=num_warps,
        num_stages=1,
        max_num_configs=1,
    )
    kernel(src, idx, output, t_size, k_size, rows * k_size)

    return output.reshape(permuted_shape).movedim(-1, dim)
