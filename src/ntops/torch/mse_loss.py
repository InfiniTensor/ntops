import functools
import math

import torch

import ntops
from ntops.torch.utils import _cached_make


@functools.lru_cache(maxsize=None)
def _launch_config():
    """Pick ``(num_warps, reduce_block_size, none_block_size)`` for this GPU.

    Performance evaluation disables auto-tuning (``max_num_configs=1``), so
    explicit values are required; the reduction block size additionally sets the
    partial-sums buffer length and so must be known host-side. Values tuned with
    ``bench/tune_mse_loss.py``. ``num_stages`` is a no-op here (one block per
    program, no inner loop) and stays 1.

    The reduction kernel's intra-block ``ntl.sum`` favors more warps on Iluvatar
    (16) but fewer on MetaX (4); on NVIDIA RTX 4090 it is warp-insensitive and
    wants the larger 8192 block, which is also the MetaX optimum, so NVIDIA and
    other unmeasured devices fall through to that 8-warp / 8192 default. This
    keys on the hardware only, never on input shapes/names.
    """
    name = torch.cuda.get_device_name().lower() if torch.cuda.is_available() else ""

    if "metax" in name:
        return 4, 8192, 1024
    if "iluvatar" in name:
        return 16, 4096, 1024
    return 8, 8192, 1024


def mse_loss(input, target, reduction="mean"):
    if reduction not in ("none", "mean", "sum"):
        raise ValueError(f"unsupported reduction: {reduction!r}")

    if input.shape != target.shape:
        input, target = torch.broadcast_tensors(input, target)
        input = input.contiguous()
        target = target.contiguous()

    num_warps, reduce_block_size, none_block_size = _launch_config()

    if reduction == "none":
        output = torch.empty_like(input)

        kernel = _cached_make(
            ntops.kernels.mse_loss.premake,
            input.ndim,
            block_size=none_block_size,
            num_warps=num_warps,
            num_stages=1,
        )
        kernel(input, target, output)

        return output

    flat_input = input.reshape(-1)
    flat_target = target.reshape(-1)

    numel = flat_input.numel()
    num_partials = max(1, math.ceil(numel / reduce_block_size))

    partials = torch.empty(num_partials, dtype=torch.float32, device=input.device)

    kernel = _cached_make(
        ntops.kernels.mse_loss.reduce_premake,
        block_size=reduce_block_size,
        num_warps=num_warps,
        num_stages=1,
    )
    kernel(flat_input, flat_target, partials)

    total = partials.sum()

    if reduction == "mean":
        total = total / numel

    return total.to(input.dtype)
