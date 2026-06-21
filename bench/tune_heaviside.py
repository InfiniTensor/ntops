"""Tune ``(num_warps, block_size)`` for the heaviside kernel.

Evaluation disables auto-tuning (``max_num_configs=1``), so the winning config
must be passed explicitly into ``premake``. This sweeps configs at a fixed large
shape under eval-like conditions and reports the best GB/s per dtype.

Run on each platform (NVIDIA / Iluvatar / MetaX) and bake the winners into a
device-adaptive ``_launch_config`` in ``ntops/torch/heaviside.py``.

    python bench/tune_heaviside.py
"""

import torch
import triton.testing

import ntops
from ntops.torch.utils import _cached_make, set_default_max_num_configs

DEVICE = "cuda"
SHAPE = (8192, 8192)
BLOCK_SIZES = (256, 512, 1024, 2048, 4096, 8192)
NUM_WARPS = (4, 8, 16)


def _tune(dtype):
    input = torch.randn(SHAPE, dtype=dtype, device=DEVICE)
    values = torch.randn(SHAPE, dtype=dtype, device=DEVICE)
    output = torch.empty_like(input)

    # heaviside touches 2 reads + 1 write per element.
    nbytes = 3 * input.numel() * input.element_size()
    best_bw, best_cfg = 0.0, None

    for block_size in BLOCK_SIZES:
        for num_warps in NUM_WARPS:
            try:
                kernel = _cached_make(
                    ntops.kernels.heaviside.premake,
                    input.ndim,
                    block_size=block_size,
                    num_warps=num_warps,
                    num_stages=1,
                )
                ms = triton.testing.do_bench(lambda k=kernel: k(input, values, output))
                bw = nbytes / ms * 1e-6  # ms -> s (1e3) and bytes -> GB (1e-9)
                print(f"  block={block_size:5d} warps={num_warps:2d}  {bw:7.0f} GB/s")
                if bw > best_bw:
                    best_bw, best_cfg = bw, (num_warps, block_size)
            except Exception as exc:  # noqa: BLE001
                print(f"  block={block_size:5d} warps={num_warps:2d}  SKIP ({type(exc).__name__})")

    return best_bw, best_cfg


def main():
    set_default_max_num_configs(1)
    print(f"device: {torch.cuda.get_device_name()}")

    for dtype in (torch.float32, torch.float16):
        print(f"\n[{dtype}]")
        bw, cfg = _tune(dtype)
        print(f"  best: num_warps={cfg[0]}, block_size={cfg[1]}  ({bw:.0f} GB/s)")


if __name__ == "__main__":
    main()
