"""Tune (num_warps, block_size) for the fractional_max_pool2d gather kernel.

The data-dependent gather is launch/occupancy-sensitive: block_size=1024 gives
too few programs for small/medium M. Sweep to find the config that maximizes
occupancy across sizes.

    python bench/tune_fractional_max_pool.py
"""

import torch
import triton.testing

import ntops
from ntops.torch.fractional_max_pool2d import _intervals
from ntops.torch.utils import _cached_make, set_default_max_num_configs

DEVICE = "cuda"
DTYPE = torch.float32
BLOCK_SIZES = (64, 128, 256, 512, 1024, 2048)
NUM_WARPS = (1, 2, 4, 8)

CASES = [
    (8, 16, 16, 16, 2, 2, 12, 12),
    (16, 64, 56, 56, 3, 3, 40, 40),
    (32, 64, 112, 112, 2, 2, 80, 80),
]


def _base_offset(input, start_h, start_w):
    n, c, h, w = input.shape
    nc_base = (torch.arange(n * c, device=input.device).reshape(n, c)) * (h * w)
    bo = nc_base[..., None, None] + (start_h * w)[..., :, None] + start_w[..., None, :]
    return bo.reshape(-1).to(torch.int64)


def main():
    set_default_max_num_configs(1)
    print(f"device: {torch.cuda.get_device_name()}")

    for n, c, h, w, kh, kw, oh, ow in CASES:
        x = torch.randn(n, c, h, w, dtype=DTYPE, device=DEVICE).reshape(-1)
        xr = x.reshape(n, c, h, w)
        s = torch.rand(n, c, 2, dtype=DTYPE, device=DEVICE)
        sh = _intervals(s[..., 1], h, oh, kh)
        sw = _intervals(s[..., 0], w, ow, kw)
        bo = _base_offset(xr, sh, sw)
        m = bo.numel()
        out = torch.empty((m,), dtype=DTYPE, device=DEVICE)

        ms_th = triton.testing.do_bench(
            lambda: torch.nn.functional.fractional_max_pool2d(
                xr, (kh, kw), output_size=(oh, ow), _random_samples=s
            )
        )
        print(f"\n({n},{c},{h},{w})->({oh},{ow})  M={m}  (torch {ms_th:.3f} ms)")
        best = (1e9, None)
        for block_size in BLOCK_SIZES:
            for num_warps in NUM_WARPS:
                try:
                    kernel = _cached_make(
                        ntops.kernels.fractional_max_pool2d.premake,
                        block_size=block_size,
                        num_warps=num_warps,
                        num_stages=1,
                        max_num_configs=1,
                    )
                    ms = triton.testing.do_bench(lambda k=kernel: k(bo, out, x, w, kh, kw, m))
                    if ms < best[0]:
                        best = (ms, (num_warps, block_size))
                    print(f"  block={block_size:5d} warps={num_warps:2d}  {ms:7.3f} ms ({ms_th/ms:.2f}x)")
                except Exception as exc:  # noqa: BLE001
                    print(f"  block={block_size:5d} warps={num_warps:2d}  SKIP ({type(exc).__name__})")
        print(f"  best: num_warps={best[1][0]}, block_size={best[1][1]}  ({ms_th/best[0]:.2f}x torch)")


if __name__ == "__main__":
    main()
