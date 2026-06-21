"""Tune scatter_add's atomic-scatter config.

The atomic scatter is latency-bound (constant ~6 GB/s regardless of size), which
means too few concurrent atomics. The fix is many concurrent threads each doing
~1 atomic: small block_size + enough warps + many programs. This sweeps to find
the config that hides atomic latency.

    python bench/tune_scatter_add.py
"""

import torch
import triton.testing

import ntops
from ntops.torch.utils import _cached_make, set_default_max_num_configs

DEVICE = "cuda"
DTYPE = torch.float32

# (input_shape, dim, k) — a few representative scatter shapes.
SHAPES = [
    ((1024, 256), 1, 256),
    ((4096, 512), 1, 256),
]
BLOCK_SIZES = (64, 128, 256, 512, 1024, 2048)
NUM_WARPS = (1, 2, 4, 8, 16)


def _next_pow2(x):
    return 1 << (x - 1).bit_length()


def _run(shape, dim, k, block_size, num_warps):
    inp = torch.randn(shape, dtype=DTYPE, device=DEVICE)
    idx_shape = list(shape)
    idx_shape[dim] = k
    t = shape[dim]
    idx = torch.randint(0, t, idx_shape, device=DEVICE).to(torch.int64)
    src = torch.randn(idx_shape, dtype=DTYPE, device=DEVICE)

    inp_p = inp.movedim(dim, -1).contiguous()
    t_size = inp_p.shape[-1]
    rows = inp_p.numel() // t_size
    output = inp_p.reshape(rows, t_size).clone()
    idx_p = idx.movedim(dim, -1).contiguous().reshape(rows, -1)
    k_size = idx_p.shape[-1]
    src_p = src.movedim(dim, -1).contiguous().reshape(rows, k_size)

    kernel = _cached_make(
        ntops.kernels.scatter_add.premake,
        block_size=block_size,
        num_warps=num_warps,
        num_stages=1,
        max_num_configs=1,
    )
    return triton.testing.do_bench(
        lambda: kernel(output, idx_p, src_p, t_size, k_size, rows * k_size)
    )


def main():
    set_default_max_num_configs(1)
    print(f"device: {torch.cuda.get_device_name()}")

    for shape, dim, k in SHAPES:
        inp = torch.randn(shape, dtype=DTYPE, device=DEVICE)
        idx_shape = list(shape)
        idx_shape[dim] = k
        idx = torch.randint(0, shape[dim], idx_shape, device=DEVICE)
        src = torch.randn(idx_shape, dtype=DTYPE, device=DEVICE)
        ms_th = triton.testing.do_bench(lambda: torch.scatter_add(inp, dim, idx, src))

        print(f"\nshape={shape} dim={dim} k={k}  (torch {ms_th:.3f} ms)")
        best = (1e9, None)
        for block_size in BLOCK_SIZES:
            for num_warps in NUM_WARPS:
                try:
                    ms = _run(shape, dim, k, block_size, num_warps)
                    print(
                        f"  block={block_size:5d} warps={num_warps:2d}  "
                        f"{ms:8.3f} ms  ({ms_th / ms:.2f}x torch)"
                    )
                    if ms < best[0]:
                        best = (ms, (num_warps, block_size))
                except Exception as exc:  # noqa: BLE001
                    print(f"  block={block_size:5d} warps={num_warps:2d}  SKIP ({type(exc).__name__})")
        print(f"  best: num_warps={best[1][0]}, block_size={best[1][1]}  "
              f"({ms_th / best[0]:.2f}x torch)")


if __name__ == "__main__":
    main()
