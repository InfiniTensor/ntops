"""Tune the pinned launch config for ``ntops.torch.pixel_unshuffle`` on the
current GPU.

The kernel is an element-wise copy of a strided (permuted) view. Performance
evaluation runs with auto-tuning disabled (``max_num_configs=1``), so the values
baked into ``ntops/torch/pixel_unshuffle.py`` decide the score. This script
sweeps ``block_size`` / ``num_warps`` / ``num_stages`` under those exact
conditions and prints, per shape, the fastest config plus the speedup over
``F.pixel_unshuffle``. ``num_stages`` is expected to be a no-op (one block per
program, no inner loop).

Usage
-----
    python bench/tune_pixel_unshuffle.py
"""

import itertools

import torch
import torch.nn.functional as F

import ntops
from ntops.torch.utils import _cached_make

# (shape, downscale_factor) -- the bandwidth-bound regime.
_CASES = [
    ([16, 128, 64, 64], 2),
    ([32, 256, 64, 64], 2),
    ([8, 64, 128, 128], 2),
    ([4, 8, 12, 12], 3),
]

_BLOCK_SIZES = [256, 512, 1024, 2048, 4096, 8192]
_NUM_WARPS = [4, 8, 16]
_NUM_STAGES = [1, 2]

_DTYPES = [torch.float32, torch.float16]


def _time(fn, n_warmup=10, n_repeat=50):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_repeat


def _runner(input, r, block_size, num_warps, num_stages):
    *batch, c, h, w = input.shape
    h_, w_ = h // r, w // r
    src = input.reshape(*batch, c, h_, r, w_, r).movedim((-3, -1), (-4, -3))
    output = torch.empty(
        (*batch, c, r, r, h_, w_), dtype=input.dtype, device=input.device
    )

    kernel = _cached_make(
        ntops.kernels.pixel_unshuffle.premake,
        src.ndim,
        block_size=block_size,
        num_warps=num_warps,
        num_stages=num_stages,
        max_num_configs=1,
    )
    return lambda: kernel(src, output)


def _check_correctness(input, r):
    expected = F.pixel_unshuffle(input, r)
    got = ntops.torch.pixel_unshuffle(input, r)
    assert torch.equal(got, expected), "pixel_unshuffle mismatch vs torch"


def tune():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    for dtype in _DTYPES:
        print(f"\n{'='*96}")
        print(
            f"pixel_unshuffle config sweep | dtype={dtype} | "
            f"device={torch.cuda.get_device_name()}"
        )
        print("=" * 96)

        for shape, r in _CASES:
            input = torch.randn(shape, dtype=dtype, device="cuda")
            _check_correctness(input, r)

            num_bytes = input.numel() * input.element_size() * 2
            torch_ms = _time(lambda: F.pixel_unshuffle(input, r))

            results = []
            for bs, nw, ns in itertools.product(
                _BLOCK_SIZES, _NUM_WARPS, _NUM_STAGES
            ):
                try:
                    ms = _time(_runner(input, r, bs, nw, ns))
                except Exception as exc:  # noqa: BLE001
                    print(f"  skip bs={bs} nw={nw} ns={ns}: {type(exc).__name__}")
                    continue
                results.append((ms, bs, nw, ns))

            results.sort()
            best_ms, bbs, bnw, bns = results[0]
            best_gbps = num_bytes / (best_ms * 1e-3) / 1e9
            torch_gbps = num_bytes / (torch_ms * 1e-3) / 1e9

            print(
                f"\nshape={shape} r={r}  (torch {torch_ms:.4f} ms / {torch_gbps:.0f} GB/s)"
            )
            print(
                f"  BEST  block_size={bbs:<5} num_warps={bnw:<3} num_stages={bns}  "
                f"-> {best_ms:.4f} ms / {best_gbps:.0f} GB/s  "
                f"(speedup vs torch {torch_ms / best_ms:.2f})"
            )
            for ms, bs, nw, ns in results[:5]:
                gbps = num_bytes / (ms * 1e-3) / 1e9
                print(
                    f"    block_size={bs:<5} num_warps={nw:<3} num_stages={ns}  "
                    f"{ms:.4f} ms / {gbps:.0f} GB/s"
                )


if __name__ == "__main__":
    tune()
