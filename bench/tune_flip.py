"""Tune the pinned launch config (block_size / num_warps / num_stages) for
``ntops.torch.flip`` on the current GPU.

Performance evaluation runs with auto-tuning disabled (``max_num_configs=1``),
so the values baked into ``ntops/torch/flip.py`` decide the score. This script
sweeps a small grid under those exact conditions and prints, per shape, the
fastest config plus the speedup over ``torch.flip``.

Usage
-----
    python bench/tune_flip.py
"""

import itertools

import torch

import ntops
from ntops.torch.utils import _cached_make

# Shapes that matter for a bandwidth-bound op: the medium case that has not yet
# saturated memory, and the large cases at the bandwidth ceiling. Small shapes
# are launch-overhead bound and not informative for config tuning.
_SHAPES = [
    ([4096, 4096], (0, 1)),
    ([4096, 4096], (1,)),
    ([8192, 8192], (1,)),
    ([8192, 8192], (0,)),
]

_BLOCK_SIZES = [512, 1024, 2048, 4096, 8192]
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


def _run_config(input, output, dims, block_size, num_warps, num_stages):
    kernel = _cached_make(
        ntops.kernels.flip.premake,
        input.ndim,
        dims,
        block_size=block_size,
        num_warps=num_warps,
        num_stages=num_stages,
        max_num_configs=1,
    )
    return lambda: kernel(input, output)


def tune():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    for dtype in _DTYPES:
        print(f"\n{'='*92}")
        print(f"flip config sweep | dtype={dtype} | device={torch.cuda.get_device_name()}")
        print("=" * 92)

        for shape, dims in _SHAPES:
            input = torch.randn(shape, dtype=dtype, device="cuda")
            output = torch.empty(shape, dtype=dtype, device="cuda")
            num_bytes = input.numel() * input.element_size() * 2

            torch_ms = _time(lambda: torch.flip(input, list(dims)))

            results = []
            for bs, nw, ns in itertools.product(
                _BLOCK_SIZES, _NUM_WARPS, _NUM_STAGES
            ):
                try:
                    fn = _run_config(input, output, dims, bs, nw, ns)
                    ms = _time(fn)
                except Exception as exc:  # noqa: BLE001
                    print(f"  skip bs={bs} nw={nw} ns={ns}: {type(exc).__name__}")
                    continue
                results.append((ms, bs, nw, ns))

            results.sort()
            best_ms, bbs, bnw, bns = results[0]
            best_gbps = num_bytes / (best_ms * 1e-3) / 1e9
            torch_gbps = num_bytes / (torch_ms * 1e-3) / 1e9

            print(
                f"\nshape={shape} dims={dims}  "
                f"(torch {torch_ms:.4f} ms / {torch_gbps:.0f} GB/s)"
            )
            print(
                f"  BEST  block_size={bbs:<5} num_warps={bnw:<3} num_stages={bns}  "
                f"-> {best_ms:.4f} ms / {best_gbps:.0f} GB/s  "
                f"(speedup vs torch {torch_ms / best_ms:.2f})"
            )
            print("  top 5:")
            for ms, bs, nw, ns in results[:5]:
                gbps = num_bytes / (ms * 1e-3) / 1e9
                print(
                    f"    block_size={bs:<5} num_warps={nw:<3} num_stages={ns}  "
                    f"{ms:.4f} ms / {gbps:.0f} GB/s"
                )


if __name__ == "__main__":
    tune()
