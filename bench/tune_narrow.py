"""Tune the pinned launch config for ``ntops.torch.narrow`` on the current GPU.

narrow materializes a strided slice with a copy kernel. Performance evaluation
runs with auto-tuning disabled (``max_num_configs=1``), so the value baked into
``ntops/torch/narrow.py`` decides the score. This sweeps
``block_size x num_warps x num_stages`` on two representative slices -- a strided
inner-dim slice (the harder memory pattern) and a contiguous leading-dim slice
-- and prints, per shape, the fastest config plus the speedup over
``torch.narrow(...).contiguous()``. ``_launch_config`` must key on hardware only,
so pick a config that is good across both patterns.

Usage
-----
    python bench/tune_narrow.py
"""

import itertools

import torch

import ntops
from ntops.torch.utils import _cached_make

_NUMELS = [1024 * 1024, 4096 * 4096, 8192 * 8192]

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


def _runner(input, dim, length, block_size, num_warps, num_stages):
    src = input.narrow(dim, 0, length)
    output = torch.empty(src.shape, dtype=input.dtype, device=input.device)

    kernel = _cached_make(
        ntops.kernels.narrow.premake,
        src.ndim,
        block_size=block_size,
        num_warps=num_warps,
        num_stages=num_stages,
        max_num_configs=1,
    )

    return lambda: kernel(src, output)


def _sweep(label, make_runner, num_bytes, torch_ms):
    results = []
    for bs, nw, ns in itertools.product(_BLOCK_SIZES, _NUM_WARPS, _NUM_STAGES):
        try:
            ms = _time(make_runner(bs, nw, ns))
        except Exception as exc:  # noqa: BLE001
            print(f"  skip bs={bs} nw={nw} ns={ns}: {type(exc).__name__}")
            continue
        results.append((ms, bs, nw, ns))

    results.sort()
    best_ms, bbs, bnw, bns = results[0]
    best_gbps = num_bytes / (best_ms * 1e-3) / 1e9
    torch_gbps = num_bytes / (torch_ms * 1e-3) / 1e9

    print(f"\n  [{label}]  (torch {torch_ms:.4f} ms / {torch_gbps:.0f} GB/s)")
    print(
        f"    BEST  block_size={bbs:<5} num_warps={bnw:<3} num_stages={bns}  "
        f"-> {best_ms:.4f} ms / {best_gbps:.0f} GB/s  "
        f"(speedup vs torch {torch_ms / best_ms:.2f})"
    )
    for ms, bs, nw, ns in results[:5]:
        gbps = num_bytes / (ms * 1e-3) / 1e9
        print(
            f"      block_size={bs:<5} num_warps={nw:<3} num_stages={ns}  "
            f"{ms:.4f} ms / {gbps:.0f} GB/s"
        )


def _check_correctness(dtype):
    x = torch.randn(64, 1000, dtype=dtype, device="cuda")
    src = x.narrow(1, 0, 500)
    out = torch.empty(src.shape, dtype=dtype, device="cuda")
    kernel = _cached_make(
        ntops.kernels.narrow.premake,
        src.ndim,
        block_size=1024,
        num_warps=4,
        num_stages=1,
        max_num_configs=1,
    )
    kernel(src, out)
    assert torch.equal(out, src.contiguous())


def tune():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    for dtype in _DTYPES:
        _check_correctness(dtype)
        itemsize = torch.empty(0, dtype=dtype).element_size()

        print(f"\n{'='*92}")
        print(
            f"narrow config sweep | dtype={dtype} | "
            f"device={torch.cuda.get_device_name()}"
        )
        print("=" * 92)

        for numel in _NUMELS:
            side = int(round(numel**0.5))
            full = torch.randn(side, side, dtype=dtype, device="cuda")
            length = side // 2
            out_bytes = side * length * itemsize * 2  # read slice + write output

            print(f"\nnumel~{numel} ({side}x{side}, {numel * itemsize / 1e6:.1f} MB)")

            torch_ms = _time(lambda: full.narrow(1, 0, length).contiguous())
            _sweep(
                "dim=1 (strided)",
                lambda bs, nw, ns: _runner(full, 1, length, bs, nw, ns),
                out_bytes,
                torch_ms,
            )

            torch_ms = _time(lambda: full.narrow(0, 0, length).contiguous())
            _sweep(
                "dim=0 (contiguous)",
                lambda bs, nw, ns: _runner(full, 0, length, bs, nw, ns),
                out_bytes,
                torch_ms,
            )


if __name__ == "__main__":
    tune()
