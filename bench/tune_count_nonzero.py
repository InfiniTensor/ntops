"""Tune the pinned launch configs for ``ntops.torch.count_nonzero`` on the
current GPU.

Two kernels are tuned independently:
  * the global path (``dim=None``) -- flattens, one partial per block;
  * the dim path (``dim`` given) -- reshapes to ``(M, N)``, one partial per
    ``(row, block)``.

Both are memory-bound partial-sum reductions reading the input once.
Performance evaluation runs with auto-tuning disabled (``max_num_configs=1``),
so the values baked into ``ntops/torch/count_nonzero.py`` decide the score; the
block size also sizes the partials buffer host-side. This sweeps
``block_size x num_warps x num_stages`` under those conditions and prints, per
shape, the fastest config plus the speedup over ``torch.count_nonzero``.

Usage
-----
    python bench/tune_count_nonzero.py
"""

import itertools
import math

import torch

import ntops
from ntops.torch.utils import _cached_make

_NUMELS = [1024 * 1024, 4096 * 4096, 8192 * 8192]

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


def _global_runner(flat, block_size, num_warps, num_stages):
    numel = flat.numel()
    num_partials = max(1, math.ceil(numel / block_size))
    partials = torch.empty(num_partials, dtype=torch.int64, device=flat.device)

    kernel = _cached_make(
        ntops.kernels.count_nonzero.global_premake,
        block_size=block_size,
        num_warps=num_warps,
        num_stages=num_stages,
        max_num_configs=1,
    )

    def run():
        kernel(flat, partials)
        return partials.sum()

    return run


def _dim_runner(x2d, block_size, num_warps, num_stages):
    m, n = x2d.shape
    num_blocks = max(1, math.ceil(n / block_size))
    partials = torch.empty((m, num_blocks), dtype=torch.int64, device=x2d.device)

    kernel = _cached_make(
        ntops.kernels.count_nonzero.dim_premake,
        block_size=block_size,
        num_warps=num_warps,
        num_stages=num_stages,
        max_num_configs=1,
    )

    def run():
        kernel(x2d, partials)
        return partials.sum(dim=1)

    return run


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


def _make_input(numel, dtype):
    x = torch.randn(numel, dtype=dtype, device="cuda")
    return torch.where(x.abs() < 0.5, torch.zeros_like(x), x)


def _check_correctness(dtype):
    x = _make_input(40000, dtype)
    got = _global_runner(x, 1024, 4, 1)()
    assert got.item() == torch.count_nonzero(x).item(), (got, torch.count_nonzero(x))

    x2 = x.reshape(200, 200)
    got2 = _dim_runner(x2, 1024, 4, 1)()
    assert torch.equal(got2, torch.count_nonzero(x2, dim=1))

    # Leading (dim=0) coalesced path, exercised end-to-end through the wrapper.
    got3 = ntops.torch.count_nonzero(x2, dim=0)
    assert torch.equal(got3, torch.count_nonzero(x2, dim=0))


def tune():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    for dtype in _DTYPES:
        _check_correctness(dtype)
        itemsize = torch.empty(0, dtype=dtype).element_size()

        print(f"\n{'='*92}")
        print(
            f"count_nonzero config sweep | dtype={dtype} | "
            f"device={torch.cuda.get_device_name()}"
        )
        print("=" * 92)

        for numel in _NUMELS:
            side = int(round(numel**0.5))
            x = _make_input(numel, dtype)

            print(f"\nnumel={numel} (~{side}^2, {numel * itemsize / 1e6:.1f} MB)")

            # Global path: reads the whole input once.
            torch_ms = _time(lambda: torch.count_nonzero(x))
            _sweep(
                "global (dim=None)",
                lambda bs, nw, ns: _global_runner(x, bs, nw, ns),
                numel * itemsize,
                torch_ms,
            )

            # Dim path: reduce the last dim of a squarish (side, side) view.
            x2d = x[: side * side].reshape(side, side)
            torch_ms = _time(lambda: torch.count_nonzero(x2d, dim=1))
            _sweep(
                "dim=1",
                lambda bs, nw, ns: _dim_runner(x2d, bs, nw, ns),
                side * side * itemsize,
                torch_ms,
            )


if __name__ == "__main__":
    tune()
