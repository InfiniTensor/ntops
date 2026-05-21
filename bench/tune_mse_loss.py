"""Tune the pinned launch configs for ``ntops.torch.mse_loss`` on the current
GPU.

Two kernels are tuned independently:
  * the reduction path (``reduction="mean"|"sum"``) -- the defining, perf
    critical kernel; partials buffer size depends on ``block_size``;
  * the element-wise path (``reduction="none"``).

Performance evaluation runs with auto-tuning disabled (``max_num_configs=1``),
so the values baked into ``ntops/torch/mse_loss.py`` decide the score. This
script sweeps a small grid under those exact conditions and prints, per shape,
the fastest config plus the speedup over ``torch.nn.functional.mse_loss``.

Usage
-----
    python bench/tune_mse_loss.py
"""

import itertools
import math

import torch
import torch.nn.functional as F

import ntops
from ntops.torch.utils import _cached_make

# Numbers of elements to tune over (bandwidth-bound regime). Small sizes are
# launch-overhead bound and not informative for config selection.
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


def _reduce_runner(flat_in, flat_tg, block_size, num_warps, num_stages):
    numel = flat_in.numel()
    num_partials = max(1, math.ceil(numel / block_size))
    partials = torch.empty(num_partials, dtype=torch.float32, device=flat_in.device)

    kernel = _cached_make(
        ntops.kernels.mse_loss.reduce_premake,
        block_size=block_size,
        num_warps=num_warps,
        num_stages=num_stages,
        max_num_configs=1,
    )

    def run():
        kernel(flat_in, flat_tg, partials)
        return partials.sum()

    return run


def _none_runner(input, target, output, block_size, num_warps, num_stages):
    kernel = _cached_make(
        ntops.kernels.mse_loss.premake,
        input.ndim,
        block_size=block_size,
        num_warps=num_warps,
        num_stages=num_stages,
        max_num_configs=1,
    )
    return lambda: kernel(input, target, output)


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


def _check_reduce_correctness(dtype):
    """Sanity check that the reduction kernel matches F.mse_loss before trusting
    any timing numbers."""
    x = torch.randn(40000, dtype=dtype, device="cuda")
    y = torch.randn(40000, dtype=dtype, device="cuda")
    run = _reduce_runner(x, y, 1024, 4, 1)
    got = (run() / x.numel()).to(dtype)
    ref = F.mse_loss(x, y, reduction="mean")
    tol = 1e-3 if dtype == torch.float32 else 1e-2
    assert torch.allclose(got, ref, rtol=tol, atol=tol), (got.item(), ref.item())


def tune():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    for dtype in _DTYPES:
        _check_reduce_correctness(dtype)
        itemsize = torch.empty(0, dtype=dtype).element_size()

        print(f"\n{'='*92}")
        print(
            f"mse_loss config sweep | dtype={dtype} | "
            f"device={torch.cuda.get_device_name()}"
        )
        print("=" * 92)

        for numel in _NUMELS:
            side = int(round(numel**0.5))
            input = torch.randn(numel, dtype=dtype, device="cuda")
            target = torch.randn(numel, dtype=dtype, device="cuda")
            output = torch.empty_like(input)

            print(f"\nnumel={numel} (~{side}^2, {numel * itemsize / 1e6:.1f} MB)")

            # Reduction path: reads input + target (2x).
            torch_ms = _time(lambda: F.mse_loss(input, target, reduction="sum"))
            _sweep(
                "reduce (sum/mean)",
                lambda bs, nw, ns: _reduce_runner(input, target, bs, nw, ns),
                numel * itemsize * 2,
                torch_ms,
            )

            # Element-wise path: reads input + target, writes output (3x).
            torch_ms = _time(lambda: F.mse_loss(input, target, reduction="none"))
            _sweep(
                "none (element-wise)",
                lambda bs, nw, ns: _none_runner(input, target, output, bs, nw, ns),
                numel * itemsize * 3,
                torch_ms,
            )


if __name__ == "__main__":
    tune()
