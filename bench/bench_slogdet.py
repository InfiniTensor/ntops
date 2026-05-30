"""Benchmark ntops.slogdet vs torch.linalg.slogdet.

Single-block in-kernel LU targets *many small matrices*. Reports GB/s and
speedup at several (batch, n) combinations so you can see where the crossover
is between ninetoothed and cuSOLVER.

    python bench/bench_slogdet.py
"""

import os
from contextlib import contextmanager

import torch
import triton.testing

import ntops


@contextmanager
def _suppress_stderr():
    # The Iluvatar backend emits "loop ... not unrolled" remarks at the C/MLIR
    # level (fd 2), which `contextlib.redirect_stderr` can't catch; dup2 the fd.
    # Benign (correctness unaffected); this just keeps the bench output clean.
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)

DEVICE = "cuda"
DTYPE = torch.float32

# (batch, n) — n must be a power of 2 after padding; we pass the exact n and
# let the wrapper pad.  Add larger n cautiously: single-block LU may fail to
# compile above n~64-128.
CASES = [
    (1, 1),
    (1, 4),
    (1, 8),
    (1, 16),
    (64, 4),
    (128, 4),
    (512, 4),
    (64, 8),
    (128, 8),
    (64, 16),
    (128, 16),
    (512, 16),
    (64, 32),
    (128, 32),
    # probe the large-n ceiling: where does single-block LU stop compiling /
    # stop winning vs cuSOLVER?
    (1, 64),
    (64, 64),
    (1, 128),
    (64, 128),
    (1, 256),
]


def main():
    print(f"device: {torch.cuda.get_device_name()}\n")
    print(f"  {'shape':>16s}  {'九齿 ms':>10s}  {'torch ms':>10s}  speedup")
    print("  " + "-" * 55)

    for batch, n in CASES:
        A = torch.randn(batch, n, n, dtype=DTYPE, device=DEVICE)

        try:
            with _suppress_stderr():
                ms_nt = triton.testing.do_bench(lambda: ntops.torch.slogdet(A))
        except Exception as exc:
            print(f"  ({batch:4d},{n:3d})  SKIP ninetoothed: {type(exc).__name__}")
            continue

        ms_th = triton.testing.do_bench(lambda: torch.linalg.slogdet(A))

        print(
            f"  ({batch:4d}, {n:3d})  "
            f"{ms_nt:10.3f} ms  {ms_th:10.3f} ms  {ms_th / ms_nt:.2f}x"
        )


if __name__ == "__main__":
    main()
