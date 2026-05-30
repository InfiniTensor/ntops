"""Compare ninetoothed ops against their torch references.

Reports achieved bandwidth (GB/s) and speedup (torch_ms / ninetoothed_ms) at a
couple of large shapes. Auto-tuning is left ON here, so this shows the
*achievable ceiling*; the eval config (max_num_configs=1) comes from the
tune_*.py scripts. Byte models are approximate — the speedup ratio is the
meaningful number.

Scoring is against other ninetoothed entries, not torch: torch winning on the
view-based op (hsplit) is expected, and slogdet (LU factorization) is omitted
because it *is* torch.

    python bench/bench_vs_torch.py
"""

import torch
import torch.nn.functional as F
import triton.testing

import ntops

DEVICE = "cuda"
DTYPE = torch.float32
SHAPES = ((4096, 4096), (8192, 8192))


def _report(name, shape, ms_nt, ms_th, nbytes):
    bw_nt = nbytes / ms_nt * 1e-6  # ms -> s (1e3) and bytes -> GB (1e-9)
    bw_th = nbytes / ms_th * 1e-6
    print(
        f"  {name:15s} {str(shape):14s} "
        f"九齿 {bw_nt:7.0f} GB/s | torch {bw_th:7.0f} GB/s | "
        f"speedup {ms_th / ms_nt:.2f}x"
    )


def _bench(name, shape, nt_fn, th_fn, byte_factor):
    ms_nt = triton.testing.do_bench(nt_fn)
    ms_th = triton.testing.do_bench(th_fn)
    numel = shape[0] * shape[1]
    _report(name, shape, ms_nt, ms_th, byte_factor * numel * torch.tensor([], dtype=DTYPE).element_size())


def main():
    print(f"device: {torch.cuda.get_device_name()}  dtype: {DTYPE}\n")

    for shape in SHAPES:
        x = torch.randn(shape, dtype=DTYPE, device=DEVICE)
        v = torch.randn(shape, dtype=DTYPE, device=DEVICE)

        _bench(
            "heaviside", shape,
            lambda: ntops.torch.heaviside(x, v),
            lambda: torch.heaviside(x, v),
            byte_factor=3,
        )

        end = shape[-1] // 2
        src = x[..., :end].contiguous()
        _bench(
            "slice_scatter", shape,
            lambda: ntops.torch.slice_scatter(x, src, dim=-1, start=0, end=end),
            lambda: torch.slice_scatter(x, src, dim=-1, start=0, end=end),
            byte_factor=2,
        )

        # torch.hsplit returns zero-copy views (≈ no work), so force torch to
        # materialize each split for a fair, apples-to-apples comparison.
        _bench(
            "hsplit", shape,
            lambda: ntops.torch.hsplit(x, 2),
            lambda: tuple(v.contiguous() for v in torch.hsplit(x, 2)),
            byte_factor=2,
        )

        _bench(
            "gumbel_softmax", shape,
            lambda: ntops.torch.gumbel_softmax(x, dim=-1),
            lambda: F.gumbel_softmax(x, dim=-1),
            byte_factor=2,
        )

        print()


if __name__ == "__main__":
    main()
