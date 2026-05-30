"""Benchmark frac, scatter_add, multilabel_margin_loss vs torch.

    python bench/bench_frac_scatter_mlml.py
"""

import torch
import torch.nn.functional as F
import triton.testing

import ntops

DEVICE = "cuda"
DTYPE = torch.float32


def _report(name, shape_str, ms_nt, ms_th, nbytes):
    bw_nt = nbytes / ms_nt * 1e-6
    bw_th = nbytes / ms_th * 1e-6
    print(
        f"  {name:22s} {shape_str:18s} "
        f"九齿 {bw_nt:7.0f} GB/s | torch {bw_th:7.0f} GB/s | "
        f"speedup {ms_th / ms_nt:.2f}x"
    )


def bench_frac():
    print("\n[frac]")
    for shape in [(4096 * 4096,), (4096, 4096), (8192, 8192)]:
        x = torch.randn(shape, dtype=DTYPE, device=DEVICE) * 5
        nbytes = x.numel() * x.element_size() * 2  # 1 read + 1 write
        ms_nt = triton.testing.do_bench(lambda: ntops.torch.frac(x))
        ms_th = triton.testing.do_bench(lambda: torch.frac(x))
        _report("frac", str(shape), ms_nt, ms_th, nbytes)


def bench_scatter_add():
    print("\n[scatter_add]")
    # (input_shape, dim, k)
    cases = [
        ((1024, 256), 1, 128),
        ((1024, 256), 1, 256),
        ((4096, 512), 1, 256),
        ((8192, 256), 0, 4096),
    ]
    for (shape, dim, k) in cases:
        inp = torch.randn(shape, dtype=DTYPE, device=DEVICE)
        idx_shape = list(shape); idx_shape[dim] = k
        t = shape[dim]
        idx = torch.randint(0, t, idx_shape, device=DEVICE)
        src = torch.randn(idx_shape, dtype=DTYPE, device=DEVICE)
        nbytes = (inp.numel() + src.numel()) * inp.element_size() * 2
        label = f"shape={shape} dim={dim} k={k}"
        try:
            ms_nt = triton.testing.do_bench(
                lambda: ntops.torch.scatter_add(inp, dim, idx, src)
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  {'scatter_add':22s} {label:18s} 九齿 SKIP ({type(exc).__name__})")
            continue
        ms_th = triton.testing.do_bench(
            lambda: torch.scatter_add(inp, dim, idx, src)
        )
        _report("scatter_add", label, ms_nt, ms_th, nbytes)


def bench_mlml():
    print("\n[multilabel_margin_loss]")
    cases = [(64, 16), (256, 32), (512, 64), (1024, 32)]
    for (n, c) in cases:
        x = torch.randn(n, c, dtype=DTYPE, device=DEVICE)
        target = torch.full((n, c), -1, dtype=torch.int64, device=DEVICE)
        for i in range(n):
            num = torch.randint(1, c // 2 + 1, (1,)).item()
            target[i, :num] = torch.randperm(c, device=DEVICE)[:num]
        nbytes = x.numel() * x.element_size() * 2
        ms_nt = triton.testing.do_bench(
            lambda: ntops.torch.multilabel_margin_loss(x, target, reduction="mean")
        )
        ms_th = triton.testing.do_bench(
            lambda: F.multilabel_margin_loss(x, target, reduction="mean")
        )
        _report("mlml", f"N={n} C={c}", ms_nt, ms_th, nbytes)


def main():
    print(f"device: {torch.cuda.get_device_name()}  dtype: {DTYPE}")
    bench_frac()
    bench_scatter_add()
    bench_mlml()


if __name__ == "__main__":
    main()
