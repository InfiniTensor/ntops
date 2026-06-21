"""Benchmark T1-1-8 operators vs torch.

    kl_div / count_nonzero / narrow / corrcoef / combinations

    python bench/bench_t1_1_8.py
"""

import torch
import torch.nn.functional as F
import triton.testing

import ntops

DEVICE = "cuda"
DTYPE = torch.float32


def _report_bw(name, shape_str, ms_nt, ms_th, nbytes):
    bw_nt = nbytes / ms_nt * 1e-6
    bw_th = nbytes / ms_th * 1e-6
    print(
        f"  {name:16s} {shape_str:24s} "
        f"九齿 {bw_nt:7.0f} GB/s | torch {bw_th:7.0f} GB/s | "
        f"speedup {ms_th / ms_nt:.2f}x"
    )


def _report_ms(name, shape_str, ms_nt, ms_th):
    print(
        f"  {name:16s} {shape_str:24s} "
        f"九齿 {ms_nt:8.3f} ms | torch {ms_th:8.3f} ms | "
        f"speedup {ms_th / ms_nt:.2f}x"
    )


def bench_kl_div():
    print("\n[kl_div]")
    for shape in [(4096, 4096), (8192, 8192), (1024, 8192)]:
        x = F.log_softmax(torch.randn(shape, dtype=DTYPE, device=DEVICE), dim=-1)
        t = F.softmax(torch.randn(shape, dtype=DTYPE, device=DEVICE), dim=-1)
        nbytes = x.numel() * x.element_size() * 2  # 2 reads
        ms_nt = triton.testing.do_bench(
            lambda: ntops.torch.kl_div(x, t, reduction="mean")
        )
        ms_th = triton.testing.do_bench(
            lambda: F.kl_div(x, t, reduction="mean")
        )
        _report_bw("kl_div", str(shape), ms_nt, ms_th, nbytes)


def bench_count_nonzero():
    print("\n[count_nonzero]")
    cases = [((4096, 4096), None), ((8192, 8192), None), ((8192, 8192), 1)]
    for shape, dim in cases:
        x = torch.randint(0, 2, shape, device=DEVICE).to(DTYPE)
        nbytes = x.numel() * x.element_size()  # 1 read
        ms_nt = triton.testing.do_bench(lambda: ntops.torch.count_nonzero(x, dim))
        ms_th = triton.testing.do_bench(lambda: torch.count_nonzero(x, dim))
        _report_bw("count_nonzero", f"{shape} dim={dim}", ms_nt, ms_th, nbytes)


def bench_narrow():
    print("\n[narrow]")
    # (shape, dim, start, length)
    cases = [
        ((8192, 8192), 0, 1024, 4096),
        ((8192, 8192), 1, 1024, 4096),
        ((4096, 4096), 0, 0, 2048),
    ]
    for shape, dim, start, length in cases:
        x = torch.randn(shape, dtype=DTYPE, device=DEVICE)
        out_numel = (length if dim == 0 else shape[0]) * (
            length if dim == 1 else shape[1]
        )
        nbytes = out_numel * x.element_size() * 2  # read slice + write
        ms_nt = triton.testing.do_bench(
            lambda: ntops.torch.narrow(x, dim, start, length)
        )
        # torch.narrow returns a zero-copy view (O(1) metadata, no memory
        # traffic), so comparing our materializing copy against it is apples to
        # oranges -- the "148437 GB/s" it reports is bytes / ~0 time, not real
        # bandwidth. Add .contiguous() so torch also materializes the slice: the
        # fair, same-work comparison (matches benchmark_narrow in test_narrow.py).
        ms_th = triton.testing.do_bench(
            lambda: torch.narrow(x, dim, start, length).contiguous()
        )
        _report_bw("narrow", f"{shape} d={dim} l={length}", ms_nt, ms_th, nbytes)


def bench_corrcoef():
    print("\n[corrcoef]")
    cases = [(64, 4096), (128, 8192), (256, 16384)]
    for m, n in cases:
        x = torch.randn(m, n, dtype=DTYPE, device=DEVICE)
        ms_nt = triton.testing.do_bench(lambda: ntops.torch.corrcoef(x))
        ms_th = triton.testing.do_bench(lambda: torch.corrcoef(x))
        _report_ms("corrcoef", f"({m}, {n})", ms_nt, ms_th)


def bench_combinations():
    print("\n[combinations]")
    for n, r in [(64, 2), (128, 2), (256, 2)]:
        x = torch.randn(n, dtype=DTYPE, device=DEVICE)
        ms_nt = triton.testing.do_bench(
            lambda: ntops.torch.combinations(x, r=r)
        )
        ms_th = triton.testing.do_bench(lambda: torch.combinations(x, r=r))
        _report_ms("combinations", f"n={n} r={r}", ms_nt, ms_th)


def main():
    print(f"device: {torch.cuda.get_device_name()}  dtype: {DTYPE}")
    bench_kl_div()
    bench_count_nonzero()
    bench_narrow()
    bench_corrcoef()
    bench_combinations()


if __name__ == "__main__":
    main()
