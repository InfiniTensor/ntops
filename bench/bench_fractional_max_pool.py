"""Benchmark fractional_max_pool2d/3d (method B) vs torch.

    python bench/bench_fractional_max_pool.py
"""

import torch
import torch.nn.functional as F
import triton.testing

import ntops

DEVICE = "cuda"
DTYPE = torch.float32


def _report(name, shape_str, ms_nt, ms_th):
    print(f"  {name:10s} {shape_str:28s} 九齿 {ms_nt:8.3f} ms | torch {ms_th:8.3f} ms | speedup {ms_th / ms_nt:.2f}x")


def main():
    print(f"device: {torch.cuda.get_device_name()}  dtype: {DTYPE}\n")

    print("[fractional_max_pool2d]")
    cases_2d = [
        (8, 16, 16, 16, 2, 2, 12, 12),
        (8, 32, 32, 32, 2, 2, 24, 24),
        (16, 64, 56, 56, 3, 3, 40, 40),
        (32, 64, 112, 112, 2, 2, 80, 80),
    ]
    for n, c, h, w, kh, kw, oh, ow in cases_2d:
        x = torch.randn(n, c, h, w, dtype=DTYPE, device=DEVICE)
        s = torch.rand(n, c, 2, dtype=DTYPE, device=DEVICE)
        ms_nt = triton.testing.do_bench(
            lambda: ntops.torch.fractional_max_pool2d(x, (kh, kw), output_size=(oh, ow), _random_samples=s)
        )
        ms_th = triton.testing.do_bench(
            lambda: F.fractional_max_pool2d(x, (kh, kw), output_size=(oh, ow), _random_samples=s)
        )
        _report("fmp2d", f"({n},{c},{h},{w})->({oh},{ow})", ms_nt, ms_th)

    print("\n[fractional_max_pool3d]")
    cases_3d = [
        (4, 16, 16, 16, 16, 2, 2, 2, 12, 12, 12),
        (8, 32, 16, 32, 32, 2, 2, 2, 12, 24, 24),
    ]
    for n, c, d, h, w, kd, kh, kw, od, oh, ow in cases_3d:
        x = torch.randn(n, c, d, h, w, dtype=DTYPE, device=DEVICE)
        s = torch.rand(n, c, 3, dtype=DTYPE, device=DEVICE)
        ms_nt = triton.testing.do_bench(
            lambda: ntops.torch.fractional_max_pool3d(x, (kd, kh, kw), output_size=(od, oh, ow), _random_samples=s)
        )
        ms_th = triton.testing.do_bench(
            lambda: F.fractional_max_pool3d(x, (kd, kh, kw), output_size=(od, oh, ow), _random_samples=s)
        )
        _report("fmp3d", f"({n},{c},{d},{h},{w})->({od},{oh},{ow})", ms_nt, ms_th)


if __name__ == "__main__":
    main()
