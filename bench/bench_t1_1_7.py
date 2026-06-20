"""Benchmark T1-1-7 operators vs torch.

    feature_alpha_dropout / mse_loss / flip / fliplr / pixel_unshuffle

    python bench/bench_t1_1_7.py
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
        f"  {name:22s} {shape_str:22s} "
        f"九齿 {bw_nt:7.0f} GB/s | torch {bw_th:7.0f} GB/s | "
        f"speedup {ms_th / ms_nt:.2f}x"
    )


def bench_feature_alpha_dropout():
    print("\n[feature_alpha_dropout]")
    for shape in [(64, 256, 32, 32), (128, 512, 16, 16), (32, 256, 64, 64)]:
        x = torch.randn(shape, dtype=DTYPE, device=DEVICE)
        nbytes = x.numel() * x.element_size() * 2
        ms_nt = triton.testing.do_bench(
            lambda: ntops.torch.feature_alpha_dropout(x, p=0.5, training=True)
        )
        ms_th = triton.testing.do_bench(
            lambda: F.feature_alpha_dropout(x, p=0.5, training=True)
        )
        _report("feature_alpha_dropout", str(shape), ms_nt, ms_th, nbytes)


def bench_mse_loss():
    print("\n[mse_loss]")
    for shape in [(4096, 4096), (8192, 8192), (4096 * 4096,)]:
        x = torch.randn(shape, dtype=DTYPE, device=DEVICE)
        t = torch.randn(shape, dtype=DTYPE, device=DEVICE)
        nbytes = x.numel() * x.element_size() * 2  # 2 reads
        ms_nt = triton.testing.do_bench(
            lambda: ntops.torch.mse_loss(x, t, reduction="mean")
        )
        ms_th = triton.testing.do_bench(
            lambda: F.mse_loss(x, t, reduction="mean")
        )
        _report("mse_loss", str(shape), ms_nt, ms_th, nbytes)


def bench_flip():
    print("\n[flip]")
    cases = [((4096, 4096), (0,)), ((4096, 4096), (1,)), ((8192, 8192), (0, 1))]
    for shape, dims in cases:
        x = torch.randn(shape, dtype=DTYPE, device=DEVICE)
        nbytes = x.numel() * x.element_size() * 2  # 1 read + 1 write
        ms_nt = triton.testing.do_bench(lambda: ntops.torch.flip(x, dims))
        ms_th = triton.testing.do_bench(lambda: torch.flip(x, dims))
        _report("flip", f"{shape} dims={dims}", ms_nt, ms_th, nbytes)


def bench_fliplr():
    print("\n[fliplr]")
    for shape in [(4096, 4096), (8192, 8192)]:
        x = torch.randn(shape, dtype=DTYPE, device=DEVICE)
        nbytes = x.numel() * x.element_size() * 2
        ms_nt = triton.testing.do_bench(lambda: ntops.torch.fliplr(x))
        ms_th = triton.testing.do_bench(lambda: torch.fliplr(x))
        _report("fliplr", str(shape), ms_nt, ms_th, nbytes)


def bench_pixel_unshuffle():
    print("\n[pixel_unshuffle]")
    cases = [((32, 64, 112, 112), 2), ((16, 128, 128, 128), 4), ((64, 64, 64, 64), 2)]
    for shape, r in cases:
        x = torch.randn(shape, dtype=DTYPE, device=DEVICE)
        nbytes = x.numel() * x.element_size() * 2
        ms_nt = triton.testing.do_bench(
            lambda: ntops.torch.pixel_unshuffle(x, r)
        )
        ms_th = triton.testing.do_bench(lambda: F.pixel_unshuffle(x, r))
        _report("pixel_unshuffle", f"{shape} r={r}", ms_nt, ms_th, nbytes)


def main():
    print(f"device: {torch.cuda.get_device_name()}  dtype: {DTYPE}")
    bench_feature_alpha_dropout()
    bench_mse_loss()
    bench_flip()
    bench_fliplr()
    bench_pixel_unshuffle()


if __name__ == "__main__":
    main()
