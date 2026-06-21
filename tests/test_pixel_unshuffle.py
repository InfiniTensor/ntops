import pytest
import torch
import torch.nn.functional as F

import ntops
from tests.skippers import skip_if_cuda_not_available

# ---------------------------------------------------------------------------
# Correctness tests (compared against torch.nn.functional.pixel_unshuffle)
# ---------------------------------------------------------------------------

_FLOAT_CASES = [
    # (shape, downscale_factor)
    ([1, 1, 4, 4], 2),
    ([2, 3, 8, 8], 2),
    ([4, 8, 12, 12], 3),
    ([2, 16, 16, 16], 4),
    ([8, 4, 32, 32], 2),
    ([3, 5, 6, 9], 3),       # H != W
    ([1, 1, 2, 2], 2),       # minimal
]


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("shape, r", _FLOAT_CASES)
def test_pixel_unshuffle_matches_torch(shape, r, dtype):
    device = "cuda"
    input = torch.randn(shape, dtype=dtype, device=device)

    output = ntops.torch.pixel_unshuffle(input, r)
    expected = F.pixel_unshuffle(input, r)

    assert output.shape == expected.shape
    assert torch.equal(output, expected)


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("shape, r", _FLOAT_CASES)
def test_pixel_unshuffle_int(shape, r, dtype):
    device = "cuda"
    input = torch.randint(-1000, 1000, shape, dtype=dtype, device=device)

    output = ntops.torch.pixel_unshuffle(input, r)
    expected = F.pixel_unshuffle(input, r)

    assert output.shape == expected.shape
    assert torch.equal(output, expected)


@skip_if_cuda_not_available
def test_pixel_unshuffle_leading_dims():
    """torch supports arbitrary leading dims: (*, C, H, W)."""
    device = "cuda"
    input = torch.randn(2, 3, 4, 16, 16, device=device)

    output = ntops.torch.pixel_unshuffle(input, 4)
    expected = F.pixel_unshuffle(input, 4)

    assert output.shape == expected.shape
    assert torch.equal(output, expected)


@skip_if_cuda_not_available
def test_pixel_unshuffle_roundtrip():
    """pixel_shuffle(pixel_unshuffle(x)) == x."""
    device = "cuda"
    r = 2
    input = torch.randn(2, 3, 8, 8, device=device)

    unshuffled = ntops.torch.pixel_unshuffle(input, r)
    restored = F.pixel_shuffle(unshuffled, r)

    assert torch.equal(restored, input)


# ---------------------------------------------------------------------------
# Performance benchmark interface
# ---------------------------------------------------------------------------

def benchmark_pixel_unshuffle(
    shape,
    downscale_factor=2,
    dtype=torch.float32,
    device="cuda",
    n_warmup=10,
    n_repeat=100,
):
    """Compare ntops.torch.pixel_unshuffle vs F.pixel_unshuffle.

    Returns timing (ms) and effective memory bandwidth (GB/s) for both,
    plus the speedup ratio.

    Example
    -------
    >>> results = benchmark_pixel_unshuffle([32, 64, 128, 128], 2)
    >>> print(results)
    """
    if not torch.cuda.is_available() and device == "cuda":
        raise RuntimeError("CUDA not available")

    r = downscale_factor
    input_tensor = torch.randn(shape, dtype=dtype, device=device)

    for _ in range(n_warmup):
        ntops.torch.pixel_unshuffle(input_tensor, r)
        F.pixel_unshuffle(input_tensor, r)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(n_repeat):
        ntops.torch.pixel_unshuffle(input_tensor, r)
    end.record()
    torch.cuda.synchronize()
    ntops_ms = start.elapsed_time(end) / n_repeat

    start.record()
    for _ in range(n_repeat):
        F.pixel_unshuffle(input_tensor, r)
    end.record()
    torch.cuda.synchronize()
    torch_ms = start.elapsed_time(end) / n_repeat

    # read input + write output == 2x input size
    num_bytes = input_tensor.numel() * input_tensor.element_size() * 2
    ntops_gbps = num_bytes / (ntops_ms * 1e-3) / 1e9
    torch_gbps = num_bytes / (torch_ms * 1e-3) / 1e9

    return {
        "shape": shape,
        "downscale_factor": r,
        "dtype": str(dtype),
        "ntops_time_ms": ntops_ms,
        "torch_time_ms": torch_ms,
        "ntops_bandwidth_GBs": ntops_gbps,
        "torch_bandwidth_GBs": torch_gbps,
        "speedup": torch_ms / ntops_ms,
    }


_SWEEP_SHAPES = [
    ([4, 16, 32, 32], 2),     # 0.25 MB
    ([8, 64, 64, 64], 2),     # 16 MB
    ([16, 128, 64, 64], 2),   # 64 MB
    ([32, 256, 64, 64], 2),   # 256 MB
]


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_benchmark_sweep(dtype):
    """Sweep tensor sizes. Run with: pytest tests/test_pixel_unshuffle.py::test_benchmark_sweep -v -s"""
    header = (
        f"{'shape':>22} {'r':>3} {'MB':>8} "
        f"{'ntops(ms)':>11} {'torch(ms)':>11} "
        f"{'ntops(GB/s)':>13} {'torch(GB/s)':>13} {'speedup':>9}"
    )
    print(f"\n{'='*len(header)}")
    print(f"pixel_unshuffle sweep | dtype={dtype}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for shape, r in _SWEEP_SHAPES:
        res = benchmark_pixel_unshuffle(shape, downscale_factor=r, dtype=dtype)
        mb = (
            res["ntops_bandwidth_GBs"] * res["ntops_time_ms"] * 1e-3 * 1e9
        ) / 2 / 1e6
        print(
            f"{str(shape):>22} {r:>3} {mb:>8.1f} "
            f"{res['ntops_time_ms']:>11.4f} {res['torch_time_ms']:>11.4f} "
            f"{res['ntops_bandwidth_GBs']:>13.1f} {res['torch_bandwidth_GBs']:>13.1f} "
            f"{res['speedup']:>9.2f}"
        )

    print("=" * len(header))


@skip_if_cuda_not_available
def test_benchmark_interface():
    """Smoke-test that benchmark interface runs without error."""
    results = benchmark_pixel_unshuffle(
        [8, 32, 32, 32], downscale_factor=2, n_warmup=2, n_repeat=5
    )
    assert results["ntops_time_ms"] > 0
    assert results["ntops_bandwidth_GBs"] > 0
