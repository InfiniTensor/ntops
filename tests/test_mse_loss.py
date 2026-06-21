import pytest
import torch
import torch.nn.functional as F

import ntops
from tests.skippers import skip_if_cuda_not_available

# ---------------------------------------------------------------------------
# Correctness tests (compared against torch.nn.functional.mse_loss)
# ---------------------------------------------------------------------------

_SHAPES = [
    [16],
    [1024],
    [4097],          # not a multiple of the reduction block size
    [32, 64],
    [8, 7, 5],
    [4, 3, 16, 16],
    [1],             # single element
]


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype, rtol, atol", [
    (torch.float32, 1e-3, 1e-3),
    (torch.float16, 1e-2, 1e-2),
])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("shape", _SHAPES)
def test_mse_loss(shape, reduction, dtype, rtol, atol):
    device = "cuda"
    input = torch.randn(shape, dtype=dtype, device=device)
    target = torch.randn(shape, dtype=dtype, device=device)

    output = ntops.torch.mse_loss(input, target, reduction=reduction)
    expected = F.mse_loss(input, target, reduction=reduction)

    assert output.shape == expected.shape
    assert output.dtype == expected.dtype
    assert torch.allclose(output, expected, rtol=rtol, atol=atol)


@skip_if_cuda_not_available
def test_mse_loss_default_reduction_is_mean():
    device = "cuda"
    input = torch.randn(2, 3, 4, device=device)
    target = torch.randn(2, 3, 4, device=device)

    output = ntops.torch.mse_loss(input, target)
    expected = F.mse_loss(input, target)

    assert torch.allclose(output, expected, rtol=1e-3, atol=1e-3)


@skip_if_cuda_not_available
def test_mse_loss_broadcast():
    device = "cuda"
    input = torch.randn(4, 3, 8, device=device)
    target = torch.randn(3, 8, device=device)

    output = ntops.torch.mse_loss(input, target, reduction="sum")
    expected = F.mse_loss(
        input, target.expand_as(input).contiguous(), reduction="sum"
    )

    assert torch.allclose(output, expected, rtol=1e-3, atol=1e-3)


@skip_if_cuda_not_available
def test_mse_loss_invalid_reduction():
    device = "cuda"
    input = torch.randn(8, device=device)
    target = torch.randn(8, device=device)

    with pytest.raises(ValueError):
        ntops.torch.mse_loss(input, target, reduction="median")


# ---------------------------------------------------------------------------
# Performance benchmark interface
# ---------------------------------------------------------------------------

def benchmark_mse_loss(
    shape,
    reduction="mean",
    dtype=torch.float32,
    device="cuda",
    n_warmup=10,
    n_repeat=100,
):
    """Compare ntops.torch.mse_loss vs F.mse_loss.

    Returns timing (ms) and effective memory bandwidth (GB/s) for both,
    plus the speedup ratio. Bandwidth assumes both ``input`` and ``target``
    are read once (2x input bytes), which is the lower bound for the op.

    Example
    -------
    >>> results = benchmark_mse_loss([4096, 4096], "mean")
    >>> print(results)
    """
    if not torch.cuda.is_available() and device == "cuda":
        raise RuntimeError("CUDA not available")

    input = torch.randn(shape, dtype=dtype, device=device)
    target = torch.randn(shape, dtype=dtype, device=device)

    for _ in range(n_warmup):
        ntops.torch.mse_loss(input, target, reduction=reduction)
        F.mse_loss(input, target, reduction=reduction)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(n_repeat):
        ntops.torch.mse_loss(input, target, reduction=reduction)
    end.record()
    torch.cuda.synchronize()
    ntops_ms = start.elapsed_time(end) / n_repeat

    start.record()
    for _ in range(n_repeat):
        F.mse_loss(input, target, reduction=reduction)
    end.record()
    torch.cuda.synchronize()
    torch_ms = start.elapsed_time(end) / n_repeat

    num_bytes = input.numel() * input.element_size() * 2
    ntops_gbps = num_bytes / (ntops_ms * 1e-3) / 1e9
    torch_gbps = num_bytes / (torch_ms * 1e-3) / 1e9

    return {
        "shape": shape,
        "reduction": reduction,
        "dtype": str(dtype),
        "ntops_time_ms": ntops_ms,
        "torch_time_ms": torch_ms,
        "ntops_bandwidth_GBs": ntops_gbps,
        "torch_bandwidth_GBs": torch_gbps,
        "speedup": torch_ms / ntops_ms,
    }


_SWEEP_SHAPES = [
    [1024, 1024],      # 4 MB
    [4096, 4096],      # 64 MB
    [8192, 8192],      # 256 MB
]


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_benchmark_sweep(reduction, dtype):
    """Sweep tensor sizes. Run with:
    pytest tests/test_mse_loss.py::test_benchmark_sweep -v -s
    """
    header = (
        f"{'shape':>16} {'MB':>8} "
        f"{'ntops(ms)':>11} {'torch(ms)':>11} "
        f"{'ntops(GB/s)':>13} {'torch(GB/s)':>13} {'speedup':>9}"
    )
    print(f"\n{'='*len(header)}")
    print(f"mse_loss sweep | reduction={reduction} | dtype={dtype}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for shape in _SWEEP_SHAPES:
        res = benchmark_mse_loss(shape, reduction=reduction, dtype=dtype)
        mb = (
            torch.empty(shape, dtype=dtype).numel()
            * torch.empty(0, dtype=dtype).element_size()
        ) / 1e6
        print(
            f"{str(shape):>16} {mb:>8.1f} "
            f"{res['ntops_time_ms']:>11.4f} {res['torch_time_ms']:>11.4f} "
            f"{res['ntops_bandwidth_GBs']:>13.1f} {res['torch_bandwidth_GBs']:>13.1f} "
            f"{res['speedup']:>9.2f}"
        )

    print("=" * len(header))


@skip_if_cuda_not_available
def test_benchmark_interface():
    """Smoke-test that the benchmark interface runs without error."""
    results = benchmark_mse_loss([512, 512], n_warmup=2, n_repeat=5)
    assert results["ntops_time_ms"] > 0
    assert results["ntops_bandwidth_GBs"] > 0
