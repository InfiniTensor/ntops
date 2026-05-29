import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available

# ---------------------------------------------------------------------------
# Correctness tests (compared against torch.corrcoef). Tolerances are loose
# because the covariance goes through the ninetoothed mm kernel (tf32 by
# default on NVIDIA) and float accumulation differs from torch's.
# ---------------------------------------------------------------------------

_SHAPES = [
    [2, 50],
    [4, 100],
    [8, 256],
    [16, 1024],
    [3, 4097],   # observations not a multiple of the mm tiling
]


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype, rtol, atol", [
    # Loose: the covariance matmul uses tf32 by default on NVIDIA (~1e-3 rel).
    (torch.float32, 1e-2, 1e-2),
])
@pytest.mark.parametrize("shape", _SHAPES)
def test_corrcoef_2d(shape, dtype, rtol, atol):
    device = "cuda"
    input = torch.randn(shape, dtype=dtype, device=device)

    output = ntops.torch.corrcoef(input)
    expected = torch.corrcoef(input)

    assert output.shape == expected.shape  # (D, D)
    assert output.dtype == expected.dtype
    assert torch.allclose(output, expected, rtol=rtol, atol=atol)


@skip_if_cuda_not_available
def test_corrcoef_1d():
    """A 1-D input yields the scalar 1.0."""
    device = "cuda"
    input = torch.randn(100, device=device)

    output = ntops.torch.corrcoef(input)
    expected = torch.corrcoef(input)

    assert output.shape == expected.shape  # ()
    assert torch.allclose(output, expected, rtol=1e-3, atol=1e-3)


@skip_if_cuda_not_available
def test_corrcoef_diagonal_is_one():
    device = "cuda"
    input = torch.randn(6, 200, device=device)

    output = ntops.torch.corrcoef(input)

    assert torch.allclose(
        output.diagonal(), torch.ones(6, device=device), rtol=1e-3, atol=1e-3
    )
    assert output.abs().max().item() <= 1.0 + 1e-5  # clamped to [-1, 1]


@skip_if_cuda_not_available
def test_corrcoef_integer_input_promotes():
    device = "cuda"
    input = torch.randint(0, 10, (4, 100), device=device)

    output = ntops.torch.corrcoef(input)
    expected = torch.corrcoef(input)

    assert output.dtype == expected.dtype  # float32
    assert torch.allclose(output, expected, rtol=1e-2, atol=1e-2)


@skip_if_cuda_not_available
def test_corrcoef_too_many_dims():
    device = "cuda"
    input = torch.randn(2, 3, 4, device=device)

    with pytest.raises(RuntimeError):
        ntops.torch.corrcoef(input)


# ---------------------------------------------------------------------------
# Performance benchmark interface
# ---------------------------------------------------------------------------

def benchmark_corrcoef(
    shape,
    dtype=torch.float32,
    device="cuda",
    n_warmup=10,
    n_repeat=100,
):
    """Compare ntops.torch.corrcoef vs torch.corrcoef.

    Example
    -------
    >>> results = benchmark_corrcoef([64, 4096])
    """
    if not torch.cuda.is_available() and device == "cuda":
        raise RuntimeError("CUDA not available")

    input = torch.randn(shape, dtype=dtype, device=device)

    for _ in range(n_warmup):
        ntops.torch.corrcoef(input)
        torch.corrcoef(input)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(n_repeat):
        ntops.torch.corrcoef(input)
    end.record()
    torch.cuda.synchronize()
    ntops_ms = start.elapsed_time(end) / n_repeat

    start.record()
    for _ in range(n_repeat):
        torch.corrcoef(input)
    end.record()
    torch.cuda.synchronize()
    torch_ms = start.elapsed_time(end) / n_repeat

    return {
        "shape": shape,
        "dtype": str(dtype),
        "ntops_time_ms": ntops_ms,
        "torch_time_ms": torch_ms,
        "speedup": torch_ms / ntops_ms,
    }


_SWEEP_SHAPES = [[32, 4096], [128, 8192], [512, 8192]]


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float32])
def test_benchmark_sweep(dtype):
    """Sweep sizes. Run with:
    pytest tests/test_corrcoef.py::test_benchmark_sweep -v -s
    """
    header = (
        f"{'shape':>16} {'ntops(ms)':>11} {'torch(ms)':>11} {'speedup':>9}"
    )
    print(f"\n{'='*len(header)}")
    print(f"corrcoef sweep | dtype={dtype}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for shape in _SWEEP_SHAPES:
        res = benchmark_corrcoef(shape, dtype=dtype)
        print(
            f"{str(shape):>16} {res['ntops_time_ms']:>11.4f} "
            f"{res['torch_time_ms']:>11.4f} {res['speedup']:>9.2f}"
        )

    print("=" * len(header))


@skip_if_cuda_not_available
def test_benchmark_interface():
    results = benchmark_corrcoef([32, 512], n_warmup=2, n_repeat=5)
    assert results["ntops_time_ms"] > 0
