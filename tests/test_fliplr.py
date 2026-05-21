import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available

# ---------------------------------------------------------------------------
# Correctness tests (compared against torch.fliplr)
# ---------------------------------------------------------------------------

_SHAPES = [
    [4, 6],
    [1, 8],          # single row
    [8, 1],          # single column
    [16, 16],
    [2, 33],         # odd inner dim
    [3, 5, 7],       # 3-D: only dim 1 is flipped
    [2, 3, 4, 5],    # 4-D
]


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("shape", _SHAPES)
def test_fliplr_float(shape, dtype):
    device = "cuda"
    input = torch.randn(shape, dtype=dtype, device=device)

    output = ntops.torch.fliplr(input)
    expected = torch.fliplr(input)

    assert output.shape == expected.shape
    assert torch.equal(output, expected)


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("shape", _SHAPES)
def test_fliplr_int(shape, dtype):
    device = "cuda"
    input = torch.randint(-1000, 1000, shape, dtype=dtype, device=device)

    output = ntops.torch.fliplr(input)
    expected = torch.fliplr(input)

    assert output.shape == expected.shape
    assert torch.equal(output, expected)


@skip_if_cuda_not_available
def test_fliplr_matches_flip_dim1():
    device = "cuda"
    input = torch.randn(4, 6, 8, device=device)

    assert torch.equal(ntops.torch.fliplr(input), ntops.torch.flip(input, (1,)))


@skip_if_cuda_not_available
def test_fliplr_non_contiguous():
    """A transposed (non-contiguous) input must still flip correctly."""
    device = "cuda"
    input = torch.randn(6, 4, 8, device=device).transpose(0, 2)

    output = ntops.torch.fliplr(input)
    expected = torch.fliplr(input)

    assert output.shape == expected.shape
    assert torch.equal(output, expected)


@skip_if_cuda_not_available
def test_fliplr_double_flip_is_identity():
    device = "cuda"
    input = torch.randn(5, 7, device=device)

    assert torch.equal(ntops.torch.fliplr(ntops.torch.fliplr(input)), input)


@skip_if_cuda_not_available
def test_fliplr_1d_raises():
    device = "cuda"
    input = torch.randn(8, device=device)

    with pytest.raises(RuntimeError):
        ntops.torch.fliplr(input)


# ---------------------------------------------------------------------------
# Performance benchmark interface
# ---------------------------------------------------------------------------

def benchmark_fliplr(
    shape,
    dtype=torch.float32,
    device="cuda",
    n_warmup=10,
    n_repeat=100,
):
    """Compare ntops.torch.fliplr vs torch.fliplr.

    Returns timing (ms) and effective memory bandwidth (GB/s) for both, plus
    the speedup ratio. Bandwidth assumes one read of the input plus one write
    of the output (2x input bytes), the lower bound for the op.

    Example
    -------
    >>> results = benchmark_fliplr([4096, 4096])
    >>> print(results)
    """
    if not torch.cuda.is_available() and device == "cuda":
        raise RuntimeError("CUDA not available")

    input = torch.randn(shape, dtype=dtype, device=device)

    for _ in range(n_warmup):
        ntops.torch.fliplr(input)
        torch.fliplr(input)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(n_repeat):
        ntops.torch.fliplr(input)
    end.record()
    torch.cuda.synchronize()
    ntops_ms = start.elapsed_time(end) / n_repeat

    start.record()
    for _ in range(n_repeat):
        torch.fliplr(input)
    end.record()
    torch.cuda.synchronize()
    torch_ms = start.elapsed_time(end) / n_repeat

    num_bytes = input.numel() * input.element_size() * 2
    ntops_gbps = num_bytes / (ntops_ms * 1e-3) / 1e9
    torch_gbps = num_bytes / (torch_ms * 1e-3) / 1e9

    return {
        "shape": shape,
        "dtype": str(dtype),
        "ntops_time_ms": ntops_ms,
        "torch_time_ms": torch_ms,
        "ntops_bandwidth_GBs": ntops_gbps,
        "torch_bandwidth_GBs": torch_gbps,
        "speedup": torch_ms / ntops_ms,
    }


_SWEEP_SHAPES = [
    [1024, 1024],
    [4096, 4096],
    [8192, 8192],
]


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_benchmark_sweep(dtype):
    """Sweep tensor sizes. Run with:
    pytest tests/test_fliplr.py::test_benchmark_sweep -v -s
    """
    header = (
        f"{'shape':>16} {'MB':>8} "
        f"{'ntops(ms)':>11} {'torch(ms)':>11} "
        f"{'ntops(GB/s)':>13} {'torch(GB/s)':>13} {'speedup':>9}"
    )
    print(f"\n{'='*len(header)}")
    print(f"fliplr sweep | dtype={dtype}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for shape in _SWEEP_SHAPES:
        res = benchmark_fliplr(shape, dtype=dtype)
        mb = (
            res["ntops_bandwidth_GBs"] * res["ntops_time_ms"] * 1e-3 * 1e9
        ) / 2 / 1e6
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
    results = benchmark_fliplr([512, 512], n_warmup=2, n_repeat=5)
    assert results["ntops_time_ms"] > 0
    assert results["ntops_bandwidth_GBs"] > 0
