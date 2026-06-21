import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available

# ---------------------------------------------------------------------------
# Correctness tests (compared against torch.count_nonzero). The count is exact,
# so torch.equal; output is always int64.
# ---------------------------------------------------------------------------

_SHAPES = [
    [16],
    [1024],
    [4097],          # not a multiple of the reduction block size
    [32, 64],
    [8, 7, 5],
    [4, 3, 16, 16],
]


def _make_input(shape, dtype, device):
    """A tensor with a healthy mix of zeros and nonzeros."""
    if dtype == torch.int64:
        x = torch.randint(-2, 3, shape, dtype=dtype, device=device)
    else:
        x = torch.randn(shape, dtype=dtype, device=device)
        x = torch.where(x.abs() < 0.5, torch.zeros_like(x), x)
    return x


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int64])
@pytest.mark.parametrize("shape", _SHAPES)
def test_count_nonzero_global(shape, dtype):
    device = "cuda"
    input = _make_input(shape, dtype, device)

    output = ntops.torch.count_nonzero(input)
    expected = torch.count_nonzero(input)

    assert output.shape == expected.shape  # scalar ()
    assert output.dtype == expected.dtype  # int64
    assert torch.equal(output, expected)


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
@pytest.mark.parametrize("shape, dim", [
    ([32, 64], 0),
    ([32, 64], 1),
    ([32, 64], -1),
    ([8, 7, 5], 0),
    ([8, 7, 5], 1),
    ([8, 7, 5], 2),
    ([8, 7, 5], (0, 1)),
    ([8, 7, 5], (1, 2)),
    ([8, 7, 5], (0, 1, 2)),
    ([4, 3, 16, 16], (2, 3)),
])
def test_count_nonzero_dim(shape, dim, dtype):
    device = "cuda"
    input = _make_input(shape, dtype, device)

    output = ntops.torch.count_nonzero(input, dim=dim)
    expected = torch.count_nonzero(input, dim=dim)

    assert output.shape == expected.shape
    assert output.dtype == expected.dtype
    assert torch.equal(output, expected)


@skip_if_cuda_not_available
def test_count_nonzero_all_zeros_and_all_nonzeros():
    device = "cuda"
    zeros = torch.zeros(100, device=device)
    ones = torch.ones(100, device=device)

    assert ntops.torch.count_nonzero(zeros).item() == 0
    assert ntops.torch.count_nonzero(ones).item() == 100


@skip_if_cuda_not_available
def test_count_nonzero_empty():
    device = "cuda"
    input = torch.randn(0, device=device)

    output = ntops.torch.count_nonzero(input)
    expected = torch.count_nonzero(input)

    assert torch.equal(output, expected)


# ---------------------------------------------------------------------------
# Performance benchmark interface
# ---------------------------------------------------------------------------

def benchmark_count_nonzero(
    shape,
    dim=None,
    dtype=torch.float32,
    device="cuda",
    n_warmup=10,
    n_repeat=100,
):
    """Compare ntops.torch.count_nonzero vs torch.count_nonzero. Bandwidth
    assumes the input is read once.

    Example
    -------
    >>> results = benchmark_count_nonzero([4096, 4096])
    """
    if not torch.cuda.is_available() and device == "cuda":
        raise RuntimeError("CUDA not available")

    input = torch.randn(shape, dtype=dtype, device=device)
    input = torch.where(input.abs() < 0.5, torch.zeros_like(input), input)

    def run_ntops():
        ntops.torch.count_nonzero(input, dim=dim)

    def run_torch():
        torch.count_nonzero(input, dim=dim)

    for _ in range(n_warmup):
        run_ntops()
        run_torch()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(n_repeat):
        run_ntops()
    end.record()
    torch.cuda.synchronize()
    ntops_ms = start.elapsed_time(end) / n_repeat

    start.record()
    for _ in range(n_repeat):
        run_torch()
    end.record()
    torch.cuda.synchronize()
    torch_ms = start.elapsed_time(end) / n_repeat

    num_bytes = input.numel() * input.element_size()

    return {
        "shape": shape,
        "dim": dim,
        "dtype": str(dtype),
        "ntops_time_ms": ntops_ms,
        "torch_time_ms": torch_ms,
        "ntops_bandwidth_GBs": num_bytes / (ntops_ms * 1e-3) / 1e9,
        "torch_bandwidth_GBs": num_bytes / (torch_ms * 1e-3) / 1e9,
        "speedup": torch_ms / ntops_ms,
    }


_SWEEP_SHAPES = [[1024, 1024], [4096, 4096], [8192, 8192]]


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("dim", [None, 0, 1])
def test_benchmark_sweep(dim, dtype):
    """Sweep sizes. Run with:
    pytest tests/test_count_nonzero.py::test_benchmark_sweep -v -s
    """
    header = (
        f"{'shape':>16} {'dim':>6} "
        f"{'ntops(ms)':>11} {'torch(ms)':>11} "
        f"{'ntops(GB/s)':>13} {'torch(GB/s)':>13} {'speedup':>9}"
    )
    print(f"\n{'='*len(header)}")
    print(f"count_nonzero sweep | dim={dim} | dtype={dtype}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for shape in _SWEEP_SHAPES:
        res = benchmark_count_nonzero(shape, dim=dim, dtype=dtype)
        print(
            f"{str(shape):>16} {str(dim):>6} "
            f"{res['ntops_time_ms']:>11.4f} {res['torch_time_ms']:>11.4f} "
            f"{res['ntops_bandwidth_GBs']:>13.1f} {res['torch_bandwidth_GBs']:>13.1f} "
            f"{res['speedup']:>9.2f}"
        )

    print("=" * len(header))


@skip_if_cuda_not_available
def test_benchmark_interface():
    results = benchmark_count_nonzero([512, 512], n_warmup=2, n_repeat=5)
    assert results["ntops_time_ms"] > 0
