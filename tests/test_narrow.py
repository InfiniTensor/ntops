import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available

# ---------------------------------------------------------------------------
# Correctness tests (compared against torch.narrow). narrow is a pure copy of a
# strided slice, so the output must match bit-for-bit -- torch.equal.
# ---------------------------------------------------------------------------

_CASES = [
    # (shape, dim, start, length)
    ([16], 0, 3, 8),
    ([16], 0, 0, 16),       # full
    ([16], -1, 2, 4),       # negative dim
    ([16], 0, -5, 3),       # negative start
    ([8, 7], 0, 1, 4),
    ([8, 7], 1, 2, 3),
    ([8, 7], -1, -4, 2),
    ([4, 5, 6], 1, 1, 3),
    ([4, 5, 6], 2, 0, 6),
    ([4, 5, 6], 0, 2, 0),   # zero length
    ([3, 4097], 1, 5, 4090),  # not a multiple of the copy block size
]


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int64])
@pytest.mark.parametrize("shape, dim, start, length", _CASES)
def test_narrow(shape, dim, start, length, dtype):
    device = "cuda"
    if dtype == torch.int64:
        input = torch.randint(-1000, 1000, shape, dtype=dtype, device=device)
    else:
        input = torch.randn(shape, dtype=dtype, device=device)

    output = ntops.torch.narrow(input, dim, start, length)
    expected = torch.narrow(input, dim, start, length)

    assert output.shape == expected.shape
    assert output.dtype == expected.dtype
    assert torch.equal(output, expected)


@skip_if_cuda_not_available
def test_narrow_tensor_start():
    """torch accepts a 0-dim tensor start."""
    device = "cuda"
    input = torch.randn(10, device=device)

    output = ntops.torch.narrow(input, 0, torch.tensor(3), 4)
    expected = torch.narrow(input, 0, 3, 4)

    assert torch.equal(output, expected)


@skip_if_cuda_not_available
def test_narrow_out_of_range_dim():
    device = "cuda"
    input = torch.randn(4, 5, device=device)

    with pytest.raises(IndexError):
        ntops.torch.narrow(input, 2, 0, 1)


@skip_if_cuda_not_available
def test_narrow_length_too_large():
    device = "cuda"
    input = torch.randn(8, device=device)

    with pytest.raises(RuntimeError):
        ntops.torch.narrow(input, 0, 5, 10)


# ---------------------------------------------------------------------------
# Performance benchmark interface
# ---------------------------------------------------------------------------

def benchmark_narrow(
    shape,
    dim=0,
    fraction=0.5,
    dtype=torch.float32,
    device="cuda",
    n_warmup=10,
    n_repeat=100,
):
    """Compare ntops.torch.narrow (materializing copy) vs torch.narrow (view) +
    .contiguous(). Bandwidth assumes the slice is read and written once.

    Example
    -------
    >>> results = benchmark_narrow([4096, 4096], dim=1)
    """
    if not torch.cuda.is_available() and device == "cuda":
        raise RuntimeError("CUDA not available")

    input = torch.randn(shape, dtype=dtype, device=device)
    length = max(1, int(shape[dim] * fraction))

    def run_ntops():
        ntops.torch.narrow(input, dim, 0, length)

    def run_torch():
        torch.narrow(input, dim, 0, length).contiguous()

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

    out = ntops.torch.narrow(input, dim, 0, length)
    num_bytes = out.numel() * out.element_size() * 2

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
@pytest.mark.parametrize("dim", [0, 1])
def test_benchmark_sweep(dim, dtype):
    """Sweep sizes. Run with:
    pytest tests/test_narrow.py::test_benchmark_sweep -v -s
    """
    header = (
        f"{'shape':>16} {'dim':>4} "
        f"{'ntops(ms)':>11} {'torch(ms)':>11} "
        f"{'ntops(GB/s)':>13} {'torch(GB/s)':>13} {'speedup':>9}"
    )
    print(f"\n{'='*len(header)}")
    print(f"narrow sweep | dim={dim} | dtype={dtype}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for shape in _SWEEP_SHAPES:
        res = benchmark_narrow(shape, dim=dim, dtype=dtype)
        print(
            f"{str(shape):>16} {dim:>4} "
            f"{res['ntops_time_ms']:>11.4f} {res['torch_time_ms']:>11.4f} "
            f"{res['ntops_bandwidth_GBs']:>13.1f} {res['torch_bandwidth_GBs']:>13.1f} "
            f"{res['speedup']:>9.2f}"
        )

    print("=" * len(header))


@skip_if_cuda_not_available
def test_benchmark_interface():
    results = benchmark_narrow([512, 512], n_warmup=2, n_repeat=5)
    assert results["ntops_time_ms"] > 0
