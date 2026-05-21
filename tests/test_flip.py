import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available

# ---------------------------------------------------------------------------
# Correctness tests (compared against torch.flip)
# ---------------------------------------------------------------------------

_CASES = [
    # (shape, dims)
    ([8], [0]),
    ([8], [-1]),
    ([16], []),              # no-op flip
    ([4, 6], [0]),
    ([4, 6], [1]),
    ([4, 6], [0, 1]),
    ([4, 6], [-1]),
    ([2, 3, 4], [0]),
    ([2, 3, 4], [0, 2]),
    ([2, 3, 4], [-1]),
    ([2, 3, 4], [0, 1, 2]),
    ([3, 5, 7, 9], [1, 3]),
    ([1, 1, 4, 4], [2, 3]),
    ([2, 16, 16], [1, 2]),
]


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("shape, dims", _CASES)
def test_flip_float(shape, dims, dtype):
    device = "cuda"
    input = torch.randn(shape, dtype=dtype, device=device)

    output = ntops.torch.flip(input, dims)
    expected = torch.flip(input, dims)

    assert output.shape == expected.shape
    assert torch.equal(output, expected)


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("shape, dims", _CASES)
def test_flip_int(shape, dims, dtype):
    device = "cuda"
    input = torch.randint(-1000, 1000, shape, dtype=dtype, device=device)

    output = ntops.torch.flip(input, dims)
    expected = torch.flip(input, dims)

    assert output.shape == expected.shape
    assert torch.equal(output, expected)


@skip_if_cuda_not_available
def test_flip_int_dim_argument():
    """A bare ``int`` for ``dims`` is accepted as a convenience."""
    device = "cuda"
    input = torch.randn(4, 6, device=device)

    output = ntops.torch.flip(input, 0)
    expected = torch.flip(input, [0])

    assert torch.equal(output, expected)


@skip_if_cuda_not_available
def test_flip_non_contiguous():
    """A transposed (non-contiguous) input must still flip correctly."""
    device = "cuda"
    input = torch.randn(4, 6, 8, device=device).transpose(0, 2)

    for dims in ([0], [2], [0, 2], [0, 1, 2]):
        output = ntops.torch.flip(input, dims)
        expected = torch.flip(input, dims)

        assert output.shape == expected.shape
        assert torch.equal(output, expected)


@skip_if_cuda_not_available
def test_flip_double_flip_is_identity():
    device = "cuda"
    input = torch.randn(3, 5, 7, device=device)

    restored = ntops.torch.flip(ntops.torch.flip(input, [0, 2]), [0, 2])

    assert torch.equal(restored, input)


@skip_if_cuda_not_available
def test_flip_duplicate_dims():
    device = "cuda"
    input = torch.randn(4, 6, device=device)

    with pytest.raises(RuntimeError):
        ntops.torch.flip(input, [0, 0])


@skip_if_cuda_not_available
def test_flip_dim_out_of_range():
    device = "cuda"
    input = torch.randn(4, 6, device=device)

    with pytest.raises(IndexError):
        ntops.torch.flip(input, [2])


# ---------------------------------------------------------------------------
# Performance benchmark interface
# ---------------------------------------------------------------------------

def benchmark_flip(
    shape,
    dims=(-1,),
    dtype=torch.float32,
    device="cuda",
    n_warmup=10,
    n_repeat=100,
):
    """Compare ntops.torch.flip vs torch.flip.

    Returns timing (ms) and effective memory bandwidth (GB/s) for both, plus
    the speedup ratio. Bandwidth assumes one read of the input plus one write
    of the output (2x input bytes), the lower bound for the op.

    Example
    -------
    >>> results = benchmark_flip([4096, 4096], dims=(0,))
    >>> print(results)
    """
    if not torch.cuda.is_available() and device == "cuda":
        raise RuntimeError("CUDA not available")

    input = torch.randn(shape, dtype=dtype, device=device)
    dims = list(dims)

    for _ in range(n_warmup):
        ntops.torch.flip(input, dims)
        torch.flip(input, dims)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(n_repeat):
        ntops.torch.flip(input, dims)
    end.record()
    torch.cuda.synchronize()
    ntops_ms = start.elapsed_time(end) / n_repeat

    start.record()
    for _ in range(n_repeat):
        torch.flip(input, dims)
    end.record()
    torch.cuda.synchronize()
    torch_ms = start.elapsed_time(end) / n_repeat

    num_bytes = input.numel() * input.element_size() * 2
    ntops_gbps = num_bytes / (ntops_ms * 1e-3) / 1e9
    torch_gbps = num_bytes / (torch_ms * 1e-3) / 1e9

    return {
        "shape": shape,
        "dims": dims,
        "dtype": str(dtype),
        "ntops_time_ms": ntops_ms,
        "torch_time_ms": torch_ms,
        "ntops_bandwidth_GBs": ntops_gbps,
        "torch_bandwidth_GBs": torch_gbps,
        "speedup": torch_ms / ntops_ms,
    }


_SWEEP_SHAPES = [
    ([1024, 1024], (0,)),       # flip outer dim (coalesced inner)
    ([1024, 1024], (1,)),       # flip inner dim (reversed reads)
    ([4096, 4096], (0, 1)),
    ([8192, 8192], (1,)),
]


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_benchmark_sweep(dtype):
    """Sweep tensor sizes/dims. Run with:
    pytest tests/test_flip.py::test_benchmark_sweep -v -s
    """
    header = (
        f"{'shape':>16} {'dims':>8} {'MB':>8} "
        f"{'ntops(ms)':>11} {'torch(ms)':>11} "
        f"{'ntops(GB/s)':>13} {'torch(GB/s)':>13} {'speedup':>9}"
    )
    print(f"\n{'='*len(header)}")
    print(f"flip sweep | dtype={dtype}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for shape, dims in _SWEEP_SHAPES:
        res = benchmark_flip(shape, dims=dims, dtype=dtype)
        mb = (
            res["ntops_bandwidth_GBs"] * res["ntops_time_ms"] * 1e-3 * 1e9
        ) / 2 / 1e6
        print(
            f"{str(shape):>16} {str(dims):>8} {mb:>8.1f} "
            f"{res['ntops_time_ms']:>11.4f} {res['torch_time_ms']:>11.4f} "
            f"{res['ntops_bandwidth_GBs']:>13.1f} {res['torch_bandwidth_GBs']:>13.1f} "
            f"{res['speedup']:>9.2f}"
        )

    print("=" * len(header))


@skip_if_cuda_not_available
def test_benchmark_interface():
    """Smoke-test that the benchmark interface runs without error."""
    results = benchmark_flip([512, 512], dims=(0,), n_warmup=2, n_repeat=5)
    assert results["ntops_time_ms"] > 0
    assert results["ntops_bandwidth_GBs"] > 0
