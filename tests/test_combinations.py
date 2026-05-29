import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available

# ---------------------------------------------------------------------------
# Correctness tests (compared against torch.combinations).
#
# combinations is a pure gather/selection, so the output must be bit-identical
# to torch's, including ordering, dtype, device, and shape -- ``torch.equal``.
# ---------------------------------------------------------------------------

_NS = [1, 2, 3, 5, 8]
_RS = [0, 1, 2, 3, 4]


def _make_input(n, dtype, device):
    if dtype == torch.int64:
        return torch.randint(-1000, 1000, (n,), dtype=dtype, device=device)
    return torch.randn(n, dtype=dtype, device=device)


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int64])
@pytest.mark.parametrize("with_replacement", [False, True])
@pytest.mark.parametrize("r", _RS)
@pytest.mark.parametrize("n", _NS)
def test_combinations(n, r, with_replacement, dtype):
    device = "cuda"
    input = _make_input(n, dtype, device)

    output = ntops.torch.combinations(input, r=r, with_replacement=with_replacement)
    expected = torch.combinations(input, r=r, with_replacement=with_replacement)

    assert output.shape == expected.shape
    assert output.dtype == expected.dtype
    assert output.device == expected.device
    assert torch.equal(output, expected)


@skip_if_cuda_not_available
def test_combinations_default_args():
    """Default r=2, with_replacement=False."""
    device = "cuda"
    input = torch.randn(6, device=device)

    output = ntops.torch.combinations(input)
    expected = torch.combinations(input)

    assert torch.equal(output, expected)


@skip_if_cuda_not_available
def test_combinations_r_greater_than_n():
    """r > n without replacement yields an empty (0, r) result."""
    device = "cuda"
    input = torch.randn(3, device=device)

    output = ntops.torch.combinations(input, r=5)
    expected = torch.combinations(input, r=5)

    assert output.shape == expected.shape  # (0, 5)
    assert torch.equal(output, expected)


@skip_if_cuda_not_available
def test_combinations_empty_input():
    device = "cuda"
    input = torch.randn(0, device=device)

    for r in (0, 1, 2):
        output = ntops.torch.combinations(input, r=r)
        expected = torch.combinations(input, r=r)
        assert output.shape == expected.shape
        assert torch.equal(output, expected)


@skip_if_cuda_not_available
def test_combinations_invalid_ndim():
    device = "cuda"
    input = torch.randn(3, 4, device=device)

    with pytest.raises(RuntimeError):
        ntops.torch.combinations(input)


@skip_if_cuda_not_available
def test_combinations_negative_r():
    device = "cuda"
    input = torch.randn(5, device=device)

    with pytest.raises(RuntimeError):
        ntops.torch.combinations(input, r=-1)


# ---------------------------------------------------------------------------
# Performance benchmark interface
# ---------------------------------------------------------------------------

def benchmark_combinations(
    n,
    r=2,
    with_replacement=False,
    dtype=torch.float32,
    device="cuda",
    n_warmup=10,
    n_repeat=100,
):
    """Compare ntops.torch.combinations vs torch.combinations.

    Returns timing (ms) for both plus the speedup ratio. combinations has no
    ninetoothed kernel (the gather is not expressible in ninetoothed); this
    interface exists for parity with the rest of the suite.

    Example
    -------
    >>> results = benchmark_combinations(256, r=2)
    >>> print(results)
    """
    if not torch.cuda.is_available() and device == "cuda":
        raise RuntimeError("CUDA not available")

    input = torch.randn(n, dtype=dtype, device=device)

    def run_ntops():
        ntops.torch.combinations(input, r=r, with_replacement=with_replacement)

    def run_torch():
        torch.combinations(input, r=r, with_replacement=with_replacement)

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

    return {
        "n": n,
        "r": r,
        "with_replacement": with_replacement,
        "dtype": str(dtype),
        "ntops_time_ms": ntops_ms,
        "torch_time_ms": torch_ms,
        "speedup": torch_ms / ntops_ms,
    }


@skip_if_cuda_not_available
@pytest.mark.parametrize("r", [2, 3])
def test_benchmark_sweep(r):
    """Sweep input sizes. Run with:
    pytest tests/test_combinations.py::test_benchmark_sweep -v -s
    """
    header = (
        f"{'n':>8} {'r':>4} {'num_comb':>12} "
        f"{'ntops(ms)':>11} {'torch(ms)':>11} {'speedup':>9}"
    )
    print(f"\n{'='*len(header)}")
    print(f"combinations sweep | r={r}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    ns = [64, 128, 256] if r == 2 else [16, 32, 48]
    for n in ns:
        res = benchmark_combinations(n, r=r)
        num_comb = torch.combinations(
            torch.arange(n), r=r
        ).shape[0]
        print(
            f"{n:>8} {r:>4} {num_comb:>12} "
            f"{res['ntops_time_ms']:>11.4f} {res['torch_time_ms']:>11.4f} "
            f"{res['speedup']:>9.2f}"
        )

    print("=" * len(header))


@skip_if_cuda_not_available
def test_benchmark_interface():
    """Smoke-test that the benchmark interface runs without error."""
    results = benchmark_combinations(64, r=2, n_warmup=2, n_repeat=5)
    assert results["ntops_time_ms"] > 0
