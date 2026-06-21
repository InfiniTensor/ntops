import pytest
import torch
import torch.nn.functional as F

import ntops
from tests.skippers import skip_if_cuda_not_available

pytestmark = pytest.mark.filterwarnings(
    "ignore:.*batchmean.*:UserWarning"
)

# ---------------------------------------------------------------------------
# Correctness tests (compared against torch.nn.functional.kl_div)
#
# ``kl_div`` expects ``input`` to be log-probabilities and ``target`` to be
# probabilities (or log-probabilities when ``log_target=True``); the helpers
# below build valid inputs via log_softmax / softmax over the last dim.
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


def _make_inputs(shape, dtype, device, log_target):
    input = torch.randn(shape, dtype=dtype, device=device).log_softmax(dim=-1)

    if log_target:
        target = torch.randn(shape, dtype=dtype, device=device).log_softmax(dim=-1)
    else:
        target = torch.randn(shape, dtype=dtype, device=device).softmax(dim=-1)

    return input, target


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype, rtol, atol", [
    (torch.float32, 1e-3, 1e-3),
    (torch.float16, 1e-2, 1e-2),
])
@pytest.mark.parametrize("log_target", [False, True])
@pytest.mark.parametrize("reduction", ["none", "sum", "mean", "batchmean"])
@pytest.mark.parametrize("shape", _SHAPES)
def test_kl_div(shape, reduction, log_target, dtype, rtol, atol):
    device = "cuda"
    input, target = _make_inputs(shape, dtype, device, log_target)

    output = ntops.torch.kl_div(
        input, target, reduction=reduction, log_target=log_target
    )
    expected = F.kl_div(
        input, target, reduction=reduction, log_target=log_target
    )

    assert output.shape == expected.shape
    assert output.dtype == expected.dtype
    assert torch.allclose(output, expected, rtol=rtol, atol=atol)


@skip_if_cuda_not_available
@pytest.mark.parametrize("reduction", ["none", "sum", "mean", "batchmean"])
def test_kl_div_target_with_zeros(reduction):
    """target == 0 must contribute 0 (the 0*log(0)=0 convention), not NaN."""
    device = "cuda"
    input = torch.randn(8, 16, device=device).log_softmax(dim=-1)
    target = torch.rand(8, 16, device=device)
    target[target < 0.3] = 0.0  # inject exact zeros

    output = ntops.torch.kl_div(input, target, reduction=reduction)
    expected = F.kl_div(input, target, reduction=reduction)

    assert not torch.isnan(output).any()
    assert torch.allclose(output, expected, rtol=1e-3, atol=1e-3)


@skip_if_cuda_not_available
def test_kl_div_default_reduction_is_mean():
    device = "cuda"
    input = torch.randn(2, 3, 4, device=device).log_softmax(dim=-1)
    target = torch.randn(2, 3, 4, device=device).softmax(dim=-1)

    output = ntops.torch.kl_div(input, target)
    expected = F.kl_div(input, target, reduction="mean")

    assert torch.allclose(output, expected, rtol=1e-3, atol=1e-3)


@skip_if_cuda_not_available
@pytest.mark.parametrize("log_target", [False, True])
def test_kl_div_broadcast(log_target):
    device = "cuda"
    input = torch.randn(4, 3, 8, device=device).log_softmax(dim=-1)
    if log_target:
        target = torch.randn(3, 8, device=device).log_softmax(dim=-1)
    else:
        target = torch.randn(3, 8, device=device).softmax(dim=-1)

    output = ntops.torch.kl_div(
        input, target, reduction="sum", log_target=log_target
    )
    expected = F.kl_div(
        input,
        target.expand_as(input).contiguous(),
        reduction="sum",
        log_target=log_target,
    )

    assert torch.allclose(output, expected, rtol=1e-3, atol=1e-3)


@skip_if_cuda_not_available
def test_kl_div_invalid_reduction():
    device = "cuda"
    input = torch.randn(8, device=device).log_softmax(dim=-1)
    target = torch.randn(8, device=device).softmax(dim=-1)

    with pytest.raises(ValueError):
        ntops.torch.kl_div(input, target, reduction="median")


# ---------------------------------------------------------------------------
# Performance benchmark interface
# ---------------------------------------------------------------------------

def benchmark_kl_div(
    shape,
    reduction="mean",
    log_target=False,
    dtype=torch.float32,
    device="cuda",
    n_warmup=10,
    n_repeat=100,
):
    """Compare ntops.torch.kl_div vs F.kl_div.

    Returns timing (ms) and effective memory bandwidth (GB/s) for both,
    plus the speedup ratio. Bandwidth assumes both ``input`` and ``target``
    are read once (2x input bytes), which is the lower bound for the op.

    Example
    -------
    >>> results = benchmark_kl_div([4096, 4096], "mean")
    >>> print(results)
    """
    if not torch.cuda.is_available() and device == "cuda":
        raise RuntimeError("CUDA not available")

    input = torch.randn(shape, dtype=dtype, device=device).log_softmax(dim=-1)
    target = torch.randn(shape, dtype=dtype, device=device).softmax(dim=-1)

    def run_ntops():
        ntops.torch.kl_div(input, target, reduction=reduction, log_target=log_target)

    def run_torch():
        F.kl_div(input, target, reduction=reduction, log_target=log_target)

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

    num_bytes = input.numel() * input.element_size() * 2
    ntops_gbps = num_bytes / (ntops_ms * 1e-3) / 1e9
    torch_gbps = num_bytes / (torch_ms * 1e-3) / 1e9

    return {
        "shape": shape,
        "reduction": reduction,
        "log_target": log_target,
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
@pytest.mark.parametrize("reduction", ["none", "mean", "sum", "batchmean"])
def test_benchmark_sweep(reduction, dtype):
    """Sweep tensor sizes. Run with:
    pytest tests/test_kl_div.py::test_benchmark_sweep -v -s
    """
    header = (
        f"{'shape':>16} {'MB':>8} "
        f"{'ntops(ms)':>11} {'torch(ms)':>11} "
        f"{'ntops(GB/s)':>13} {'torch(GB/s)':>13} {'speedup':>9}"
    )
    print(f"\n{'='*len(header)}")
    print(f"kl_div sweep | reduction={reduction} | dtype={dtype}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for shape in _SWEEP_SHAPES:
        res = benchmark_kl_div(shape, reduction=reduction, dtype=dtype)
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
    results = benchmark_kl_div([512, 512], n_warmup=2, n_repeat=5)
    assert results["ntops_time_ms"] > 0
    assert results["ntops_bandwidth_GBs"] > 0
