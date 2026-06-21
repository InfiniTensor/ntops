import random

import pytest
import torch
import torch.nn.functional as F

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments

_ALPHA_PRIME = -1.7580993408473766
_ATOL = 1e-3


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------

@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_feature_alpha_dropout_training_false(shape, dtype, device, rtol, atol):
    input = torch.randn(shape, dtype=dtype, device=device)

    output = ntops.torch.feature_alpha_dropout(input, p=0.5, training=False)

    assert torch.equal(output, input)


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_feature_alpha_dropout_p_zero(shape, dtype, device, rtol, atol):
    input = torch.randn(shape, dtype=dtype, device=device)

    output = ntops.torch.feature_alpha_dropout(input, p=0.0, training=True)

    assert torch.equal(output, input)


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_feature_alpha_dropout_p_one(shape, dtype, device, rtol, atol):
    input = torch.randn(shape, dtype=dtype, device=device)

    output = ntops.torch.feature_alpha_dropout(input, p=1.0, training=True)

    expected = torch.full_like(input, _ALPHA_PRIME)
    assert torch.allclose(output, expected, atol=_ATOL)


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_feature_alpha_dropout_kept_values(shape, dtype, device, rtol, atol):
    """Non-dropped elements must equal input * scale."""
    input = torch.randn(shape, dtype=dtype, device=device)
    p = random.uniform(0.05, 0.5)
    scale = 1.0 / (1.0 - p)

    output = ntops.torch.feature_alpha_dropout(input, p=p, training=True)

    alpha_prime_t = torch.tensor(_ALPHA_PRIME, dtype=dtype, device=device)
    kept = ~torch.isclose(output, alpha_prime_t.expand_as(output), atol=_ATOL)

    if kept.any():
        assert torch.allclose(
            output[kept],
            (input * scale)[kept],
            rtol=rtol,
            atol=atol,
        )


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_feature_alpha_dropout_dropped_values(shape, dtype, device, rtol, atol):
    """Dropped elements must equal alpha_prime."""
    input = torch.randn(shape, dtype=dtype, device=device)
    p = random.uniform(0.05, 0.5)

    output = ntops.torch.feature_alpha_dropout(input, p=p, training=True)

    alpha_prime_t = torch.tensor(_ALPHA_PRIME, dtype=dtype, device=device)
    dropped = torch.isclose(output, alpha_prime_t.expand_as(output), atol=_ATOL)

    if dropped.any():
        assert torch.allclose(
            output[dropped],
            alpha_prime_t.expand_as(output)[dropped],
            atol=_ATOL,
        )


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("shape", [[4, 8, 16, 16], [2, 16, 32], [8, 4]])
def test_feature_alpha_dropout_channel_consistency(shape, dtype):
    """Within each (sample, channel) all spatial positions share the same mask.

    Use all-ones input so kept elements become scale > 0 and dropped elements
    become alpha_prime < 0 — reliably separated by sign, no tolerance needed.
    """
    device = "cuda"
    input = torch.ones(shape, dtype=dtype, device=device)
    p = 0.4

    output = ntops.torch.feature_alpha_dropout(input, p=p, training=True)

    # dropped: output < 0  (alpha_prime ≈ -1.758)
    # kept:    output > 0  (scale = 1/(1-p) ≈ 1.667)
    dropped = (output < 0).reshape(shape[0], shape[1], -1)  # (N, C, S)

    all_dropped = dropped.all(dim=-1)   # (N, C)
    any_dropped = dropped.any(dim=-1)   # (N, C)
    assert torch.equal(all_dropped, any_dropped), (
        "Channel consistency violated: some channels partially dropped"
    )


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_feature_alpha_dropout_drop_rate(dtype):
    """Observed channel drop rate should be close to p."""
    device = "cuda"
    shape = [32, 64, 8, 8]
    p = 0.3
    input = torch.ones(shape, dtype=dtype, device=device)

    output = ntops.torch.feature_alpha_dropout(input, p=p, training=True)

    # all-ones input: dropped channels are all negative, kept are all positive
    dropped_channels = (output < 0).reshape(shape[0], shape[1], -1).all(dim=-1)
    observed_rate = dropped_channels.float().mean().item()

    assert abs(observed_rate - p) < 0.1, (
        f"Drop rate {observed_rate:.3f} too far from p={p}"
    )


# ---------------------------------------------------------------------------
# Performance benchmark interface
# ---------------------------------------------------------------------------

def benchmark_feature_alpha_dropout(
    shape,
    p=0.5,
    dtype=torch.float32,
    device="cuda",
    n_warmup=10,
    n_repeat=100,
):
    """Compare ntops.torch.feature_alpha_dropout vs F.feature_alpha_dropout.

    Returns timing (ms) and effective memory bandwidth (GB/s) for both,
    plus the speedup ratio.

    Example
    -------
    >>> results = benchmark_feature_alpha_dropout([32, 64, 128, 128])
    >>> print(results)
    """
    if not torch.cuda.is_available() and device == "cuda":
        raise RuntimeError("CUDA not available")

    input_tensor = torch.randn(shape, dtype=dtype, device=device)

    for _ in range(n_warmup):
        ntops.torch.feature_alpha_dropout(input_tensor, p=p, training=True)
        F.feature_alpha_dropout(input_tensor, p=p, training=True)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(n_repeat):
        ntops.torch.feature_alpha_dropout(input_tensor, p=p, training=True)
    end.record()
    torch.cuda.synchronize()
    ntops_ms = start.elapsed_time(end) / n_repeat

    start.record()
    for _ in range(n_repeat):
        F.feature_alpha_dropout(input_tensor, p=p, training=True)
    end.record()
    torch.cuda.synchronize()
    torch_ms = start.elapsed_time(end) / n_repeat

    num_bytes = input_tensor.numel() * input_tensor.element_size() * 2
    ntops_gbps = num_bytes / (ntops_ms * 1e-3) / 1e9
    torch_gbps = num_bytes / (torch_ms * 1e-3) / 1e9

    return {
        "shape": shape,
        "p": p,
        "dtype": str(dtype),
        "ntops_time_ms": ntops_ms,
        "torch_time_ms": torch_ms,
        "ntops_bandwidth_GBs": ntops_gbps,
        "torch_bandwidth_GBs": torch_gbps,
        "speedup": torch_ms / ntops_ms,
    }


_SWEEP_SHAPES = [
    [4, 16, 32, 32],     # 0.25 MB
    [8, 64, 64, 64],     # 16 MB
    [16, 128, 64, 64],   # 64 MB
    [32, 256, 64, 64],   # 256 MB
]


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_benchmark_sweep(dtype):
    """Sweep tensor sizes. Run with: pytest tests/test_feature_alpha_dropout.py::test_benchmark_sweep -v -s"""
    header = (
        f"{'shape':>22} {'MB':>8} "
        f"{'ntops(ms)':>11} {'torch(ms)':>11} "
        f"{'ntops(GB/s)':>13} {'torch(GB/s)':>13} {'speedup':>9}"
    )
    print(f"\n{'='*len(header)}")
    print(f"feature_alpha_dropout sweep | dtype={dtype} | p=0.5")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for shape in _SWEEP_SHAPES:
        r = benchmark_feature_alpha_dropout(shape, dtype=dtype)
        mb = (r["ntops_bandwidth_GBs"] * r["ntops_time_ms"] * 1e-3 * 1e9) / 2 / 1e6
        print(
            f"{str(shape):>22} {mb:>8.1f} "
            f"{r['ntops_time_ms']:>11.4f} {r['torch_time_ms']:>11.4f} "
            f"{r['ntops_bandwidth_GBs']:>13.1f} {r['torch_bandwidth_GBs']:>13.1f} "
            f"{r['speedup']:>9.2f}"
        )

    print("=" * len(header))


@skip_if_cuda_not_available
def test_benchmark_interface():
    """Smoke-test that benchmark interface runs without error."""
    results = benchmark_feature_alpha_dropout(
        [8, 32, 32, 32], n_warmup=2, n_repeat=5
    )
    assert results["ntops_time_ms"] > 0
    assert results["ntops_bandwidth_GBs"] > 0
