import pytest
import math
import torch

import ntops


def kl_div_cpu(input, target, reduction="sum", log_target=False, eps=1e-10):
    """CPU reference matching the spec."""
    if log_target:
        log_p = target
        p = torch.exp(log_p)
    else:
        p = torch.clamp(target, min=eps, max=1.0)
        log_p = torch.log(p)
    p = torch.clamp(p, min=eps, max=1.0)
    loss = p * (log_p - input)
    if reduction == "none":
        return loss
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "mean":
        return loss.mean()
    elif reduction == "batchmean":
        return loss.sum() / loss.shape[0]


DTYPE_TOLERANCES = [
    (torch.float32, 1e-5, 1e-5),
    (torch.float16, 1e-3, 1e-3),
]


@pytest.mark.parametrize("dtype, rtol, atol", DTYPE_TOLERANCES)
def test_kl_div_identical(dtype, rtol, atol):
    """KL(q||q) = 0 when distributions are identical."""
    log_q = torch.tensor([-0.6931, -0.6931, -1.0986], dtype=dtype, device="cuda")
    target = torch.tensor([0.5, 0.5, 0.333], dtype=dtype, device="cuda")
    result = ntops.torch.kl_div(log_q, target, reduction="sum")
    expected = kl_div_cpu(log_q, target, reduction="sum")
    assert torch.allclose(result, expected, rtol=rtol, atol=atol)
    assert not torch.isnan(result).any()


@pytest.mark.parametrize("dtype, rtol, atol", DTYPE_TOLERANCES)
def test_kl_div_log_target(dtype, rtol, atol):
    """KL divergence with log_target=True."""
    log_q = torch.tensor([-1.0, -0.5, -0.2], dtype=dtype, device="cuda")
    log_target = torch.tensor([-1.0, -0.5, -0.2], dtype=dtype, device="cuda")
    result = ntops.torch.kl_div(log_q, log_target, reduction="sum", log_target=True)
    expected = kl_div_cpu(log_q, log_target, reduction="sum", log_target=True)
    assert torch.allclose(result, expected, rtol=rtol, atol=atol)
    assert not torch.isnan(result).any()


@pytest.mark.parametrize("dtype, rtol, atol", DTYPE_TOLERANCES)
def test_kl_div_different(dtype, rtol, atol):
    """KL divergence between different distributions."""
    log_q = torch.tensor([-0.6931, -0.6931], dtype=dtype, device="cuda")  # log(0.5), log(0.5)
    target = torch.tensor([0.9, 0.1], dtype=dtype, device="cuda")
    result = ntops.torch.kl_div(log_q, target, reduction="sum")
    expected = kl_div_cpu(log_q, target, reduction="sum")
    assert torch.allclose(result, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype, rtol, atol", DTYPE_TOLERANCES)
def test_kl_div_reduction_none(dtype, rtol, atol):
    """No reduction — return element-wise loss."""
    log_q = torch.tensor([-1.0, -0.5], dtype=dtype, device="cuda")
    target = torch.tensor([0.2, 0.8], dtype=dtype, device="cuda")
    result = ntops.torch.kl_div(log_q, target, reduction="none")
    expected = kl_div_cpu(log_q, target, reduction="none")
    assert torch.allclose(result, expected, rtol=rtol, atol=atol)
    assert result.shape == log_q.shape


@pytest.mark.parametrize("dtype, rtol, atol", DTYPE_TOLERANCES)
def test_kl_div_reduction_mean(dtype, rtol, atol):
    """Mean reduction."""
    log_q = torch.tensor([-0.6931, -0.6931, -0.5108, -0.5108], dtype=dtype, device="cuda")
    target = torch.tensor([0.9, 0.1, 0.5, 0.5], dtype=dtype, device="cuda")
    result = ntops.torch.kl_div(log_q, target, reduction="mean")
    expected = kl_div_cpu(log_q, target, reduction="mean")
    assert torch.allclose(result, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype, rtol, atol", DTYPE_TOLERANCES)
def test_kl_div_reduction_batchmean(dtype, rtol, atol):
    """Batchmean reduction."""
    log_q = torch.randn(4, 3, dtype=dtype, device="cuda").log_softmax(dim=1)
    target = torch.randn(4, 3, dtype=dtype, device="cuda").softmax(dim=1)
    result = ntops.torch.kl_div(log_q, target, reduction="batchmean")
    expected = kl_div_cpu(log_q, target, reduction="batchmean")
    assert torch.allclose(result, expected, rtol=rtol, atol=atol)


def test_kl_div_edge_cases():
    """Edge cases."""
    # Empty tensor
    x = torch.empty(0, 3, device="cuda")
    result = ntops.torch.kl_div(x, x, reduction="sum")
    assert result.item() == 0.0

    # Target at boundaries (0 and 1) — should be clamped
    log_q = torch.tensor([-0.6931, -0.6931], device="cuda")
    target = torch.tensor([0.0, 1.0], device="cuda")
    result = ntops.torch.kl_div(log_q, target, reduction="sum")
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()

    # Invalid reduction
    with pytest.raises(ValueError):
        ntops.torch.kl_div(log_q, target, reduction="invalid")


def test_kl_div_float64():
    """float64 precision."""
    log_q = torch.tensor([-0.693147, -0.693147], device="cuda", dtype=torch.float64)
    target = torch.tensor([0.5, 0.5], device="cuda", dtype=torch.float64)
    result = ntops.torch.kl_div(log_q, target, reduction="sum")
    expected = kl_div_cpu(log_q, target, reduction="sum")
    assert torch.allclose(result, expected, rtol=1e-7, atol=1e-7)
