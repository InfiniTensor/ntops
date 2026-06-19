import pytest
import torch

import ntops

DTYPE_TOLERANCES = [
    (torch.float32, 1e-5, 1e-5),
    (torch.float16, 1e-3, 1e-3),
]


def logit_cpu(x, eps=1e-6):
    """CPU reference: clip then log(x / (1-x))."""
    x = x.clamp(eps, 1 - eps)
    return torch.log(x / (1 - x))


@pytest.mark.parametrize("dtype, rtol, atol", DTYPE_TOLERANCES)
def test_logit_basic(dtype, rtol, atol):
    """Basic logit: values in (0, 1)."""
    x = torch.tensor([0.1, 0.5, 0.9], dtype=dtype, device="cuda")
    result = ntops.torch.logit(x)
    expected = logit_cpu(x)
    assert torch.allclose(result, expected, rtol=rtol, atol=atol)
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()


@pytest.mark.parametrize("dtype, rtol, atol", DTYPE_TOLERANCES)
def test_logit_boundaries(dtype, rtol, atol):
    """Boundary values: 0 and 1 clamped to [eps, 1-eps].

    Note: float16 `1.0 - eps` rounds to 1.0, causing log(0)=inf.
    Our float32 intermediate avoids this — compare against float32 reference.
    """
    x = torch.tensor([0.0, 1.0, 0.5], dtype=dtype, device="cuda")
    result = ntops.torch.logit(x)
    # Use float32 reference since float16 reference degrades at boundaries
    expected_f32 = logit_cpu(x.float())
    assert torch.allclose(result, expected_f32.to(dtype), rtol=rtol, atol=atol)
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()


@pytest.mark.parametrize("dtype, rtol, atol", DTYPE_TOLERANCES)
def test_logit_symmetric(dtype, rtol, atol):
    """logit(1-x) = -logit(x) for symmetric values."""
    x = torch.tensor([0.2, 0.4, 0.6, 0.8], dtype=dtype, device="cuda")
    result = ntops.torch.logit(x)
    result_complement = ntops.torch.logit(1 - x)
    assert torch.allclose(result, -result_complement, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype, rtol, atol", DTYPE_TOLERANCES)
def test_logit_custom_eps(dtype, rtol, atol):
    """Custom epsilon value. Compare against float32 reference."""
    x = torch.tensor([0.0, 1.0, 0.5], dtype=dtype, device="cuda")
    result = ntops.torch.logit(x, eps=1e-3)
    expected_f32 = logit_cpu(x.float(), eps=1e-3)
    assert torch.allclose(result, expected_f32.to(dtype), rtol=rtol, atol=atol)
    assert not torch.isnan(result).any()


@pytest.mark.parametrize("dtype, rtol, atol", DTYPE_TOLERANCES)
def test_logit_large(dtype, rtol, atol):
    """Large tensor."""
    x = torch.rand(10000, dtype=dtype, device="cuda")
    result = ntops.torch.logit(x)
    expected = logit_cpu(x)
    assert torch.allclose(result, expected, rtol=rtol, atol=atol)
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()


def test_logit_edge_cases():
    """Edge cases."""
    # Empty tensor
    x = torch.empty(0, device="cuda")
    result = ntops.torch.logit(x)
    assert result.numel() == 0

    # 2D tensor
    x = torch.tensor([[0.1, 0.9], [0.5, 0.0]], device="cuda")
    result = ntops.torch.logit(x)
    expected = logit_cpu(x)
    assert torch.allclose(result, expected)

    # Values far outside [0, 1] — should be clamped
    x = torch.tensor([-10.0, 10.0, 0.5], device="cuda")
    result = ntops.torch.logit(x)
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()


def test_logit_float64():
    """float64 precision."""
    x = torch.tensor([0.1, 0.5, 0.9], device="cuda", dtype=torch.float64)
    result = ntops.torch.logit(x)
    expected = logit_cpu(x)
    # float32 intermediate limits float64 output precision to ~1e-7
    assert torch.allclose(result, expected, rtol=1e-7, atol=1e-7)
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()
