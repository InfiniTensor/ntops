import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


@skip_if_cuda_not_available
def test_corrcoef_basic():
    """Basic correlation coefficient computation."""
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="cuda")
    result = ntops.torch.corrcoef(x)
    expected = torch.corrcoef(x)
    assert torch.allclose(result, expected)
    assert result.shape == (2, 2)


@skip_if_cuda_not_available
def test_corrcoef_identity():
    """Perfect correlation with itself — diagonal should be 1."""
    x = torch.randn(3, 100, device="cuda")
    result = ntops.torch.corrcoef(x)
    expected = torch.corrcoef(x)
    assert torch.allclose(result, expected)
    assert torch.allclose(result.diag(), torch.ones(3, device="cuda"))


@skip_if_cuda_not_available
def test_corrcoef_constant():
    """Constant input — should produce NaN (division by zero variance)."""
    x = torch.ones(3, 5, device="cuda")
    result = ntops.torch.corrcoef(x)
    expected = torch.corrcoef(x)
    assert torch.equal(torch.isnan(result), torch.isnan(expected))


@skip_if_cuda_not_available
def test_corrcoef_float16():
    """float16 precision."""
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float16, device="cuda")
    result = ntops.torch.corrcoef(x)
    expected = torch.corrcoef(x)
    assert torch.allclose(result, expected, rtol=1e-3, atol=1e-3)


@skip_if_cuda_not_available
def test_corrcoef_float64():
    """float64 precision."""
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64, device="cuda")
    result = ntops.torch.corrcoef(x)
    expected = torch.corrcoef(x)
    assert torch.allclose(result, expected)


@skip_if_cuda_not_available
def test_corrcoef_negative_correlation():
    """Test negative correlation."""
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]], device="cuda")
    result = ntops.torch.corrcoef(x)
    expected = torch.corrcoef(x)
    assert torch.allclose(result, expected)
    # Off-diagonal should be negative
    assert result[0, 1] < 0


@skip_if_cuda_not_available
def test_corrcoef_single_variable():
    """Single variable — returns a scalar 1.0 (same as torch.corrcoef)."""
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], device="cuda")
    result = ntops.torch.corrcoef(x)
    expected = torch.corrcoef(x)
    assert torch.allclose(result, expected)
    assert result.ndim == 0  # torch.corrcoef returns scalar for single var
