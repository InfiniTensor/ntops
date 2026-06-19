import pytest
import torch

import ntops


def trapezoid_cpu(y, x=None, dim=-1):
    """CPU reference matching the PyTorch API on this system."""
    if x is None:
        return torch.trapezoid(y, dx=1, dim=dim)
    return torch.trapezoid(y, x=x, dim=dim)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_trapezoid_1d(dtype):
    """1D tensor integration."""
    y = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=dtype, device="cuda")
    result = ntops.torch.trapezoid(y)
    expected = trapezoid_cpu(y)
    rtol = 1e-5 if dtype == torch.float32 else 1e-3
    atol = 1e-5 if dtype == torch.float32 else 1e-3
    assert torch.allclose(result, expected, rtol=rtol, atol=atol)
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_trapezoid_2d_dim0(dtype):
    """2D tensor, integrate along dim=0."""
    y = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=dtype, device="cuda")
    result = ntops.torch.trapezoid(y, dim=0)
    expected = trapezoid_cpu(y, dim=0)
    rtol = 1e-5 if dtype == torch.float32 else 1e-3
    atol = 1e-5 if dtype == torch.float32 else 1e-3
    assert torch.allclose(result, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_trapezoid_2d_dim1(dtype):
    """2D tensor, integrate along dim=1."""
    y = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=dtype, device="cuda")
    result = ntops.torch.trapezoid(y, dim=1)
    expected = trapezoid_cpu(y, dim=1)
    rtol = 1e-5 if dtype == torch.float32 else 1e-3
    atol = 1e-5 if dtype == torch.float32 else 1e-3
    assert torch.allclose(result, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_trapezoid_with_x(dtype):
    """Trapezoid with custom x coordinates."""
    y = torch.tensor([1.0, 2.0, 3.0], dtype=dtype, device="cuda")
    x = torch.tensor([0.0, 1.0, 3.0], dtype=dtype, device="cuda")
    result = ntops.torch.trapezoid(y, x=x)
    expected = trapezoid_cpu(y, x=x)
    rtol = 1e-5 if dtype == torch.float32 else 1e-3
    atol = 1e-5 if dtype == torch.float32 else 1e-3
    assert torch.allclose(result, expected, rtol=rtol, atol=atol)


def test_trapezoid_edge_cases():
    """Edge cases."""
    # Single element
    y = torch.tensor([5.0], device="cuda")
    result = ntops.torch.trapezoid(y)
    assert result.numel() == 0 or result.item() == 0.0

    # Two elements
    y = torch.tensor([1.0, 3.0], device="cuda")
    result = ntops.torch.trapezoid(y)
    expected = torch.tensor(2.0, device="cuda")  # (1+3)/2 * 1 = 2
    assert torch.allclose(result, expected)

    # 3D tensor
    y = torch.randn(2, 3, 4, device="cuda")
    result = ntops.torch.trapezoid(y, dim=1)
    expected = trapezoid_cpu(y, dim=1)
    assert torch.allclose(result, expected)
    assert result.shape == (2, 4)

    # Negative dim
    y = torch.randn(2, 3, 4, device="cuda")
    result = ntops.torch.trapezoid(y, dim=-1)
    expected = trapezoid_cpu(y, dim=-1)
    assert torch.allclose(result, expected)
    assert result.shape == (2, 3)


def test_trapezoid_float64():
    """float64 precision."""
    y = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda", dtype=torch.float64)
    result = ntops.torch.trapezoid(y)
    expected = trapezoid_cpu(y)
    assert torch.allclose(result, expected, rtol=1e-10, atol=1e-10)
