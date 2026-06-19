import pytest
import torch
import ntops


# =============================================================================
# Basic functionality tests
# =============================================================================

def test_column_stack_1d_two_tensors():
    """Stack two 1D tensors as columns."""
    a = torch.tensor([1, 2, 3], device="cuda")
    b = torch.tensor([4, 5, 6], device="cuda")
    result = ntops.torch.column_stack((a, b))
    expected = torch.tensor([[1, 4], [2, 5], [3, 6]], device="cuda")
    assert torch.equal(result, expected)


def test_column_stack_1d_three_tensors():
    """Stack three 1D tensors as columns."""
    a = torch.tensor([1, 2], device="cuda")
    b = torch.tensor([3, 4], device="cuda")
    c = torch.tensor([5, 6], device="cuda")
    result = ntops.torch.column_stack((a, b, c))
    expected = torch.tensor([[1, 3, 5], [2, 4, 6]], device="cuda")
    assert torch.equal(result, expected)


def test_column_stack_2d_two_tensors():
    """Stack two 2D tensors along columns."""
    a = torch.arange(6, device="cuda").reshape(2, 3).float()
    b = torch.arange(6, 10, device="cuda").reshape(2, 2).float()
    result = ntops.torch.column_stack((a, b))
    expected = torch.column_stack((a, b))
    assert torch.equal(result, expected)
    assert result.shape == (2, 5)


def test_column_stack_3d():
    """Stack 3D tensors along the second-to-last dim."""
    a = torch.randn(2, 3, 4, device="cuda")
    b = torch.randn(2, 5, 4, device="cuda")
    result = ntops.torch.column_stack((a, b))
    expected = torch.column_stack((a, b))
    assert torch.equal(result, expected)
    assert result.shape == (2, 8, 4)


def test_column_stack_single_tensor():
    """Stack a single 1D tensor (should become a column)."""
    a = torch.tensor([1, 2, 3, 4], device="cuda")
    result = ntops.torch.column_stack((a,))
    expected = torch.tensor([[1], [2], [3], [4]], device="cuda")
    assert torch.equal(result, expected)


# =============================================================================
# Edge cases
# =============================================================================

def test_column_stack_list_input():
    """Stack should accept list input."""
    a = torch.tensor([1, 2, 3], device="cuda")
    b = torch.tensor([4, 5, 6], device="cuda")
    result = ntops.torch.column_stack([a, b])
    expected = torch.tensor([[1, 4], [2, 5], [3, 6]], device="cuda")
    assert torch.equal(result, expected)


def test_column_stack_empty_sequence():
    """Empty sequence should raise RuntimeError."""
    with pytest.raises(RuntimeError):
        ntops.torch.column_stack([])


def test_column_stack_shape_mismatch():
    """Tensors with mismatched shapes should raise RuntimeError."""
    a = torch.randn(3, 4, device="cuda")
    b = torch.randn(5, 4, device="cuda")
    with pytest.raises(RuntimeError):
        ntops.torch.column_stack((a, b))


# =============================================================================
# Dtype and device tests
# =============================================================================

@pytest.mark.parametrize("dtype", [
    torch.float32,
    torch.float16,
    torch.float64,
    torch.int32,
    torch.int64,
])
def test_column_stack_dtype_preservation(dtype):
    """Column stack should preserve dtype."""
    a = torch.arange(5, device="cuda").to(dtype)
    b = torch.arange(5, 10, device="cuda").to(dtype)
    result = ntops.torch.column_stack((a, b))
    assert result.dtype == dtype


def test_column_stack_device_preservation():
    """Column stack should preserve the device."""
    a = torch.randn(5, device="cuda")
    b = torch.randn(5, device="cuda")
    result = ntops.torch.column_stack((a, b))
    assert result.device == a.device


# =============================================================================
# Gradient test
# =============================================================================

def test_column_stack_gradient():
    """Column stack should support gradient propagation."""
    a = torch.randn(3, 4, device="cuda", requires_grad=True)
    b = torch.randn(3, 5, device="cuda", requires_grad=True)
    y = ntops.torch.column_stack((a, b))
    loss = y.sum()
    loss.backward()
    assert a.grad is not None
    assert a.grad.shape == a.shape
    assert b.grad is not None
    assert b.grad.shape == b.shape


# =============================================================================
# Four mandatory checks
# =============================================================================

def test_column_stack_no_nan():
    """Output should not contain NaN."""
    a = torch.randn(100, 50, device="cuda")
    b = torch.randn(100, 30, device="cuda")
    result = ntops.torch.column_stack((a, b))
    assert not torch.isnan(result).any()


def test_column_stack_no_inf():
    """Output should not contain Inf."""
    a = torch.randn(100, 50, device="cuda")
    b = torch.randn(100, 30, device="cuda")
    result = ntops.torch.column_stack((a, b))
    assert not torch.isinf(result).any()


def test_column_stack_int_exact():
    """Integer column_stack should be exact match."""
    a = torch.randint(0, 100, (5, 3), device="cuda")
    b = torch.randint(0, 100, (5, 2), device="cuda")
    result = ntops.torch.column_stack((a, b))
    expected = torch.column_stack((a, b))
    assert torch.equal(result, expected)
