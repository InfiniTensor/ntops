import os
import pytest
import torch
import ntops


# =============================================================================
# Basic functionality tests
# =============================================================================

def test_roll_1d_positive_shift():
    """Roll 1D tensor with positive shift."""
    x = torch.tensor([1, 2, 3, 4, 5], device="cuda")
    result = ntops.torch.roll(x, shifts=2, dims=0)
    expected = torch.tensor([4, 5, 1, 2, 3], device="cuda")
    assert torch.equal(result, expected)


def test_roll_1d_negative_shift():
    """Roll 1D tensor with negative shift (forward roll)."""
    x = torch.tensor([1, 2, 3, 4, 5], device="cuda")
    result = ntops.torch.roll(x, shifts=-2, dims=0)
    expected = torch.tensor([3, 4, 5, 1, 2], device="cuda")
    assert torch.equal(result, expected)


def test_roll_2d_dim0():
    """Roll 2D tensor along dimension 0."""
    x = torch.arange(12, device="cuda").reshape(3, 4)
    result = ntops.torch.roll(x, shifts=1, dims=0)
    expected = torch.tensor([
        [8, 9, 10, 11],
        [0, 1, 2, 3],
        [4, 5, 6, 7],
    ], device="cuda")
    assert torch.equal(result, expected)


def test_roll_2d_dim1():
    """Roll 2D tensor along dimension 1."""
    x = torch.arange(12, device="cuda").reshape(3, 4)
    result = ntops.torch.roll(x, shifts=1, dims=1)
    expected = torch.tensor([
        [3, 0, 1, 2],
        [7, 4, 5, 6],
        [11, 8, 9, 10],
    ], device="cuda")
    assert torch.equal(result, expected)


def test_roll_multi_dim():
    """Roll along multiple dimensions simultaneously."""
    x = torch.arange(12, device="cuda").reshape(3, 4)
    result = ntops.torch.roll(x, shifts=(1, -1), dims=(0, 1))
    expected = torch.roll(x, shifts=(1, -1), dims=(0, 1))
    assert torch.equal(result, expected)


def test_roll_3d():
    """Roll a 3D tensor."""
    x = torch.arange(24, device="cuda").reshape(2, 3, 4)
    result = ntops.torch.roll(x, shifts=1, dims=1)
    expected = torch.roll(x, shifts=1, dims=1)
    assert torch.equal(result, expected)


def test_roll_4d():
    """Roll a 4D tensor."""
    x = torch.arange(120, device="cuda").reshape(2, 3, 4, 5)
    result = ntops.torch.roll(x, shifts=2, dims=2)
    expected = torch.roll(x, shifts=2, dims=2)
    assert torch.equal(result, expected)


# =============================================================================
# Edge cases
# =============================================================================

def test_roll_zero_shift():
    """Roll with shift=0 should return the same tensor."""
    x = torch.randn(3, 4, device="cuda")
    result = ntops.torch.roll(x, shifts=0, dims=0)
    assert torch.equal(result, x)


def test_roll_shift_larger_than_dim():
    """Roll with shift > dimension length (should apply modulo)."""
    x = torch.tensor([1, 2, 3, 4, 5], device="cuda")
    result = ntops.torch.roll(x, shifts=7, dims=0)  # 7 % 5 = 2
    expected = torch.tensor([4, 5, 1, 2, 3], device="cuda")
    assert torch.equal(result, expected)


def test_roll_int_shifts():
    """Roll with int shifts (not list/tuple)."""
    x = torch.randn(3, 4, device="cuda")
    result = ntops.torch.roll(x, shifts=2, dims=1)
    expected = torch.roll(x, shifts=2, dims=1)
    assert torch.equal(result, expected)


def test_roll_int_dims():
    """Roll with int dims (not list/tuple)."""
    x = torch.randn(3, 4, device="cuda")
    result = ntops.torch.roll(x, shifts=(1,), dims=1)
    expected = torch.roll(x, shifts=(1,), dims=1)
    assert torch.equal(result, expected)


def test_roll_single_element():
    """Roll a single-element tensor (should be a no-op)."""
    x = torch.tensor([42], device="cuda")
    result = ntops.torch.roll(x, shifts=5, dims=0)
    assert torch.equal(result, x)


def test_roll_symmetry():
    """Roll forward then backward should return original."""
    x = torch.randn(5, 10, device="cuda")
    result = ntops.torch.roll(x, shifts=3, dims=1)
    result = ntops.torch.roll(result, shifts=-3, dims=1)
    assert torch.equal(result, x)


def test_roll_very_large_shift():
    """Roll with shift much larger than dimension."""
    x = torch.tensor([1, 2, 3], device="cuda")
    result = ntops.torch.roll(x, shifts=1000003, dims=0)  # 1000003 % 3 = 1
    expected = torch.tensor([3, 1, 2], device="cuda")
    assert torch.equal(result, expected)


# =============================================================================
# Dtype and device tests
# =============================================================================

@pytest.mark.parametrize("dtype", [
    torch.float32,
    torch.float16,
    torch.float64,
    torch.int32,
    torch.int64,
    torch.uint8,
    torch.bool,
])
def test_roll_dtype_preservation(dtype):
    """Roll should preserve the input dtype."""
    if dtype == torch.bool:
        x = torch.tensor([True, False, True, False, True], device="cuda")
    else:
        x = torch.arange(10, device="cuda").to(dtype)
    result = ntops.torch.roll(x, shifts=3, dims=0)
    assert result.dtype == dtype


def test_roll_device_preservation():
    """Roll should preserve the device."""
    x = torch.randn(10, device="cuda")
    result = ntops.torch.roll(x, shifts=3, dims=0)
    assert result.device == x.device


# =============================================================================
# Gradient test
# =============================================================================

def test_roll_gradient():
    """Roll should support gradient propagation."""
    x = torch.randn(3, 4, device="cuda", requires_grad=True)
    y = ntops.torch.roll(x, shifts=1, dims=1)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


# =============================================================================
# Four mandatory checks
# =============================================================================

def test_roll_no_nan():
    """Output should not contain NaN."""
    x = torch.randn(100, 100, device="cuda")
    result = ntops.torch.roll(x, shifts=10, dims=0)
    assert not torch.isnan(result).any()


def test_roll_no_inf():
    """Output should not contain Inf."""
    x = torch.randn(100, 100, device="cuda")
    result = ntops.torch.roll(x, shifts=10, dims=0)
    assert not torch.isinf(result).any()


def test_roll_int_exact():
    """Integer roll should be exact match (no precision loss)."""
    x = torch.randint(0, 100, (10, 10), device="cuda")
    result = ntops.torch.roll(x, shifts=3, dims=0)
    expected = torch.roll(x, shifts=3, dims=0)
    assert torch.equal(result, expected)


# =============================================================================
# Negative shifts on all dims
# =============================================================================

def test_roll_negative_shift_last_dim():
    """Negative shift on last dimension."""
    x = torch.arange(12, device="cuda").reshape(3, 4)
    result = ntops.torch.roll(x, shifts=-1, dims=-1)
    expected = torch.roll(x, shifts=-1, dims=-1)
    assert torch.equal(result, expected)
