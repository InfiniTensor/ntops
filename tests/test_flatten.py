"""
flatten 算子测试脚本
"""
import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


@skip_if_cuda_not_available
def test_flatten_start_dim_0():
    """Test flattening from dimension 0 (complete flatten)"""
    x = torch.randn(2, 3, 4, device="cuda")
    result = ntops.torch.flatten(x, start_dim=0)
    expected = torch.flatten(x, start_dim=0)

    assert result.shape == expected.shape == (24,)
    assert torch.equal(result, expected)


@skip_if_cuda_not_available
def test_flatten_start_dim_1():
    """Test flattening from dimension 1"""
    x = torch.randn(2, 3, 4, device="cuda")
    result = ntops.torch.flatten(x, start_dim=1)
    expected = torch.flatten(x, start_dim=1)

    assert result.shape == expected.shape == (2, 12)
    assert torch.equal(result, expected)


@skip_if_cuda_not_available
def test_flatten_start_dim_1_4d():
    """Test flattening 4D tensor from dimension 1"""
    x = torch.randn(2, 3, 4, 5, device="cuda")
    result = ntops.torch.flatten(x, start_dim=1)
    expected = torch.flatten(x, start_dim=1)

    assert result.shape == (2, 60)
    assert torch.equal(result, expected)


@skip_if_cuda_not_available
def test_flatten_start_dim_2():
    """Test flattening from dimension 2"""
    x = torch.randn(2, 3, 4, device="cuda")
    result = ntops.torch.flatten(x, start_dim=2)
    expected = torch.flatten(x, start_dim=2)

    assert result.shape == (2, 3, 4)
    assert torch.equal(result, expected)


@skip_if_cuda_not_available
def test_flatten_1d_input():
    """Test flattening a 1D tensor (no change)"""
    x = torch.randn(10, device="cuda")
    result = ntops.torch.flatten(x, start_dim=0)
    expected = torch.flatten(x, start_dim=0)

    assert result.shape == expected.shape == (10,)
    assert torch.equal(result, expected)


@skip_if_cuda_not_available
def test_flatten_start_dim_equals_ndim():
    """Test when start_dim >= ndim (should return copy)"""
    x = torch.randn(2, 3, 4, device="cuda")
    result = ntops.torch.flatten(x, start_dim=3)

    # When start_dim >= ndim, our implementation returns a copy
    expected = x.clone()

    assert result.shape == expected.shape == (2, 3, 4)
    assert torch.equal(result, expected)


@skip_if_cuda_not_available
def test_flatten_default_start_dim():
    """Test default start_dim=0"""
    x = torch.randn(2, 3, 4, device="cuda")
    result = ntops.torch.flatten(x)
    expected = torch.flatten(x)

    assert result.shape == expected.shape == (24,)
    assert torch.equal(result, expected)


@skip_if_cuda_not_available
def test_flatten_5d_tensor():
    """Test 5D tensor"""
    x = torch.randn(2, 3, 4, 5, 6, device="cuda")
    result = ntops.torch.flatten(x, start_dim=2)
    expected = torch.flatten(x, start_dim=2)

    assert result.shape == (2, 3, 120)
    assert torch.equal(result, expected)


@skip_if_cuda_not_available
def test_flatten_contiguous():
    """Test that flatten works with contiguous tensors"""
    x = torch.randn(2, 3, 4, device="cuda").contiguous()
    result = ntops.torch.flatten(x, start_dim=1)
    expected = torch.flatten(x, start_dim=1)

    assert result.shape == (2, 12)
    assert torch.equal(result, expected)


@skip_if_cuda_not_available
def test_flatten_non_contiguous():
    """Test that flatten works with non-contiguous tensors"""
    x = torch.randn(3, 4, 2, device="cuda")
    x_t = x.permute(2, 0, 1)  # Non-contiguous
    result = ntops.torch.flatten(x_t, start_dim=1)
    expected = torch.flatten(x_t, start_dim=1)

    assert result.shape == expected.shape
    assert torch.equal(result, expected)


@skip_if_cuda_not_available
def test_flatten_data_unchanged():
    """Verify that flatten doesn't change the data, only the shape"""
    x = torch.randn(2, 3, 4, device="cuda")
    result = ntops.torch.flatten(x, start_dim=1)

    # Modify the flattened tensor
    result[0, 0] = 999.0

    # The original should also be affected (they share memory)
    assert x[0, 0, 0].item() == pytest.approx(999.0, abs=1e-5)


@skip_if_cuda_not_available
def test_flatten_dtype_preservation():
    """Test that dtype is preserved"""
    for dtype in [torch.float16, torch.float32, torch.float64]:
        x = torch.randn(2, 3, 4, device="cuda", dtype=dtype)
        result = ntops.torch.flatten(x, start_dim=1)
        assert result.dtype == dtype


@skip_if_cuda_not_available
def test_flatten_gradient():
    """Test that gradients flow through flatten correctly"""
    x = torch.randn(2, 3, 4, device="cuda", requires_grad=True)
    result = ntops.torch.flatten(x, start_dim=1)
    loss = result.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
