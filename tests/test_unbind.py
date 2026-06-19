"""
unbind 算子测试脚本
"""
import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


@skip_if_cuda_not_available
def test_unbind_basic():
    """Test basic unbind functionality"""
    x = torch.arange(12, device="cuda").reshape(3, 4)
    result = ntops.torch.unbind(x, dim=0)

    assert len(result) == 3
    # Each result has shape (4,) - dim 0 removed
    assert result[0].shape == (4,)
    assert result[1].shape == (4,)
    assert result[2].shape == (4,)

    # Verify data
    expected_0 = x[0]
    expected_1 = x[1]
    expected_2 = x[2]
    assert torch.equal(result[0], expected_0)
    assert torch.equal(result[1], expected_1)
    assert torch.equal(result[2], expected_2)


@skip_if_cuda_not_available
def test_unbind_dim_1():
    """Test unbinding along dimension 1"""
    x = torch.arange(12, device="cuda").reshape(3, 4)
    result = ntops.torch.unbind(x, dim=1)

    assert len(result) == 4
    # Each result has shape (3,) - dim 1 removed
    assert result[0].shape == (3,)
    assert result[1].shape == (3,)
    assert result[2].shape == (3,)
    assert result[3].shape == (3,)


@skip_if_cuda_not_available
def test_unbind_dim_minus_1():
    """Test unbinding along last dimension"""
    x = torch.randn(3, 5, 4, device="cuda")
    result = ntops.torch.unbind(x, dim=-1)

    assert len(result) == 4
    # Each result has shape (3, 5) - last dim removed
    for tensor in result:
        assert tensor.shape == (3, 5)


@skip_if_cuda_not_available
def test_unbind_3d_tensor():
    """Test unbinding 3D tensor"""
    x = torch.randn(2, 3, 4, device="cuda")
    result = ntops.torch.unbind(x, dim=1)

    assert len(result) == 3
    # Each result has shape (2, 4) - dim 1 removed
    for tensor in result:
        assert tensor.shape == (2, 4)


@skip_if_cuda_not_available
def test_unbind_4d_tensor():
    """Test unbinding 4D tensor"""
    x = torch.randn(2, 3, 4, 5, device="cuda")
    result = ntops.torch.unbind(x, dim=2)

    assert len(result) == 4
    # Each result has shape (2, 3, 5) - dim 2 removed
    for tensor in result:
        assert tensor.shape == (2, 3, 5)


@skip_if_cuda_not_available
def test_unbind_single_element_dim():
    """Test unbinding dimension with size 1"""
    x = torch.randn(1, 5, 3, device="cuda")
    result = ntops.torch.unbind(x, dim=0)

    assert len(result) == 1
    assert result[0].shape == (5, 3)


@skip_if_cuda_not_available
def test_unbind_data_integrity():
    """Verify that unbound data matches original"""
    x = torch.arange(20, device="cuda").reshape(4, 5)
    result = ntops.torch.unbind(x, dim=0)

    # Reconstruct by stacking
    reconstructed = torch.stack(result, dim=0)
    assert torch.equal(reconstructed, x)


@skip_if_cuda_not_available
def test_unbind_preserves_dtype():
    """Test that dtype is preserved in all tensors"""
    for dtype in [torch.float16, torch.float32, torch.float64]:
        x = torch.randn(3, 5, device="cuda", dtype=dtype)
        result = ntops.torch.unbind(x, dim=0)
        for tensor in result:
            assert tensor.dtype == dtype


@skip_if_cuda_not_available
def test_unbind_preserves_device():
    """Test that device is preserved in all tensors"""
    x = torch.randn(3, 5, device="cuda")
    result = ntops.torch.unbind(x, dim=0)
    for tensor in result:
        assert tensor.device.type == "cuda"


@skip_if_cuda_not_available
def test_unbind_gradient():
    """Test that gradients flow through unbind correctly"""
    x = torch.randn(3, 4, device="cuda", requires_grad=True)
    result = ntops.torch.unbind(x, dim=0)

    # Sum all tensors and backprop
    loss = sum(t.sum() for t in result)
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    # All gradients should be 1
    assert torch.allclose(x.grad, torch.ones_like(x))


@skip_if_cuda_not_available
def test_unbind_returns_tuple():
    """Test that unbind returns a tuple (not a list)"""
    x = torch.randn(3, 4, device="cuda")
    result = ntops.torch.unbind(x, dim=0)

    assert isinstance(result, tuple)


@skip_if_cuda_not_available
def test_unbind_non_contiguous():
    """Test unbinding non-contiguous (transposed) tensor"""
    x = torch.randn(3, 4, 2, device="cuda")
    x_t = x.permute(2, 0, 1)  # Non-contiguous, shape (2, 3, 4)
    result = ntops.torch.unbind(x_t, dim=1)

    assert len(result) == 3
    # Each result has shape (2, 4) - dim 1 removed
    for tensor in result:
        assert tensor.shape == (2, 4)


@skip_if_cuda_not_available
def test_unbind_default_dim():
    """Test default dim=0"""
    x = torch.arange(12, device="cuda").reshape(3, 4)
    result = ntops.torch.unbind(x)

    assert len(result) == 3
    for tensor in result:
        assert tensor.shape == (4,)


@skip_if_cuda_not_available
def test_unbind_large_dimension():
    """Test unbinding with many elements along dimension"""
    x = torch.randn(100, 5, device="cuda")
    result = ntops.torch.unbind(x, dim=0)

    assert len(result) == 100
    for tensor in result:
        assert tensor.shape == (5,)


@skip_if_cuda_not_available
def test_unbind_index_access():
    """Test that indexed access works correctly"""
    x = torch.arange(12, device="cuda").reshape(3, 4)
    result = ntops.torch.unbind(x, dim=0)

    # result[i] should equal x[i]
    for i in range(3):
        assert torch.equal(result[i], x[i])


@skip_if_cuda_not_available
def test_unbind_vs_chunk():
    """Compare unbind with chunk (chunk_size=1) - note the shape difference"""
    x = torch.randn(5, 10, device="cuda")

    # unbind along dim 0 - removes the dimension
    unbind_result = ntops.torch.unbind(x, dim=0)

    # chunk with chunks=5 (each chunk has 1 element) - keeps dimension
    chunk_result = ntops.torch.chunk(x, chunks=5, dim=0)

    # Both should have 5 elements
    assert len(unbind_result) == len(chunk_result) == 5

    # unbind removes dimension, chunk keeps it
    # unbind_result[i].shape: (10,)
    # chunk_result[i].shape: (1, 10)
    for i in range(5):
        assert unbind_result[i].shape == (10,)
        assert chunk_result[i].shape == (1, 10)
        # After squeezing chunk result, they should be equal
        assert torch.equal(unbind_result[i], chunk_result[i].squeeze(0))
