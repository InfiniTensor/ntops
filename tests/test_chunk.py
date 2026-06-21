"""
chunk 算子测试脚本
"""
import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


@skip_if_cuda_not_available
def test_chunk_basic():
    """Test basic chunk functionality"""
    x = torch.arange(10, device="cuda").reshape(5, 2)
    result = ntops.torch.chunk(x, chunks=2, dim=0)

    assert len(result) == 2
    # 5 // 2 = 2, 5 % 2 = 1
    # First chunk: 2 + 1 = 3 rows
    # Second chunk: 2 rows
    assert result[0].shape == (3, 2)
    assert result[1].shape == (2, 2)

    # Verify data
    expected_0 = x[:3]
    expected_1 = x[3:]
    assert torch.equal(result[0], expected_0)
    assert torch.equal(result[1], expected_1)


@skip_if_cuda_not_available
def test_chunk_exact_division():
    """Test when size is exactly divisible by chunks"""
    x = torch.arange(12, device="cuda").reshape(6, 2)
    result = ntops.torch.chunk(x, chunks=3, dim=0)

    assert len(result) == 3
    # 6 // 3 = 2, 6 % 3 = 0
    # All chunks have 2 rows
    assert result[0].shape == (2, 2)
    assert result[1].shape == (2, 2)
    assert result[2].shape == (2, 2)


@skip_if_cuda_not_available
def test_chunk_dim_1():
    """Test chunking along dimension 1"""
    x = torch.arange(20, device="cuda").reshape(4, 5)
    result = ntops.torch.chunk(x, chunks=2, dim=1)

    assert len(result) == 2
    # 5 // 2 = 2, 5 % 2 = 1
    # First chunk: 2 + 1 = 3 columns
    # Second chunk: 2 columns
    assert result[0].shape == (4, 3)
    assert result[1].shape == (4, 2)


@skip_if_cuda_not_available
def test_chunk_dim_minus_1():
    """Test chunking along last dimension"""
    x = torch.arange(15, device="cuda").reshape(3, 5)
    result = ntops.torch.chunk(x, chunks=3, dim=-1)

    assert len(result) == 3
    # 5 // 3 = 1, 5 % 3 = 2
    # First two chunks: 1 + 1 = 2 columns
    # Third chunk: 1 column
    assert result[0].shape == (3, 2)
    assert result[1].shape == (3, 2)
    assert result[2].shape == (3, 1)


@skip_if_cuda_not_available
def test_chunk_3d_tensor():
    """Test chunking 3D tensor"""
    x = torch.randn(4, 6, 8, device="cuda")
    result = ntops.torch.chunk(x, chunks=2, dim=1)

    assert len(result) == 2
    # 6 // 2 = 3, 6 % 2 = 0
    # Both chunks have 3 elements in dim 1
    assert result[0].shape == (4, 3, 8)
    assert result[1].shape == (4, 3, 8)


@skip_if_cuda_not_available
def test_chunk_large_remainder():
    """Test when remainder is large"""
    x = torch.arange(17, device="cuda").reshape(17, 1)
    result = ntops.torch.chunk(x, chunks=5, dim=0)

    assert len(result) == 5
    # 17 // 5 = 3, 17 % 5 = 2
    # First two chunks: 3 + 1 = 4 elements
    # Last three chunks: 3 elements
    assert result[0].shape == (4, 1)
    assert result[1].shape == (4, 1)
    assert result[2].shape == (3, 1)
    assert result[3].shape == (3, 1)
    assert result[4].shape == (3, 1)


@skip_if_cuda_not_available
def test_chunk_size_equals_chunks():
    """Test when size equals chunks"""
    x = torch.arange(5, device="cuda")
    result = ntops.torch.chunk(x, chunks=5, dim=0)

    assert len(result) == 5
    # 5 // 5 = 1, 5 % 5 = 0
    # All chunks have 1 element
    for chunk in result:
        assert chunk.shape == (1,)


@skip_if_cuda_not_available
def test_chunk_data_integrity():
    """Verify that chunked data matches original"""
    x = torch.arange(20, device="cuda").reshape(5, 4)
    result = ntops.torch.chunk(x, chunks=3, dim=0)

    # Reconstruct by concatenating
    reconstructed = torch.cat(result, dim=0)
    assert torch.equal(reconstructed, x)


@skip_if_cuda_not_available
def test_chunk_preserves_dtype():
    """Test that dtype is preserved in all chunks"""
    for dtype in [torch.float16, torch.float32, torch.float64]:
        x = torch.randn(10, 5, device="cuda", dtype=dtype)
        result = ntops.torch.chunk(x, chunks=2, dim=0)
        for chunk in result:
            assert chunk.dtype == dtype


@skip_if_cuda_not_available
def test_chunk_preserves_device():
    """Test that device is preserved in all chunks"""
    x = torch.randn(10, 5, device="cuda")
    result = ntops.torch.chunk(x, chunks=2, dim=0)
    for chunk in result:
        assert chunk.device.type == "cuda"


@skip_if_cuda_not_available
def test_chunk_gradient():
    """Test that gradients flow through chunk correctly"""
    x = torch.randn(6, 4, device="cuda", requires_grad=True)
    result = ntops.torch.chunk(x, chunks=2, dim=0)

    # Sum both chunks and backprop
    loss = result[0].sum() + result[1].sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    # All gradients should be 1
    assert torch.allclose(x.grad, torch.ones_like(x))


@skip_if_cuda_not_available
def test_chunk_single_element():
    """Test chunking with single element result"""
    x = torch.arange(3, device="cuda").reshape(3, 1)
    result = ntops.torch.chunk(x, chunks=3, dim=0)

    assert len(result) == 3
    for i, chunk in enumerate(result):
        assert chunk.shape == (1, 1)
        assert chunk[0, 0].item() == i


@skip_if_cuda_not_available
def test_chunk_non_contiguous():
    """Test chunking non-contiguous (transposed) tensor"""
    x = torch.randn(3, 5, 2, device="cuda")
    x_t = x.permute(2, 0, 1)  # Non-contiguous, shape (2, 3, 5)
    result = ntops.torch.chunk(x_t, chunks=2, dim=1)

    assert len(result) == 2
    # dim 1 has size 3
    # 3 // 2 = 1, 3 % 2 = 1
    # First chunk: 1 + 1 = 2 elements in dim 1
    # Second chunk: 1 element in dim 1
    assert result[0].shape == (2, 2, 5)
    assert result[1].shape == (2, 1, 5)


@skip_if_cuda_not_available
def test_chunk_default_dim():
    """Test default dim=0"""
    x = torch.arange(8, device="cuda")
    result = ntops.torch.chunk(x, chunks=2)

    assert len(result) == 2
    # 8 // 2 = 4, 8 % 2 = 0
    # Both chunks have 4 elements
    assert result[0].shape == (4,)
    assert result[1].shape == (4,)
