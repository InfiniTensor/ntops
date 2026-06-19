import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


@skip_if_cuda_not_available
def test_count_nonzero_basic():
    """Basic counting of nonzero elements."""
    x = torch.tensor([[1, 0, 3], [0, 5, 0]], device="cuda")
    result = ntops.torch.count_nonzero(x)
    expected = torch.count_nonzero(x)
    assert result.item() == expected.item()
    assert result.item() == 3


@skip_if_cuda_not_available
def test_count_nonzero_all_zero():
    """All zero input."""
    x = torch.zeros(3, 4, device="cuda")
    result = ntops.torch.count_nonzero(x)
    expected = torch.count_nonzero(x)
    assert result.item() == expected.item()
    assert result.item() == 0


@skip_if_cuda_not_available
def test_count_nonzero_all_nonzero():
    """All nonzero input."""
    x = torch.ones(3, 4, device="cuda")
    result = ntops.torch.count_nonzero(x)
    expected = torch.count_nonzero(x)
    assert result.item() == expected.item()
    assert result.item() == 12


@skip_if_cuda_not_available
@pytest.mark.parametrize("dim", [0, 1])
def test_count_nonzero_dim(dim):
    """Counting along a specific dimension."""
    x = torch.tensor([[1, 0, 3], [0, 5, 0]], device="cuda")
    result = ntops.torch.count_nonzero(x, dim=dim)
    expected = torch.count_nonzero(x, dim=dim)
    assert torch.equal(result, expected)


@skip_if_cuda_not_available
def test_count_nonzero_keepdim():
    """Counting with keepdim=True."""
    x = torch.tensor([[1, 0, 3], [0, 5, 0]], device="cuda")

    result0 = ntops.torch.count_nonzero(x, dim=0, keepdim=True)
    expected0 = torch.count_nonzero(x, dim=0).unsqueeze(0)
    assert torch.equal(result0, expected0)
    assert result0.ndim == x.ndim

    result1 = ntops.torch.count_nonzero(x, dim=1, keepdim=True)
    expected1 = torch.count_nonzero(x, dim=1).unsqueeze(1)
    assert torch.equal(result1, expected1)
    assert result1.ndim == x.ndim


@skip_if_cuda_not_available
def test_count_nonzero_float():
    """Float tensor with zeros."""
    x = torch.tensor([0.0, 1.5, -2.3, 0.0, 3.14], device="cuda")
    result = ntops.torch.count_nonzero(x)
    expected = torch.count_nonzero(x)
    assert result.item() == expected.item()
    assert result.item() == 3


@skip_if_cuda_not_available
def test_count_nonzero_3d():
    """3D tensor."""
    x = torch.tensor([[[1, 0], [0, 0]], [[0, 2], [3, 0]]], device="cuda")
    result = ntops.torch.count_nonzero(x)
    expected = torch.count_nonzero(x)
    assert result.item() == expected.item()
    assert result.item() == 3


@skip_if_cuda_not_available
def test_count_nonzero_3d_dim():
    """3D tensor with dim."""
    x = torch.tensor([[[1, 0], [0, 0]], [[0, 2], [3, 0]]], device="cuda")
    for dim in range(3):
        result = ntops.torch.count_nonzero(x, dim=dim)
        expected = torch.count_nonzero(x, dim=dim)
        assert torch.equal(result, expected)


@skip_if_cuda_not_available
def test_count_nonzero_large():
    """Large random tensor."""
    x = torch.randint(0, 5, (100, 100), device="cuda")
    result = ntops.torch.count_nonzero(x)
    expected = torch.count_nonzero(x)
    assert result.item() == expected.item()


@skip_if_cuda_not_available
def test_count_nonzero_empty():
    """Empty tensor."""
    x = torch.empty(0, 3, device="cuda")
    result = ntops.torch.count_nonzero(x)
    expected = torch.count_nonzero(x)
    assert result.item() == expected.item()
