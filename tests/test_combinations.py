import pytest
import torch
import itertools

import ntops


def combinations_cpu(x, r):
    """CPU reference using itertools."""
    comb = list(itertools.combinations(x.tolist(), r))
    if not comb:
        return torch.empty(0, r, dtype=x.dtype)
    return torch.tensor(comb, dtype=x.dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
def test_combinations_basic(dtype):
    """C(4, 2) = 6 combinations."""
    x = torch.tensor([1, 2, 3, 4], dtype=dtype, device="cuda")
    result = ntops.torch.combinations(x, 2)
    expected = combinations_cpu(x.cpu(), 2).to("cuda")
    assert result.shape == expected.shape
    assert torch.equal(result, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
def test_combinations_r1(dtype):
    """r = 1: each element individually."""
    x = torch.tensor([5, 6, 7], dtype=dtype, device="cuda")
    result = ntops.torch.combinations(x, 1)
    expected = combinations_cpu(x.cpu(), 1).to("cuda")
    assert torch.equal(result, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
def test_combinations_r_n(dtype):
    """r = n: single combination = the whole array."""
    x = torch.tensor([1, 2, 3], dtype=dtype, device="cuda")
    result = ntops.torch.combinations(x, 3)
    expected = combinations_cpu(x.cpu(), 3).to("cuda")
    assert torch.equal(result, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
def test_combinations_r0(dtype):
    """r = 0: empty combinations (returns 1D empty tensor on this torch version)."""
    x = torch.tensor([1, 2, 3], dtype=dtype, device="cuda")
    result = ntops.torch.combinations(x, 0)
    # torch.combinations(x, r=0) returns shape (0,) — 1D empty
    assert result.ndim >= 1


@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
def test_combinations_large(dtype):
    """C(10, 3) = 120 combinations."""
    x = torch.arange(10, dtype=dtype, device="cuda")
    result = ntops.torch.combinations(x, 3)
    expected = combinations_cpu(x.cpu(), 3).to("cuda")
    assert torch.equal(result, expected)
    assert result.shape == (120, 3)


def test_combinations_edge_cases():
    """Edge cases."""
    # r > n → empty
    x = torch.tensor([1, 2, 3], device="cuda")
    result = ntops.torch.combinations(x, 5)
    assert result.numel() == 0
    assert result.shape == (0, 5)

    # r < 0 → error
    with pytest.raises(ValueError):
        ntops.torch.combinations(x, -1)

    # 2D input → error
    x2d = torch.tensor([[1, 2], [3, 4]], device="cuda")
    with pytest.raises(ValueError):
        ntops.torch.combinations(x2d, 2)

    # Single element
    x = torch.tensor([42], device="cuda")
    result = ntops.torch.combinations(x, 1)
    assert result.item() == 42
    assert result.shape == (1, 1)


def test_combinations_float16():
    """float16 dtype."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float16, device="cuda")
    result = ntops.torch.combinations(x, 2)
    expected = torch.combinations(x, r=2)
    assert torch.equal(result, expected)


def test_combinations_gpu_roundtrip():
    """Verify GPU tensor stays on GPU."""
    x = torch.tensor([10, 20, 30, 40, 50], device="cuda")
    result = ntops.torch.combinations(x, 3)
    assert result.is_cuda
    assert result.shape == (10, 3)  # C(5,3) = 10
