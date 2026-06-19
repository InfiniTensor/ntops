import numpy as np
import pytest
import torch
import ntops


# =============================================================================
# CPU reference implementation
# =============================================================================

def cartesian_prod_cpu(*tensors):
    arrs = [np.asarray(x).flatten() for x in tensors]
    ndim = len(arrs)
    shapes = []
    for i in range(ndim):
        shp = [1] * ndim
        shp[i] = -1
        shapes.append(shp)
    grids = [arr.reshape(shp) for arr, shp in zip(arrs, shapes)]
    out = np.broadcast_arrays(*grids)
    flat = [g.reshape(-1, 1) for g in out]
    return np.concatenate(flat, axis=1)


# =============================================================================
# Basic functionality tests
# =============================================================================

def test_cartesian_prod_two_1d():
    """Cartesian product of two 1D tensors."""
    x_np = np.array([1, 2])
    y_np = np.array([3, 4, 5])
    xt = torch.tensor([1, 2], device="cuda")
    yt = torch.tensor([3, 4, 5], device="cuda")
    result = ntops.torch.cartesian_prod(xt, yt)
    expected = cartesian_prod_cpu(x_np, y_np)
    assert torch.equal(result, torch.tensor(expected, device="cuda"))


def test_cartesian_prod_three_1d():
    """Cartesian product of three 1D tensors."""
    x_np = np.array([1, 2])
    y_np = np.array([3, 4])
    z_np = np.array([5, 6])
    xt = torch.tensor([1, 2], device="cuda")
    yt = torch.tensor([3, 4], device="cuda")
    zt = torch.tensor([5, 6], device="cuda")
    result = ntops.torch.cartesian_prod(xt, yt, zt)
    expected = cartesian_prod_cpu(x_np, y_np, z_np)
    assert torch.equal(result, torch.tensor(expected, device="cuda"))


def test_cartesian_prod_multidim_input():
    """Multi-dimensional inputs are flattened."""
    x_np = np.array([[1, 2], [3, 4]])
    y_np = np.array([5, 6])
    xt = torch.tensor([[1, 2], [3, 4]], device="cuda")
    yt = torch.tensor([5, 6], device="cuda")
    result = ntops.torch.cartesian_prod(xt, yt)
    expected = cartesian_prod_cpu(x_np, y_np)
    assert torch.equal(result, torch.tensor(expected, device="cuda"))


def test_cartesian_prod_single_input():
    """Cartesian product of a single tensor."""
    x_np = np.array([1, 2, 3])
    xt = torch.tensor([1, 2, 3], device="cuda")
    result = ntops.torch.cartesian_prod(xt)
    expected = cartesian_prod_cpu(x_np)
    assert torch.equal(result, torch.tensor(expected, device="cuda"))


def test_cartesian_prod_large():
    """Large Cartesian product."""
    x = torch.arange(10, device="cuda")
    y = torch.arange(20, device="cuda")
    result = ntops.torch.cartesian_prod(x, y)
    assert result.shape == (200, 2)
    # First row: (0, 0)
    assert result[0, 0] == 0 and result[0, 1] == 0
    # Last row: (9, 19)
    assert result[-1, 0] == 9 and result[-1, 1] == 19


# =============================================================================
# Dtype and device tests
# =============================================================================

@pytest.mark.parametrize("dtype", [
    torch.float32,
    torch.float16,
    torch.int32,
    torch.int64,
])
def test_cartesian_prod_dtype(dtype):
    """Cartesian product should preserve input dtype."""
    x = torch.tensor([1, 2], device="cuda").to(dtype)
    y = torch.tensor([3, 4], device="cuda").to(dtype)
    result = ntops.torch.cartesian_prod(x, y)
    assert result.dtype == dtype


def test_cartesian_prod_device():
    """Output should be on the input device."""
    x = torch.tensor([1, 2], device="cuda")
    y = torch.tensor([3, 4], device="cuda")
    result = ntops.torch.cartesian_prod(x, y)
    assert result.device == x.device


def test_cartesian_prod_mixed_dtype():
    """Mixed dtypes should upcast to common dtype."""
    x = torch.tensor([1, 2], device="cuda")  # int64
    y = torch.tensor([3.0, 4.0], device="cuda")  # float32
    result = ntops.torch.cartesian_prod(x, y)
    assert result.dtype == torch.float32


# =============================================================================
# Four mandatory checks
# =============================================================================

def test_cartesian_prod_no_nan():
    """Output should not contain NaN."""
    x = torch.randn(10, device="cuda")
    y = torch.randn(20, device="cuda")
    result = ntops.torch.cartesian_prod(x, y)
    assert not torch.isnan(result).any()


def test_cartesian_prod_no_inf():
    """Output should not contain Inf."""
    x = torch.arange(10, device="cuda").float()
    y = torch.arange(20, device="cuda").float()
    result = ntops.torch.cartesian_prod(x, y)
    assert not torch.isinf(result).any()


def test_cartesian_prod_int_exact():
    """Integer Cartesian product should be exact match."""
    x_np = np.array([10, 20, 30])
    y_np = np.array([1, 2, 3, 4, 5])
    xt = torch.tensor([10, 20, 30], device="cuda")
    yt = torch.tensor([1, 2, 3, 4, 5], device="cuda")
    result = ntops.torch.cartesian_prod(xt, yt)
    expected = cartesian_prod_cpu(x_np, y_np)
    assert torch.equal(result, torch.tensor(expected, device="cuda"))
