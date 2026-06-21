import numpy as np
import pytest
import torch
import ntops


# =============================================================================
# CPU reference implementation
# =============================================================================

def meshgrid_cpu(*xs, indexing="xy"):
    arrs = [np.asarray(x) for x in xs]
    ndim = len(arrs)
    shapes = []
    for i in range(ndim):
        shp = [1] * ndim
        shp[i] = -1
        shapes.append(shp)
    grids = [arr.reshape(shp) for arr, shp in zip(arrs, shapes)]
    out = np.broadcast_arrays(*grids)
    if indexing == "xy" and ndim >= 2:
        out = list(out)
        out[0], out[1] = out[1], out[0]
    return out


# =============================================================================
# Basic functionality tests
# =============================================================================

def test_meshgrid_2d_ij():
    """2D meshgrid with 'ij' indexing."""
    x_np = np.array([1, 2, 3])
    y_np = np.array([4, 5, 6, 7])
    xt = torch.tensor([1, 2, 3], device="cuda")
    yt = torch.tensor([4, 5, 6, 7], device="cuda")
    result = ntops.torch.meshgrid(xt, yt, indexing="ij")
    expected = meshgrid_cpu(x_np, y_np, indexing="ij")
    assert len(result) == 2
    for r, e in zip(result, expected):
        assert torch.equal(r, torch.tensor(e, device="cuda"))


def test_meshgrid_2d_xy():
    """2D meshgrid with 'xy' indexing."""
    x_np = np.array([1, 2, 3])
    y_np = np.array([4, 5, 6, 7])
    xt = torch.tensor([1, 2, 3], device="cuda")
    yt = torch.tensor([4, 5, 6, 7], device="cuda")
    result = ntops.torch.meshgrid(xt, yt, indexing="xy")
    expected = meshgrid_cpu(x_np, y_np, indexing="xy")
    assert len(result) == 2
    for r, e in zip(result, expected):
        assert torch.equal(r, torch.tensor(e, device="cuda"))


def test_meshgrid_3d_ij():
    """3D meshgrid with 'ij' indexing."""
    x_np = np.array([1, 2, 3])
    y_np = np.array([4, 5, 6, 7])
    z_np = np.array([8, 9])
    xt = torch.tensor([1, 2, 3], device="cuda")
    yt = torch.tensor([4, 5, 6, 7], device="cuda")
    zt = torch.tensor([8, 9], device="cuda")
    result = ntops.torch.meshgrid(xt, yt, zt, indexing="ij")
    expected = meshgrid_cpu(x_np, y_np, z_np, indexing="ij")
    for r, e in zip(result, expected):
        assert torch.equal(r, torch.tensor(e, device="cuda"))


def test_meshgrid_3d_xy():
    """3D meshgrid with 'xy' indexing."""
    x_np = np.array([1, 2])
    y_np = np.array([3, 4, 5])
    z_np = np.array([6, 7, 8, 9])
    xt = torch.tensor([1, 2], device="cuda")
    yt = torch.tensor([3, 4, 5], device="cuda")
    zt = torch.tensor([6, 7, 8, 9], device="cuda")
    result = ntops.torch.meshgrid(xt, yt, zt, indexing="xy")
    expected = meshgrid_cpu(x_np, y_np, z_np, indexing="xy")
    for r, e in zip(result, expected):
        assert torch.equal(r, torch.tensor(e, device="cuda"))


def test_meshgrid_1d():
    """1D meshgrid (single input)."""
    x = torch.tensor([1, 2, 3], device="cuda")
    result = ntops.torch.meshgrid(x, indexing="ij")
    assert len(result) == 1
    assert torch.equal(result[0], x)


def test_meshgrid_default_indexing():
    """Default indexing should be 'xy'."""
    x_np = np.array([1, 2, 3])
    y_np = np.array([4, 5, 6, 7])
    xt = torch.tensor([1, 2, 3], device="cuda")
    yt = torch.tensor([4, 5, 6, 7], device="cuda")
    result = ntops.torch.meshgrid(xt, yt)
    expected = meshgrid_cpu(x_np, y_np, indexing="xy")
    for r, e in zip(result, expected):
        assert torch.equal(r, torch.tensor(e, device="cuda"))


# =============================================================================
# Edge cases
# =============================================================================

def test_meshgrid_single_element():
    """Meshgrid with single-element inputs."""
    x = torch.tensor([42], device="cuda")
    y = torch.tensor([7], device="cuda")
    result = ntops.torch.meshgrid(x, y, indexing="ij")
    assert result[0].shape == (1, 1)
    assert result[0].item() == 42
    assert result[1].item() == 7


def test_meshgrid_large():
    """Meshgrid with larger inputs."""
    x = torch.arange(100, device="cuda")
    y = torch.arange(200, device="cuda")
    result = ntops.torch.meshgrid(x, y, indexing="ij")
    assert result[0].shape == (100, 200)
    # Verify a few positions
    assert result[0][0, 0] == 0
    assert result[0][50, 0] == 50
    assert result[0][0, 100] == 0
    assert result[1][0, 0] == 0
    assert result[1][0, 50] == 50


# =============================================================================
# Output is view (strided, not contiguous)
# =============================================================================

def test_meshgrid_output_is_view():
    """Meshgrid outputs should be broadcast views (zero-copy)."""
    x = torch.arange(10, device="cuda")
    y = torch.arange(20, device="cuda")
    result = ntops.torch.meshgrid(x, y, indexing="ij")
    # Broadcast views are not contiguous
    assert not result[0].is_contiguous()
    assert not result[1].is_contiguous()
    # But data is correct
    assert result[0][5, :].sum() == 20 * 5  # 20 copies of value 5 along dim 1
    assert result[1][0, :].sum() == 20 * 19 / 2  # values 0-19


# =============================================================================
# Dtype and device tests
# =============================================================================

@pytest.mark.parametrize("dtype", [
    torch.float32,
    torch.float16,
    torch.int32,
    torch.int64,
])
def test_meshgrid_dtype_preservation(dtype):
    """Meshgrid should preserve input dtype."""
    x = torch.tensor([1, 2, 3], device="cuda").to(dtype)
    y = torch.tensor([4, 5, 6, 7], device="cuda").to(dtype)
    result = ntops.torch.meshgrid(x, y, indexing="ij")
    for r in result:
        assert r.dtype == dtype


def test_meshgrid_device_preservation():
    """Meshgrid output should be on the input device."""
    x = torch.tensor([1, 2, 3], device="cuda")
    y = torch.tensor([4, 5, 6, 7], device="cuda")
    result = ntops.torch.meshgrid(x, y)
    for r in result:
        assert r.device == x.device


# =============================================================================
# Gradient test
# =============================================================================

def test_meshgrid_gradient():
    """Meshgrid should support gradient propagation."""
    x = torch.tensor([1.0, 2.0, 3.0], device="cuda", requires_grad=True)
    y = torch.tensor([4.0, 5.0, 6.0, 7.0], device="cuda", requires_grad=True)
    gx, gy = ntops.torch.meshgrid(x, y, indexing="ij")
    loss = gx.sum() + gy.sum()
    loss.backward()
    assert x.grad is not None
    # x is broadcast 4 times along dim 1 → grad = 4 per element
    assert torch.equal(x.grad, torch.tensor([4.0, 4.0, 4.0], device="cuda"))
    assert y.grad is not None
    # y is broadcast 3 times along dim 0 → grad = 3 per element
    assert torch.equal(y.grad, torch.tensor([3.0, 3.0, 3.0, 3.0], device="cuda"))


# =============================================================================
# Four mandatory checks
# =============================================================================

def test_meshgrid_no_nan():
    """Output should not contain NaN."""
    x = torch.randn(10, device="cuda")
    y = torch.randn(20, device="cuda")
    result = ntops.torch.meshgrid(x, y)
    for r in result:
        assert not torch.isnan(r).any()


def test_meshgrid_no_inf():
    """Output should not contain Inf."""
    x = torch.arange(10, device="cuda").float()
    y = torch.arange(20, device="cuda").float()
    result = ntops.torch.meshgrid(x, y)
    for r in result:
        assert not torch.isinf(r).any()


def test_meshgrid_int_exact():
    """Integer meshgrid should be exact match."""
    x_np = np.array([10, 20, 30])
    y_np = np.array([1, 2, 3, 4, 5])
    xt = torch.tensor([10, 20, 30], device="cuda")
    yt = torch.tensor([1, 2, 3, 4, 5], device="cuda")
    result = ntops.torch.meshgrid(xt, yt, indexing="ij")
    expected = meshgrid_cpu(x_np, y_np, indexing="ij")
    for r, e in zip(result, expected):
        assert torch.equal(r, torch.tensor(e, device="cuda"))
