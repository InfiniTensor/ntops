import numpy as np
import pytest
import torch
import ntops


# =============================================================================
# CPU reference implementation
# =============================================================================

def mode_cpu(x_np: np.ndarray, dim: int, keepdim=False):
    """CPU reference matching the specification."""
    ndim = x_np.ndim
    perm = [i for i in range(ndim) if i != dim] + [dim]
    x_t = np.transpose(x_np, perm)
    out_shape = list(x_t.shape[:-1])
    vals = np.zeros(out_shape, dtype=x_np.dtype)
    cnts = np.zeros(out_shape, dtype=np.int64)
    for idx in np.ndindex(*out_shape):
        vec = x_t[idx]
        unique, counts = np.unique(vec, return_counts=True)
        max_idx = np.argmax(counts)
        vals[idx] = unique[max_idx]
        cnts[idx] = counts[max_idx]
    if keepdim:
        vals = np.expand_dims(vals, axis=dim)
        cnts = np.expand_dims(cnts, axis=dim)
    return vals, cnts


def assert_mode_valid(result_vals, result_cnts, x, dim, keepdim):
    """
    Verify that:
    1. The returned value is indeed a mode (its count = max count for that position)
    2. Counts are correct
    """
    # Check that the mode value actually appears `cnts` times along dim
    if keepdim:
        mode_vals = result_vals
    else:
        mode_vals = result_vals.unsqueeze(dim)

    computed_cnts = (x == mode_vals).sum(dim=dim, keepdim=keepdim).to(torch.int64)

    # The count computed from the mode value must match the returned count
    flat_result = result_cnts.reshape(-1)
    flat_computed = computed_cnts.reshape(-1)
    for i in range(flat_result.shape[0]):
        assert flat_result[i] == flat_computed[i], (
            f"Count mismatch at position {i}: "
            f"reported={flat_result[i]}, computed from value={flat_computed[i]}"
        )

    # The count must be at least 1 (mode always exists)
    assert (result_cnts >= 1).all(), "Counts must be >= 1"


# =============================================================================
# Basic functionality tests
# =============================================================================

def test_mode_1d():
    """Mode of a 1D tensor."""
    x_np = np.array([1, 2, 2, 3, 3, 3], dtype=np.int64)
    x = torch.from_numpy(x_np).cuda()
    vals, cnts = ntops.torch.mode(x, dim=0)
    _, c_ref = mode_cpu(x_np, dim=0)
    assert torch.equal(cnts, torch.tensor(c_ref, device="cuda"))
    assert_mode_valid(vals, cnts, x, dim=0, keepdim=False)


def test_mode_2d_dim0():
    """Mode of 2D tensor along dim=0 (column-wise)."""
    np.random.seed(42)
    x_np = np.random.randint(0, 5, (4, 3))
    x = torch.from_numpy(x_np).cuda()
    vals, cnts = ntops.torch.mode(x, dim=0)
    _, c_ref = mode_cpu(x_np, dim=0)
    assert torch.equal(cnts, torch.tensor(c_ref, device="cuda"))
    assert_mode_valid(vals, cnts, x, dim=0, keepdim=False)


def test_mode_2d_dim1():
    """Mode of 2D tensor along dim=1 (row-wise)."""
    np.random.seed(42)
    x_np = np.random.randint(0, 5, (4, 3))
    x = torch.from_numpy(x_np).cuda()
    vals, cnts = ntops.torch.mode(x, dim=1)
    _, c_ref = mode_cpu(x_np, dim=1)
    assert torch.equal(cnts, torch.tensor(c_ref, device="cuda"))
    assert_mode_valid(vals, cnts, x, dim=1, keepdim=False)


def test_mode_keepdim():
    """Mode with keepdim=True."""
    np.random.seed(42)
    x_np = np.random.randint(0, 5, (4, 3))
    x = torch.from_numpy(x_np).cuda()
    vals, cnts = ntops.torch.mode(x, dim=0, keepdim=True)
    _, c_ref = mode_cpu(x_np, dim=0, keepdim=True)
    assert vals.shape == (1, 3)
    assert cnts.shape == (1, 3)
    assert torch.equal(cnts, torch.tensor(c_ref, device="cuda"))
    assert_mode_valid(vals, cnts, x, dim=0, keepdim=True)


def test_mode_3d():
    """Mode of a 3D tensor."""
    np.random.seed(42)
    x_np = np.random.randint(0, 5, (3, 4, 5))
    x = torch.from_numpy(x_np).cuda()
    for dim in [0, 1, 2]:
        vals, cnts = ntops.torch.mode(x, dim=dim)
        _, c_ref = mode_cpu(x_np, dim=dim)
        assert torch.equal(cnts, torch.tensor(c_ref, device="cuda")), f"dim={dim} counts"
        assert_mode_valid(vals, cnts, x, dim=dim, keepdim=False)


# =============================================================================
# Float type tests
# =============================================================================

def test_mode_float32():
    """Mode with float32 inputs."""
    np.random.seed(42)
    x_np = np.random.randn(3, 5).astype(np.float32)
    x = torch.from_numpy(x_np).cuda()
    vals, cnts = ntops.torch.mode(x, dim=0)
    _, c_ref = mode_cpu(x_np, dim=0)
    assert torch.equal(cnts, torch.tensor(c_ref, device="cuda"))
    assert_mode_valid(vals, cnts, x, dim=0, keepdim=False)


def test_mode_float16():
    """Mode with float16 inputs."""
    np.random.seed(42)
    x_np = np.random.randint(0, 10, (4, 5)).astype(np.float16)
    x = torch.from_numpy(x_np.astype(np.float32)).cuda().to(torch.float16)
    vals, cnts = ntops.torch.mode(x, dim=0)
    assert vals.dtype == torch.float16
    assert cnts.dtype == torch.int64
    assert_mode_valid(vals, cnts, x, dim=0, keepdim=False)


# =============================================================================
# Edge cases
# =============================================================================

def test_mode_all_same():
    """Mode when all values are the same."""
    x = torch.ones(5, 3, device="cuda")
    vals, cnts = ntops.torch.mode(x, dim=0)
    assert torch.equal(vals, torch.ones(3, device="cuda"))
    assert torch.equal(cnts, torch.full((3,), 5, dtype=torch.int64, device="cuda"))


def test_mode_all_unique():
    """Mode when all values are unique."""
    x = torch.tensor([[1, 4], [2, 5], [3, 6]], device="cuda")
    vals, cnts = ntops.torch.mode(x, dim=0)
    # Any value is a valid mode (all have count=1)
    assert torch.equal(cnts, torch.ones(2, dtype=torch.int64, device="cuda"))
    assert_mode_valid(vals, cnts, x, dim=0, keepdim=False)


def test_mode_single_row():
    """Mode on a single-row tensor."""
    x = torch.tensor([[1, 2, 3]], device="cuda")
    vals, cnts = ntops.torch.mode(x, dim=0)
    assert_mode_valid(vals, cnts, x, dim=0, keepdim=False)


def test_mode_negative_dim():
    """Mode with negative dim indexing."""
    np.random.seed(42)
    x_np = np.random.randint(0, 5, (4, 3))
    x = torch.from_numpy(x_np).cuda()
    vals1, cnts1 = ntops.torch.mode(x, dim=0)
    vals2, cnts2 = ntops.torch.mode(x, dim=-2)
    assert torch.equal(cnts1, cnts2)


# =============================================================================
# Dtype and device tests
# =============================================================================

@pytest.mark.parametrize("dtype", [
    torch.int32,
    torch.int64,
    torch.float32,
    torch.float64,
])
def test_mode_dtype_preservation(dtype):
    """Mode values should preserve input dtype."""
    np.random.seed(42)
    if dtype in (torch.int32, torch.int64):
        x_np = np.random.randint(0, 10, (5, 4))
    else:
        x_np = np.random.randn(5, 4).astype(
            np.float32 if dtype == torch.float32 else np.float64
        )
    x = torch.from_numpy(x_np).cuda().to(dtype)
    vals, cnts = ntops.torch.mode(x, dim=0)
    assert vals.dtype == dtype
    assert cnts.dtype == torch.int64


def test_mode_device_preservation():
    """Mode output should be on the input device."""
    x = torch.randint(0, 10, (5, 4), device="cuda")
    vals, cnts = ntops.torch.mode(x, dim=0)
    assert vals.device == x.device
    assert cnts.device == x.device


# =============================================================================
# Gradient test
# =============================================================================

def test_mode_gradient():
    """Mode should support gradient propagation on values."""
    x = torch.randn(3, 5, device="cuda", requires_grad=True)
    vals, _ = ntops.torch.mode(x, dim=0)
    loss = vals.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


# =============================================================================
# Large tensor test
# =============================================================================

def test_mode_large():
    """Mode on a larger tensor."""
    np.random.seed(42)
    x_np = np.random.randint(0, 20, (128, 256))
    x = torch.from_numpy(x_np).cuda()
    vals, cnts = ntops.torch.mode(x, dim=0)
    _, c_ref = mode_cpu(x_np, dim=0)
    assert torch.equal(cnts, torch.tensor(c_ref, device="cuda"))
    assert_mode_valid(vals, cnts, x, dim=0, keepdim=False)


# =============================================================================
# Four mandatory checks
# =============================================================================

def test_mode_no_nan():
    """Output should not contain NaN."""
    x = torch.randint(0, 10, (100, 100), device="cuda")
    vals, cnts = ntops.torch.mode(x, dim=0)
    assert not torch.isnan(vals).any()
    assert not torch.isnan(cnts.float()).any()


def test_mode_no_inf():
    """Output should not contain Inf."""
    x = torch.randint(0, 10, (100, 100), device="cuda")
    vals, cnts = ntops.torch.mode(x, dim=0)
    assert not torch.isinf(vals).any()
    assert not torch.isinf(cnts.float()).any()


def test_mode_int_exact():
    """Mode counts should be exact match with CPU reference."""
    np.random.seed(42)
    x_np = np.random.randint(0, 50, (20, 30))
    x = torch.from_numpy(x_np).cuda()
    vals, cnts = ntops.torch.mode(x, dim=0)
    _, c_ref = mode_cpu(x_np, dim=0)
    assert torch.equal(cnts, torch.tensor(c_ref, device="cuda"))
    assert_mode_valid(vals, cnts, x, dim=0, keepdim=False)


# =============================================================================
# Count output type test
# =============================================================================

def test_mode_counts_dtype():
    """Counts should always be int64."""
    x = torch.randint(0, 5, (4, 3), device="cuda")
    _, cnts = ntops.torch.mode(x, dim=0)
    assert cnts.dtype == torch.int64
