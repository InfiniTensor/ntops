import pytest
import torch

import ntops

DTYPE_TOLERANCES = [
    (torch.float32, 1e-5, 1e-5),
    (torch.float16, 1e-3, 1e-3),
]


def make_special_tensor(device, dtype):
    """Create a tensor with NaN, +Inf, -Inf, zero, and normal values."""
    data = [
        float("nan"), float("inf"), float("-inf"),
        0.0, -0.0, 1.0, -1.0, 42.5, -3.14,
    ]
    return torch.tensor(data, dtype=dtype, device=device)


@pytest.mark.parametrize("dtype, rtol, atol", DTYPE_TOLERANCES)
def test_nan_to_num_default(dtype, rtol, atol):
    """Default replacements: NaN→0, Inf→max, -Inf→min."""
    x = make_special_tensor("cuda", dtype)
    result = ntops.torch.nan_to_num(x)

    # Check NaN replaced with 0
    assert not torch.isnan(result).any()
    # Check +Inf replaced
    assert not torch.isposinf(result).any()
    # Check -Inf replaced
    assert not torch.isneginf(result).any()
    # Normal values unchanged
    assert result[3].item() == 0.0
    assert result[4].item() == 0.0  # -0.0 → 0.0 in comparison
    assert result[5].item() == pytest.approx(1.0, rel=rtol)
    assert result[6].item() == pytest.approx(-1.0, rel=rtol)


@pytest.mark.parametrize("dtype, rtol, atol", DTYPE_TOLERANCES)
def test_nan_to_num_custom_values(dtype, rtol, atol):
    """Custom replacement values."""
    x = make_special_tensor("cuda", dtype)
    result = ntops.torch.nan_to_num(x, nan=-1.0, posinf=100.0, neginf=-100.0)

    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()
    # NaN replaced with -1.0
    assert result[0].item() == pytest.approx(-1.0, rel=rtol)
    # +Inf replaced with 100.0
    assert result[1].item() == pytest.approx(100.0, rel=rtol)
    # -Inf replaced with -100.0
    assert result[2].item() == pytest.approx(-100.0, rel=rtol)
    # Normal values unchanged
    assert result[5].item() == pytest.approx(1.0, rel=rtol)


@pytest.mark.parametrize("dtype, rtol, atol", DTYPE_TOLERANCES)
def test_nan_to_num_no_special_values(dtype, rtol, atol):
    """Input with no special values is unchanged."""
    x = torch.tensor([1.0, 2.0, 3.0, -4.0], dtype=dtype, device="cuda")
    result = ntops.torch.nan_to_num(x)
    assert torch.allclose(result, x, rtol=rtol, atol=atol)
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()


@pytest.mark.parametrize("dtype, rtol, atol", DTYPE_TOLERANCES)
def test_nan_to_num_all_nan(dtype, rtol, atol):
    """All NaN input."""
    x = torch.full((10,), float("nan"), dtype=dtype, device="cuda")
    result = ntops.torch.nan_to_num(x, nan=5.0)
    assert not torch.isnan(result).any()
    expected = torch.full((10,), 5.0, dtype=dtype, device="cuda")
    assert torch.allclose(result, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype, rtol, atol", DTYPE_TOLERANCES)
def test_nan_to_num_large(dtype, rtol, atol):
    """Large tensor with mixed special values."""
    x = torch.randn(10000, dtype=dtype, device="cuda")
    # Inject special values
    x[0] = float("nan")
    x[1] = float("inf")
    x[2] = float("-inf")
    x[100] = float("nan")
    x[200] = float("inf")

    result = ntops.torch.nan_to_num(x)
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()
    # Normal values unchanged (check a few)
    for i in [3, 4, 5, 10, 50, 500]:
        if not torch.isnan(x[i]) and not torch.isinf(x[i]):
            assert result[i].item() == pytest.approx(x[i].item(), rel=rtol)


def test_nan_to_num_edge_cases():
    """Edge cases."""
    # Empty tensor
    x = torch.empty(0, device="cuda")
    result = ntops.torch.nan_to_num(x)
    assert result.numel() == 0

    # Scalar NaN
    x = torch.tensor(float("nan"), device="cuda")
    result = ntops.torch.nan_to_num(x)
    assert result.item() == 0.0

    # Scalar Inf
    x = torch.tensor(float("inf"), device="cuda")
    result = ntops.torch.nan_to_num(x)
    assert result.item() == torch.finfo(torch.float32).max

    # 2D tensor
    x = torch.tensor([[float("nan"), 1.0], [float("inf"), -1.0]], device="cuda")
    result = ntops.torch.nan_to_num(x)
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()
    assert result[0, 1].item() == 1.0
    assert result[1, 1].item() == -1.0


def test_nan_to_num_float64():
    """float64 precision test."""
    x = torch.tensor([float("nan"), float("inf"), float("-inf"), 1.0], device="cuda", dtype=torch.float64)
    result = ntops.torch.nan_to_num(x)
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()
    assert result[0].item() == 0.0
    assert result[1].item() == torch.finfo(torch.float64).max
    assert result[2].item() == torch.finfo(torch.float64).min
    assert result[3].item() == 1.0


def test_nan_to_num_int():
    """Integer input returns clone (ints can't be NaN/Inf)."""
    x = torch.tensor([1, 2, 3, -4], dtype=torch.int32, device="cuda")
    result = ntops.torch.nan_to_num(x)
    assert torch.equal(result, x)
    # Should be a different tensor (clone)
    result[0] = 99
    assert x[0] == 1  # original unchanged
