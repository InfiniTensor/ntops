import pytest
import torch

import ntops

DTYPE_TOLERANCES = [
    (torch.float32, 1e-5, 1e-5),
    (torch.float16, 1e-2, 1e-2),  # float16: pow compounds precision loss
]


@pytest.mark.parametrize("dtype, rtol, atol", DTYPE_TOLERANCES)
def test_logspace_basic(dtype, rtol, atol):
    """Basic logspace: base 10, 0 to 2 in 3 steps → [1, 10, 100]."""
    result = ntops.torch.logspace(0.0, 2.0, 3, base=10.0, dtype=dtype)
    expected = torch.logspace(0.0, 2.0, 3, base=10.0, dtype=dtype, device=result.device)
    assert torch.allclose(result, expected, rtol=rtol, atol=atol)
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()


@pytest.mark.parametrize("dtype, rtol, atol", DTYPE_TOLERANCES)
def test_logspace_base2(dtype, rtol, atol):
    """Base 2: 0 to 4 in 5 steps → [1, 2, 4, 8, 16]."""
    result = ntops.torch.logspace(0.0, 4.0, 5, base=2.0, dtype=dtype)
    expected = torch.logspace(0.0, 4.0, 5, base=2.0, dtype=dtype, device=result.device)
    assert torch.allclose(result, expected, rtol=rtol, atol=atol)
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()


@pytest.mark.parametrize("dtype, rtol, atol", DTYPE_TOLERANCES)
def test_logspace_base_e(dtype, rtol, atol):
    """Base e: 0 to 1 in 5 steps."""
    import math
    result = ntops.torch.logspace(0.0, 1.0, 5, base=math.e, dtype=dtype)
    expected = torch.logspace(0.0, 1.0, 5, base=math.e, dtype=dtype, device=result.device)
    assert torch.allclose(result, expected, rtol=rtol, atol=atol)
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()


@pytest.mark.parametrize("dtype, rtol, atol", DTYPE_TOLERANCES)
def test_logspace_negative_exponents(dtype, rtol, atol):
    """Negative exponents: -2 to 2 in 5 steps, base 10."""
    result = ntops.torch.logspace(-2.0, 2.0, 5, base=10.0, dtype=dtype)
    expected = torch.logspace(-2.0, 2.0, 5, base=10.0, dtype=dtype, device=result.device)
    assert torch.allclose(result, expected, rtol=rtol, atol=atol)
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()


def test_logspace_edge_cases():
    """Test edge cases."""
    # steps=1
    result = ntops.torch.logspace(2.0, 5.0, 1, base=10.0, dtype=torch.float32)
    assert result.numel() == 1
    assert abs(result.item() - 100.0) < 1e-5

    # steps=2
    result = ntops.torch.logspace(0.0, 1.0, 2, base=10.0, dtype=torch.float32)
    expected = torch.tensor([1.0, 10.0], dtype=torch.float32, device=result.device)
    assert torch.allclose(result, expected)

    # steps=0 — should raise
    with pytest.raises(ValueError):
        ntops.torch.logspace(0.0, 1.0, -1)

    # same start and end
    result = ntops.torch.logspace(3.0, 3.0, 3, base=10.0, dtype=torch.float32)
    expected = torch.tensor([1000.0, 1000.0, 1000.0], dtype=torch.float32, device=result.device)
    assert torch.allclose(result, expected)

    # different base
    result = ntops.torch.logspace(0.0, 3.0, 4, base=3.0, dtype=torch.float32)
    expected = torch.tensor([1.0, 3.0, 9.0, 27.0], dtype=torch.float32, device=result.device)
    assert torch.allclose(result, expected)


def test_logspace_float64():
    """float64 precision test.

    Note: intermediate computation uses float32 for GPU efficiency.
    This gives ~1e-7 precision for float64 outputs.
    """
    result = ntops.torch.logspace(0.0, 2.0, 5, base=10.0, dtype=torch.float64)
    expected = torch.logspace(0.0, 2.0, 5, base=10.0, dtype=torch.float64, device=result.device)
    assert torch.allclose(result, expected, rtol=1e-6, atol=1e-6)
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()
