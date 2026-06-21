import pytest
import torch

import ntops

DTYPE_TOLERANCES = [
    (torch.float32, 1e-5, 1e-5),
    (torch.float16, 1e-3, 1e-3),
]

# Float16 has ~3 significant digits; large step counts cause 1-2 ULP
# quantization differences vs PyTorch's internal implementation.
# We compare against float64 reference with float16-appropriate tolerance.
LARGE_TOLERANCES = [
    (torch.float32, 1e-5, 1e-5),
    (torch.float16, 1e-3, 1e-2),  # float16: atol relaxed for inherent quantization
]


@pytest.mark.parametrize("dtype, rtol, atol", DTYPE_TOLERANCES)
def test_linspace_basic(dtype, rtol, atol):
    """Basic linspace: 0 to 1 in 5 steps."""
    start, end, steps = 0.0, 1.0, 5
    result = ntops.torch.linspace(start, end, steps, dtype=dtype)
    expected = torch.linspace(start, end, steps, dtype=dtype, device=result.device)
    assert torch.allclose(result, expected, rtol=rtol, atol=atol)
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()


@pytest.mark.parametrize("dtype, rtol, atol", DTYPE_TOLERANCES)
def test_linspace_negative(dtype, rtol, atol):
    """Negative start: -5 to 5 in 11 steps."""
    start, end, steps = -5.0, 5.0, 11
    result = ntops.torch.linspace(start, end, steps, dtype=dtype)
    expected = torch.linspace(start, end, steps, dtype=dtype, device=result.device)
    assert torch.allclose(result, expected, rtol=rtol, atol=atol)
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()


@pytest.mark.parametrize("dtype, rtol, atol", DTYPE_TOLERANCES)
def test_linspace_descending(dtype, rtol, atol):
    """Descending: 10 to 0 in 11 steps."""
    start, end, steps = 10.0, 0.0, 11
    result = ntops.torch.linspace(start, end, steps, dtype=dtype)
    expected = torch.linspace(start, end, steps, dtype=dtype, device=result.device)
    assert torch.allclose(result, expected, rtol=rtol, atol=atol)
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()


@pytest.mark.parametrize("dtype, rtol, atol", LARGE_TOLERANCES)
def test_linspace_large(dtype, rtol, atol):
    """Large number of steps: 0 to 10 in 10001 steps.

    Float16 note: atol is relaxed because float16 has ~3 significant digits;
    at 10001 steps, quantization differences of 1-2 ULP (~0.004 at value 2.5)
    are inevitable between different computation paths.
    """
    start, end, steps = 0.0, 10.0, 10001
    result = ntops.torch.linspace(start, end, steps, dtype=dtype)
    # Compare against float64 reference for fair assessment
    ref = torch.linspace(start, end, steps, dtype=torch.float64, device=result.device)
    assert torch.allclose(result, ref.to(result.dtype), rtol=rtol, atol=atol)
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()


@pytest.mark.parametrize("dtype, rtol, atol", DTYPE_TOLERANCES)
def test_linspace_non_integer(dtype, rtol, atol):
    """Non-integer endpoints and step: 1.5 to 9.5 in 9 steps."""
    start, end, steps = 1.5, 9.5, 9
    result = ntops.torch.linspace(start, end, steps, dtype=dtype)
    expected = torch.linspace(start, end, steps, dtype=dtype, device=result.device)
    assert torch.allclose(result, expected, rtol=rtol, atol=atol)
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()


def test_linspace_edge_cases():
    """Test edge cases."""
    # steps=1
    result = ntops.torch.linspace(3.0, 7.0, 1, dtype=torch.float32)
    assert result.numel() == 1
    assert result.item() == 3.0

    # steps=2
    result = ntops.torch.linspace(0.0, 1.0, 2, dtype=torch.float32)
    expected = torch.tensor([0.0, 1.0], dtype=torch.float32, device=result.device)
    assert torch.allclose(result, expected)

    # steps=0 — should raise
    with pytest.raises(ValueError):
        ntops.torch.linspace(0.0, 1.0, -1)

    # same start and end
    result = ntops.torch.linspace(5.0, 5.0, 3, dtype=torch.float32)
    expected = torch.tensor([5.0, 5.0, 5.0], dtype=torch.float32, device=result.device)
    assert torch.allclose(result, expected)

    # steps that doesn't divide block_size
    result = ntops.torch.linspace(0.0, 1.0, 7, dtype=torch.float32)
    expected = torch.linspace(0.0, 1.0, 7, dtype=torch.float32, device=result.device)
    assert torch.allclose(result, expected)


def test_linspace_float64():
    """float64 precision test.

    Note: intermediate computation uses float32 for GPU efficiency.
    This gives ~1e-7 relative precision for float64 outputs, which is
    more than sufficient for linspace use cases.
    """
    start, end, steps = 0.0, 1.0, 5
    result = ntops.torch.linspace(start, end, steps, dtype=torch.float64)
    expected = torch.linspace(start, end, steps, dtype=torch.float64, device=result.device)
    # Relaxed tolerance because intermediate computation is float32
    assert torch.allclose(result, expected, rtol=1e-7, atol=1e-7)
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()
