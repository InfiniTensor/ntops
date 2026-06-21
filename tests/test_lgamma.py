import math
import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_lgamma(shape, dtype, device, rtol, atol):
    # lgamma requires positive inputs
    input = torch.rand(shape, dtype=dtype, device=device) * 5 + 0.1  # [0.1, 5.1)

    ninetoothed_output = ntops.torch.lgamma(input)
    reference_output = torch.lgamma(input)

    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)
    assert not torch.isnan(ninetoothed_output).any()


@skip_if_cuda_not_available
def test_lgamma_edge_cases():
    device = "cuda"
    dtype = torch.float32

    # Test: lgamma(1) = 0 (gamma(1) = 1, log(1) = 0)
    x = torch.tensor([1.0], dtype=dtype, device=device)
    result = ntops.torch.lgamma(x)
    expected = torch.lgamma(x)
    assert torch.equal(result, expected)
    assert result.item() == pytest.approx(0.0, abs=1e-5)

    # Test: lgamma(2) = 0 (gamma(2) = 1, log(1) = 0)
    x = torch.tensor([2.0], dtype=dtype, device=device)
    result = ntops.torch.lgamma(x)
    expected = torch.lgamma(x)
    assert torch.equal(result, expected)
    assert result.item() == pytest.approx(0.0, abs=1e-5)

    # Test: lgamma(3) = log(2) ≈ 0.693
    x = torch.tensor([3.0], dtype=dtype, device=device)
    result = ntops.torch.lgamma(x)
    expected = torch.lgamma(x)
    assert torch.equal(result, expected)
    assert result.item() == pytest.approx(math.log(2), abs=1e-5)

    # Test: lgamma(0.5) = log(sqrt(pi)) ≈ 0.572
    x = torch.tensor([0.5], dtype=dtype, device=device)
    result = ntops.torch.lgamma(x)
    expected = torch.lgamma(x)
    assert torch.equal(result, expected)
    assert result.item() == pytest.approx(0.5 * math.log(math.pi), abs=1e-5)

    # Test: small positive values
    x = torch.tensor([0.1, 0.5, 1.5, 2.5], dtype=dtype, device=device)
    result = ntops.torch.lgamma(x)
    expected = torch.lgamma(x)
    assert torch.allclose(result, expected, rtol=1e-5, atol=1e-5)

    # Test: larger values
    x = torch.tensor([10.0, 50.0, 100.0], dtype=dtype, device=device)
    result = ntops.torch.lgamma(x)
    expected = torch.lgamma(x)
    assert torch.allclose(result, expected, rtol=1e-4, atol=1e-4)

    # Test: 2D tensors
    x = torch.tensor([[1.0, 2.0], [3.0, 0.5]], dtype=dtype, device=device)
    result = ntops.torch.lgamma(x)
    expected = torch.lgamma(x)
    assert torch.allclose(result, expected, rtol=1e-5, atol=1e-5)


@skip_if_cuda_not_available
def test_lgamma_nan_inf():
    device = "cuda"
    dtype = torch.float32

    # Test: lgamma(0) should return inf (gamma has poles at non-positive integers)
    x = torch.tensor([0.0], dtype=dtype, device=device)
    result = ntops.torch.lgamma(x)
    expected = torch.lgamma(x)
    # Both should be inf
    assert torch.isinf(result).all() == torch.isinf(expected).all()

    # Test: lgamma(negative) should return nan
    x = torch.tensor([-1.0, -2.5, -10.0], dtype=dtype, device=device)
    result = ntops.torch.lgamma(x)
    expected = torch.lgamma(x)
    # Both should have nan
    assert torch.isnan(result).all() == torch.isnan(expected).all()


@skip_if_cuda_not_available
def test_lgamma_float16():
    # Test float16 support
    x = torch.tensor([1.0, 2.0, 3.0, 0.5, 5.0], dtype=torch.float16, device="cuda")
    result = ntops.torch.lgamma(x)
    expected = torch.lgamma(x)
    assert torch.allclose(result, expected, rtol=1e-2, atol=1e-2)
