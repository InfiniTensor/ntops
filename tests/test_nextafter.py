import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_nextafter(shape, dtype, device, rtol, atol):
    x = torch.randn(shape, dtype=dtype, device=device).abs()
    y = torch.randn(shape, dtype=dtype, device=device).abs()

    ninetoothed_output = ntops.torch.nextafter(x, y)
    reference_output = torch.nextafter(x, y)

    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)
    assert not torch.isnan(ninetoothed_output).any()


@skip_if_cuda_not_available
def test_nextafter_edge_cases():
    device = "cuda"
    dtype = torch.float32

    # Test: nextafter(x, x) should return x
    x = torch.tensor([1.0, -1.0, 0.0], dtype=dtype, device=device)
    y = x.clone()
    result = ntops.torch.nextafter(x, y)
    expected = torch.nextafter(x, y)
    assert torch.equal(result, expected)

    # Test: toward positive direction
    x = torch.tensor([1.0, -1.0, 0.0], dtype=dtype, device=device)
    y = torch.tensor([2.0, 0.0, 1.0], dtype=dtype, device=device)
    result = ntops.torch.nextafter(x, y)
    expected = torch.nextafter(x, y)
    assert torch.equal(result, expected)

    # Test: toward negative direction
    x = torch.tensor([1.0, -1.0, 0.0], dtype=dtype, device=device)
    y = torch.tensor([0.0, -2.0, -1.0], dtype=dtype, device=device)
    result = ntops.torch.nextafter(x, y)
    expected = torch.nextafter(x, y)
    assert torch.equal(result, expected)

    # Test: around zero (subnormal numbers)
    x = torch.tensor([0.0], dtype=dtype, device=device)
    y = torch.tensor([1.0], dtype=dtype, device=device)
    result = ntops.torch.nextafter(x, y)
    expected = torch.nextafter(x, y)
    assert torch.equal(result, expected)
    assert result > 0  # Smallest positive subnormal

    # Test: 2D tensors
    x = torch.tensor([[1.0, 2.0], [0.0, -1.0]], dtype=dtype, device=device)
    y = torch.tensor([[2.0, 3.0], [1.0, 0.0]], dtype=dtype, device=device)
    result = ntops.torch.nextafter(x, y)
    expected = torch.nextafter(x, y)
    assert torch.equal(result, expected)
