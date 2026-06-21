import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_copysign(shape, dtype, device, rtol, atol):
    x = torch.randn(shape, dtype=dtype, device=device)
    y = torch.randn(shape, dtype=dtype, device=device)

    ninetoothed_output = ntops.torch.copysign(x, y)
    reference_output = torch.copysign(x, y)

    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)
    assert not torch.isnan(ninetoothed_output).any()
    assert not torch.isinf(ninetoothed_output).any()


@skip_if_cuda_not_available
def test_copysign_edge_cases():
    device = "cuda"
    dtype = torch.float32

    # Test: x positive, y positive -> positive
    x = torch.tensor([1.5, 2.5, 3.5], dtype=dtype, device=device)
    y = torch.tensor([1.0, 2.0, 3.0], dtype=dtype, device=device)
    result = ntops.torch.copysign(x, y)
    expected = torch.copysign(x, y)
    assert torch.equal(result, expected)

    # Test: x positive, y negative -> negative
    x = torch.tensor([1.5, 2.5, 3.5], dtype=dtype, device=device)
    y = torch.tensor([-1.0, -2.0, -3.0], dtype=dtype, device=device)
    result = ntops.torch.copysign(x, y)
    expected = torch.copysign(x, y)
    assert torch.equal(result, expected)

    # Test: x negative, y positive -> positive
    x = torch.tensor([-1.5, -2.5, -3.5], dtype=dtype, device=device)
    y = torch.tensor([1.0, 2.0, 3.0], dtype=dtype, device=device)
    result = ntops.torch.copysign(x, y)
    expected = torch.copysign(x, y)
    assert torch.equal(result, expected)

    # Test: x negative, y negative -> negative
    x = torch.tensor([-1.5, -2.5, -3.5], dtype=dtype, device=device)
    y = torch.tensor([-1.0, -2.0, -3.0], dtype=dtype, device=device)
    result = ntops.torch.copysign(x, y)
    expected = torch.copysign(x, y)
    assert torch.equal(result, expected)

    # Test: zero values
    x = torch.tensor([0.0, -0.0, 1.0], dtype=dtype, device=device)
    y = torch.tensor([1.0, -1.0, 0.0], dtype=dtype, device=device)
    result = ntops.torch.copysign(x, y)
    expected = torch.copysign(x, y)
    assert torch.equal(result, expected)

    # Test: large values
    x = torch.tensor([1e10, -1e10], dtype=dtype, device=device)
    y = torch.tensor([1.0, -1.0], dtype=dtype, device=device)
    result = ntops.torch.copysign(x, y)
    expected = torch.copysign(x, y)
    assert torch.equal(result, expected)
