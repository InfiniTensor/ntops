import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
@pytest.mark.parametrize(
    "sizes",
    [
        [3, 5],
        [4, 4],
        [2, 3, 4],
        [1, 10],
        [7, 1],
        [2, 2, 2],
    ],
)
def test_cartesian_prod(sizes, dtype):
    tensors = [torch.arange(s, dtype=dtype, device="cuda") for s in sizes]

    ninetoothed_output = ntops.torch.cartesian_prod(*tensors)
    reference_output = torch.cartesian_prod(*tensors)

    assert ninetoothed_output.shape == reference_output.shape
    assert ninetoothed_output.dtype == reference_output.dtype
    assert torch.equal(ninetoothed_output, reference_output)


@skip_if_cuda_not_available
def test_cartesian_prod_single_input():
    """Single 1D input returns the input tensor itself."""
    x = torch.arange(3, dtype=torch.float32, device="cuda")
    out = ntops.torch.cartesian_prod(x)
    expected = torch.cartesian_prod(x)
    assert out.shape == expected.shape  # (3,), not (3, 1)
    assert torch.equal(out, expected)


@skip_if_cuda_not_available
def test_cartesian_prod_2d_raises():
    """Non-1D inputs must raise RuntimeError."""
    x = torch.randn(3, 4, device="cuda")
    with pytest.raises(RuntimeError, match="expected 1D"):
        ntops.torch.cartesian_prod(x)
