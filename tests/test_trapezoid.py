import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


@skip_if_cuda_not_available
class TestTrapezoid:
    def test_1d_basic(self):
        y = torch.tensor([1.0, 2.0, 3.0], device="cuda")

        ninetoothed_output = ntops.torch.trapezoid(y)
        reference_output = torch.trapezoid(y)

        assert torch.allclose(ninetoothed_output, reference_output)
        assert ninetoothed_output.shape == ()

    def test_1d_with_dx(self):
        y = torch.tensor([1.0, 2.0, 3.0], device="cuda")

        ninetoothed_output = ntops.torch.trapezoid(y, dx=2.0)
        reference_output = torch.trapezoid(y, dx=2.0)

        assert torch.allclose(ninetoothed_output, reference_output)

    def test_1d_with_x(self):
        y = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        x = torch.tensor([0.0, 2.0, 5.0], device="cuda")

        ninetoothed_output = ntops.torch.trapezoid(y, x=x)
        reference_output = torch.trapezoid(y, x=x)

        assert torch.allclose(ninetoothed_output, reference_output)

    def test_2d_default_dim(self):
        y = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="cuda")

        ninetoothed_output = ntops.torch.trapezoid(y)
        reference_output = torch.trapezoid(y)

        assert torch.allclose(ninetoothed_output, reference_output)
        assert ninetoothed_output.shape == (2,)

    def test_2d_with_dim(self):
        y = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="cuda")

        ninetoothed_output = ntops.torch.trapezoid(y, dim=0)
        reference_output = torch.trapezoid(y, dim=0)

        assert torch.allclose(ninetoothed_output, reference_output)
        assert ninetoothed_output.shape == (3,)

    def test_2d_with_x_and_dim(self):
        y = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="cuda")
        x = torch.tensor([0.0, 1.0, 3.0], device="cuda")

        ninetoothed_output = ntops.torch.trapezoid(y, x=x, dim=-1)
        reference_output = torch.trapezoid(y, x=x, dim=-1)

        assert torch.allclose(ninetoothed_output, reference_output)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_dtype(self, dtype):
        y = torch.tensor([1.0, 2.0, 3.0], device="cuda", dtype=dtype)

        ninetoothed_output = ntops.torch.trapezoid(y)
        reference_output = torch.trapezoid(y)

        assert torch.allclose(ninetoothed_output, reference_output, atol=0.1, rtol=0.1)
