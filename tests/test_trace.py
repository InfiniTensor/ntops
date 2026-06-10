import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


@skip_if_cuda_not_available
class TestTrace:
    @pytest.mark.parametrize("n", [1, 3, 5, 10])
    def test_square(self, n):
        input = torch.randn(n, n, device="cuda")

        ninetoothed_output = ntops.torch.trace(input)
        reference_output = torch.trace(input)

        assert torch.allclose(ninetoothed_output, reference_output)
        assert ninetoothed_output.shape == ()

    @pytest.mark.parametrize("n, m", [(3, 5), (5, 3)])
    def test_rectangular(self, n, m):
        input = torch.randn(n, m, device="cuda")

        ninetoothed_output = ntops.torch.trace(input)
        reference_output = torch.trace(input)

        assert torch.allclose(ninetoothed_output, reference_output)
        assert ninetoothed_output.shape == ()

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int32, torch.int64])
    def test_dtype(self, dtype):
        input = torch.randn(4, 4, device="cuda").to(dtype)

        ninetoothed_output = ntops.torch.trace(input)
        reference_output = torch.trace(input)

        assert torch.allclose(ninetoothed_output, reference_output.to(dtype))

    def test_zero_diagonal(self):
        input = torch.zeros(0, 0, device="cuda")

        ninetoothed_output = ntops.torch.trace(input)
        reference_output = torch.trace(input)

        assert ninetoothed_output.item() == 0.0
