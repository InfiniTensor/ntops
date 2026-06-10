import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


@skip_if_cuda_not_available
class TestOuter:
    @pytest.mark.parametrize("m, n", [(1, 1), (3, 5), (5, 3), (10, 1), (1, 10)])
    def test_shapes(self, m, n):
        input = torch.randn(m, device="cuda")
        other = torch.randn(n, device="cuda")

        ninetoothed_output = ntops.torch.outer(input, other)
        reference_output = torch.outer(input, other)

        assert torch.allclose(ninetoothed_output, reference_output)
        assert ninetoothed_output.shape == (m, n)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_dtype(self, dtype):
        input = torch.randn(5, device="cuda").to(dtype)
        other = torch.randn(7, device="cuda").to(dtype)

        ninetoothed_output = ntops.torch.outer(input, other)
        reference_output = torch.outer(input, other)

        assert torch.allclose(ninetoothed_output, reference_output, atol=0.01, rtol=0.01)

    def test_known_values(self):
        input = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        other = torch.tensor([4.0, 5.0], device="cuda")

        ninetoothed_output = ntops.torch.outer(input, other)
        reference_output = torch.outer(input, other)

        expected = torch.tensor([[4.0, 5.0], [8.0, 10.0], [12.0, 15.0]], device="cuda")
        assert torch.equal(ninetoothed_output, expected)
        assert torch.equal(ninetoothed_output, reference_output)
