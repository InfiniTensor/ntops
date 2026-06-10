import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


@skip_if_cuda_not_available
class TestTril:
    @pytest.mark.parametrize("n", [1, 3, 5, 10])
    def test_square(self, n):
        input = torch.randn(n, n, device="cuda")

        ninetoothed_output = ntops.torch.tril(input)
        reference_output = torch.tril(input)

        assert torch.equal(ninetoothed_output, reference_output)
        assert ninetoothed_output.shape == input.shape

    @pytest.mark.parametrize("n, m", [(3, 5), (5, 3), (1, 10), (10, 1)])
    def test_rectangular(self, n, m):
        input = torch.randn(n, m, device="cuda")

        ninetoothed_output = ntops.torch.tril(input)
        reference_output = torch.tril(input)

        assert torch.equal(ninetoothed_output, reference_output)

    @pytest.mark.parametrize("diagonal", [-2, -1, 0, 1, 2])
    def test_diagonal(self, diagonal):
        input = torch.randn(5, 5, device="cuda")

        ninetoothed_output = ntops.torch.tril(input, diagonal)
        reference_output = torch.tril(input, diagonal)

        assert torch.equal(ninetoothed_output, reference_output)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int32, torch.int64])
    def test_dtype(self, dtype):
        input = torch.randn(4, 4, device="cuda").to(dtype)

        ninetoothed_output = ntops.torch.tril(input)
        reference_output = torch.tril(input)

        assert torch.equal(ninetoothed_output, reference_output)

    def test_3d(self):
        input = torch.randn(2, 3, 3, device="cuda")

        ninetoothed_output = ntops.torch.tril(input)
        reference_output = torch.tril(input)

        assert torch.equal(ninetoothed_output, reference_output)
