import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


@skip_if_cuda_not_available
class TestRoll:
    @pytest.mark.parametrize("n", [1, 5, 10, 30])
    def test_1d(self, n):
        input = torch.randn(n, device="cuda")

        for shift in [0, 1, -1, n // 2, n - 1]:
            ninetoothed_output = ntops.torch.roll(input, shift)
            reference_output = torch.roll(input, shift)

            assert torch.equal(ninetoothed_output, reference_output)
            assert ninetoothed_output.shape == input.shape

    @pytest.mark.parametrize("shape", [(3, 5), (5, 3), (2, 4, 6)])
    def test_2d_3d(self, shape):
        input = torch.randn(*shape, device="cuda")

        for shift, dims in [(1, 0), (2, 1), (1, -1), (2, -2)]:
            if abs(dims) < input.ndim:
                ninetoothed_output = ntops.torch.roll(input, shift, dims)
                reference_output = torch.roll(input, shift, dims)

                assert torch.equal(ninetoothed_output, reference_output)

    def test_multi_dim_shifts(self):
        input = torch.randn(4, 5, 6, device="cuda")

        ninetoothed_output = ntops.torch.roll(input, (1, 2), (0, 1))
        reference_output = torch.roll(input, (1, 2), (0, 1))

        assert torch.equal(ninetoothed_output, reference_output)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int32, torch.int64])
    def test_dtype(self, dtype):
        input = torch.randn(10, device="cuda").to(dtype)

        ninetoothed_output = ntops.torch.roll(input, 3)
        reference_output = torch.roll(input, 3)

        assert torch.equal(ninetoothed_output, reference_output)

    def test_full_roll(self):
        input = torch.randn(5, 5, device="cuda")

        ninetoothed_output = ntops.torch.roll(input, (5, 5), (0, 1))
        reference_output = torch.roll(input, (5, 5), (0, 1))

        assert torch.equal(ninetoothed_output, reference_output)
        assert torch.equal(ninetoothed_output, input)
