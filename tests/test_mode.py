import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


@skip_if_cuda_not_available
class TestMode:
    @pytest.mark.parametrize("shape, dim", [
        ((10,), 0),
        ((5, 5), 0),
        ((5, 5), 1),
        ((2, 3, 4), 0),
        ((2, 3, 4), 2),
    ])
    def test_basic(self, shape, dim):
        input = torch.randn(*shape, device="cuda")

        ninetoothed_values, ninetoothed_indices = ntops.torch.mode(input, dim)
        reference_values, reference_indices = torch.mode(input, dim)

        assert torch.equal(ninetoothed_values, reference_values)
        assert torch.equal(ninetoothed_indices, reference_indices)

    def test_keepdim(self):
        input = torch.randn(3, 4, 5, device="cuda")

        ninetoothed_values, ninetoothed_indices = ntops.torch.mode(input, dim=1, keepdim=True)
        reference_values, reference_indices = torch.mode(input, dim=1, keepdim=True)

        assert torch.equal(ninetoothed_values, reference_values)
        assert ninetoothed_values.shape == reference_values.shape
        assert ninetoothed_indices.shape == reference_indices.shape

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int32, torch.int64])
    def test_dtype(self, dtype):
        input = torch.randn(4, 4, device="cuda").to(dtype)

        ninetoothed_values, ninetoothed_indices = ntops.torch.mode(input)
        reference_values, reference_indices = torch.mode(input)

        assert torch.equal(ninetoothed_values, reference_values)
        assert torch.equal(ninetoothed_indices, reference_indices)

    def test_integer_mode(self):
        input = torch.tensor([[1, 2, 2, 3], [4, 4, 4, 5]], device="cuda")

        ninetoothed_values, ninetoothed_indices = ntops.torch.mode(input, dim=1)
        reference_values, reference_indices = torch.mode(input, dim=1)

        assert torch.equal(ninetoothed_values, reference_values)
        assert torch.equal(ninetoothed_indices, reference_indices)
