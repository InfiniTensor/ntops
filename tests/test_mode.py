import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


@skip_if_cuda_not_available
class TestMode:
    @pytest.mark.parametrize("shape, dim, values", [
        ((10,), 0, torch.tensor([1, 5, 5, 3, 5, 2, 3, 5, 8, 9])),
        ((5, 5), 0, torch.tensor([[1, 1, 1, 1, 1],
                                   [1, 2, 1, 1, 2],
                                   [2, 1, 2, 2, 1],
                                   [1, 2, 1, 1, 2],
                                   [2, 1, 2, 2, 1]])),
        ((5, 5), 1, torch.tensor([[1, 2, 2, 3, 3], [2, 3, 3, 4, 5], [1, 4, 4, 4, 2], [3, 3, 5, 1, 5], [2, 2, 2, 3, 3]])),
        ((2, 3, 4), 0, torch.tensor([[[1, 2, 1, 2], [1, 2, 2, 1], [1, 1, 2, 2]],
                                     [[1, 2, 1, 2], [1, 2, 2, 1], [1, 1, 2, 2]]])),
        ((2, 3, 4), 2, torch.tensor([[[1, 1, 1, 2], [2, 2, 2, 1], [3, 3, 3, 1]],
                                     [[2, 2, 2, 1], [1, 1, 1, 3], [1, 1, 2, 1]]])),
    ])
    def test_basic(self, shape, dim, values):
        input = values.to(dtype=torch.float32, device="cuda")

        ninetoothed_values, ninetoothed_indices = ntops.torch.mode(input, dim)
        reference_values, reference_indices = torch.mode(input, dim)

        assert torch.equal(ninetoothed_values, reference_values)
        assert torch.equal(ninetoothed_indices, reference_indices)

    def test_keepdim(self):
        # Each dim=1 group of 3 values has a clear mode (no ties)
        input = torch.tensor([[[1, 2, 2, 1], [2, 2, 2, 2], [1, 1, 1, 2]],
                              [[1, 2, 1, 1], [2, 3, 2, 2], [1, 2, 2, 2]]],
                             device="cuda", dtype=torch.float32)

        ninetoothed_values, ninetoothed_indices = ntops.torch.mode(input, dim=1, keepdim=True)
        reference_values, reference_indices = torch.mode(input, dim=1, keepdim=True)

        assert torch.equal(ninetoothed_values, reference_values)
        assert ninetoothed_values.shape == reference_values.shape
        assert ninetoothed_indices.shape == reference_indices.shape

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_dtype(self, dtype):
        input = torch.tensor([[1, 2, 2, 3], [4, 4, 4, 5], [3, 3, 3, 1]], device="cuda").to(dtype)

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
