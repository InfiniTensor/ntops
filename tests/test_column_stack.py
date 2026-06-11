import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


@skip_if_cuda_not_available
class TestColumnStack:
    def test_two_1d_tensors(self):
        a = torch.tensor([1, 2, 3], device="cuda")
        b = torch.tensor([4, 5, 6], device="cuda")

        ninetoothed_output = ntops.torch.column_stack([a, b])
        reference_output = torch.column_stack([a, b])

        assert torch.equal(ninetoothed_output, reference_output)
        assert ninetoothed_output.shape == (3, 2)

    def test_three_1d_tensors(self):
        a = torch.randn(5, device="cuda")
        b = torch.randn(5, device="cuda")
        c = torch.randn(5, device="cuda")

        ninetoothed_output = ntops.torch.column_stack([a, b, c])
        reference_output = torch.column_stack([a, b, c])

        assert torch.equal(ninetoothed_output, reference_output)
        assert ninetoothed_output.shape == (5, 3)

    def test_2d_tensors(self):
        a = torch.randn(3, 2, device="cuda")
        b = torch.randn(3, 4, device="cuda")

        ninetoothed_output = ntops.torch.column_stack([a, b])
        reference_output = torch.column_stack([a, b])

        assert torch.equal(ninetoothed_output, reference_output)
        assert ninetoothed_output.shape == (3, 6)

    def test_single_tensor(self):
        a = torch.randn(5, device="cuda")

        ninetoothed_output = ntops.torch.column_stack([a])
        reference_output = torch.column_stack([a])

        assert torch.equal(ninetoothed_output, reference_output)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int32, torch.int64])
    def test_dtype(self, dtype):
        a = torch.randn(4, device="cuda").to(dtype)
        b = torch.randn(4, device="cuda").to(dtype)

        ninetoothed_output = ntops.torch.column_stack([a, b])
        reference_output = torch.column_stack([a, b])

        assert torch.equal(ninetoothed_output, reference_output)
