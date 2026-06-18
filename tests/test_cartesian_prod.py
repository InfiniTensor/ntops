import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


@skip_if_cuda_not_available
class TestCartesianProd:
    def test_two_1d_tensors(self):
        a = torch.tensor([1, 2], device="cuda")
        b = torch.tensor([3, 4], device="cuda")

        ninetoothed_output = ntops.torch.cartesian_prod(a, b)
        reference_output = torch.cartesian_prod(a, b)

        assert torch.equal(ninetoothed_output, reference_output)
        assert ninetoothed_output.shape == (4, 2)

    def test_three_1d_tensors(self):
        a = torch.tensor([1, 2], device="cuda")
        b = torch.tensor([3, 4], device="cuda")
        c = torch.tensor([5, 6], device="cuda")

        ninetoothed_output = ntops.torch.cartesian_prod(a, b, c)
        reference_output = torch.cartesian_prod(a, b, c)

        assert torch.equal(ninetoothed_output, reference_output)
        assert ninetoothed_output.shape == (8, 3)

    def test_single_tensor(self):
        a = torch.tensor([1, 2, 3], device="cuda")

        ninetoothed_output = ntops.torch.cartesian_prod(a)
        reference_output = torch.cartesian_prod(a)

        assert torch.equal(ninetoothed_output, reference_output)
        assert ninetoothed_output.shape == reference_output.shape

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int32, torch.int64])
    def test_dtype(self, dtype):
        a = torch.tensor([1, 2], device="cuda", dtype=dtype)
        b = torch.tensor([3, 4], device="cuda", dtype=dtype)

        ninetoothed_output = ntops.torch.cartesian_prod(a, b)
        reference_output = torch.cartesian_prod(a, b)

        assert torch.equal(ninetoothed_output, reference_output)

    def test_larger_inputs(self):
        a = torch.arange(5, device="cuda")
        b = torch.arange(6, device="cuda")

        ninetoothed_output = ntops.torch.cartesian_prod(a, b)
        reference_output = torch.cartesian_prod(a, b)

        assert torch.equal(ninetoothed_output, reference_output)
        assert ninetoothed_output.shape == (30, 2)
