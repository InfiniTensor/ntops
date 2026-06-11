import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


@skip_if_cuda_not_available
class TestNanToNum:
    def test_no_nan_or_inf(self):
        input = torch.tensor([1.0, 2.0, 3.0], device="cuda")

        ninetoothed_output = ntops.torch.nan_to_num(input)
        reference_output = torch.nan_to_num(input)

        assert torch.equal(ninetoothed_output, reference_output)

    def test_nan_replacement(self):
        input = torch.tensor([1.0, float("nan"), 3.0], device="cuda")

        ninetoothed_output = ntops.torch.nan_to_num(input)
        reference_output = torch.nan_to_num(input)

        assert torch.allclose(ninetoothed_output, reference_output, equal_nan=True)
        assert not torch.isnan(ninetoothed_output).any()

    def test_inf_replacement(self):
        input = torch.tensor([1.0, float("inf"), float("-inf")], device="cuda")

        ninetoothed_output = ntops.torch.nan_to_num(input)
        reference_output = torch.nan_to_num(input)

        assert torch.allclose(ninetoothed_output, reference_output, equal_nan=True)
        assert not torch.isinf(ninetoothed_output).any()

    def test_custom_nan_value(self):
        input = torch.tensor([1.0, float("nan"), 3.0], device="cuda")

        ninetoothed_output = ntops.torch.nan_to_num(input, nan=-1.0)
        reference_output = torch.nan_to_num(input, nan=-1.0)

        assert torch.allclose(ninetoothed_output, reference_output, equal_nan=True)

    def test_custom_posinf_neginf(self):
        input = torch.tensor([1.0, float("inf"), float("-inf")], device="cuda")

        ninetoothed_output = ntops.torch.nan_to_num(input, posinf=1e6, neginf=-1e6)
        reference_output = torch.nan_to_num(input, posinf=1e6, neginf=-1e6)

        assert torch.allclose(ninetoothed_output, reference_output)

    def test_mixed_nan_inf(self):
        input = torch.tensor([float("nan"), float("inf"), float("-inf"), 42.0], device="cuda")

        ninetoothed_output = ntops.torch.nan_to_num(input)
        reference_output = torch.nan_to_num(input)

        assert torch.allclose(ninetoothed_output, reference_output, equal_nan=True)

    def test_2d_tensor(self):
        input = torch.tensor([[1.0, float("nan")], [float("inf"), 4.0]], device="cuda")

        ninetoothed_output = ntops.torch.nan_to_num(input)
        reference_output = torch.nan_to_num(input)

        assert torch.allclose(ninetoothed_output, reference_output, equal_nan=True)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_dtype(self, dtype):
        input = torch.tensor([1.0, float("nan")], device="cuda", dtype=dtype)

        ninetoothed_output = ntops.torch.nan_to_num(input)
        reference_output = torch.nan_to_num(input)

        assert torch.allclose(ninetoothed_output, reference_output, equal_nan=True, atol=0.1, rtol=0.1)
