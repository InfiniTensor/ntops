import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


@skip_if_cuda_not_available
class TestLogit:
    def test_basic(self):
        input = torch.tensor([0.1, 0.5, 0.9], device="cuda")

        ninetoothed_output = ntops.torch.logit(input)
        reference_output = torch.logit(input)

        assert torch.allclose(ninetoothed_output, reference_output)

    def test_with_eps(self):
        input = torch.tensor([0.0, 0.5, 1.0], device="cuda")

        ninetoothed_output = ntops.torch.logit(input, eps=1e-6)
        reference_output = torch.logit(input, eps=1e-6)

        assert torch.allclose(ninetoothed_output, reference_output)

    def test_2d_tensor(self):
        input = torch.tensor([[0.2, 0.8], [0.3, 0.7]], device="cuda")

        ninetoothed_output = ntops.torch.logit(input)
        reference_output = torch.logit(input)

        assert torch.allclose(ninetoothed_output, reference_output)

    def test_close_to_bounds(self):
        input = torch.tensor([1e-6, 1 - 1e-6], device="cuda")

        ninetoothed_output = ntops.torch.logit(input)
        reference_output = torch.logit(input)

        assert torch.allclose(ninetoothed_output, reference_output, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_dtype(self, dtype):
        input = torch.tensor([0.1, 0.5, 0.9], device="cuda", dtype=dtype)

        ninetoothed_output = ntops.torch.logit(input)
        reference_output = torch.logit(input)

        assert torch.allclose(ninetoothed_output, reference_output, atol=0.1, rtol=0.1)
