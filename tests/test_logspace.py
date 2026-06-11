import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


@skip_if_cuda_not_available
class TestLogspace:
    def test_basic(self):
        ninetoothed_output = ntops.torch.logspace(0, 2, 5, device="cuda")
        reference_output = torch.logspace(0, 2, 5, device="cuda")

        assert torch.allclose(ninetoothed_output, reference_output)
        assert ninetoothed_output.shape == (5,)
        assert ninetoothed_output.device.type == "cuda"

    def test_negative_exponents(self):
        ninetoothed_output = ntops.torch.logspace(-2, 0, 5, device="cuda")
        reference_output = torch.logspace(-2, 0, 5, device="cuda")

        assert torch.allclose(ninetoothed_output, reference_output)
        assert ninetoothed_output.shape == (5,)

    def test_single_step(self):
        ninetoothed_output = ntops.torch.logspace(3, 3, 1, device="cuda")
        reference_output = torch.logspace(3, 3, 1, device="cuda")

        assert torch.allclose(ninetoothed_output, reference_output, atol=1e-3, rtol=1e-3)
        assert ninetoothed_output.shape == (1,)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_dtype(self, dtype):
        ninetoothed_output = ntops.torch.logspace(0, 1, 10, dtype=dtype, device="cuda")
        reference_output = torch.logspace(0, 1, 10, dtype=dtype, device="cuda")

        assert torch.allclose(ninetoothed_output, reference_output, atol=1e-2, rtol=1e-2)
        assert ninetoothed_output.dtype == dtype
