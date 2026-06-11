import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


@skip_if_cuda_not_available
class TestLinspace:
    def test_basic(self):
        ninetoothed_output = ntops.torch.linspace(0, 1, 5, device="cuda")
        reference_output = torch.linspace(0, 1, 5, device="cuda")

        assert torch.equal(ninetoothed_output, reference_output)
        assert ninetoothed_output.shape == (5,)
        assert ninetoothed_output.device.type == "cuda"

    def test_negative_to_positive(self):
        ninetoothed_output = ntops.torch.linspace(-3, 3, 7, device="cuda")
        reference_output = torch.linspace(-3, 3, 7, device="cuda")

        assert torch.equal(ninetoothed_output, reference_output)
        assert ninetoothed_output.shape == (7,)

    def test_single_step(self):
        ninetoothed_output = ntops.torch.linspace(5, 5, 1, device="cuda")
        reference_output = torch.linspace(5, 5, 1, device="cuda")

        assert torch.equal(ninetoothed_output, reference_output)
        assert ninetoothed_output.shape == (1,)

    def test_many_steps(self):
        ninetoothed_output = ntops.torch.linspace(0, 1, 100, device="cuda")
        reference_output = torch.linspace(0, 1, 100, device="cuda")

        assert torch.allclose(ninetoothed_output, reference_output)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_dtype(self, dtype):
        ninetoothed_output = ntops.torch.linspace(0, 1, 10, dtype=dtype, device="cuda")
        reference_output = torch.linspace(0, 1, 10, dtype=dtype, device="cuda")

        assert torch.allclose(ninetoothed_output, reference_output, atol=1e-3, rtol=1e-3)
        assert ninetoothed_output.dtype == dtype
