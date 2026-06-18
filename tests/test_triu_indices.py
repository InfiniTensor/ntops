import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


@skip_if_cuda_not_available
class TestTriuIndices:
    @pytest.mark.parametrize("n", [1, 3, 5, 10])
    def test_square(self, n):
        ninetoothed_output = ntops.torch.triu_indices(n, device="cuda")
        reference_output = torch.triu_indices(n, n, device="cuda")

        assert torch.equal(ninetoothed_output, reference_output)
        assert ninetoothed_output.shape == reference_output.shape

    @pytest.mark.parametrize("n, m", [(3, 5), (5, 3), (1, 10), (10, 1)])
    def test_rectangular(self, n, m):
        ninetoothed_output = ntops.torch.triu_indices(n, m, device="cuda")
        reference_output = torch.triu_indices(n, m, device="cuda")

        assert torch.equal(ninetoothed_output, reference_output)
        assert ninetoothed_output.shape == reference_output.shape

    @pytest.mark.parametrize("offset", [-3, -1, 0, 1, 3])
    def test_offset(self, offset):
        n, m = 5, 5

        ninetoothed_output = ntops.torch.triu_indices(n, m, offset, device="cuda")
        reference_output = torch.triu_indices(n, m, offset, device="cuda")

        assert torch.equal(ninetoothed_output, reference_output)

    def test_default_device(self):
        if torch.cuda.is_available():
            result = ntops.torch.triu_indices(3, device="cuda")
            assert result.device.type == "cuda"
