import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


@skip_if_cuda_not_available
class TestEye:
    @pytest.mark.parametrize("n", [1, 3, 5, 10])
    def test_square(self, n):
        ninetoothed_output = ntops.torch.eye(n, device="cuda")
        reference_output = torch.eye(n, device="cuda")

        assert torch.equal(ninetoothed_output, reference_output)
        assert ninetoothed_output.device.type == "cuda"
        assert ninetoothed_output.shape == (n, n)

    @pytest.mark.parametrize("n, m", [(3, 5), (5, 3), (1, 10), (10, 1)])
    def test_rectangular(self, n, m):
        ninetoothed_output = ntops.torch.eye(n, m, device="cuda")
        reference_output = torch.eye(n, m, device="cuda")

        assert torch.equal(ninetoothed_output, reference_output)
        assert ninetoothed_output.shape == (n, m)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int32, torch.int64])
    def test_dtype(self, dtype):
        n = 4
        ninetoothed_output = ntops.torch.eye(n, dtype=dtype, device="cuda")
        reference_output = torch.eye(n, dtype=dtype, device="cuda")

        assert torch.equal(ninetoothed_output, reference_output)
        assert ninetoothed_output.dtype == dtype

    def test_default_device(self):
        if torch.cuda.is_available():
            result = ntops.torch.eye(3)
            assert result.device.type == "cuda"
