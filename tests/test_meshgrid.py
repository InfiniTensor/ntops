import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


@skip_if_cuda_not_available
class TestMeshgrid:
    def test_two_vectors(self):
        x = torch.tensor([1, 2, 3], device="cuda")
        y = torch.tensor([4, 5], device="cuda")

        ninetoothed_x, ninetoothed_y = ntops.torch.meshgrid(x, y, indexing="ij")
        reference_x, reference_y = torch.meshgrid(x, y, indexing="ij")

        assert torch.equal(ninetoothed_x, reference_x)
        assert torch.equal(ninetoothed_y, reference_y)
        assert ninetoothed_x.shape == (3, 2)
        assert ninetoothed_y.shape == (3, 2)

    def test_xy_indexing(self):
        x = torch.tensor([1, 2, 3], device="cuda")
        y = torch.tensor([4, 5], device="cuda")

        ninetoothed_x, ninetoothed_y = ntops.torch.meshgrid(x, y, indexing="xy")
        reference_x, reference_y = torch.meshgrid(x, y, indexing="xy")

        assert torch.equal(ninetoothed_x, reference_x)
        assert torch.equal(ninetoothed_y, reference_y)
        assert ninetoothed_x.shape == (2, 3)

    def test_three_vectors(self):
        x = torch.randn(3, device="cuda")
        y = torch.randn(4, device="cuda")
        z = torch.randn(5, device="cuda")

        ninetoothed_grid = ntops.torch.meshgrid(x, y, z, indexing="ij")
        reference_grid = torch.meshgrid(x, y, z, indexing="ij")

        for ninetoothed_out, reference_out in zip(ninetoothed_grid, reference_grid):
            assert torch.equal(ninetoothed_out, reference_out)
            assert ninetoothed_out.shape == reference_out.shape

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int32, torch.int64])
    def test_dtype(self, dtype):
        x = torch.tensor([1, 2, 3], device="cuda", dtype=dtype)
        y = torch.tensor([4, 5, 6], device="cuda", dtype=dtype)

        ninetoothed_x, ninetoothed_y = ntops.torch.meshgrid(x, y, indexing="ij")
        reference_x, reference_y = torch.meshgrid(x, y, indexing="ij")

        assert torch.equal(ninetoothed_x, reference_x)
        assert torch.equal(ninetoothed_y, reference_y)

    def test_default_indexing(self):
        x = torch.randn(3, device="cuda")
        y = torch.randn(4, device="cuda")

        ninetoothed_x, ninetoothed_y = ntops.torch.meshgrid(x, y)
        reference_x, reference_y = torch.meshgrid(x, y, indexing="ij")

        assert torch.equal(ninetoothed_x, reference_x)
        assert torch.equal(ninetoothed_y, reference_y)
