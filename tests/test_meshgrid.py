import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
@pytest.mark.parametrize("indexing", ["ij", "xy"])
@pytest.mark.parametrize(
    "sizes",
    [
        [3, 5],
        [4, 4],
        [2, 3, 4],
        [1, 10],
        [7, 1],
        [2, 2, 2],
    ],
)
def test_meshgrid(sizes, indexing, dtype):
    tensors = [torch.arange(s, dtype=dtype, device="cuda") for s in sizes]

    ninetoothed_outputs = ntops.torch.meshgrid(*tensors, indexing=indexing)
    reference_outputs = torch.meshgrid(*tensors, indexing=indexing)

    assert len(ninetoothed_outputs) == len(reference_outputs)
    for nt_out, ref_out in zip(ninetoothed_outputs, reference_outputs):
        assert nt_out.shape == ref_out.shape
        assert nt_out.dtype == ref_out.dtype
        assert torch.equal(nt_out, ref_out)


@skip_if_cuda_not_available
def test_meshgrid_0d():
    """0-D tensors treated as length-1 1-D."""
    s = torch.tensor(3.0, device="cuda")  # 0D scalar
    v = torch.tensor([1.0, 2.0], device="cuda")  # 1D
    nt = ntops.torch.meshgrid(s, v, indexing="ij")
    ref = torch.meshgrid(s, v, indexing="ij")
    for a, b in zip(nt, ref):
        assert a.shape == b.shape
        assert torch.equal(a, b)


@skip_if_cuda_not_available
def test_meshgrid_list_arg():
    """Single list/tuple argument: torch.meshgrid([x, y])."""
    a = torch.arange(3, dtype=torch.float32, device="cuda")
    b = torch.arange(5, dtype=torch.float32, device="cuda")
    nt = ntops.torch.meshgrid([a, b], indexing="ij")
    ref = torch.meshgrid([a, b], indexing="ij")
    for x, y in zip(nt, ref):
        assert x.shape == y.shape
        assert torch.equal(x, y)
