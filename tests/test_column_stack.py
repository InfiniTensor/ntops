import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


def _make_tensors(shapes, dtype, device):
    if dtype in (torch.int32, torch.int64, torch.bool):
        return [torch.randint(0, 100, s, dtype=dtype, device=device) for s in shapes]
    return [torch.randn(s, dtype=dtype, device=device) for s in shapes]


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
@pytest.mark.parametrize(
    "shapes",
    [
        [(5,), (5,)],
        [(3,), (3,), (3,)],
        [(3, 4), (3, 2)],
        [(2, 3), (2, 3), (2, 3)],
        [(4,), (4,), (4,), (4,)],
    ],
)
def test_column_stack(shapes, dtype):
    tensors = _make_tensors(shapes, dtype, "cuda")
    ninetoothed_output = ntops.torch.column_stack(tensors)
    reference_output = torch.column_stack(tensors)
    assert ninetoothed_output.shape == reference_output.shape
    assert ninetoothed_output.dtype == reference_output.dtype
    assert torch.equal(ninetoothed_output, reference_output)


@skip_if_cuda_not_available
def test_column_stack_0d():
    """0-D tensors treated as scalars (→ (1, 1) columns)."""
    a = torch.tensor(3.0, device="cuda")
    b = torch.tensor(5.0, device="cuda")
    out = ntops.torch.column_stack((a, b))
    expected = torch.column_stack((a, b))
    assert out.shape == expected.shape
    assert torch.equal(out, expected)


@skip_if_cuda_not_available
def test_column_stack_3d():
    """3-D tensors stacked along dim=1."""
    a = torch.randn(2, 3, 4, device="cuda")
    b = torch.randn(2, 5, 4, device="cuda")
    out = ntops.torch.column_stack((a, b))
    expected = torch.column_stack((a, b))
    assert out.shape == expected.shape
    assert torch.equal(out, expected)


@skip_if_cuda_not_available
def test_column_stack_shape_mismatch():
    """Non-concatenating dims must match."""
    a = torch.randn(3, 2, device="cuda")
    b = torch.randn(4, 2, device="cuda")
    with pytest.raises(RuntimeError):
        ntops.torch.column_stack((a, b))
