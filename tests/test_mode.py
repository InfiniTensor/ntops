"""Mode tests with semantic verification for tie cases.

For unique mode: exact value and index match.
For tied modes: verify returned value has max frequency and
input[returned_index] == returned_value.
"""

import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


def _unsqueeze_like(value, target, dim):
    """Unsqueeze `value` at `dim` until it has same ndim as `target`."""
    while value.ndim < target.ndim:
        value = value.unsqueeze(dim)
    return value


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("dim", [0, -1])
@pytest.mark.parametrize("shape", [(10,), (3, 8), (4, 5, 6)])
def test_mode(shape, dim, keepdim, dtype):
    if dtype in (torch.int32,):
        input = torch.randint(0, 10, shape, dtype=dtype, device="cuda")
    else:
        input = torch.randn(shape, dtype=dtype, device="cuda")

    nt_vals, nt_inds = ntops.torch.mode(input, dim=dim, keepdim=keepdim)
    ref_vals, ref_inds = torch.mode(input, dim=dim, keepdim=keepdim)

    # Shape and dtype must match
    assert nt_vals.shape == ref_vals.shape
    assert nt_inds.shape == ref_inds.shape
    assert nt_vals.dtype == ref_vals.dtype

    # input[returned_index] must equal returned value
    # Gather requires same ndim: unsqueeze index at `dim` if needed
    idx_for_gather = _unsqueeze_like(nt_inds.long(), input, dim)
    val_for_gather = _unsqueeze_like(nt_vals, input, dim)
    gathered = input.gather(dim, idx_for_gather)
    assert torch.equal(gathered, val_for_gather), (
        f"input[nt_inds] != nt_vals"
    )

    # Count of our returned value must equal reference count
    val_for_count = _unsqueeze_like(nt_vals, input, dim)
    ref_for_count = _unsqueeze_like(ref_vals, input, dim)
    nt_counts = (input == val_for_count).sum(dim=dim)
    ref_counts = (input == ref_for_count).sum(dim=dim)
    assert torch.equal(nt_counts, ref_counts), (
        f"nt count != ref count"
    )

    # If values match, indices must match too
    if torch.equal(nt_vals, ref_vals):
        assert torch.equal(nt_inds, ref_inds), (
            f"same value but different index"
        )
