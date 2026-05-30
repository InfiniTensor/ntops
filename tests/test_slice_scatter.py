import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available

_CASES = [
    # (shape, dim, start, end, step)
    ((8,), 0, 2, 6, 1),
    ((8,), 0, None, None, 1),
    ((4, 6), 1, 1, 5, 1),
    ((4, 6), 1, 0, 6, 2),
    ((4, 6), 0, 1, 3, 1),
    ((4, 6), -1, 2, None, 1),
    ((2, 3, 5), 2, 1, 4, 1),
    ((2, 3, 5), 0, None, 2, 1),
]


@skip_if_cuda_not_available
@pytest.mark.parametrize("shape, dim, start, end, step", _CASES)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16))
def test_slice_scatter(shape, dim, start, end, step, dtype):
    device = "cuda"
    input = torch.randn(shape, dtype=dtype, device=device)

    index = [slice(None)] * len(shape)
    index[dim] = slice(start, end, step)
    src_shape = input[tuple(index)].shape
    src = torch.randn(src_shape, dtype=dtype, device=device)

    ninetoothed_output = ntops.torch.slice_scatter(
        input, src, dim, start, end, step
    )
    reference_output = torch.slice_scatter(input, src, dim, start, end, step)

    assert torch.equal(ninetoothed_output, reference_output)
