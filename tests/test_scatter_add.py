import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available

_CASES = [
    # (input_shape, dim, k) — index/src share input's non-dim dims, dim-size = k
    ((8,), 0, 5),
    ((8,), 0, 8),
    ((4, 6), 0, 4),
    ((4, 6), 1, 6),
    ((4, 6), 1, 3),
    ((2, 3, 5), 2, 5),
    ((2, 3, 5), 0, 2),
]


@skip_if_cuda_not_available
@pytest.mark.parametrize("shape, dim, k", _CASES)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16))
def test_scatter_add(shape, dim, k, dtype):
    device = "cuda"
    input = torch.randn(shape, dtype=dtype, device=device)

    index_shape = list(shape)
    index_shape[dim] = k
    t = shape[dim]
    index = torch.randint(0, t, index_shape, device=device)
    src = torch.randn(index_shape, dtype=dtype, device=device)

    ninetoothed_output = ntops.torch.scatter_add(input, dim, index, src)
    reference_output = torch.scatter_add(input, dim, index, src)

    assert torch.allclose(ninetoothed_output, reference_output, rtol=1e-2, atol=1e-2)
