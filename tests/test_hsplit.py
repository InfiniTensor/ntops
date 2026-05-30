import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available

_CASES = [
    # (shape, indices_or_sections)
    ((6,), 3),
    ((8,), 2),
    ((8,), [2, 5]),
    ((4, 6), 2),
    ((4, 6), 3),
    ((4, 6), [1, 4]),
    ((2, 8, 3), 4),
    ((2, 9, 3), [3, 6]),
]


@skip_if_cuda_not_available
@pytest.mark.parametrize("shape, indices_or_sections", _CASES)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16))
def test_hsplit(shape, indices_or_sections, dtype):
    device = "cuda"
    input = torch.randn(shape, dtype=dtype, device=device)

    ninetoothed_outputs = ntops.torch.hsplit(input, indices_or_sections)
    reference_outputs = torch.hsplit(input, indices_or_sections)

    assert len(ninetoothed_outputs) == len(reference_outputs)

    for ninetoothed_output, reference_output in zip(
        ninetoothed_outputs, reference_outputs
    ):
        assert ninetoothed_output.shape == reference_output.shape
        assert ninetoothed_output.is_contiguous()
        assert torch.equal(ninetoothed_output, reference_output)
