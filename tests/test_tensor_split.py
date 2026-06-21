import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
@pytest.mark.parametrize("sections", [1, 2, 3, 4])
def test_tensor_split_sections(shape, dtype, device, rtol, atol, sections):
    input = torch.randn(shape, dtype=dtype, device=device)

    if input.ndim == 0:
        pytest.skip("tensor_split does not support scalar input")

    for dim in range(-input.ndim, input.ndim):
        ninetoothed_outputs = ntops.torch.tensor_split(input, sections, dim=dim)
        reference_outputs = torch.tensor_split(input, sections, dim=dim)

        assert len(ninetoothed_outputs) == len(reference_outputs)

        for ninetoothed_output, reference_output in zip(
            ninetoothed_outputs,
            reference_outputs,
        ):
            assert torch.allclose(
                ninetoothed_output,
                reference_output,
                rtol=rtol,
                atol=atol,
            )