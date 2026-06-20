import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


# Filter out float16 since Triton floor/cast does not support it
_argument_names, _argument_values = generate_arguments()
_filtered_values = [v for v in _argument_values if v[1] != torch.float16]


@skip_if_cuda_not_available
@pytest.mark.parametrize(_argument_names, _filtered_values)
def test_frac(shape, dtype, device, rtol, atol):
    input = torch.randn(shape, dtype=dtype, device=device)

    ninetoothed_output = ntops.torch.frac(input)
    reference_output = torch.frac(input)

    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)
