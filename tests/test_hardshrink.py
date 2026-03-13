import pytest
import torch
import random

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


# def generate_arguments():
#     return 'shape,dtype,device,rtol,atol', [
#         ((334,), torch.float16, 'cuda', 1e-2, 1e-2),]
@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_hardshrink(shape, dtype, device, rtol, atol):
    input = torch.randn(shape, dtype=dtype, device=device) * 10
    # lambd = torch.randn((), dtype=dtype, device=device).abs()
    lambd = random.uniform(0.0, 2.0)

    output = ntops.torch.hardshrink(input, lambd)

    expected_output = torch.nn.functional.hardshrink(input, lambd)

    torch.testing.assert_close(output, expected_output, rtol=rtol, atol=atol)
