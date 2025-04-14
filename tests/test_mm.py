import random

import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


def generate_arguments():
    arguments = []

    for dtype in (torch.float32, torch.float16):
        if dtype is torch.float32:
            atol = 0.05
            rtol = 0.05
        else:
            atol = 0.025
            rtol = 0.025

        def generate_random_size():
            return random.randint(1, 1024)

        m = generate_random_size()
        n = generate_random_size()
        k = generate_random_size()

        arguments.append((m, n, k, dtype, atol, rtol))

    return "m, n, k, dtype, atol, rtol", arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_cuda(m, n, k, dtype, atol, rtol):
    device = "cuda"

    input = torch.randn((m, k), dtype=dtype, device=device)
    other = torch.randn((k, n), dtype=dtype, device=device)

    ninetoothed_output = ntops.mm(input, other)
    reference_output = torch.mm(input, other)

    assert torch.allclose(ninetoothed_output, reference_output, atol=atol, rtol=rtol)
