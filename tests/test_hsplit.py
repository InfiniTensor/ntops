import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_hsplit_sections(shape, dtype, device, rtol, atol):
    if len(shape) == 0:
        return

    input = torch.randn(shape, dtype=dtype, device=device)

    dim = 0 if input.ndim == 1 else 1
    size = input.shape[dim]

    # 为了避免某些实现要求整除，这里选 sections=1，最稳
    sections = 1

    ninetoothed_outputs = ntops.torch.hsplit(input, sections)
    reference_outputs = torch.hsplit(input, sections)

    assert len(ninetoothed_outputs) == len(reference_outputs)

    for ninetoothed_output, reference_output in zip(
        ninetoothed_outputs,
        reference_outputs,
    ):
        assert torch.equal(ninetoothed_output, reference_output)


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_hsplit_indices(shape, dtype, device, rtol, atol):
    if len(shape) == 0:
        return

    input = torch.randn(shape, dtype=dtype, device=device)

    dim = 0 if input.ndim == 1 else 1
    size = input.shape[dim]

    # indices_or_sections 为 list 的情况
    indices = [
        size // 3,
        2 * size // 3,
    ]

    ninetoothed_outputs = ntops.torch.hsplit(input, indices)
    reference_outputs = torch.hsplit(input, indices)

    assert len(ninetoothed_outputs) == len(reference_outputs)

    for ninetoothed_output, reference_output in zip(
        ninetoothed_outputs,
        reference_outputs,
    ):
        assert torch.equal(ninetoothed_output, reference_output)