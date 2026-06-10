import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_chunk(shape, dtype, device, rtol, atol):
    # TODO: Test for `float16` later.
    if dtype is torch.float16:
        return

    input = torch.randn(shape, dtype=dtype, device=device)
    chunks = max(1, input.shape[0] // 2)

    ninetoothed_output = ntops.torch.chunk(input, chunks)
    reference_output = torch.chunk(input, chunks)

    assert len(ninetoothed_output) == len(reference_output)

    for ninetoothed_chunk, reference_chunk in zip(ninetoothed_output, reference_output):
        assert torch.allclose(ninetoothed_chunk, reference_chunk, rtol=rtol, atol=atol)
        assert ninetoothed_chunk.shape == reference_chunk.shape


@skip_if_cuda_not_available
@pytest.mark.parametrize("ndim", [1, 2, 3, 4])
def test_chunk_dims(ndim):
    shape = tuple(range(3, ndim + 3))
    input = torch.randn(shape, device="cuda")

    for dim in range(ndim):
        chunks = max(1, input.shape[dim] // 2)

        ninetoothed_output = ntops.torch.chunk(input, chunks, dim)
        reference_output = torch.chunk(input, chunks, dim)

        assert len(ninetoothed_output) == len(reference_output)

        for ninetoothed_chunk, reference_chunk in zip(ninetoothed_output, reference_output):
            assert torch.allclose(ninetoothed_chunk, reference_chunk)
            assert ninetoothed_chunk.shape == reference_chunk.shape
