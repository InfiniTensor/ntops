import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available

_SHAPES = [(1, 1), (4, 4), (16, 16), (8, 5, 5), (2, 3, 7, 7)]


@skip_if_cuda_not_available
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", (torch.float32,))
def test_slogdet(shape, dtype):
    device = "cuda"
    A = torch.randn(shape, dtype=dtype, device=device)

    ninetoothed_sign, ninetoothed_logabsdet = ntops.torch.slogdet(A)
    reference_sign, reference_logabsdet = torch.linalg.slogdet(A)

    assert torch.allclose(ninetoothed_sign, reference_sign)
    assert torch.allclose(ninetoothed_logabsdet, reference_logabsdet)


@skip_if_cuda_not_available
def test_slogdet_singular():
    device = "cuda"
    A = torch.zeros((3, 3), dtype=torch.float32, device=device)

    sign, logabsdet = ntops.torch.slogdet(A)

    assert sign.item() == 0.0
    assert logabsdet.item() == float("-inf")


@skip_if_cuda_not_available
def test_slogdet_non_square_raises():
    device = "cuda"
    A = torch.randn((3, 4), dtype=torch.float32, device=device)

    with pytest.raises(AssertionError):
        ntops.torch.slogdet(A)
