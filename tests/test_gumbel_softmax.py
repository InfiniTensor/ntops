import pytest
import torch
import torch.nn.functional as F

import ntops
from tests.skippers import skip_if_cuda_not_available

_SHAPES = [(8, 16), (4, 1024), (2, 3, 32)]


def _seeded(fn, seed, *args, **kwargs):
    torch.manual_seed(seed)
    return fn(*args, **kwargs)


@skip_if_cuda_not_available
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16))
@pytest.mark.parametrize("dim", (-1, 1))
def test_gumbel_softmax_soft(shape, dtype, dim):
    device = "cuda"
    logits = torch.randn(shape, dtype=dtype, device=device)

    # Identical noise via a shared seed; only the softmax backend differs.
    ninetoothed_output = _seeded(
        ntops.torch.gumbel_softmax, 0, logits, tau=1.0, hard=False, dim=dim
    )
    reference_output = _seeded(
        F.gumbel_softmax, 0, logits, tau=1.0, hard=False, dim=dim
    )

    rtol, atol = (1e-3, 1e-3) if dtype is torch.float32 else (1e-2, 1e-2)
    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)


@skip_if_cuda_not_available
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16))
def test_gumbel_softmax_hard(shape, dtype):
    device = "cuda"
    logits = torch.randn(shape, dtype=dtype, device=device)

    output = _seeded(
        ntops.torch.gumbel_softmax, 0, logits, tau=1.0, hard=True, dim=-1
    )

    # `hard=True` yields a one-hot along `dim`.
    assert torch.equal(output.sum(-1), torch.ones(shape[:-1], device=device))
    assert torch.equal(
        ((output == 0) | (output == 1)),
        torch.ones_like(output, dtype=torch.bool),
    )
