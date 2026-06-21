import pytest
import torch
import torch.nn.functional as F

import ntops
from tests.skippers import skip_if_cuda_not_available


def _random_target(shape, c, device):
    # A contiguous non-negative prefix of label indices, then -1 padding.
    target = torch.full(shape, -1, dtype=torch.int64, device=device)
    n = shape[0] if len(shape) == 2 else 1
    rows = target.reshape(n, c)
    for r in range(n):
        num = torch.randint(1, c + 1, (1,)).item()
        labels = torch.randperm(c, device=device)[:num]
        rows[r, :num] = labels
    return target


@skip_if_cuda_not_available
@pytest.mark.parametrize("shape", [(4, 8), (2, 16), (8, 32)])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_multilabel_margin_loss(shape, reduction):
    device = "cuda"
    c = shape[-1]
    input = torch.randn(shape, dtype=torch.float32, device=device)
    target = _random_target(shape, c, device)

    ninetoothed_output = ntops.torch.multilabel_margin_loss(
        input, target, reduction=reduction
    )
    reference_output = F.multilabel_margin_loss(input, target, reduction=reduction)

    assert torch.allclose(ninetoothed_output, reference_output, rtol=1e-3, atol=1e-3)


@skip_if_cuda_not_available
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_multilabel_margin_loss_1d(reduction):
    device = "cuda"
    c = 8
    input = torch.randn((c,), dtype=torch.float32, device=device)
    target = _random_target((c,), c, device)

    ninetoothed_output = ntops.torch.multilabel_margin_loss(
        input, target, reduction=reduction
    )
    reference_output = F.multilabel_margin_loss(input, target, reduction=reduction)

    assert torch.allclose(ninetoothed_output, reference_output, rtol=1e-3, atol=1e-3)
