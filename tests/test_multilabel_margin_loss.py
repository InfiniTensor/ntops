import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@pytest.mark.parametrize("n_labels", [3, 5])
def test_multilabel_margin_loss(n_labels, reduction, dtype):
    torch.manual_seed(42)
    batch_size = 2
    input = torch.randn(batch_size, n_labels, dtype=dtype, device="cuda")
    target = torch.zeros(batch_size, n_labels, dtype=torch.long, device="cuda")
    target[:, 0] = 1  # first label is positive

    nt_out = ntops.torch.multilabel_margin_loss(input, target, reduction=reduction)
    ref_out = torch.nn.functional.multilabel_margin_loss(input, target, reduction=reduction)
    assert torch.allclose(nt_out, ref_out)
