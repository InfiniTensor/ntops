import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


def manual_cosine_embedding_loss(x1, x2, y, margin=0.0, reduction="mean"):
    if x1.dim() == 1:
        x1 = x1.unsqueeze(0)  # (D,) -> (1, D)
        x2 = x2.unsqueeze(0)
        if y.dim() == 0:
            y = y.unsqueeze(0)  # () -> (1,)

    cosine = torch.nn.functional.cosine_similarity(x1, x2, dim=-1, eps=1e-8)

    loss = torch.where(y == 1, 1.0 - cosine, torch.clamp(cosine - margin, min=0.0))

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:  # 'none'
        return loss


def generate_arguments():
    return "shape,dtype,device,rtol,atol", [
        (
            (
                15,
                411,
            ),
            torch.float16,
            "cuda",
            1e-3,
            1e-2,
        ),
    ]


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_cosine_embedding_loss(shape, dtype, device, rtol, atol):
    if len(shape) > 2:
        pytest.skip("Skipping test for tensors with more than 2 dimensions.")
    else:
        x1 = torch.randn(shape, dtype=dtype, device=device)
        x2 = torch.randn(shape, dtype=dtype, device=device)
        if len(shape) == 1:
            y = torch.randint(-1, 2, (1,), device=device).float()
            y[y == 0] = 1
            y = y.squeeze()
        else:
            y = torch.randint(-1, 2, shape[:-1], device=device).float()
            y[y == 0] = 1
        margin = 0.5

        manual_output = manual_cosine_embedding_loss(
            x1.clone(), x2.clone(), y.clone(), margin=margin, reduction="mean"
        )
        ninetoothed_output = ntops.torch.cosine_embedding_loss(
            x1.clone(), x2.clone(), y.clone(), margin=margin, reduction="mean"
        )
        # reference_output = torch.nn.functional.cosine_embedding_loss(x1, x2, y, margin=margin, reduction='mean')
        assert torch.allclose(ninetoothed_output, manual_output, rtol=rtol, atol=atol)
