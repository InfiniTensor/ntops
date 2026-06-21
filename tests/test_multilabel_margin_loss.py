import random
import math

import pytest
import torch
import torch.nn.functional as F

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


def _make_multilabel_margin_target(shape, device):
    c = shape[-1]

    outer = 1
    for s in shape[:-1]:
        outer *= s

    target_2d = torch.full(
        (outer, c),
        -1,
        dtype=torch.long,
        device=device,
    )

    for i in range(outer):
        num_pos = random.randint(0, c)

        if num_pos > 0:
            labels = torch.randperm(
                c,
                device=device,
                dtype=torch.long,
            )[:num_pos]

            target_2d[i, :num_pos] = labels

    return target_2d.reshape(shape)


def _reference_multilabel_margin_loss(input, target, reduction):
    c = input.shape[-1]

    input_2d = input.reshape(-1, c)
    target_2d = target.reshape(-1, c)

    output = F.multilabel_margin_loss(
        input_2d,
        target_2d,
        reduction=reduction,
    )

    if reduction == "none":
        return output.reshape(input.shape[:-1])

    return output


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_multilabel_margin_loss(shape, dtype, device, rtol, atol):
    # 不 skip 高维，统一按最后一维为类别维 C
    assert len(shape) >= 1

    input = torch.randn(
        shape,
        dtype=dtype,
        device=device,
    )

    target = _make_multilabel_margin_target(
        shape,
        device,
    )

    reduction = random.choice(("none", "mean", "sum"))

    ninetoothed_output = ntops.torch.multilabel_margin_loss(
        input,
        target,
        reduction=reduction,
    )

    reference_output = _reference_multilabel_margin_loss(
        input,
        target,
        reduction,
    )

    assert torch.allclose(
        ninetoothed_output,
        reference_output,
        rtol=rtol,
        atol=atol,
    )