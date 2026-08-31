import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


def _reference(
    input,
    gate,
    weight,
    eps,
    group_size,
    norm_before_gate,
    activation,
):
    if activation in ("silu", "swish"):
        activated_gate = torch.nn.functional.silu(gate.float())
    else:
        activated_gate = torch.sigmoid(gate.float())

    value = input.float()
    if not norm_before_gate:
        value = value * activated_gate

    grouped = value.reshape(*value.shape[:-1], -1, group_size)
    variance = grouped.square().mean(dim=-1, keepdim=True)
    normalized = (grouped * torch.rsqrt(variance + eps)).reshape_as(value)
    if norm_before_gate:
        normalized = normalized * activated_gate
    if weight is not None:
        normalized = normalized * weight.float()

    return normalized.to(input.dtype)


@skip_if_cuda_not_available
@pytest.mark.parametrize(
    "shape,group_size",
    (
        ((64, 128), 128),
        ((128, 512), 512),
        ((16, 4096), 4096),
        ((2, 8, 16, 64), 32),
    ),
)
@pytest.mark.parametrize("activation", ("swish", "sigmoid"))
@pytest.mark.parametrize("norm_before_gate", (False, True))
@pytest.mark.parametrize("elementwise_affine", (False, True))
def test_rms_norm_gated(
    shape, group_size, activation, norm_before_gate, elementwise_affine
):
    dtype = torch.bfloat16
    device = "cuda"
    eps = 1e-5
    input = torch.randn(shape, dtype=dtype, device=device)
    gate = torch.randn_like(input)
    weight = (
        torch.randn(shape[-1], dtype=dtype, device=device)
        if elementwise_affine
        else None
    )

    actual = ntops.torch.rms_norm_gated(
        input,
        gate,
        weight,
        eps=eps,
        group_size=group_size,
        norm_before_gate=norm_before_gate,
        activation=activation,
    )
    expected = _reference(
        input,
        gate,
        weight,
        eps,
        group_size,
        norm_before_gate,
        activation,
    )

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
