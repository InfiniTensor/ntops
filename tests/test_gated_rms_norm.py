import pytest
import torch
import torch.nn.functional as F

import ntops
from tests.skippers import skip_if_cuda_not_available


def _reference(
    input, gate, weight, eps, group_size, norm_before_gate, activation
):
    input_float = input.float()
    gate_float = gate.float()
    activation_fn = torch.sigmoid if activation == "sigmoid" else F.silu

    if not norm_before_gate:
        input_float = input_float * activation_fn(gate_float)

    effective_group_size = group_size or input.shape[-1]
    grouped = input_float.reshape(*input.shape[:-1], -1, effective_group_size)
    variance = grouped.square().mean(dim=-1, keepdim=True)
    output = grouped * torch.rsqrt(variance + eps)
    output = output.reshape_as(input_float) * weight.float()

    if norm_before_gate:
        output = output * activation_fn(gate_float)

    return output.to(input.dtype)


@skip_if_cuda_not_available
@pytest.mark.parametrize("shape", ((3, 64), (2, 5, 128), (2, 3, 7, 256)))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("group_size", (None, 32, 64))
@pytest.mark.parametrize("norm_before_gate", (False, True))
@pytest.mark.parametrize("activation", ("swish", "sigmoid"))
def test_gated_rms_norm(
    shape, dtype, group_size, norm_before_gate, activation
):
    input = torch.randn(shape, device="cuda", dtype=dtype)
    gate = torch.randn_like(input)
    weight = torch.randn(shape[-1], device="cuda", dtype=dtype)

    actual = ntops.torch.gated_rms_norm(
        input,
        gate,
        weight,
        eps=1e-5,
        group_size=group_size,
        norm_before_gate=norm_before_gate,
        activation=activation,
    )
    expected = _reference(
        input, gate, weight, 1e-5, group_size, norm_before_gate, activation
    )

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@skip_if_cuda_not_available
def test_gated_rms_norm_non_contiguous_input():
    input = torch.randn((4, 3, 128), device="cuda", dtype=torch.float16).transpose(
        0, 1
    )
    gate = torch.randn_like(input)
    weight = torch.randn(128, device="cuda", dtype=torch.float16)

    actual = ntops.torch.gated_rms_norm(input, gate, weight, group_size=64)
    expected = _reference(input, gate, weight, 1e-5, 64, False, "swish")

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@skip_if_cuda_not_available
def test_gated_rms_norm_non_power_of_two_group():
    input = torch.randn((2, 7, 96), device="cuda", dtype=torch.float16)
    gate = torch.randn_like(input)
    weight = torch.randn(96, device="cuda", dtype=torch.float16)

    actual = ntops.torch.gated_rms_norm(input, gate, weight, group_size=48)
    expected = _reference(input, gate, weight, 1e-5, 48, False, "swish")

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
