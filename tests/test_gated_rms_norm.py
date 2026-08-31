import pytest
import torch
import torch.nn.functional as F

import ntops
from tests.skippers import skip_if_cuda_not_available


def _reference(
    input, gate, weight, eps, group_size, norm_before_gate, activation
):
    input_float = input.float()
    gate_float = None if gate is None else gate.float()
    activation_fn = torch.sigmoid if activation == "sigmoid" else F.silu

    if gate is not None and not norm_before_gate:
        input_float = input_float * activation_fn(gate_float)

    effective_group_size = group_size or input.shape[-1]
    grouped = input_float.reshape(*input.shape[:-1], -1, effective_group_size)
    variance = grouped.square().mean(dim=-1, keepdim=True)
    output = grouped * torch.rsqrt(variance + eps)
    output = output.reshape_as(input_float) * weight.float()

    if gate is not None and norm_before_gate:
        output = output * activation_fn(gate_float)

    return output.to(input.dtype)


@skip_if_cuda_not_available
@pytest.mark.parametrize("shape", ((3, 64), (2, 5, 128), (2, 3, 7, 256)))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
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

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
@pytest.mark.parametrize(
    ("shape", "group_size"),
    (
        ((3, 64), None),
        ((3, 64), 32),
        ((3, 64), 64),
        ((2, 5, 128), None),
        ((2, 5, 128), 32),
        ((2, 5, 128), 64),
        ((2, 7, 96), None),
        ((2, 7, 96), 48),
    ),
)
def test_gated_rms_norm_without_gate(shape, dtype, group_size):

    input = torch.randn(shape, device="cuda", dtype=dtype)
    weight = torch.randn(shape[-1], device="cuda", dtype=dtype)

    actual = ntops.torch.gated_rms_norm(
        input, None, weight, eps=1e-5, group_size=group_size
    )
    expected = _reference(input, None, weight, 1e-5, group_size, False, "swish")

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


@skip_if_cuda_not_available
def test_gated_rms_norm_non_contiguous_input():
    input = torch.randn((4, 3, 128), device="cuda", dtype=torch.float16).transpose(
        0, 1
    )
    gate = torch.randn_like(input)
    weight = torch.randn(128, device="cuda", dtype=torch.float16)

    actual = ntops.torch.gated_rms_norm(input, gate, weight, group_size=64)
    expected = _reference(input, gate, weight, 1e-5, 64, False, "swish")

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


@skip_if_cuda_not_available
def test_gated_rms_norm_non_power_of_two_group():
    input = torch.randn((2, 7, 96), device="cuda", dtype=torch.float16)
    gate = torch.randn_like(input)
    weight = torch.randn(96, device="cuda", dtype=torch.float16)

    actual = ntops.torch.gated_rms_norm(input, gate, weight, group_size=48)
    expected = _reference(input, gate, weight, 1e-5, 48, False, "swish")

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


def test_gated_rms_norm_rejects_invalid_contract_on_cpu():
    input = torch.randn(2, 8)
    gate = torch.randn(2, 7)
    weight = torch.randn(8)

    with pytest.raises(ValueError, match="input.*gate"):
        ntops.torch.gated_rms_norm(input, gate, weight)

    with pytest.raises(ValueError, match="weight"):
        ntops.torch.gated_rms_norm(input, None, torch.randn(7))

    with pytest.raises(ValueError, match="group_size"):
        ntops.torch.gated_rms_norm(input, None, weight, group_size=3)

    with pytest.raises(ValueError, match="activation"):
        ntops.torch.gated_rms_norm(input, None, weight, activation="gelu")
