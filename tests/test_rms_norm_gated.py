import pytest
import torch
import torch.nn.functional as F

import ntops
from tests.skippers import skip_if_cuda_not_available


def _reference_rms_norm_gated(
    input,
    z,
    weight,
    eps,
    group_size,
    norm_before_gate,
    activation,
):
    x = input.float()
    weight = weight.float()
    z = None if z is None else z.float()
    act_fn = torch.sigmoid if activation == "sigmoid" else F.silu

    if z is not None and not norm_before_gate:
        x = x * act_fn(z)

    if group_size is None:
        variance = x.square().mean(dim=-1, keepdim=True)
        output = x * torch.rsqrt(variance + eps) * weight
    else:
        x_group = x.reshape(*x.shape[:-1], -1, group_size)
        variance = x_group.square().mean(dim=-1, keepdim=True)
        output = (x_group * torch.rsqrt(variance + eps)).reshape_as(x) * weight

    if z is not None and norm_before_gate:
        output = output * act_fn(z)

    return output.to(input.dtype)


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
@pytest.mark.parametrize("group_size", (None, 4))
@pytest.mark.parametrize("norm_before_gate", (False, True))
@pytest.mark.parametrize("activation", ("swish", "sigmoid"))
def test_rms_norm_gated(dtype, group_size, norm_before_gate, activation):
    input = torch.randn((2, 3, 8), dtype=dtype, device="cuda")
    z = torch.randn_like(input)
    weight = torch.randn((8,), dtype=dtype, device="cuda")

    output = ntops.torch.rms_norm_gated(
        input,
        z,
        weight,
        group_size=group_size,
        norm_before_gate=norm_before_gate,
        activation=activation,
    )
    reference = _reference_rms_norm_gated(
        input,
        z,
        weight,
        1e-5,
        group_size,
        norm_before_gate,
        activation,
    )

    torch.testing.assert_close(output, reference, rtol=2e-3, atol=2e-3)


@skip_if_cuda_not_available
def test_rms_norm_gated_without_gate_or_weight():
    input = torch.randn((2, 8), dtype=torch.float32, device="cuda")

    output = ntops.torch.rms_norm_gated(input)
    reference = torch.rsqrt(input.square().mean(dim=-1, keepdim=True) + 1e-5) * input

    torch.testing.assert_close(output, reference, rtol=2e-3, atol=2e-3)


@skip_if_cuda_not_available
def test_rms_norm_gated_hidden_size_larger_than_block_size():
    input = torch.randn((2, 257), dtype=torch.float32, device="cuda")
    z = torch.randn_like(input)
    weight = torch.randn((257,), dtype=torch.float32, device="cuda")

    output = ntops.torch.rms_norm_gated(input, z, weight, block_size=128)
    reference = _reference_rms_norm_gated(
        input,
        z,
        weight,
        1e-5,
        None,
        False,
        "swish",
    )

    torch.testing.assert_close(output, reference, rtol=2e-3, atol=2e-3)


def test_rms_norm_gated_validates_arguments():
    input = torch.randn((2, 8))

    with pytest.raises(ValueError, match="activation"):
        ntops.torch.rms_norm_gated(input, activation="gelu")

    with pytest.raises(ValueError, match="positive integer"):
        ntops.torch.rms_norm_gated(input, group_size=0)

    with pytest.raises(ValueError, match="divisible"):
        ntops.torch.rms_norm_gated(input, group_size=3)
