import argparse

import torch
import torch.nn.functional as F

import ntops


def reference(
    input, gate, weight, eps, group_size, norm_before_gate, activation
):
    input_float = input.float()
    gate_float = gate.float()
    activation_fn = torch.sigmoid if activation == "sigmoid" else F.silu

    if not norm_before_gate:
        input_float = input_float * activation_fn(gate_float)

    grouped = input_float.reshape(*input.shape[:-1], -1, group_size)
    variance = grouped.square().mean(dim=-1, keepdim=True)
    output = grouped * torch.rsqrt(variance + eps)
    output = output.reshape_as(input_float) * weight.float()

    if norm_before_gate:
        output = output * activation_fn(gate_float)

    return output.to(input.dtype)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.manual_seed(42)
    cases = (
        ((3, 64), torch.float16, 64, False, "swish"),
        ((2, 5, 128), torch.float16, 32, True, "swish"),
        ((2, 3, 256), torch.float16, 64, False, "sigmoid"),
        ((2, 7, 96), torch.float16, 48, False, "swish"),
        ((4, 128), torch.bfloat16, 128, True, "sigmoid"),
    )

    for shape, dtype, group_size, norm_before_gate, activation in cases:
        input = torch.randn(shape, device=args.device, dtype=dtype)
        gate = torch.randn_like(input)
        weight = torch.randn(shape[-1], device=args.device, dtype=torch.float16)
        actual = ntops.torch.gated_rms_norm(
            input,
            gate,
            weight,
            group_size=group_size,
            norm_before_gate=norm_before_gate,
            activation=activation,
        )
        expected = reference(
            input,
            gate,
            weight,
            1e-5,
            group_size,
            norm_before_gate,
            activation,
        )
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        print(
            "PASS",
            f"shape={shape}",
            f"dtype={dtype}",
            f"group_size={group_size}",
            f"norm_before_gate={norm_before_gate}",
            f"activation={activation}",
        )


if __name__ == "__main__":
    main()
