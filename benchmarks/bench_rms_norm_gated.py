"""Benchmark the autotuned RMSNormGated implementation."""

import torch
import torch.nn.functional as F
from _benchmark import benchmark_mean, selected_config

import ntops
from ntops.torch.utils import _cached_make

DEVICE = "cuda"
DTYPE = torch.bfloat16
HIDDEN_SIZE = 128
LOCAL_VALUE_HEADS = 8  # Qwen3-Next: 32 value heads with TP=4.
EPS = 1e-5
BLOCK_SIZE = 128
AUTOTUNE_WARPS = (1, 2, 4, 8)
AUTOTUNE_STAGES = (1, 2)
MAX_NUM_CONFIGS = 8


def pytorch_reference(input, z, weight):
    x = input.float()
    gate = F.silu(z.float())
    output = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + EPS)
    return (output * weight.float() * gate).to(input.dtype)


def _kernel_handle(input, z, weight):
    return _cached_make(
        ntops.kernels.rms_norm_gated.premake,
        input.ndim,
        HIDDEN_SIZE,
        None,
        True,
        "silu",
        input_dtype=input.dtype,
        gate_dtype=z.dtype,
        weight_dtype=weight.dtype,
        output_dtype=input.dtype,
        block_size=BLOCK_SIZE,
        has_gate=True,
        num_warps=AUTOTUNE_WARPS,
        num_stages=AUTOTUNE_STAGES,
        max_num_configs=MAX_NUM_CONFIGS,
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    print(f"torch={torch.__version__}")
    print(f"device={torch.cuda.get_device_name()}")
    print(
        "scenario,tokens,rows,hidden,dtype,best_num_warps,best_num_stages,"
        "autotuned_mean_us,pytorch_mean_us,"
        "speedup_vs_pytorch,max_abs_error"
    )

    scenarios = (
        ("decode", 1),
        ("concurrent_decode", 10),
        ("prefill_2048", 2048),
    )
    for name, num_tokens in scenarios:
        rows = num_tokens * LOCAL_VALUE_HEADS
        input = torch.randn((rows, HIDDEN_SIZE), device=DEVICE, dtype=DTYPE)
        z = torch.randn_like(input)
        weight = torch.randn(HIDDEN_SIZE, device=DEVICE, dtype=torch.float32)

        def autotuned_function():
            return ntops.torch.rms_norm_gated(
                input,
                z,
                weight,
                eps=EPS,
                norm_before_gate=True,
                activation="silu",
            )

        def pytorch_function():
            return pytorch_reference(input, z, weight)

        with torch.inference_mode():
            autotuned_output = autotuned_function()
            pytorch_output = pytorch_function()
            torch.cuda.synchronize()
            max_abs_error = (
                (autotuned_output.float() - pytorch_output.float()).abs().max().item()
            )

        warps, stages = selected_config(_kernel_handle(input, z, weight))
        autotuned_mean = benchmark_mean(autotuned_function)
        pytorch_mean = benchmark_mean(pytorch_function)
        print(
            f"{name},{num_tokens},{rows},{HIDDEN_SIZE},{DTYPE},"
            f"{warps},{stages},{autotuned_mean:.3f},{pytorch_mean:.3f},"
            f"{pytorch_mean / autotuned_mean:.3f},{max_abs_error:.6f}"
        )


if __name__ == "__main__":
    main()
