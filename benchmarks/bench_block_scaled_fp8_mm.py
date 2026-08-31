"""Benchmark BlockWise1x128 x BlockWise128x128 FP8 projection shapes."""

import importlib

import torch
import torch.nn.functional as F
from _benchmark import benchmark_mean, selected_config

import ntops

_block_scaled_fp8_mm_module = importlib.import_module("ntops.torch.block_scaled_fp8_mm")


def _column_major(value):
    return value.t().contiguous().t()


def _make_inputs(m, n, k):
    mat_a = torch.randn((m, k), device="cuda").clamp(-3, 3).to(torch.float8_e4m3fn)
    weight = torch.randn((n, k), device="cuda").clamp(-3, 3).to(torch.float8_e4m3fn)
    k_blocks = k // 128
    padded_k_blocks = ((k_blocks + 3) // 4) * 4
    scale_a = _column_major(
        torch.ones((m, k_blocks), device="cuda", dtype=torch.float32)
    )
    scale_b = _column_major(
        torch.ones(
            (padded_k_blocks, n // 128),
            device="cuda",
            dtype=torch.float32,
        )
    )
    return mat_a, weight.t(), scale_a, scale_b


def _torch_dtype_reference(mat_a, mat_b, scale_a, scale_b):
    """Dequantize through PyTorch's FP8 dtype conversion, then run FP32 MM."""
    m, k = mat_a.shape
    n = mat_b.shape[1]
    k_blocks = k // 128

    dequant_a = (
        mat_a.float().reshape(m, k_blocks, 128) * scale_a.float().unsqueeze(-1)
    ).reshape(m, k)
    expanded_scale_b = scale_b[:k_blocks].float().repeat_interleave(128, dim=1)
    dequant_b = (
        mat_b.float().reshape(k_blocks, 128, n) * expanded_scale_b.unsqueeze(1)
    ).reshape(k, n)
    return (dequant_a @ dequant_b).to(torch.bfloat16)


def _select_reference(mat_a, mat_b, scale_a, scale_b):
    native = getattr(F, "scaled_mm", None)
    if callable(native):

        def run_native():
            return native(
                mat_a,
                mat_b,
                scale_a,
                ntops.torch.ScalingType.BlockWise1x128,
                scale_b,
                ntops.torch.ScalingType.BlockWise128x128,
            )

        try:
            return "torch.nn.functional.scaled_mm", run_native, run_native()
        except (NotImplementedError, RuntimeError, TypeError, ValueError):
            pass

    def run_dtype_reference():
        return _torch_dtype_reference(mat_a, mat_b, scale_a, scale_b)

    return "torch_dtype_dequant_mm", run_dtype_reference, run_dtype_reference()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA-compatible accelerator is required")

    print(f"torch={torch.__version__}")
    print(f"hip={torch.version.hip}")
    print(f"device={torch.cuda.get_device_name()}")
    print(
        "scenario,M,N,K,reference_provider,reference_mean_us,ntops_mean_us,"
        "best_num_warps,best_num_stages,speedup_vs_reference,tflops,"
        "max_abs_error"
    )
    scenarios = (
        ("attention_decode", 1, 4096, 4096),
        ("moe_expert", 32, 14336, 4096),
        ("linear_prefill", 128, 4096, 4096),
    )
    for name, m, n, k in scenarios:
        mat_a, mat_b, scale_a, scale_b = _make_inputs(m, n, k)

        def run_ntops():
            return ntops.torch.block_scaled_fp8_mm(
                mat_a,
                mat_b,
                scale_a,
                ntops.torch.ScalingType.BlockWise1x128,
                scale_b,
                ntops.torch.ScalingType.BlockWise128x128,
            )

        with torch.inference_mode():
            output = run_ntops()
            reference_provider, run_reference, reference = _select_reference(
                mat_a, mat_b, scale_a, scale_b
            )
            torch.testing.assert_close(output, reference, rtol=0.03, atol=0.03)
            max_abs_error = (output.float() - reference.float()).abs().max().item()
            torch.cuda.synchronize()

        kernel = _block_scaled_fp8_mm_module._make_kernel(
            mat_a.dtype,
            torch.bfloat16,
            None,
            mat_a.device,
        )
        _, fixed_warps, fixed_stages, _ = (
            _block_scaled_fp8_mm_module._kernel_tuning_config(mat_a.device)
        )
        fallback_warps = fixed_warps if isinstance(fixed_warps, int) else fixed_warps[0]
        warps, stages = selected_config(kernel, fallback=(fallback_warps, fixed_stages))

        reference_us = benchmark_mean(run_reference)
        ntops_us = benchmark_mean(run_ntops)
        tflops = 2.0 * m * n * k / (ntops_us * 1.0e6)
        print(
            f"{name},{m},{n},{k},{reference_provider},{reference_us:.3f},"
            f"{ntops_us:.3f},{warps},{stages},{reference_us / ntops_us:.3f},"
            f"{tflops:.3f},{max_abs_error:.6f}"
        )


if __name__ == "__main__":
    main()
