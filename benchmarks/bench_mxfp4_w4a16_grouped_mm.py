"""Benchmark W4A16 MXFP4 scaled grouped matrix multiplication."""

import importlib

import torch
import torch.nn.functional as F
from _benchmark import benchmark_mean, selected_config

import ntops

_mxfp4_w4a16_grouped_mm_module = importlib.import_module(
    "ntops.torch.mxfp4_w4a16_grouped_mm"
)

DEVICE = "cuda"
DTYPE = torch.bfloat16
GROUP_COUNT = 8
K = 4096
N = 4096

SCENARIOS = (
    ("uniform_decode", 1),
    ("uniform_concurrent", 16),
    ("routed_uneven", (1, 0, 3, 8, 16, 24, 32, 44)),
)


def _native_packed_dtypes():
    return (
        getattr(torch, "float4_e2m1fn_x2", None),
        getattr(torch, "float8_e8m0fnu", None),
    )


def _decode_nibble(code):
    magnitude_code = code & 0x7
    exponent = (magnitude_code >> 1).to(torch.int32)
    mantissa = (magnitude_code & 1).to(torch.float32)
    normal = (1.0 + 0.5 * mantissa) * torch.exp2(exponent.to(torch.float32) - 1.0)
    magnitude = torch.where(exponent == 0, 0.5 * mantissa, normal)
    sign = torch.where((code & 0x8) == 0, 1.0, -1.0)
    return sign * magnitude


def _manual_decode_mxfp4(packed, scales):
    if packed.dtype != torch.uint8:
        packed = packed.view(torch.uint8)
    if scales.dtype != torch.uint8:
        scales = scales.view(torch.uint8)

    group_count, packed_k, n = packed.shape
    weight = torch.empty(
        group_count,
        packed_k * 2,
        n,
        dtype=torch.bfloat16,
        device=packed.device,
    )
    block_scales = torch.exp2(scales.to(torch.float32) - 127.0)
    block_scales = block_scales.repeat_interleave(32, dim=1)
    weight[:, 0::2] = (_decode_nibble(packed & 0xF) * block_scales[:, 0::2]).to(
        torch.bfloat16
    )
    weight[:, 1::2] = (_decode_nibble(packed >> 4) * block_scales[:, 1::2]).to(
        torch.bfloat16
    )
    return weight


def _make_inputs(rows):
    packed = torch.randint(
        0,
        256,
        (GROUP_COUNT, K // 2, N),
        dtype=torch.uint8,
        device=DEVICE,
    )
    scales = torch.randint(
        124,
        128,
        (GROUP_COUNT, K // 32, N),
        dtype=torch.uint8,
        device=DEVICE,
    )
    packed_dtype, scale_dtype = _native_packed_dtypes()
    native_dtypes = packed_dtype is not None and scale_dtype is not None
    mat_b = packed.view(packed_dtype) if native_dtypes else packed
    scale_b = scales.view(scale_dtype) if native_dtypes else scales

    if isinstance(rows, int):
        mat_a = torch.randn(
            GROUP_COUNT, rows, K, dtype=DTYPE, device=DEVICE
        ).contiguous()
        return mat_a, mat_b, scale_b, None, None, native_dtypes

    row_ends = tuple(torch.tensor(rows).cumsum(0).tolist())
    mat_a = torch.randn(sum(rows), K, dtype=DTYPE, device=DEVICE).contiguous()
    offs = torch.tensor(row_ends, dtype=torch.int32, device=DEVICE)
    return mat_a, mat_b, scale_b, offs, row_ends, native_dtypes


def _torch_matmul(mat_a, weight, row_ends):
    if row_ends is None:
        return torch.bmm(mat_a.float(), weight.float()).to(torch.bfloat16)

    outputs = []
    start = 0
    for expert, end in enumerate(row_ends):
        outputs.append(
            (mat_a[start:end].float() @ weight[expert].float()).to(torch.bfloat16)
        )
        start = end
    return torch.cat(outputs)


def _ntops_mxfp4_w4a16_grouped_mm(mat_a, mat_b, scale_b, offs):
    def run():
        return ntops.torch.mxfp4_w4a16_grouped_mm(
            mat_a,
            mat_b,
            None,
            None,
            scale_b,
            ntops.torch.ScalingType.BlockWise1x32,
            offs=offs,
        )

    if offs is None:
        return run()

    # PyTorch's jagged nested-tensor constructor requires a version counter.
    with torch.inference_mode(False):
        return run()


def _select_reference(mat_a, mat_b, scale_b, offs, row_ends, native_dtypes):
    native = getattr(F, "scaled_grouped_mm", None)
    fallback_reason = "native_dtype_unavailable"
    if native_dtypes and callable(native):

        def run_native():
            return native(
                mat_a,
                mat_b,
                None,
                None,
                scale_b,
                ntops.torch.ScalingType.BlockWise1x32,
                offs=offs,
            )

        try:
            return "torch.nn.functional.scaled_grouped_mm", run_native, run_native()
        except (NotImplementedError, RuntimeError, TypeError, ValueError):
            fallback_reason = "native_operator_unavailable"
    elif native_dtypes:
        fallback_reason = "native_operator_unavailable"

    def run_manual_reference():
        weight = _manual_decode_mxfp4(mat_b, scale_b)
        return _torch_matmul(mat_a, weight, row_ends)

    provider = f"manual_dequant_mm_{fallback_reason}"
    return provider, run_manual_reference, run_manual_reference()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-compatible accelerator is required for this benchmark")

    torch.manual_seed(0)
    backend = "hip_reduction" if torch.version.hip is not None else "block_dot"
    print(f"torch={torch.__version__}")
    print(f"device={torch.cuda.get_device_name()}")
    print(
        "scenario,mode,groups,total_rows,max_rows,k,n,backend,input_storage,"
        "reference_provider,reference_mean_us,ntops_mean_us,"
        "best_num_warps,best_num_stages,speedup_vs_reference,effective_tflops,"
        "max_abs_error"
    )

    for scenario, rows in SCENARIOS:
        mat_a, mat_b, scale_b, offs, row_ends, native_dtypes = _make_inputs(rows)

        def run_ntops():
            return _ntops_mxfp4_w4a16_grouped_mm(mat_a, mat_b, scale_b, offs)

        with torch.inference_mode():
            output = run_ntops()
            reference_provider, run_reference, reference = _select_reference(
                mat_a, mat_b, scale_b, offs, row_ends, native_dtypes
            )
            if reference is not None:
                torch.testing.assert_close(output, reference, rtol=0.03, atol=0.03)
                max_abs_error = (output.float() - reference.float()).abs().max().item()
            else:
                max_abs_error = float("nan")
            torch.cuda.synchronize()

        kernel = _mxfp4_w4a16_grouped_mm_module._make_kernel(offs is not None)
        _, _, _, fixed_warps, fixed_stages, _ = (
            _mxfp4_w4a16_grouped_mm_module._kernel_launch_config(offs is not None)
        )
        fallback_warps = fixed_warps if isinstance(fixed_warps, int) else fixed_warps[0]
        warps, stages = selected_config(kernel, fallback=(fallback_warps, fixed_stages))

        reference_us = (
            benchmark_mean(run_reference) if run_reference is not None else float("nan")
        )
        ntops_us = benchmark_mean(run_ntops)

        if isinstance(rows, int):
            mode = "uniform"
            total_rows = GROUP_COUNT * rows
            max_rows = rows
        else:
            mode = "routed"
            total_rows = sum(rows)
            max_rows = max(rows)
        input_storage = "native_packed" if native_dtypes else "raw_uint8"
        speedup = reference_us / ntops_us
        effective_tflops = 2 * total_rows * K * N / (ntops_us * 1e6)

        print(
            f"{scenario},{mode},{GROUP_COUNT},{total_rows},{max_rows},{K},{N},"
            f"{backend},{input_storage},{reference_provider},{reference_us:.3f},"
            f"{ntops_us:.3f},{warps},{stages},{speedup:.3f},"
            f"{effective_tflops:.3f},{max_abs_error:.6f}"
        )


if __name__ == "__main__":
    main()
