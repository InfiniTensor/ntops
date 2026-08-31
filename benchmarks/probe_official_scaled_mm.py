"""Probe PyTorch/vLLM T2 baseline paths on a target accelerator.

The eager implementation in this file is used only as a correctness oracle. It
is never timed and is never reported as a performance baseline.

``vllm_portable`` is a separately labelled vLLM-derived compatibility baseline;
the ``pytorch`` and ``vllm_triton`` modes continue to execute installed code
without changing its timed kernel.
"""

import argparse
import time

import torch


SCALE_BLOCK_SIZE = 128


def _ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def _make_inputs(m, n, k, fp8_dtype):
    input = torch.randn((m, k), device="cuda").clamp(-2, 2).to(fp8_dtype)
    weight = torch.randn((n, k), device="cuda").clamp(-2, 2).to(fp8_dtype)
    scale_a = torch.rand(
        (m, _ceil_div(k, SCALE_BLOCK_SIZE)),
        device="cuda",
        dtype=torch.float32,
    ) + 0.25
    weight_scale = torch.rand(
        (
            _ceil_div(n, SCALE_BLOCK_SIZE),
            _ceil_div(k, SCALE_BLOCK_SIZE),
        ),
        device="cuda",
        dtype=torch.float32,
    ) + 0.25
    return input, weight, scale_a, weight_scale


def _reference(input, weight, scale_a, weight_scale):
    m, k = input.shape
    n = weight.shape[0]
    output = torch.zeros((m, n), dtype=torch.float32, device=input.device)

    for block in range(_ceil_div(k, SCALE_BLOCK_SIZE)):
        start = block * SCALE_BLOCK_SIZE
        end = min(start + SCALE_BLOCK_SIZE, k)
        scale_b_block = weight_scale[:, block].repeat_interleave(
            SCALE_BLOCK_SIZE
        )[:n]
        partial = input[:, start:end].float() @ weight[:, start:end].float().t()
        output += partial * scale_a[:, block, None] * scale_b_block[None, :]

    return output.to(torch.bfloat16)


def _make_candidate(
    mode,
    input,
    weight,
    scale_a,
    weight_scale,
    vllm_config=None,
    ntops_max_block_m=None,
):
    if mode == "ntops":
        import ntops

        if ntops_max_block_m is not None:
            ntops.kernels.scaled_mm.MAX_SPECIALIZED_BLOCK_SIZE_M = (
                ntops_max_block_m
            )

        mat2 = weight.t()
        scale_b = weight_scale.t().contiguous()

        def run():
            return ntops.torch.scaled_mm(
                input,
                mat2,
                scale_a,
                scale_b,
                out_dtype=torch.bfloat16,
            )

        return run

    if mode == "pytorch":
        mat2 = weight.t()
        scale_b = weight_scale.t().contiguous()

        def run():
            return torch._scaled_mm(
                input,
                mat2,
                scale_a,
                scale_b,
                out_dtype=torch.bfloat16,
            )

        return run

    if mode == "vllm_portable":
        from vllm_portable_scaled_mm import (
            fp8_dot_fallback_name,
            w8a8_portable_block_scaled_mm,
        )

        print(
            "stage=vllm_portable "
            f"fp8_dot_fallback={fp8_dot_fallback_name()} source=vllm",
            flush=True,
        )

        def run():
            return w8a8_portable_block_scaled_mm(
                input,
                weight,
                scale_a,
                weight_scale,
                [SCALE_BLOCK_SIZE, SCALE_BLOCK_SIZE],
                torch.bfloat16,
                vllm_config,
            )

        return run

    if mode == "vllm_triton":
        restore_platform_probe = _select_vllm_rocm_platform()

        try:
            from vllm.model_executor.layers.quantization.utils import fp8_utils
        finally:
            restore_platform_probe()

        if vllm_config is not None:
            fp8_utils.get_w8a8_block_fp8_configs = (
                lambda _n, _k, _block_n, _block_k: {input.shape[0]: vllm_config}
            )

        function = getattr(fp8_utils, "w8a8_triton_block_scaled_mm", None)

        if function is None:
            function = getattr(fp8_utils, "w8a8_block_fp8_matmul", None)

        if function is None:
            raise RuntimeError("vLLM does not expose a block-scaled FP8 GEMM.")

        def run():
            return function(
                input,
                weight,
                scale_a,
                weight_scale,
                [SCALE_BLOCK_SIZE, SCALE_BLOCK_SIZE],
                torch.bfloat16,
            )

        return run

    if mode == "vllm_vendor":
        restore_platform_probe = _select_vllm_rocm_platform()

        try:
            from vllm.model_executor.layers.quantization.utils.fp8_utils import (
                cutlass_scaled_mm,
            )
        finally:
            restore_platform_probe()

        def run():
            return cutlass_scaled_mm(
                input,
                weight,
                scale_a,
                weight_scale,
                [SCALE_BLOCK_SIZE, SCALE_BLOCK_SIZE],
                torch.bfloat16,
            )

        return run

    raise ValueError(f"Unknown mode: {mode}")


def _select_vllm_rocm_platform():
    """Prevent the DTK NVML shim from activating CUDA beside ROCm.

    The DTK vLLM wheel otherwise activates both built-in platform plugins on a
    BW device.  This changes platform discovery only; the timed FP8 kernel is
    imported unchanged from the installed vLLM package.
    """

    if getattr(torch, "corex", False):
        return lambda: None

    try:
        from vllm.utils import import_pynvml
    except ImportError:
        from vllm.utils.import_utils import import_pynvml

    pynvml = import_pynvml()
    original = pynvml.nvmlDeviceGetCount
    pynvml.nvmlDeviceGetCount = lambda: 0

    def restore():
        pynvml.nvmlDeviceGetCount = original

    return restore


def _correctness(output, expected, rtol, atol):
    close = torch.isclose(output, expected, rtol=rtol, atol=atol)
    difference = (output.float() - expected.float()).abs()
    return {
        "passed": bool(close.all()),
        "mismatched": int((~close).sum().item()),
        "elements": close.numel(),
        "max_abs": float(difference.max().item()),
    }


def _latency_ms(function, warmup, iterations):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        function()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / iterations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=(
            "ntops",
            "pytorch",
            "vllm_portable",
            "vllm_triton",
            "vllm_vendor",
        ),
    )
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--fp8-dtype",
        choices=("e4m3fn", "e5m2", "e4m3fnuz", "e5m2fnuz"),
        default="e4m3fn",
    )
    parser.add_argument("--rtol", type=float, default=0.05)
    parser.add_argument("--atol", type=float, default=0.05)
    parser.add_argument("--vllm-block-m", type=int)
    parser.add_argument("--vllm-block-n", type=int)
    parser.add_argument("--vllm-num-warps", type=int, default=4)
    parser.add_argument("--vllm-group-m", type=int, default=16)
    parser.add_argument("--vllm-matrix-instr-nonkdim", type=int, default=16)
    parser.add_argument("--vllm-kpack", type=int, default=1)
    parser.add_argument("--ntops-max-block-m", type=int)
    args = parser.parse_args()

    if args.ntops_max_block_m is not None and (
        args.mode != "ntops"
        or args.ntops_max_block_m < 16
        or args.ntops_max_block_m > 128
        or args.ntops_max_block_m & (args.ntops_max_block_m - 1)
    ):
        raise ValueError(
            "--ntops-max-block-m requires --mode ntops and a power of two "
            "between 16 and 128."
        )

    if args.n % SCALE_BLOCK_SIZE or args.k % SCALE_BLOCK_SIZE:
        raise ValueError("The official baseline probe requires N and K / 128.")

    if args.ntops_max_block_m is not None:
        print(
            f"stage=ntops_config max_block_m={args.ntops_max_block_m}",
            flush=True,
        )

    print(
        f"stage=setup mode={args.mode} shape={args.m}x{args.n}x{args.k} "
        f"fp8_dtype={args.fp8_dtype}",
        flush=True,
    )
    torch.manual_seed(0)
    fp8_dtype = getattr(torch, f"float8_{args.fp8_dtype}")
    input, weight, scale_a, weight_scale = _make_inputs(
        args.m,
        args.n,
        args.k,
        fp8_dtype,
    )
    vllm_config = None

    if args.vllm_block_m is not None or args.vllm_block_n is not None:
        if args.vllm_block_m is None or args.vllm_block_n is None:
            raise ValueError("Both vLLM block dimensions must be provided.")
        vllm_config = {
            "BLOCK_SIZE_M": args.vllm_block_m,
            "BLOCK_SIZE_N": args.vllm_block_n,
            "BLOCK_SIZE_K": SCALE_BLOCK_SIZE,
            "GROUP_SIZE_M": args.vllm_group_m,
            "num_warps": args.vllm_num_warps,
            "matrix_instr_nonkdim": args.vllm_matrix_instr_nonkdim,
            "kpack": args.vllm_kpack,
        }
        print(f"stage=vllm_config config={vllm_config}", flush=True)

    candidate = _make_candidate(
        args.mode,
        input,
        weight,
        scale_a,
        weight_scale,
        vllm_config,
        args.ntops_max_block_m,
    )
    torch.cuda.synchronize()
    print("stage=first_call", flush=True)
    first_start = time.perf_counter()
    output = candidate()
    torch.cuda.synchronize()
    first_call_ms = (time.perf_counter() - first_start) * 1000
    print("stage=reference", flush=True)
    expected = _reference(input, weight, scale_a, weight_scale)
    result = _correctness(output, expected, args.rtol, args.atol)
    print(
        "stage=correctness "
        f"passed={result['passed']} mismatched={result['mismatched']}/"
        f"{result['elements']} max_abs={result['max_abs']:.6g} "
        f"first_call_ms={first_call_ms:.4f}",
        flush=True,
    )

    if not result["passed"]:
        raise SystemExit(2)

    latency_ms = _latency_ms(candidate, args.warmup, args.iterations)
    role = "submission" if args.mode == "ntops" else "baseline"
    print(
        f"stage=latency {role}={args.mode} latency_ms={latency_ms:.6f} "
        f"warmup={args.warmup} iterations={args.iterations}",
        flush=True,
    )


if __name__ == "__main__":
    main()
