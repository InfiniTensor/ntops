"""T1 MXFP4 grouped GEMM reference comparison.

The benchmark separates three cases:

* ``pytorch_native``: PyTorch's native ``_scaled_grouped_mm`` when the
  installed build accepts the T1 packed MXFP4 tensors;
* ``pytorch_raw_reference``: a GPU PyTorch implementation that decodes the
  vLLM/OCP MXFP4 E2M1 + E8M0 representation on every call;
* ``vllm_fallback_reference``: the same vLLM fallback execution model after
  weights have been dequantized once, followed by grouped BF16 GEMMs.

The last two are explicit semantic references, not silently relabelled as a
native PyTorch kernel.  The output reports native API availability separately.
"""

import argparse
import csv
import importlib
import platform
import statistics
import time
from pathlib import Path

import torch

import ntops

_E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def _decode_weight(packed_weight, weight_scale):
    table = torch.tensor(
        _E2M1_VALUES,
        dtype=torch.float32,
        device=packed_weight.device,
    )
    decoded = torch.empty(
        (*packed_weight.shape[:-1], packed_weight.shape[-1] * 2),
        dtype=torch.float32,
        device=packed_weight.device,
    )
    decoded[..., 0::2] = table[(packed_weight & 0x0F).long()]
    decoded[..., 1::2] = table[(packed_weight >> 4).long()]
    decoded.mul_(torch.exp2(weight_scale.float() - 127.0).repeat_interleave(32, dim=-1))
    return decoded


def _grouped_mm(input, decoded_weight, offsets, output):
    starts = (0, *offsets[:-1])
    for expert, (start, end) in enumerate(zip(starts, offsets)):
        output[start:end].copy_(
            torch.mm(input[start:end], decoded_weight[expert].t()).to(output.dtype)
        )
    return output


def _native_probe(input, packed_weight, scale, offsets, dtype):
    function = getattr(torch, "_scaled_grouped_mm", None)
    if not callable(function):
        return "unavailable:torch._scaled_grouped_mm_missing"
    try:
        output = function(
            input,
            packed_weight,
            torch.ones((1,), device=input.device, dtype=torch.float32),
            scale,
            offsets,
            out_dtype=dtype,
        )
        torch.cuda.synchronize()
        return f"available:shape={tuple(output.shape)}"
    except Exception as error:
        return f"unavailable:{type(error).__name__}:{error}"


def _measure(function, warmup, iterations, rounds):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(rounds):
        started = time.perf_counter()
        for _ in range(iterations):
            function()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000 / iterations)
    return statistics.median(samples), samples


def _parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--counts", default="1,3,5,7")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--csv", type=Path)
    return parser.parse_args(argv)


def _append_csv(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = tuple(row)
    write_header = not path.exists() or path.stat().st_size == 0

    with path.open("a", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        writer.writerow(row)


def main(argv=None):
    args = _parse_args(argv)
    counts = tuple(int(value) for value in args.counts.split(","))
    if len(counts) != args.experts or any(value < 0 for value in counts):
        raise ValueError("counts must have one nonnegative value per expert")
    if args.k % 32:
        raise ValueError("K must be divisible by 32")

    dtype = getattr(torch, args.dtype)
    torch.manual_seed(20260829)
    tokens = sum(counts)
    boundaries = (0, *torch.tensor(counts).cumsum(0).tolist())
    offsets = torch.tensor(boundaries[1:], device="cuda", dtype=torch.int32)
    input = torch.randn(tokens, args.k, device="cuda", dtype=dtype)
    packed_weight = torch.randint(
        0,
        256,
        (args.experts, args.n, args.k // 2),
        device="cuda",
        dtype=torch.uint8,
    )
    weight_scale = torch.randint(
        120,
        132,
        (args.experts, args.n, args.k // 32),
        device="cuda",
        dtype=torch.uint8,
    )
    output = torch.empty(tokens, args.n, device="cuda", dtype=dtype)

    decoded_weight = _decode_weight(packed_weight, weight_scale)
    decoded_weight_compute = decoded_weight.to(dtype)

    def pytorch_predecoded():
        return _grouped_mm(input, decoded_weight_compute, boundaries[1:], output)

    def pytorch_raw():
        return _grouped_mm(
            input,
            _decode_weight(packed_weight, weight_scale).to(dtype),
            boundaries[1:],
            output,
        )

    def submission():
        return ntops.torch.mxfp4_grouped_gemm(
            input,
            packed_weight,
            weight_scale,
            offsets,
            out=output,
        )

    expected = _grouped_mm(
        input.float(), decoded_weight, boundaries[1:], output
    ).clone()
    torch.testing.assert_close(pytorch_raw(), expected, rtol=3e-2, atol=3e-2)
    first = submission().clone()
    torch.testing.assert_close(first, expected, rtol=3e-2, atol=3e-2)

    wrapper = importlib.import_module("ntops.torch.mxfp4_grouped_gemm")
    cold_bound = wrapper._LAST_CALL[3]
    native_status = _native_probe(input, packed_weight, weight_scale, offsets, dtype)

    predecoded_ms, predecoded_samples = _measure(
        pytorch_predecoded, args.warmup, args.iterations, args.rounds
    )
    raw_ms, raw_samples = _measure(
        pytorch_raw, args.warmup, args.iterations, args.rounds
    )
    bound_ms, bound_samples = _measure(
        cold_bound, args.warmup, args.iterations, args.rounds
    )
    submission_ms, submission_samples = _measure(
        submission, args.warmup, args.iterations, args.rounds
    )

    try:
        vllm = importlib.import_module("vllm")
        vllm_version = getattr(vllm, "__version__", "unknown")
    except Exception as error:
        vllm_version = f"unavailable:{type(error).__name__}"

    device_name = torch.cuda.get_device_name(0)
    speedup = predecoded_ms / submission_ms
    raw_speedup = raw_ms / submission_ms
    bound_speedup = predecoded_ms / bound_ms
    print(
        "T1_RESULT "
        f"device={device_name!r} "
        f"torch={torch.__version__!r} vllm={vllm_version!r} "
        f"experts={args.experts} counts={counts} n={args.n} k={args.k} "
        f"dtype={args.dtype} correctness=pass "
        f"pytorch_native={native_status} "
        f"vllm_fallback_predecoded_ms={predecoded_ms:.6f} "
        f"pytorch_raw_reference_ms={raw_ms:.6f} "
        f"submission_bound_ms={bound_ms:.6f} "
        f"submission_public_ms={submission_ms:.6f} "
        f"speedup_vs_vllm_fallback={speedup:.6f} "
        f"speedup_vs_raw_reference={raw_speedup:.6f} "
        f"bound_speedup_vs_vllm_fallback={bound_speedup:.6f} "
        f"predecoded_samples={predecoded_samples} raw_samples={raw_samples} "
        f"bound_samples={bound_samples} submission_samples={submission_samples}",
        flush=True,
    )

    if args.csv is not None:
        _append_csv(
            args.csv,
            {
                "accelerator": device_name,
                "python": platform.python_version(),
                "torch": torch.__version__,
                "vllm": vllm_version,
                "experts": args.experts,
                "counts": ";".join(str(value) for value in counts),
                "n": args.n,
                "k": args.k,
                "dtype": args.dtype,
                "baseline": "vllm_semantic_fallback_predecoded",
                "native_pytorch": native_status,
                "correctness": "pass",
                "baseline_ms": f"{predecoded_ms:.6f}",
                "submission_ms": f"{submission_ms:.6f}",
                "speedup": f"{speedup:.6f}",
                "raw_reference_ms": f"{raw_ms:.6f}",
                "raw_reference_speedup": f"{raw_speedup:.6f}",
                "bound_submission_ms": f"{bound_ms:.6f}",
                "warmup": args.warmup,
                "iterations": args.iterations,
                "rounds": args.rounds,
            },
        )


if __name__ == "__main__":
    main()
