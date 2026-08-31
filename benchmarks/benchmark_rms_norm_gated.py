"""Benchmark the fused T3 kernel against the unmodified vLLM operator.

The reported speedup follows the competition definition:

    baseline_latency / submission_latency

Both paths are invoked through their public Python interfaces in the same
process.  PyTorch eager expressions are used for correctness only and are
never timed.
"""

import argparse
import csv
import statistics
import time
from pathlib import Path

import torch

import ntops


def _import_vllm_rms_norm_gated():
    # Some BW images expose both ROCm SMI and an NVML compatibility shim.
    # vLLM 0.9 then detects both built-in platforms and aborts before the
    # operator can be imported.  Suppress only the spurious CUDA discovery
    # while vLLM resolves its platform; the timed operator remains unchanged.
    # This is the unchanged operator called by
    # ``Mixer2RMSNormGated.forward_cuda``.  Calling it directly avoids
    # initializing vLLM's unrelated distributed model context, which is not
    # registered as a standard CUDA platform by some CoreX vLLM builds.
    try:
        from vllm.utils import import_pynvml
    except ImportError:
        from vllm.utils.import_utils import import_pynvml

    pynvml = import_pynvml()
    original = pynvml.nvmlDeviceGetCount
    pynvml.nvmlDeviceGetCount = lambda: 0

    try:
        try:
            from vllm.model_executor.layers.mamba.ops.layernorm_gated import (
                rms_norm_gated,
            )
        except ImportError:
            # vLLM 0.9 only exposes the operation through the layer wrapper.
            from vllm.distributed.parallel_state import (
                init_distributed_environment,
                initialize_model_parallel,
            )
            from vllm.model_executor.layers.mamba.mamba_mixer2 import (
                Mixer2RMSNormGated,
            )

            torch.cuda.set_device(0)
            init_distributed_environment(
                world_size=1,
                rank=0,
                local_rank=0,
                distributed_init_method="tcp://127.0.0.1:29532",
            )
            initialize_model_parallel(1, 1)
            return Mixer2RMSNormGated, False

        return rms_norm_gated, True
    finally:
        pynvml.nvmlDeviceGetCount = original


def _reference(input, gate, weight, eps):
    gated = input.float() * torch.nn.functional.silu(gate.float())
    variance = gated.square().mean(dim=-1, keepdim=True)
    return (gated * torch.rsqrt(variance + eps) * weight.float()).to(input.dtype)


def _time_once(function, iterations):
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        function()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000 / iterations


def _measure_pair(baseline, submission, warmup=50, iterations=500, rounds=7):
    for _ in range(warmup):
        baseline()
        submission()
    torch.cuda.synchronize()

    baseline_samples = []
    submission_samples = []
    for round_index in range(rounds):
        if round_index % 2 == 0:
            baseline_samples.append(_time_once(baseline, iterations))
            submission_samples.append(_time_once(submission, iterations))
        else:
            submission_samples.append(_time_once(submission, iterations))
            baseline_samples.append(_time_once(baseline, iterations))

    return (
        statistics.median(baseline_samples),
        statistics.median(submission_samples),
        baseline_samples,
        submission_samples,
    )


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Benchmark the T3 submission against the vLLM baseline."
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument(
        "--csv",
        type=Path,
        help="Optionally write the median latency and speedup rows as CSV.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    torch.manual_seed(0)
    vllm_entry, direct_vllm_op = _import_vllm_rms_norm_gated()
    eps = 1e-5
    results = []

    for tokens, hidden_size in ((64, 128), (128, 512), (16, 4096)):
        if direct_vllm_op:
            layer = None
            weight = torch.empty(
                hidden_size, device="cuda", dtype=torch.bfloat16
            ).uniform_(-1, 1)
        else:
            layer = vllm_entry(
                full_hidden_size=hidden_size,
                full_n_groups=1,
                use_rms_norm=True,
                eps=eps,
            ).to(device="cuda", dtype=torch.bfloat16)
            layer.weight.data.uniform_(-1, 1)
            weight = layer.weight
        input = torch.randn(
            tokens, hidden_size, device="cuda", dtype=torch.bfloat16
        )
        gate = torch.randn_like(input)

        def baseline():
            if direct_vllm_op:
                return vllm_entry(
                    input,
                    weight,
                    bias=None,
                    z=gate,
                    eps=eps,
                    norm_before_gate=False,
                )
            return layer.forward_cuda(input, gate)

        def submission():
            return ntops.torch.rms_norm_gated(
                input,
                gate,
                weight,
                eps=eps,
                group_size=hidden_size,
                norm_before_gate=False,
                activation="swish",
            )

        baseline_output = baseline()
        submission_output = submission()
        expected = _reference(input, gate, weight, eps)
        torch.testing.assert_close(
            baseline_output, expected, rtol=2e-2, atol=2e-2
        )
        torch.testing.assert_close(
            submission_output, expected, rtol=2e-2, atol=2e-2
        )

        (
            baseline_ms,
            submission_ms,
            baseline_samples,
            submission_samples,
        ) = _measure_pair(
            baseline,
            submission,
            warmup=args.warmup,
            iterations=args.iterations,
            rounds=args.rounds,
        )
        speedup = baseline_ms / submission_ms
        baseline_path = (
            "direct_op" if direct_vllm_op else "layer_forward_cuda"
        )
        print(
            f"shape={tokens}x{hidden_size} correctness=pass "
            f"vllm_path={baseline_path} "
            f"vllm_ms={baseline_ms:.6f} "
            f"submission_ms={submission_ms:.6f} speedup={speedup:.4f} "
            f"vllm_samples={baseline_samples} "
            f"submission_samples={submission_samples}",
            flush=True,
        )
        results.append(
            {
                "shape": f"{tokens}x{hidden_size}",
                "tokens": tokens,
                "hidden_size": hidden_size,
                "correctness": "pass",
                "baseline": "vLLM",
                "baseline_path": baseline_path,
                "baseline_ms": f"{baseline_ms:.6f}",
                "submission_ms": f"{submission_ms:.6f}",
                "speedup": f"{speedup:.4f}",
                "warmup": args.warmup,
                "iterations": args.iterations,
                "rounds": args.rounds,
            }
        )

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=tuple(results[0]))
            writer.writeheader()
            writer.writerows(results)


if __name__ == "__main__":
    main()
