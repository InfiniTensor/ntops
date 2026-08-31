"""Micro-ablate the checked and prevalidated NineToothed launch paths.

This is not a competition speedup benchmark.  It isolates the B-part runtime
launch optimization while keeping the generated T3 kernel and tensor objects
fixed.  The competition result remains the vLLM-versus-submission measurement
in ``benchmark_rms_norm_gated.py``.
"""

import statistics
import time

import torch

import ntops
from ntops.torch.utils import _cached_make


def _time_once(function, iterations):
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        function()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000 / iterations


def _measure_pair(lhs, rhs, warmup=50, iterations=500, rounds=7):
    for _ in range(warmup):
        lhs()
        rhs()
    torch.cuda.synchronize()

    lhs_samples = []
    rhs_samples = []
    for round_index in range(rounds):
        if round_index % 2 == 0:
            lhs_samples.append(_time_once(lhs, iterations))
            rhs_samples.append(_time_once(rhs, iterations))
        else:
            rhs_samples.append(_time_once(rhs, iterations))
            lhs_samples.append(_time_once(lhs, iterations))

    return (
        statistics.median(lhs_samples),
        statistics.median(rhs_samples),
        lhs_samples,
        rhs_samples,
    )


def _make_kernel(input, gate, weight, output):
    hidden_size = input.shape[-1]
    return _cached_make(
        ntops.kernels.rms_norm_gated.premake,
        input.ndim,
        hidden_size,
        ntops.kernels.rms_norm_gated.ActivationVariant.SILU,
        False,
        input_dtype=input.dtype,
        gate_dtype=gate.dtype,
        weight_dtype=weight.dtype,
        output_dtype=output.dtype,
        block_size=hidden_size,
        input_shape=tuple(input.shape),
        gate_shape=tuple(gate.shape),
        weight_shape=tuple(weight.shape),
        output_shape=tuple(output.shape),
        input_strides=tuple(input.stride()),
        gate_strides=tuple(gate.stride()),
        weight_strides=tuple(weight.stride()),
        output_strides=tuple(output.stride()),
        num_warps=ntops.kernels.rms_norm_gated.NUM_WARPS,
        num_stages=ntops.kernels.rms_norm_gated.NUM_STAGES,
    )


def main():
    torch.manual_seed(0)
    eps = 1e-5

    for tokens, hidden_size in ((64, 128), (128, 512), (16, 4096)):
        input = torch.randn(
            tokens, hidden_size, device="cuda", dtype=torch.bfloat16
        )
        gate = torch.randn_like(input)
        weight = torch.empty(
            hidden_size, device="cuda", dtype=torch.bfloat16
        ).uniform_(-1, 1)
        output = torch.empty_like(input)
        kernel = _make_kernel(input, gate, weight, output)
        arguments = (input, gate, weight, eps, output)

        def checked():
            return kernel(*arguments)

        prevalidated = getattr(kernel, "_launch_prevalidated_noalias", None)
        if prevalidated is None:
            raise RuntimeError("The prevalidated no-alias launcher is unavailable.")

        def fast():
            return prevalidated(*arguments)

        checked()
        fast()
        torch.testing.assert_close(
            output,
            ntops.torch.rms_norm_gated(
                input,
                gate,
                weight,
                eps=eps,
                group_size=hidden_size,
                norm_before_gate=False,
                activation="swish",
            ),
            rtol=2e-2,
            atol=2e-2,
        )

        checked_ms, fast_ms, checked_samples, fast_samples = _measure_pair(
            checked, fast
        )
        print(
            f"shape={tokens}x{hidden_size} correctness=pass "
            f"checked_ms={checked_ms:.6f} fast_ms={fast_ms:.6f} "
            f"runtime_speedup={checked_ms / fast_ms:.4f} "
            f"checked_samples={checked_samples} fast_samples={fast_samples}",
            flush=True,
        )


if __name__ == "__main__":
    main()
