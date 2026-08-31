"""Compare fused NineToothed MLA RoPE/cache write against the vLLM chain."""

import argparse
import statistics
import time

import torch

import ntops
import ntops.kernels.mla_rope_cache as primary_kernel
import ntops.kernels.mla_rope_cache_pair as pair_kernel
from ntops.torch.mla_rope_cache_pair import mla_rope_cache_pair


def _import_vllm_mla():
    """Import the unchanged vLLM MLA operators on supported vendor stacks."""
    # CoreX exposes its CUDA-compatible device through NVML.  Hiding that
    # device makes older vendor vLLM builds fall back to an unspecified
    # platform, so preserve the normal discovery path on IX-ML.
    if getattr(torch, "corex", False):
        from vllm import _custom_ops as ops
        from vllm.model_executor.layers.rotary_embedding import get_rope

        return ops, get_rope

    # Some BW images expose both ROCm SMI and an NVML compatibility shim.
    # vLLM then detects two built-in platforms and aborts during import.  Hide
    # only the spurious CUDA discovery while vLLM resolves its platform; the
    # timed RoPE and concat/cache operators remain the unmodified vLLM code.
    try:
        from vllm.utils import import_pynvml
    except ImportError:
        from vllm.utils.import_utils import import_pynvml

    pynvml = import_pynvml()
    original = pynvml.nvmlDeviceGetCount
    pynvml.nvmlDeviceGetCount = lambda: 0

    try:
        from vllm import _custom_ops as ops
        from vllm.model_executor.layers.rotary_embedding import get_rope

        return ops, get_rope
    finally:
        pynvml.nvmlDeviceGetCount = original


def _rotate_interleaved(value, cos_sin):
    cos, sin = cos_sin.chunk(2, dim=-1)
    cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2).float()
    sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2).float()
    even = value[..., 0::2].float()
    odd = value[..., 1::2].float()
    output = torch.empty_like(value, dtype=torch.float32)
    output[..., 0::2] = even * cos[..., 0::2] - odd * sin[..., 0::2]
    output[..., 1::2] = odd * cos[..., 1::2] + even * sin[..., 1::2]
    return output.to(value.dtype)


def _time(function, iterations):
    torch.cuda.synchronize()
    started = time.perf_counter()

    for _ in range(iterations):
        function()

    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000 / iterations


def _paired_median(first, second, warmup=100, iterations=1000, rounds=9):
    for _ in range(warmup):
        first()
        second()

    timings = ([], [])

    for round_index in range(rounds):
        order = (0, 1) if round_index % 2 == 0 else (1, 0)

        for index in order:
            function = first if index == 0 else second
            timings[index].append(_time(function, iterations))

    return tuple(statistics.median(values) for values in timings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument(
        "--implementation",
        choices=("primary", "pair"),
        default="primary",
    )
    parser.add_argument("--num-warps", type=int)
    parser.add_argument("--tokens", type=int, choices=(1, 16, 128))
    args = parser.parse_args()
    if args.num_warps is not None:
        primary_kernel.NUM_WARPS = args.num_warps
        pair_kernel.NUM_WARPS = args.num_warps
    submission_op = (
        ntops.torch.mla_rope_cache
        if args.implementation == "primary"
        else mla_rope_cache_pair
    )

    ops, get_rope = _import_vllm_mla()
    torch.manual_seed(0)
    dtype = torch.bfloat16
    rope_dim = 64
    kv_lora_rank = 512
    num_heads = args.num_heads
    max_position = 16384
    rope = get_rope(
        rope_dim,
        rope_dim,
        max_position,
        10000.0,
        is_neox_style=False,
        dtype=dtype,
    )
    rope.cos_sin_cache = rope.cos_sin_cache.to(device="cuda", dtype=dtype)

    cases = ((1, 16), (16, 16), (128, 64))
    if args.tokens is not None:
        cases = tuple(case for case in cases if case[0] == args.tokens)

    for tokens, block_size in cases:
        num_blocks = 64
        positions = torch.randperm(max_position, device="cuda")[:tokens]
        slots = torch.randperm(num_blocks * block_size, device="cuda")[:tokens]
        q_input = torch.randn(
            tokens, num_heads, rope_dim, device="cuda", dtype=dtype
        )
        k_input = torch.randn(tokens, 1, rope_dim, device="cuda", dtype=dtype)
        kv_c = torch.randn(tokens, kv_lora_rank, device="cuda", dtype=dtype)
        cache_shape = (num_blocks, block_size, kv_lora_rank + rope_dim)
        scale = torch.ones(1, device="cuda", dtype=torch.float32)

        cos_sin = rope.cos_sin_cache.index_select(0, positions)
        expected_q = _rotate_interleaved(q_input, cos_sin)
        expected_k = _rotate_interleaved(k_input, cos_sin).squeeze(1)
        expected_cache = torch.zeros(cache_shape, device="cuda", dtype=dtype)
        combined = torch.cat((kv_c, expected_k), dim=-1)

        for token, slot in enumerate(slots.tolist()):
            expected_cache[slot // block_size, slot % block_size] = combined[token]

        q_vllm = q_input.clone()
        k_vllm = k_input.clone()
        # IX-ML's vLLM ABI retains the singleton KV-head dimension, while the
        # ROCm vLLM ABI accepts the equivalent flattened PE row.
        k_vllm_cache = (
            k_vllm if getattr(torch, "corex", False) else k_vllm.squeeze(1)
        )
        cache_vllm = torch.zeros(cache_shape, device="cuda", dtype=dtype)
        rope.forward_cuda(positions, q_vllm, k_vllm)
        ops.concat_and_cache_mla(
            kv_c, k_vllm_cache, cache_vllm, slots, "auto", scale
        )
        torch.testing.assert_close(q_vllm, expected_q, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(cache_vllm, expected_cache, rtol=2e-2, atol=2e-2)

        q_nt = q_input.clone()
        cache_nt = torch.zeros(cache_shape, device="cuda", dtype=dtype)
        submission_op(
            q_nt,
            k_input,
            kv_c,
            positions,
            rope.cos_sin_cache,
            slots,
            cache_nt,
        )
        torch.testing.assert_close(q_nt, expected_q, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(cache_nt, expected_cache, rtol=2e-2, atol=2e-2)

        def vllm_baseline():
            rope.forward_cuda(positions, q_vllm, k_vllm)
            ops.concat_and_cache_mla(
                kv_c, k_vllm_cache, cache_vllm, slots, "auto", scale
            )

        def submission():
            submission_op(
                q_nt,
                k_input,
                kv_c,
                positions,
                rope.cos_sin_cache,
                slots,
                cache_nt,
            )

        vllm_ms, submission_ms = _paired_median(vllm_baseline, submission)
        print(
            f"shape=seq{tokens}_heads{num_heads}_block{block_size} "
            f"correctness=pass vllm_ms={vllm_ms:.6f} "
            f"submission_ms={submission_ms:.6f} "
            f"formal_speedup={vllm_ms / submission_ms:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
