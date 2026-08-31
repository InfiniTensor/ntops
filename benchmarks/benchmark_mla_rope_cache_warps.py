"""Sweep Triton warp counts for the fused MLA RoPE/cache kernel."""

import argparse
import importlib
import statistics
import time

import torch

import ntops
from ntops.torch import utils

kernel_module = importlib.import_module("ntops.kernels.mla_rope_cache")
wrapper_module = importlib.import_module("ntops.torch.mla_rope_cache")


def _median(function, *, warmup=100, iterations=1000, rounds=5):
    for _ in range(warmup):
        function()

    timings = []

    for _ in range(rounds):
        torch.cuda.synchronize()
        started = time.perf_counter()

        for _ in range(iterations):
            function()

        torch.cuda.synchronize()
        timings.append((time.perf_counter() - started) * 1000 / iterations)

    return statistics.median(timings)


def _reset_wrapper():
    utils._cached_make.cache_clear()
    wrapper_module._LAST_VALIDATED = None
    wrapper_module._LAST_VIEWS = None
    wrapper_module._LAST_KERNEL = None
    wrapper_module._LAST_CALL = None


def _inputs(tokens, block_size, num_heads):
    dtype = torch.bfloat16
    rope_dim = 64
    kv_lora_rank = 512
    max_position = 16384
    num_blocks = 64
    frequencies = 1.0 / (
        10000.0
        ** (torch.arange(0, rope_dim, 2, device="cuda").float() / rope_dim)
    )
    angles = torch.outer(
        torch.arange(max_position, device="cuda").float(), frequencies
    )
    cos_sin_cache = torch.cat((angles.cos(), angles.sin()), dim=-1).to(dtype)
    positions = torch.randperm(max_position, device="cuda")[:tokens]
    slots = torch.randperm(num_blocks * block_size, device="cuda")[:tokens]
    q = torch.randn(tokens, num_heads, rope_dim, device="cuda", dtype=dtype)
    k = torch.randn(tokens, 1, rope_dim, device="cuda", dtype=dtype)
    kv_c = torch.randn(tokens, kv_lora_rank, device="cuda", dtype=dtype)
    cache = torch.zeros(
        num_blocks,
        block_size,
        kv_lora_rank + rope_dim,
        device="cuda",
        dtype=dtype,
    )
    return q, k, kv_c, positions, cos_sin_cache, slots, cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-heads", type=int, default=32)
    args = parser.parse_args()

    torch.manual_seed(0)

    for warps in (1, 2, 4, 8):
        kernel_module.NUM_WARPS = warps

        for tokens, block_size in ((1, 16), (16, 16), (128, 64)):
            _reset_wrapper()
            inputs = _inputs(tokens, block_size, args.num_heads)

            def submission():
                ntops.torch.mla_rope_cache(*inputs)

            latency_ms = _median(submission)
            print(
                f"warps={warps} seq={tokens} heads={args.num_heads} "
                f"latency_ms={latency_ms:.6f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
