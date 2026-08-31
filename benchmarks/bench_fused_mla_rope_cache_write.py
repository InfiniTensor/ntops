"""Benchmark MLA RoPE + compressed KV-cache write fusion."""

import torch
from _benchmark import benchmark_mean, selected_config

import ntops
from ntops.torch.utils import _cached_make

DEVICE = "cuda"
DTYPE = torch.bfloat16
NUM_HEADS = 128  # documented MLA geometry; cache writer itself is head-shared
KV_LORA_RANK = 512
ROPE_DIM = 64
CACHE_BLOCK_SIZE = 16
TILE_SIZE = 128
AUTOTUNE_WARPS = (1, 2, 4, 8)
AUTOTUNE_STAGES = (1, 2)
MAX_NUM_CONFIGS = 8


def _make_inputs(tokens):
    kv_c = torch.randn(tokens, KV_LORA_RANK, device=DEVICE, dtype=DTYPE)
    k_pe = torch.randn(tokens, ROPE_DIM, device=DEVICE, dtype=DTYPE)
    num_blocks = (tokens + CACHE_BLOCK_SIZE - 1) // CACHE_BLOCK_SIZE + 4
    kv_cache = torch.empty(
        num_blocks,
        CACHE_BLOCK_SIZE,
        KV_LORA_RANK + ROPE_DIM,
        device=DEVICE,
        dtype=DTYPE,
    )
    slot_mapping = torch.arange(tokens, device=DEVICE, dtype=torch.int64)
    positions = torch.arange(tokens, device=DEVICE, dtype=torch.int64)
    half = ROPE_DIM // 2
    theta = 10000 ** (-2 * torch.arange(half, device=DEVICE) / ROPE_DIM)
    phase = positions.to(torch.float32)[:, None] * theta[None, :]
    cos_sin_cache = torch.cat((phase.cos(), phase.sin()), dim=-1)
    return kv_c, k_pe, kv_cache, slot_mapping, positions, cos_sin_cache


def _torch_rope(k_pe, positions, cos_sin_cache):
    half = ROPE_DIM // 2
    table = cos_sin_cache.index_select(0, positions)
    cos = table[:, :half].to(torch.float32)
    sin = table[:, half:].to(torch.float32)
    x0 = k_pe[:, 0::2].to(torch.float32)
    x1 = k_pe[:, 1::2].to(torch.float32)
    output = torch.empty_like(k_pe, dtype=torch.float32)
    output[:, 0::2] = x0 * cos - x1 * sin
    output[:, 1::2] = x0 * sin + x1 * cos
    return output.to(k_pe.dtype)


def _torch_unfused(inputs):
    kv_c, k_pe, cache, slots, positions, table = inputs
    k_rot = _torch_rope(k_pe, positions, table)
    block = slots // CACHE_BLOCK_SIZE
    offset = slots % CACHE_BLOCK_SIZE
    cache[block, offset, :KV_LORA_RANK] = kv_c
    cache[block, offset, KV_LORA_RANK:] = k_rot


def _torch_rope_only(inputs):
    _, k_pe, _, slots, positions, table = inputs
    del slots
    return _torch_rope(k_pe, positions, table)


def _torch_cache_only(inputs, k_rot):
    kv_c, _, cache, slots, _, _ = inputs
    block = slots // CACHE_BLOCK_SIZE
    offset = slots % CACHE_BLOCK_SIZE
    cache[block, offset, :KV_LORA_RANK] = kv_c
    cache[block, offset, KV_LORA_RANK:] = k_rot


def _kernel_handle(inputs):
    entry_dim = KV_LORA_RANK + ROPE_DIM
    tile_size = max(
        1 << (TILE_SIZE - 1).bit_length(),
        1 << (entry_dim - 1).bit_length(),
    )
    return _cached_make(
        ntops.kernels.fused_mla_rope_cache_write.premake,
        KV_LORA_RANK,
        ROPE_DIM,
        dtype=DTYPE,
        block_size=tile_size,
        cache_block_size=CACHE_BLOCK_SIZE,
        cos_dtype=inputs[-1].dtype,
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
        "scenario,tokens,heads,latent,rope,torch_rope_mean_us,"
        "torch_cache_mean_us,torch_unfused_mean_us,best_num_warps,"
        "best_num_stages,autotuned_mean_us,"
        "speedup_vs_pytorch"
    )

    for scenario, tokens in (
        ("decode", 1),
        ("concurrent_decode", 10),
        ("long_context", 2048),
    ):
        inputs = _make_inputs(tokens)

        def rope_only():
            return _torch_rope_only(inputs)

        rotated = _torch_rope_only(inputs)

        def cache_only():
            return _torch_cache_only(inputs, rotated)

        def unfused():
            return _torch_unfused(inputs)

        def autotuned():
            return ntops.torch.fused_mla_rope_cache_write(
                *inputs,
                block_size=TILE_SIZE,
                num_warps=AUTOTUNE_WARPS,
                num_stages=AUTOTUNE_STAGES,
                max_num_configs=MAX_NUM_CONFIGS,
            )

        autotuned()
        torch.cuda.synchronize()
        warps, stages = selected_config(_kernel_handle(inputs))
        rope_us = benchmark_mean(rope_only)
        cache_us = benchmark_mean(cache_only)
        unfused_us = benchmark_mean(unfused)
        auto_us = benchmark_mean(autotuned)
        print(
            f"{scenario},{tokens},{NUM_HEADS},{KV_LORA_RANK},{ROPE_DIM},"
            f"{rope_us:.3f},{cache_us:.3f},{unfused_us:.3f},{warps},{stages},"
            f"{auto_us:.3f},{unfused_us / auto_us:.3f}"
        )


if __name__ == "__main__":
    main()
