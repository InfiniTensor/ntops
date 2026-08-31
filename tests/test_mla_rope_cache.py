import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


def _rotate(value, cos_sin):
    cos, sin = cos_sin.chunk(2, dim=-1)
    cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2).float()
    sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2).float()
    even = value[..., 0::2].float()
    odd = value[..., 1::2].float()
    output = torch.empty_like(value, dtype=torch.float32)
    output[..., 0::2] = even * cos[..., 0::2] - odd * sin[..., 0::2]
    output[..., 1::2] = odd * cos[..., 1::2] + even * sin[..., 1::2]
    return output.to(value.dtype)


@skip_if_cuda_not_available
@pytest.mark.parametrize(
    "dtype",
    (
        pytest.param(torch.bfloat16, id="bf16"),
        pytest.param(torch.float16, id="fp16"),
    ),
)
@pytest.mark.parametrize(
    "index_dtype",
    (
        pytest.param(torch.int64, id="int64"),
        pytest.param(torch.int32, id="int32"),
    ),
)
@pytest.mark.parametrize(
    "tokens,heads,block_size",
    (
        (1, 32, 16),
        (16, 32, 16),
        (128, 32, 64),
        (1, 128, 16),
        (16, 128, 16),
        (128, 128, 64),
    ),
)
def test_mla_rope_cache(tokens, heads, block_size, dtype, index_dtype):
    torch.manual_seed(0)
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
    positions = torch.randperm(
        max_position, device="cuda", dtype=index_dtype
    )[:tokens]
    slots = torch.randperm(
        num_blocks * block_size, device="cuda", dtype=index_dtype
    )[:tokens]
    q = torch.randn(tokens, heads, rope_dim, device="cuda", dtype=dtype)
    k = torch.randn(tokens, 1, rope_dim, device="cuda", dtype=dtype)
    kv_c = torch.randn(tokens, kv_lora_rank, device="cuda", dtype=dtype)
    cache = torch.zeros(
        num_blocks,
        block_size,
        kv_lora_rank + rope_dim,
        device="cuda",
        dtype=dtype,
    )

    cos_sin = cos_sin_cache.index_select(0, positions)
    expected_q = _rotate(q, cos_sin)
    expected_k = _rotate(k, cos_sin).squeeze(1)
    expected_cache = cache.clone()
    combined = torch.cat((kv_c, expected_k), dim=-1)
    for token, slot in enumerate(slots.tolist()):
        expected_cache[slot // block_size, slot % block_size] = combined[token]

    result = ntops.torch.mla_rope_cache(
        q, k, kv_c, positions, cos_sin_cache, slots, cache
    )
    assert result is q
    torch.testing.assert_close(q, expected_q, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(cache, expected_cache, rtol=2e-2, atol=2e-2)

    expected_q = _rotate(expected_q, cos_sin)
    result = ntops.torch.mla_rope_cache(
        q, k, kv_c, positions, cos_sin_cache, slots, cache
    )

    assert result is q
    torch.testing.assert_close(q, expected_q, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(cache, expected_cache, rtol=2e-2, atol=2e-2)

    # The third invocation exercises graph replay on backends which support
    # it.  Updating position and slot tensors in place verifies that replay
    # consumes current runtime data instead of embedding case-specific values.
    new_positions = (positions + 1) % max_position
    new_slots = (slots + block_size) % (num_blocks * block_size)
    positions.copy_(new_positions)
    slots.copy_(new_slots)
    new_cos_sin = cos_sin_cache.index_select(0, new_positions.long())
    expected_q = _rotate(expected_q, new_cos_sin)
    replay_expected_k = _rotate(k, new_cos_sin).squeeze(1)
    replay_combined = torch.cat((kv_c, replay_expected_k), dim=-1)

    for token, slot in enumerate(new_slots.tolist()):
        expected_cache[slot // block_size, slot % block_size] = replay_combined[token]

    result = ntops.torch.mla_rope_cache(
        q, k, kv_c, positions, cos_sin_cache, slots, cache
    )

    assert result is q
    torch.testing.assert_close(q, expected_q, rtol=4e-2, atol=4e-2)
    torch.testing.assert_close(cache, expected_cache, rtol=2e-2, atol=2e-2)


@skip_if_cuda_not_available
@pytest.mark.parametrize(
    "dtype",
    (
        pytest.param(torch.bfloat16, id="bf16"),
        pytest.param(torch.float16, id="fp16"),
    ),
)
def test_mla_rope_cache_ignores_padding_slot(dtype):
    torch.manual_seed(0)
    tokens = 2
    heads = 32
    rope_dim = 64
    kv_lora_rank = 512
    max_position = 128
    block_size = 16
    num_blocks = 2
    frequencies = 1.0 / (
        10000.0
        ** (torch.arange(0, rope_dim, 2, device="cuda").float() / rope_dim)
    )
    angles = torch.outer(
        torch.arange(max_position, device="cuda").float(), frequencies
    )
    cos_sin_cache = torch.cat((angles.cos(), angles.sin()), dim=-1).to(dtype)
    positions = torch.tensor((3, 7), device="cuda", dtype=torch.int32)
    slots = torch.tensor((0, -1), device="cuda", dtype=torch.int32)
    q = torch.randn(tokens, heads, rope_dim, device="cuda", dtype=dtype)
    k = torch.randn(tokens, 1, rope_dim, device="cuda", dtype=dtype)
    kv_c = torch.randn(tokens, kv_lora_rank, device="cuda", dtype=dtype)
    cache = torch.zeros(
        num_blocks,
        block_size,
        kv_lora_rank + rope_dim,
        device="cuda",
        dtype=dtype,
    )

    cos_sin = cos_sin_cache.index_select(0, positions.long())
    expected_q = _rotate(q, cos_sin)
    expected_k = _rotate(k, cos_sin).squeeze(1)
    expected_cache = cache.clone()
    expected_cache[0, 0] = torch.cat((kv_c[0], expected_k[0]), dim=-1)

    ntops.torch.mla_rope_cache(
        q, k, kv_c, positions, cos_sin_cache, slots, cache
    )

    torch.testing.assert_close(q, expected_q, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(cache, expected_cache, rtol=2e-2, atol=2e-2)
