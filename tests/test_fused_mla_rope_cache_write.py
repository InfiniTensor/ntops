import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


def _tables(max_position, rope_dim, dtype, device):
    return torch.randn(max_position, rope_dim, dtype=dtype, device=device)


def test_fused_mla_rope_cache_write_validates_contract():
    kv_c = torch.randn(2, 8)
    k_pe = torch.randn(2, 8)
    cache = torch.randn(2, 4, 16)
    slots = torch.tensor((0, 1), dtype=torch.int64)
    positions = torch.tensor((0, 1), dtype=torch.int64)
    table = _tables(4, 8, torch.float32, "cpu")

    with pytest.raises(TypeError, match="float16, bfloat16, or float32"):
        ntops.torch.fused_mla_rope_cache_write(
            kv_c.double(),
            k_pe.double(),
            cache.double(),
            slots,
            positions,
            table,
        )

    with pytest.raises(ValueError, match="positive even"):
        ntops.torch.fused_mla_rope_cache_write(
            kv_c,
            torch.randn(2, 7),
            torch.randn(2, 4, 15),
            slots,
            positions,
            torch.randn(4, 7),
        )

    with pytest.raises(TypeError, match="slot_mapping"):
        ntops.torch.fused_mla_rope_cache_write(
            kv_c,
            k_pe,
            cache,
            slots.to(torch.int32),
            positions,
            table,
        )


@skip_if_cuda_not_available
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
@pytest.mark.parametrize("kpe_rank3", (False, True))
def test_fused_mla_rope_cache_write(dtype, kpe_rank3):
    device = "cuda"
    tokens, latent, rope = 3, 8, 8
    kv_c = torch.randn(tokens, latent, device=device, dtype=dtype)
    k_pe = torch.randn(tokens, rope, device=device, dtype=dtype)
    if kpe_rank3:
        k_pe = k_pe.unsqueeze(1)
    kv_cache = torch.randn(6, 4, latent + rope, device=device, dtype=dtype)
    reference_cache = kv_cache.clone()
    slot_mapping = torch.tensor((0, -1, 7), device=device, dtype=torch.int64)
    positions = torch.tensor((0, 2, 5), device=device, dtype=torch.int64)
    cos_sin_cache = _tables(16, rope, torch.float32, device)

    result = ntops.torch.fused_mla_rope_cache_write(
        kv_c,
        k_pe,
        kv_cache,
        slot_mapping,
        positions,
        cos_sin_cache,
        block_size=8,
        num_warps=1,
        num_stages=1,
    )
    reference = ntops.torch.fused_mla_rope_cache_write_reference(
        kv_c,
        k_pe,
        reference_cache,
        slot_mapping,
        positions,
        cos_sin_cache,
    )

    assert result is None
    assert reference is None
    atol = 2e-3 if dtype != torch.bfloat16 else 2e-2
    assert torch.allclose(kv_cache, reference_cache, rtol=atol, atol=atol)


@skip_if_cuda_not_available
def test_fused_mla_rope_cache_write_supports_padding_source_rows():
    device = "cuda"
    source_tokens, latent, rope = 4, 8, 8
    kv_c = torch.randn(source_tokens, latent, device=device, dtype=torch.float16)
    k_pe = torch.randn(source_tokens, 1, rope, device=device, dtype=torch.float16)
    cache = torch.zeros(4, 4, latent + rope, device=device, dtype=torch.float16)
    reference_cache = cache.clone()
    slots = torch.tensor((0, 5), device=device, dtype=torch.int64)
    positions = torch.tensor((1, 3), device=device, dtype=torch.int64)
    table = _tables(8, rope, torch.float32, device)

    ntops.torch.fused_mla_rope_cache_write(
        kv_c,
        k_pe,
        cache,
        slots,
        positions,
        table,
        block_size=8,
        num_warps=1,
        num_stages=1,
    )
    ntops.torch.fused_mla_rope_cache_write_reference(
        kv_c,
        k_pe,
        reference_cache,
        slots,
        positions,
        table,
    )
    assert torch.allclose(cache, reference_cache, rtol=2e-3, atol=2e-3)


@skip_if_cuda_not_available
def test_fused_mla_rope_cache_write_strided_inputs():
    device = "cuda"
    tokens, latent, rope = 2, 8, 8
    kv_c = torch.randn(tokens, latent * 2, device=device, dtype=torch.float16)[:, ::2]
    k_pe = torch.randn(tokens, rope * 2, device=device, dtype=torch.float16)[:, ::2]
    cache = torch.randn(4, 4, (latent + rope) * 2, device=device, dtype=torch.float16)[
        ..., ::2
    ]
    reference_cache = cache.clone()
    slots = torch.tensor((0, 5), device=device, dtype=torch.int64)
    positions = torch.tensor((0, 1), device=device, dtype=torch.int64)
    table = _tables(16, rope, torch.float32, device)

    result = ntops.torch.fused_mla_rope_cache_write(
        kv_c,
        k_pe,
        cache,
        slots,
        positions,
        table,
        block_size=128,
        num_warps=(1, 2, 4, 8),
        num_stages=(1, 2),
        max_num_configs=8,
    )
    reference = ntops.torch.fused_mla_rope_cache_write_reference(
        kv_c,
        k_pe,
        reference_cache,
        slots,
        positions,
        table,
    )
    assert result is None
    assert reference is None
    assert torch.allclose(cache, reference_cache, rtol=2e-3, atol=2e-3)


@skip_if_cuda_not_available
def test_fused_mla_rope_cache_write_output_anchor():
    source_tokens, latent, rope = 4, 8, 8
    kv_c = torch.randn(source_tokens, latent, device="cuda", dtype=torch.float16)
    k_pe = torch.randn(source_tokens, rope, device="cuda", dtype=torch.float16)
    cache = torch.randn(4, 4, latent + rope, device="cuda", dtype=torch.float16)
    reference_cache = cache.clone()
    slots = torch.tensor((0, -1, 5), device="cuda", dtype=torch.int64)
    positions = torch.tensor((0, 1, 2), device="cuda", dtype=torch.int64)
    table = _tables(8, rope, torch.float32, "cuda")

    result = ntops.torch.fused_mla_rope_cache_write(
        kv_c,
        k_pe,
        cache,
        slots,
        positions,
        table,
        block_size=8,
        num_warps=1,
        num_stages=1,
    )
    ntops.torch.fused_mla_rope_cache_write_reference(
        kv_c,
        k_pe,
        reference_cache,
        slots,
        positions,
        table,
    )

    assert result is None
    assert torch.allclose(cache, reference_cache, rtol=2e-3, atol=2e-3)
