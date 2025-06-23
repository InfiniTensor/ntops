import itertools
import math
import random

import pytest
import torch
import torch.nn.functional as F

import ntops.torch
from tests.skippers import skip_if_cuda_not_available


def generate_arguments():
    def _generate_random_size():
        return random.randint(1, 512)

    arguments = []

    attn_mask_types = (None, torch.bool, torch.float32)
    scales = (None, random.uniform(0.05, 0.5))
    dtypes = (torch.float32, torch.float16)
    with_kv_cache_values = (False, True)

    for attn_mask_type, scale, dtype, with_kv_cache in itertools.product(
        attn_mask_types, scales, dtypes, with_kv_cache_values
    ):
        batch_size = random.randint(1, 4)
        num_heads_q = 2 ** random.randint(1, 5)
        seq_len_q = _generate_random_size()
        head_dim = random.choice([32, 64])
        num_heads_kv = 2 ** random.randint(1, math.floor(math.log2(num_heads_q)))
        seq_len_kv = _generate_random_size()

        enable_gqa = True

        if dtype is torch.float32:
            atol = 0.01
            rtol = 0.01
        else:
            atol = 0.025
            rtol = 0.025

        arguments.append(
            (
                batch_size,
                num_heads_q,
                seq_len_q,
                head_dim,
                num_heads_kv,
                seq_len_kv,
                attn_mask_type,
                scale,
                enable_gqa,
                with_kv_cache,
                dtype,
                atol,
                rtol,
            )
        )

    return (
        "batch_size, num_heads_q, seq_len_q, head_dim, num_heads_kv, seq_len_kv, attn_mask_type, scale, enable_gqa, with_kv_cache, dtype, atol, rtol",
        arguments,
    )


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_cuda(
    batch_size,
    num_heads_q,
    seq_len_q,
    head_dim,
    num_heads_kv,
    seq_len_kv,
    attn_mask_type,
    scale,
    enable_gqa,
    with_kv_cache,
    dtype,
    atol,
    rtol,
):
    device = "cuda"

    shape_q = (batch_size, num_heads_q, seq_len_q, head_dim)
    shape_kv = (batch_size, num_heads_kv, seq_len_kv, head_dim)

    query = torch.randn(shape_q, dtype=dtype, device=device)
    key = torch.randn(shape_kv, dtype=dtype, device=device)
    value = torch.randn(shape_kv, dtype=dtype, device=device)

    if attn_mask_type is not None:
        attn_mask = torch.rand(
            (query.shape[-2], key.shape[-2]), dtype=query.dtype, device=query.device
        )

        if attn_mask_type is torch.bool:
            attn_mask = attn_mask > 0.5
        # TODO: Non-infinite floating-point masks may cause
        # precision issues. Revisit here later.
        else:
            attn_mask = torch.where(attn_mask > 0.5, 0, float("-inf"))
            attn_mask = attn_mask.to(query.dtype)
    else:
        attn_mask = None

    key_cloned = key.clone()
    value_cloned = value.clone()

    def _generate_present_and_slot(tensor):
        present = tensor[:, :, -1:, :].clone()
        present_slot = tensor[:, :, -1:, :]
        present_slot[...] = 0

        return present, present_slot

    if with_kv_cache:
        present_key, present_key_slot = _generate_present_and_slot(key)
        present_value, present_value_slot = _generate_present_and_slot(value)
    else:
        present_key = None
        present_value = None
        present_key_slot = None
        present_value_slot = None

    ninetoothed_output = ntops.torch.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        scale=scale,
        enable_gqa=enable_gqa,
        present_key=present_key,
        present_value=present_value,
        present_key_slot=present_key_slot,
        present_value_slot=present_value_slot,
    )
    reference_output = F.scaled_dot_product_attention(
        query,
        key_cloned,
        value_cloned,
        attn_mask=attn_mask,
        scale=scale,
        enable_gqa=enable_gqa,
    )

    assert torch.allclose(ninetoothed_output, reference_output, atol=atol, rtol=rtol)
