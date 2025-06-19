import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

BLOCK_SIZE_M = ninetoothed.block_size()
BLOCK_SIZE_N = ninetoothed.block_size()


def arrangement(
    query,
    key,
    value,
    present_key,
    present_value,
    present_key_slot,
    present_value_slot,
    attn_mask,
    scale,
    output,
    with_kv_cache,
    BLOCK_SIZE_M=BLOCK_SIZE_M,
    BLOCK_SIZE_N=BLOCK_SIZE_N,
):
    def arrange_query_or_output(input):
        arranged = input.tile((1, 1, BLOCK_SIZE_M, -1)).tile(
            (1, query.shape[-3] // key.shape[-3], 1, 1)
        )
        arranged.dtype = arranged.dtype.squeeze((0, 2, 3))
        arranged.dtype.dtype = arranged.dtype.dtype.squeeze((0, 1))

        return arranged

    def arrange_key_or_value(input):
        arranged = (
            input.tile((1, 1, BLOCK_SIZE_N, -1))
            .tile((1, 1, -1, -1))
            .expand((-1, -1, query_arranged.shape[-2], -1))
        )
        arranged.dtype = arranged.dtype.squeeze((0, 1, 3))
        arranged.dtype.dtype = arranged.dtype.dtype.squeeze((0, 1))

        return arranged

    def arrange_present_key_or_present_value(input):
        arranged = input.tile((1, 1, -1, -1))
        arranged.dtype = arranged.dtype.squeeze((0, 1))

        return arranged

    def arrange_attn_mask(input):
        arranged = input.tile((1, 1, BLOCK_SIZE_M, BLOCK_SIZE_N)).tile((1, 1, 1, -1))
        arranged.dtype = arranged.dtype.squeeze((0, 1, 2))
        arranged.dtype.dtype = arranged.dtype.dtype.squeeze((0, 1))

        return arranged

    query_arranged = arrange_query_or_output(query)
    key_arranged = arrange_key_or_value(key)
    value_arranged = arrange_key_or_value(value)
    present_key_arranged = arrange_present_key_or_present_value(present_key)
    present_value_arranged = arrange_present_key_or_present_value(present_value)
    present_key_slot_arranged = arrange_present_key_or_present_value(present_key_slot)
    present_value_slot_arranged = arrange_present_key_or_present_value(
        present_value_slot
    )
    attn_mask_arranged = arrange_attn_mask(attn_mask)
    scale_arranged = scale
    output_arranged = arrange_query_or_output(output)

    if with_kv_cache:
        return (
            query_arranged,
            key_arranged,
            value_arranged,
            present_key_arranged,
            present_value_arranged,
            present_key_slot_arranged,
            present_value_slot_arranged,
            attn_mask_arranged,
            scale_arranged,
            output_arranged,
        )

    return (
        query_arranged,
        key_arranged,
        value_arranged,
        attn_mask_arranged,
        scale_arranged,
        output_arranged,
    )


def application_with_kv_cache(
    query,
    key,
    value,
    present_key,
    present_value,
    present_key_slot,
    present_value_slot,
    attn_mask,
    scale,
    output,
):
    present_key_slot = present_key  # noqa: F841
    present_value_slot = present_value  # noqa: F841

    application_without_kv_cache(query, key, value, attn_mask, scale, output)


def application_without_kv_cache(query, key, value, attn_mask, scale, output):
    for i in range(query.shape[0]):
        query_i = (1.4426950408889634 * scale * query[i]).to(query[i].dtype)

        acc = ntl.zeros((query_i.shape[-2], query_i.shape[-1]), dtype=ntl.float32)
        lse = ntl.full((query_i.shape[-2],), 1, dtype=ntl.float32)
        max = ntl.full((query_i.shape[-2],), float("-inf"), dtype=ntl.float32)

        for j in range(key.shape[0]):
            qk = ntl.dot(query_i, ntl.trans(key[j])) + attn_mask[j]
            qk = ntl.where(key[j].offsets(-2) < key.source.shape[-2], qk, float("-inf"))

            next_max = ntl.maximum(max, ntl.max(qk, 1))
            stable_qk = ntl.exp2(qk - next_max[:, None])

            alpha = ntl.exp2(max - next_max)
            acc = acc * alpha[:, None] + ntl.dot(stable_qk.to(value[i].dtype), value[j])
            max = next_max
            lse = lse * alpha + ntl.sum(stable_qk, 1)

        acc /= lse[:, None]
        output[i] = acc  # noqa: F841


@functools.cache
def make(with_kv_cache):
    query, key, value, attn_mask, output = (
        Tensor(
            4, shape_options=(None, None, None, {"constexpr": True, "upper_bound": 128})
        )
        for _ in range(5)
    )
    present_key, present_value, present_key_slot, present_value_slot = (
        Tensor(
            4,
            shape_options=(
                None,
                None,
                {"constexpr": True, "upper_bound": 1},
                {"constexpr": True, "upper_bound": 128},
            ),
        )
        for _ in range(4)
    )
    scale = Tensor(0)

    if with_kv_cache:
        application = application_with_kv_cache
    else:
        application = application_without_kv_cache

    tensors = (
        query,
        key,
        value,
        present_key,
        present_value,
        present_key_slot,
        present_value_slot,
        attn_mask,
        scale,
        output,
    )

    return ninetoothed.make(
        functools.partial(arrangement, with_kv_cache=with_kv_cache),
        application,
        tensors,
    )
