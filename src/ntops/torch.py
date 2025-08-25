import functools
import math
import random

import ninetoothed
import torch

import ntops.kernels.abs
import ntops.kernels.add
import ntops.kernels.addmm
import ntops.kernels.bitwise_and
import ntops.kernels.bitwise_not
import ntops.kernels.bitwise_or
import ntops.kernels.bmm
import ntops.kernels.clamp
import ntops.kernels.cos
import ntops.kernels.div
import ntops.kernels.dropout
import ntops.kernels.eq
import ntops.kernels.exp
import ntops.kernels.ge
import ntops.kernels.gelu
import ntops.kernels.gt
import ntops.kernels.isinf
import ntops.kernels.isnan
import ntops.kernels.layer_norm
import ntops.kernels.le
import ntops.kernels.lt
import ntops.kernels.mm
import ntops.kernels.mul
import ntops.kernels.ne
import ntops.kernels.neg
import ntops.kernels.pow
import ntops.kernels.relu
import ntops.kernels.rms_norm
import ntops.kernels.rotary_position_embedding
import ntops.kernels.rsqrt
import ntops.kernels.scaled_dot_product_attention
import ntops.kernels.sigmoid
import ntops.kernels.silu
import ntops.kernels.sin
import ntops.kernels.softmax
import ntops.kernels.sub
import ntops.kernels.tanh
from ntops.kernels.scaled_dot_product_attention import CausalVariant


def abs(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.abs.premake, input.ndim)

    kernel(input, out)

    return out


def add(input, other, *, alpha=1, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.add.premake, input.ndim)

    kernel(input, other, alpha, out)

    return out


def addmm(input, mat1, mat2, *, beta=1, alpha=1, out=None):
    m, _ = mat1.shape
    _, n = mat2.shape

    if out is None:
        out = torch.empty((m, n), dtype=input.dtype, device=input.device)

    kernel = _cached_make(ntops.kernels.addmm.premake)

    kernel(input, mat1, mat2, beta, alpha, out)

    return out


def bitwise_and(input, other, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.bitwise_and.premake, input.ndim)

    kernel(input, other, out)

    return out


def bitwise_not(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = _cached_make(
        ntops.kernels.bitwise_not.premake, input.ndim, input.dtype == torch.bool
    )

    kernel(input, out)

    return out


def bitwise_or(input, other, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.bitwise_or.premake, input.ndim)

    kernel(input, other, out)

    return out


def bmm(input, mat2, *, out=None):
    b, m, _ = input.shape
    _, _, n = mat2.shape

    if out is None:
        out = torch.empty((b, m, n), dtype=input.dtype, device=input.device)

    kernel = _cached_make(ntops.kernels.bmm.premake)

    kernel(input, mat2, out)

    return out


def clamp(input, min=None, max=None, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.clamp.premake, input.ndim)

    kernel(input, min, max, out)

    return out


def cos(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.cos.premake, input.ndim)

    kernel(input, out)

    return out


def div(input, other, *, rounding_mode=None, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.div.premake, input.ndim, rounding_mode)

    kernel(input, other, out)

    return out


def dropout(input, p=0.5, training=True, inplace=False):
    if not training or p == 0:
        if inplace:
            return input
        else:
            return input.clone()

    seed = random.randrange(0, 2**31)

    if inplace:
        output = input
    else:
        output = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.dropout.premake, input.ndim)

    kernel(input, p, seed, output)

    return output


def exp(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.exp.premake, input.ndim)

    kernel(input, out)

    return out


def ge(input, other, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.ge.premake, input.ndim)

    kernel(input, other, out)

    return out


def eq(input, other, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.eq.premake, input.ndim)

    kernel(input, other, out)

    return out


def gelu(input, approximate="none"):
    output = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.gelu.premake, input.ndim, approximate)

    kernel(input, output)

    return output


def gt(input, other, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.gt.premake, input.ndim)

    kernel(input, other, out)

    return out


def isinf(input):
    output = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.isinf.premake, input.ndim)

    kernel(input, output)

    return output


def isnan(input):
    output = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.isnan.premake, input.ndim)

    kernel(input, output)

    return output


def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)

    normalized_shape = tuple(normalized_shape)

    if weight is None:
        weight = torch.ones_like(input)
    else:
        weight = weight.expand_as(input)

    if bias is None:
        bias = torch.zeros_like(input)
    else:
        bias = bias.expand_as(input)

    output = torch.empty_like(input)

    kernel = _cached_make(
        ntops.kernels.layer_norm.premake, input.ndim, normalized_shape
    )

    kernel(input, weight, bias, eps, output, math.prod(normalized_shape))

    return output


def mm(input, mat2, *, out=None):
    m, _ = input.shape
    _, n = mat2.shape

    if out is None:
        out = torch.empty((m, n), dtype=input.dtype, device=input.device)

    kernel = _cached_make(ntops.kernels.mm.premake)

    kernel(input, mat2, out)

    return out


def le(input, other, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.le.premake, input.ndim)

    kernel(input, other, out)

    return out


def lt(input, other, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.lt.premake, input.ndim)

    kernel(input, other, out)

    return out


def mul(input, other, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.mul.premake, input.ndim)

    kernel(input, other, out)

    return out


def ne(input, other, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.ne.premake, input.ndim)

    kernel(input, other, out)

    return out


def neg(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.neg.premake, input.ndim)

    kernel(input, out)

    return out


def pow(input, exponent, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.pow.premake, input.ndim)

    kernel(input, exponent, out)

    return out


def relu(input, inplace=False):
    if inplace:
        output = input
    else:
        output = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.relu.premake, input.ndim)

    kernel(input, output)

    return output


def rms_norm(input, normalized_shape, weight=None, eps=None):
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)

    normalized_shape = tuple(normalized_shape)

    if weight is None:
        weight = torch.ones_like(input)
    else:
        weight = weight.expand_as(input)

    if eps is None:
        eps = torch.finfo(input.dtype).eps

    output = torch.empty_like(input)

    kernel = _cached_make(
        ntops.kernels.rms_norm.premake, input.ndim, len(normalized_shape)
    )

    kernel(input, weight, eps, output, math.prod(normalized_shape))

    return output


def rotary_position_embedding(
    input, sin_table, cos_table, interleaved=True, inplace=False
):
    if inplace:
        output = input
    else:
        output = torch.empty_like(input)

    batch_size, _, num_heads, _ = input.shape

    sin_table = sin_table[None, :, None, :].expand(batch_size, -1, num_heads, -1)
    cos_table = cos_table[None, :, None, :].expand(batch_size, -1, num_heads, -1)

    kernel = _cached_make(
        ntops.kernels.rotary_position_embedding.premake,
        input.ndim,
        interleaved=interleaved,
        num_warps=1,
    )

    kernel(input, sin_table, cos_table, output)

    return output


def rsqrt(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.rsqrt.premake, input.ndim)

    kernel(input, out)

    return out


def scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
    causal_variant=None,
    present_key=None,
    present_value=None,
    present_key_slot=None,
    present_value_slot=None,
):
    # TODO: Support `dropout_p`.
    assert dropout_p == 0, "`dropout_p` is not supported yet."

    assert attn_mask is None or not is_causal, (
        "Cannot use `attn_mask` and `is_causal` together."
    )

    num_heads_q = query.shape[-3]
    num_heads_kv = key.shape[-3]

    assert num_heads_kv == value.shape[-3], (
        "Number of heads in `key` and `value` must be the same."
    )

    if not enable_gqa:
        assert num_heads_q == num_heads_kv, (
            "Number of heads in `query`, `key`, and `value` must be the same when GQA is not enabled."
        )
    else:
        assert num_heads_q % num_heads_kv == 0, (
            "Number of heads in `query` must be divisible by number of heads in `key` and `value` when GQA is enabled."
        )

    mask_shape = query.shape[:-1] + (key.shape[-2],)

    if attn_mask is not None:
        with_attn_mask = True

        if attn_mask.dtype == torch.bool:
            attn_mask = torch.where(attn_mask, 0, float("-inf"))

        attn_mask = attn_mask.expand(mask_shape)
    else:
        with_attn_mask = False

        attn_mask = torch.empty(mask_shape, device="meta")

    if scale is None:
        scale = 1 / math.sqrt(query.shape[-1])

    if causal_variant is None:
        causal_variant = CausalVariant.UPPER_LEFT

    if present_key is not None:
        with_kv_cache = True
    else:
        with_kv_cache = False

    output = torch.empty_like(query, dtype=value.dtype)

    kernel = _cached_make(
        ntops.kernels.scaled_dot_product_attention.premake, with_kv_cache
    )

    if with_kv_cache:
        kernel(
            query,
            key,
            value,
            present_key,
            present_value,
            present_key_slot,
            present_value_slot,
            attn_mask,
            is_causal,
            scale,
            output,
            with_attn_mask,
            causal_variant,
        )
    else:
        kernel(
            query,
            key,
            value,
            attn_mask,
            is_causal,
            scale,
            output,
            with_attn_mask,
            causal_variant,
        )

    return output


def sigmoid(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.sigmoid.premake, input.ndim)

    kernel(input, out)

    return out


def silu(input, inplace=False):
    if inplace:
        output = input
    else:
        output = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.silu.premake, input.ndim)

    kernel(input, output)

    return output


def sin(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.sin.premake, input.ndim)

    kernel(input, out)

    return out


def softmax(input, dim, dtype=None):
    tensor_dtype = dtype if dtype is not None else input.dtype

    output = torch.empty_like(input, dtype=tensor_dtype)

    kernel = _cached_make(ntops.kernels.softmax.premake, input.ndim, dim)

    kernel(input, output)

    return output


def sub(input, other, *, alpha=1, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.sub.premake, input.ndim)

    kernel(input, other, alpha, out)

    return out


def tanh(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = _cached_make(ntops.kernels.tanh.premake, input.ndim)

    kernel(input, out)

    return out


@functools.cache
def _cached_make(
    premake, *args, num_warps=None, num_stages=None, max_num_configs=None, **keywords
):
    return ninetoothed.make(
        *premake(*args, **keywords),
        num_warps=num_warps,
        num_stages=num_stages,
        max_num_configs=max_num_configs,
    )
