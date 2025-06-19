import math

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
import ntops.kernels.eq
import ntops.kernels.exp
import ntops.kernels.ge
import ntops.kernels.gelu
import ntops.kernels.gt
import ntops.kernels.isinf
import ntops.kernels.isnan
import ntops.kernels.le
import ntops.kernels.lt
import ntops.kernels.mm
import ntops.kernels.mul
import ntops.kernels.ne
import ntops.kernels.neg
import ntops.kernels.relu
import ntops.kernels.rsqrt
import ntops.kernels.scaled_dot_product_attention
import ntops.kernels.sigmoid
import ntops.kernels.sin
import ntops.kernels.softmax
import ntops.kernels.tanh


def abs(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = ntops.kernels.abs.make(input.ndim)

    kernel(input, out)

    return out


def add(input, other, *, alpha=1, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = ntops.kernels.add.make(input.ndim)

    kernel(input, other, alpha, out)

    return out


def addmm(input, mat1, mat2, *, beta=1, alpha=1, out=None):
    m, _ = mat1.shape
    _, n = mat2.shape

    if out is None:
        out = torch.empty((m, n), dtype=input.dtype, device=input.device)

    kernel = ntops.kernels.addmm.make()

    kernel(input, mat1, mat2, beta, alpha, out)

    return out


def bitwise_and(input, other, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = ntops.kernels.bitwise_and.make(input.ndim)

    kernel(input, other, out)

    return out


def bitwise_not(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = ntops.kernels.bitwise_not.make(input.ndim, input.dtype == torch.bool)

    kernel(input, out)

    return out


def bitwise_or(input, other, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = ntops.kernels.bitwise_or.make(input.ndim)

    kernel(input, other, out)

    return out


def bmm(input, mat2, *, out=None):
    b, m, _ = input.shape
    _, _, n = mat2.shape

    if out is None:
        out = torch.empty((b, m, n), dtype=input.dtype, device=input.device)

    kernel = ntops.kernels.bmm.make()

    kernel(input, mat2, out)

    return out


def clamp(input, min=None, max=None, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = ntops.kernels.clamp.make(input.ndim)

    kernel(input, min, max, out)

    return out


def cos(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = ntops.kernels.cos.make(input.ndim)

    kernel(input, out)

    return out


def div(input, other, *, rounding_mode=None, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = ntops.kernels.div.make(input.ndim, rounding_mode)

    kernel(input, other, out)

    return out


def exp(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = ntops.kernels.exp.make(input.ndim)

    kernel(input, out)

    return out


def ge(input, other, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = ntops.kernels.ge.make(input.ndim)

    kernel(input, other, out)

    return out


def eq(input, other, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = ntops.kernels.eq.make(input.ndim)

    kernel(input, other, out)

    return out


def gelu(input, approximate="none"):
    output = torch.empty_like(input)

    kernel = ntops.kernels.gelu.make(input.ndim, approximate)

    kernel(input, output)

    return output


def gt(input, other, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = ntops.kernels.gt.make(input.ndim)

    kernel(input, other, out)

    return out


def isinf(input):
    output = torch.empty_like(input)

    kernel = ntops.kernels.isinf.make(input.ndim)

    kernel(input, output)

    return output


def isnan(input):
    output = torch.empty_like(input)

    kernel = ntops.kernels.isnan.make(input.ndim)

    kernel(input, output)

    return output


def mm(input, mat2, *, out=None):
    m, _ = input.shape
    _, n = mat2.shape

    if out is None:
        out = torch.empty((m, n), dtype=input.dtype, device=input.device)

    kernel = ntops.kernels.mm.make()

    kernel(input, mat2, out)

    return out


def le(input, other, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = ntops.kernels.le.make(input.ndim)

    kernel(input, other, out)

    return out


def lt(input, other, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = ntops.kernels.lt.make(input.ndim)

    kernel(input, other, out)

    return out


def mul(input, other, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = ntops.kernels.mul.make(input.ndim)

    kernel(input, other, out)

    return out


def ne(input, other, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = ntops.kernels.ne.make(input.ndim)

    kernel(input, other, out)

    return out


def neg(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = ntops.kernels.neg.make(input.ndim)

    kernel(input, out)

    return out


def relu(input, inplace=False):
    if inplace:
        output = input
    else:
        output = torch.empty_like(input)

    kernel = ntops.kernels.relu.make(input.ndim)

    kernel(input, output)

    return output


def rsqrt(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = ntops.kernels.rsqrt.make(input.ndim)

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
    # The default value here differs from that of
    # `torch.nn.functional.scaled_dot_product_attention`
    # because GQA cannot be disabled at the moment.
    enable_gqa=True,
    present_key=None,
    present_value=None,
    present_key_slot=None,
    present_value_slot=None,
):
    # TODO: Support `dropout_p`.
    assert dropout_p == 0, "`dropout_p` is not supported yet."
    # TODO: Support `is_causal`.
    assert not is_causal, "`is_causal` is not supported yet."
    assert enable_gqa, "GQA must be enabled for now."

    mask_shape = query.shape[:-1] + (key.shape[-2],)

    if attn_mask is None:
        attn_mask = torch.zeros(mask_shape, dtype=query.dtype, device=query.device)
    elif attn_mask.dtype == torch.bool:
        attn_mask = torch.where(attn_mask, 0, float("-inf"))

    attn_mask = attn_mask.expand(mask_shape)

    if scale is None:
        scale = 1 / math.sqrt(query.shape[-1])

    if present_key is not None:
        with_kv_cache = True
    else:
        with_kv_cache = False

    output = torch.empty_like(query, dtype=value.dtype)

    kernel = ntops.kernels.scaled_dot_product_attention.make(with_kv_cache)

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
            scale,
            output,
        )
    else:
        kernel(query, key, value, attn_mask, scale, output)

    return output


def sigmoid(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = ntops.kernels.sigmoid.make(input.ndim)

    kernel(input, out)

    return out


def sin(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = ntops.kernels.sin.make(input.ndim)

    kernel(input, out)

    return out


def softmax(input, dim, dtype=None):
    tensor_dtype = dtype if dtype is not None else input.dtype

    output = torch.empty_like(input, dtype=tensor_dtype)

    kernel = ntops.kernels.softmax.make(input.ndim, dim)

    kernel(input, output)

    return output


def tanh(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = ntops.kernels.tanh.make(input.ndim)

    kernel(input, out)

    return out
