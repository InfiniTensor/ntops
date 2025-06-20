import random

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
import ntops.kernels.le
import ntops.kernels.lt
import ntops.kernels.mm
import ntops.kernels.mul
import ntops.kernels.ne
import ntops.kernels.neg
import ntops.kernels.pow
import ntops.kernels.relu
import ntops.kernels.rsqrt
import ntops.kernels.sigmoid
import ntops.kernels.silu
import ntops.kernels.sin
import ntops.kernels.softmax
import ntops.kernels.sub
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

    kernel = ntops.kernels.dropout.make(input.ndim)

    kernel(input, p, seed, output)

    return output


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


def pow(input, exponent, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = ntops.kernels.pow.make(input.ndim)

    kernel(input, exponent, out)

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


def sigmoid(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = ntops.kernels.sigmoid.make(input.ndim)

    kernel(input, out)

    return out


def silu(input, inplace=False):
    if inplace:
        output = input
    else:
        output = torch.empty_like(input)

    kernel = ntops.kernels.silu.make(input.ndim)

    kernel(input, output)

    return output


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


def sub(input, other, *, alpha=1, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = ntops.kernels.sub.make(input.ndim)

    kernel(input, other, alpha, out)

    return out


def tanh(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = ntops.kernels.tanh.make(input.ndim)

    kernel(input, out)

    return out
