import torch

import ntops.kernels.abs
import ntops.kernels.add
import ntops.kernels.addmm
import ntops.kernels.bmm
import ntops.kernels.cos
import ntops.kernels.div
import ntops.kernels.eq
import ntops.kernels.exp
import ntops.kernels.gelu
import ntops.kernels.isinf
import ntops.kernels.mm
import ntops.kernels.mul
import ntops.kernels.ne
import ntops.kernels.relu
import ntops.kernels.rsqrt
import ntops.kernels.sigmoid
import ntops.kernels.sin
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


def bmm(input, mat2, *, out=None):
    b, m, _ = input.shape
    _, _, n = mat2.shape

    if out is None:
        out = torch.empty((b, m, n), dtype=input.dtype, device=input.device)

    kernel = ntops.kernels.bmm.make()

    kernel(input, mat2, out)

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


def isinf(input):
    output = torch.empty_like(input)

    kernel = ntops.kernels.isinf.make(input.ndim)

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


def sin(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = ntops.kernels.sin.make(input.ndim)

    kernel(input, out)

    return out


def tanh(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    kernel = ntops.kernels.tanh.make(input.ndim)

    kernel(input, out)

    return out
