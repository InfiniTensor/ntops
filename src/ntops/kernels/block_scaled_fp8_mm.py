import functools

import ninetoothed.language as ntl
import torch
from ninetoothed import Tensor

BLOCK_SIZE_M = 16
BLOCK_SIZE_N = 16
BLOCK_SIZE_K = 32


def _arrange_mat_a(mat_a, output_arranged, block_size_m, block_size_k):
    arranged = mat_a.tile((1, block_size_m, block_size_k))
    arranged = arranged.tile((1, 1, -1))
    arranged = arranged.expand((-1, -1, output_arranged.shape[-1]))
    arranged.dtype = arranged.dtype.squeeze((0, 1))
    arranged.dtype.dtype = arranged.dtype.dtype.squeeze(0)
    return arranged


def _arrange_mat_b(mat_b, output_arranged, block_size_n, block_size_k):
    arranged = mat_b.tile((1, block_size_k, block_size_n))
    arranged = arranged.tile((1, -1, 1))
    arranged = arranged.expand((-1, output_arranged.shape[-2], -1))
    arranged.dtype = arranged.dtype.squeeze((0, 2))
    arranged.dtype.dtype = arranged.dtype.dtype.squeeze(0)
    return arranged


def _arrange_scale_a(scale_a, output_arranged, block_size_m):
    arranged = scale_a.tile((1, block_size_m, 1))
    arranged = arranged.tile((1, 1, -1))
    arranged = arranged.expand((-1, -1, output_arranged.shape[-1]))
    arranged.dtype = arranged.dtype.squeeze((0, 1))
    arranged.dtype.dtype = arranged.dtype.dtype.squeeze(0)
    return arranged


def _arrange_scale_b(scale_b, output_arranged, block_size_n):
    arranged = scale_b.tile((1, 1, block_size_n))
    arranged = arranged.tile((1, -1, 1))
    arranged = arranged.expand((-1, output_arranged.shape[-2], -1))
    arranged.dtype = arranged.dtype.squeeze((0, 2))
    arranged.dtype.dtype = arranged.dtype.dtype.squeeze(0)
    return arranged


def arrangement(
    mat_a,
    mat_b,
    scale_a,
    scale_b,
    output,
    block_size_m=BLOCK_SIZE_M,
    block_size_n=BLOCK_SIZE_N,
    block_size_k=BLOCK_SIZE_K,
):
    output_arranged = output.tile((1, block_size_m, block_size_n))
    output_arranged.dtype = output_arranged.dtype.squeeze(0)

    return (
        _arrange_mat_a(mat_a, output_arranged, block_size_m, block_size_k),
        _arrange_mat_b(mat_b, output_arranged, block_size_n, block_size_k),
        _arrange_scale_a(scale_a, output_arranged, block_size_m),
        _arrange_scale_b(scale_b, output_arranged, block_size_n),
        output_arranged,
    )


def arrangement_with_bias(
    mat_a,
    mat_b,
    scale_a,
    scale_b,
    bias,
    output,
    block_size_m=BLOCK_SIZE_M,
    block_size_n=BLOCK_SIZE_N,
    block_size_k=BLOCK_SIZE_K,
):
    arranged = arrangement(
        mat_a,
        mat_b,
        scale_a,
        scale_b,
        output,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
        block_size_k=block_size_k,
    )
    output_arranged = arranged[-1]
    bias_arranged = bias.tile((1, 1, block_size_n))
    bias_arranged = bias_arranged.expand((-1, output_arranged.shape[-2], -1))
    bias_arranged.dtype = bias_arranged.dtype.squeeze((0, 1))
    return (*arranged[:-1], bias_arranged, output_arranged)


def _scaled_dot(mat_a, mat_b, scale_a, scale_b, output, dots_per_scale):
    accumulator = ntl.zeros(output.shape, dtype=ntl.float32)

    for scale_index in range(scale_a.shape[0]):
        block_accumulator = ntl.zeros(output.shape, dtype=ntl.float32)
        for k_offset in range(dots_per_scale):
            k = scale_index * dots_per_scale + k_offset
            block_accumulator += ntl.dot(mat_a[k], mat_b[k])
        scale = (scale_a[scale_index] + 0).to(ntl.float32) * (
            scale_b[scale_index] + 0
        ).to(ntl.float32)
        accumulator += block_accumulator * scale

    return accumulator


def _scaled_dot_bf16(mat_a, mat_b, scale_a, scale_b, output, dots_per_scale):
    accumulator = ntl.zeros(output.shape, dtype=ntl.float32)

    for scale_index in range(scale_a.shape[0]):
        block_accumulator = ntl.zeros(output.shape, dtype=ntl.float32)
        for k_offset in range(dots_per_scale):
            k = scale_index * dots_per_scale + k_offset
            activation = mat_a[k].to(ntl.bfloat16)
            weight = mat_b[k].to(ntl.bfloat16)
            block_accumulator += ntl.dot(activation, weight)
        scale = (scale_a[scale_index] + 0).to(ntl.float32) * (
            scale_b[scale_index] + 0
        ).to(ntl.float32)
        accumulator += block_accumulator * scale

    return accumulator


def block_scaled_fp8_mm_k16(mat_a, mat_b, scale_a, scale_b, output):
    output = _scaled_dot_bf16(  # noqa: F841
        mat_a, mat_b, scale_a, scale_b, output, 8
    )


def block_scaled_fp8_mm_k32(mat_a, mat_b, scale_a, scale_b, output):
    output = _scaled_dot(mat_a, mat_b, scale_a, scale_b, output, 4)  # noqa: F841


def block_scaled_fp8_mm_with_bias_k16(mat_a, mat_b, scale_a, scale_b, bias, output):
    accumulator = _scaled_dot_bf16(mat_a, mat_b, scale_a, scale_b, output, 8)
    output = accumulator + (bias + 0).to(ntl.float32)  # noqa: F841


def block_scaled_fp8_mm_with_bias_k32(mat_a, mat_b, scale_a, scale_b, bias, output):
    accumulator = _scaled_dot(mat_a, mat_b, scale_a, scale_b, output, 4)
    output = accumulator + (bias + 0).to(ntl.float32)  # noqa: F841


def premake(
    input_dtype,
    output_dtype,
    bias_dtype=None,
    block_size_m=BLOCK_SIZE_M,
    block_size_n=BLOCK_SIZE_N,
    block_size_k=BLOCK_SIZE_K,
):
    if block_size_k not in (16, 32):
        raise ValueError("block_size_k must be 16 or 32")

    has_bias = bias_dtype is not None
    arrangement_function = arrangement_with_bias if has_bias else arrangement
    arrangement_ = functools.partial(
        arrangement_function,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
        block_size_k=block_size_k,
    )
    tensors = (
        Tensor(3, dtype=input_dtype, other=0.0),
        Tensor(3, dtype=input_dtype, other=0.0),
        Tensor(3, dtype=torch.float32, other=0.0),
        Tensor(3, dtype=torch.float32, other=0.0),
    )

    if has_bias:
        tensors += (Tensor(3, dtype=bias_dtype, other=0.0),)

    tensors += (Tensor(3, dtype=output_dtype),)
    applications = {
        (16, False): block_scaled_fp8_mm_k16,
        (16, True): block_scaled_fp8_mm_with_bias_k16,
        (32, False): block_scaled_fp8_mm_k32,
        (32, True): block_scaled_fp8_mm_with_bias_k32,
    }
    application_function = applications[(block_size_k, has_bias)]
    return arrangement_, application_function, tensors
