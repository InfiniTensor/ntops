import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor

import ntops.kernels.mm as mm


def arrangement(
    input,
    weight,
    bias,
    output,
    stride,
    dilation,
    block_size_m=None,
    block_size_n=None,
    block_size_k=None,
):
    if block_size_m is None:
        block_size_m = mm.BLOCK_SIZE_M

    if block_size_n is None:
        block_size_n = mm.BLOCK_SIZE_N

    if block_size_k is None:
        block_size_k = mm.BLOCK_SIZE_K

    mm_arrangement = functools.partial(
        mm.arrangement,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
        block_size_k=block_size_k,
    )

    input_arranged = input.tile(
        (1, *weight.shape[1:]),
        strides=(-1, -1, *stride),
        dilation=(1, 1, *dilation),
        floor_mode=True,
    )
    input_arranged = input_arranged.squeeze(1)
    input_arranged.dtype = input_arranged.dtype.squeeze(0)
    input_arranged = input_arranged.ravel()
    input_arranged = input_arranged.flatten(end_dim=3).flatten(start_dim=1)

    weight_arranged = weight.flatten(start_dim=1)
    weight_arranged = weight_arranged.permute((1, 0))

    bias_arranged = bias.permute((0, 2, 3, 1)).flatten(end_dim=3)

    _, _, bias_arranged = mm_arrangement(input_arranged, weight_arranged, bias_arranged)

    output_arranged = output.permute((0, 2, 3, 1)).flatten(end_dim=3)

    input_arranged, weight_arranged, output_arranged = mm_arrangement(
        input_arranged, weight_arranged, output_arranged
    )

    return input_arranged, weight_arranged, bias_arranged, output_arranged


def application(input, weight, bias, output):
    mm_output = ntl.zeros(output.shape, dtype=ntl.float32)
    mm.application(input, weight, mm_output)
    output = mm_output + bias


def premake(
    stride,
    dilation,
    dtype=None,
    block_size_m=None,
    block_size_n=None,
    block_size_k=None,
):
    arrangement_ = functools.partial(
        arrangement,
        stride=stride,
        dilation=dilation,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
        block_size_k=block_size_k,
    )

    tensors = tuple(Tensor(4, dtype=dtype) for _ in range(4))

    return arrangement_, application, tensors
