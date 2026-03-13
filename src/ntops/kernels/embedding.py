import functools
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor
from ninetoothed.language import libdevice

BLOCK_SIZE = ninetoothed.block_size()


def arrangement(
    input,
    weight,
    output,
    max_norm,
    norm_type,
    block_size_m=None,
    block_size_n=None,
    embedding_dim=None,
):
    if block_size_m is None:
        block_size_m = BLOCK_SIZE
    if block_size_n is None:
        block_size_n = BLOCK_SIZE

    output = output.flatten(start_dim=0, end_dim=-1)
    input = input.flatten()

    output_arranged = output.tile((1, embedding_dim))
    output_arranged.dtype = output_arranged.dtype.squeeze(0)
    output_arranged = output_arranged.tile((1, 1))
    output_arranged.dtype = output_arranged.dtype.squeeze(0)
    output_arranged = output_arranged.squeeze(1)

    input_arranged = input.tile((1,))

    weight_arranged = weight.tile((1, embedding_dim))
    weight_arranged.dtype = weight_arranged.dtype.squeeze(0)
    weight_arranged = weight_arranged.squeeze(1)
    weight_arranged = weight_arranged.tile((-1,))
    weight_arranged = weight_arranged.expand((output_arranged.shape[0],))

    return input_arranged, weight_arranged, output_arranged, max_norm, norm_type


def application(input, weight, output, max_norm, norm_type):
    idx = input[0]
    tmp = ntl.zeros(weight[0].shape, dtype=ntl.float32)
    tmp = ntl.abs(weight[idx])
    tmp = libdevice.pow(tmp, norm_type)
    sum = ntl.sum(tmp)
    norm = libdevice.pow(sum, 1.0 / norm_type)

    if norm > max_norm:
        scale = max_norm / norm
        weight[idx] = weight[idx] * scale
    output[0] = weight[idx]


def premake(ndim, embedding_dim=None, dtype=None, block_size_m=None, block_size_n=None):
    import math

    embedding_dim_power_of_2 = (
        2 ** math.ceil(math.log2(embedding_dim)) if embedding_dim > 0 else 1
    )
    arrangement_ = functools.partial(
        arrangement,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
        embedding_dim=embedding_dim_power_of_2,
    )

    tensors = (
        Tensor(ndim, dtype=ninetoothed.int64),
        Tensor(2, dtype=dtype),
        Tensor(ndim + 1, dtype=dtype),
        Tensor(0, dtype=ninetoothed.float32),
        Tensor(0, dtype=ninetoothed.float32),
    )

    return arrangement_, application, tensors


def arrangement_without_norm(
    input,
    weight,
    output,
    block_size_m=None,
    block_size_n=None,
):
    if block_size_m is None:
        block_size_m = BLOCK_SIZE
    if block_size_n is None:
        block_size_n = BLOCK_SIZE

    output = output.flatten(start_dim=0, end_dim=-1)
    input = input.flatten()
    # below commetned out code can be ussed for optional params == None
    # output:  [N, embedding_dim] -> tile to [block_size_m, block_size_n]
    output_arranged = output.tile((1, block_size_n))
    output_arranged = output_arranged.tile((block_size_m, 1))
    output_arranged.dtype = output_arranged.dtype.squeeze(1)

    # input: [N] -> tile to [block_size_m], 这是索引
    input = input.unsqueeze(1)  # 增加一个维度以便进行expand
    input_arranged = input.tile((block_size_m, 1))
    input_arranged = input_arranged.expand((-1, output_arranged.shape[1]))
    input_arranged.dtype = input_arranged.dtype.squeeze(1)

    # weight: [vocab_size, embedding_dim] -> tile embedding_dim dimension
    # 然后 expand 到匹配 output 的 batch 维度
    weight_arranged = weight.tile((1, block_size_n))
    weight_arranged = weight_arranged.tile((-1, 1))  # 保持所有 vocab 行
    weight_arranged = weight_arranged.expand((output_arranged.shape[0], -1))
    weight_arranged.dtype = weight_arranged.dtype.squeeze(1)

    return input_arranged, weight_arranged, output_arranged


def application_without_norm(input, weight, output):
    for i in range(output.shape[0]):
        idx = input[i]
        output[i] = weight[idx]


def premake_without_norm(ndim, dtype=None, block_size_m=None, block_size_n=None):
    arrangement_ = functools.partial(
        arrangement_without_norm,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
    )

    tensors = (
        Tensor(ndim, dtype=ninetoothed.int64),
        Tensor(2, dtype=dtype),
        Tensor(ndim + 1, dtype=dtype),
    )

    return arrangement_, application_without_norm, tensors
