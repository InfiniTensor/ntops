import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.reduction import arrangement


def application(input, index, src, output):
    dtype = output.dtype.dtype
    acc_dtype = ntl.float32 if dtype == ntl.float16 else dtype
    index_dtype = index.dtype.dtype

    # block_size=1 时，input[0] 是 [1] 向量
    # 所以 zero 也必须初始化成 [1] 向量，不能用 scalar 0
    zero = ntl.cast(input[0] * 0, acc_dtype)

    # input.shape[0] 是 scatter_add 的 dim_size
    for out_i in range(input.shape[0]):
        out_i_t = ntl.cast(out_i, index_dtype)

        # scatter_add 语义：
        # output 先等于 input，再把 src 按 index 累加进去
        acc = ntl.cast(input[out_i], acc_dtype)

        for src_i in range(input.shape[0]):
            idx = index[src_i]
            val = ntl.cast(src[src_i], acc_dtype)

            add_val = ntl.where(
                idx == out_i_t,
                val,
                zero,
            )

            acc += add_val

        output[out_i] = ntl.cast(acc, dtype)


def premake(ndim, dim, dtype=None, block_size=None):
    # scatter_add 沿 dim 做一维 scatter
    # 这里强制 block_size=1，避免 index / src 的 lane 类型不一致
    arrangement_ = functools.partial(
        arrangement,
        dim=dim,
        block_size=1,
    )

    input = Tensor(
        ndim,
        dtype=dtype,
        shape_options={"constexpr": True},
    )

    index = Tensor(
        ndim,
        dtype=ninetoothed.int64,
        shape_options={"constexpr": True},
    )

    src = Tensor(
        ndim,
        dtype=dtype,
        shape_options={"constexpr": True},
    )

    output = Tensor(
        ndim,
        dtype=dtype,
        shape_options={"constexpr": True},
    )

    tensors = (
        input,
        index,
        src,
        output,
    )

    return arrangement_, application, tensors