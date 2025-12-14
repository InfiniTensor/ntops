
import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed.language import libdevice
from ninetoothed import Tensor


def arrangement(*tensors, block_size):
    # input, output, min, max = tensors
    input, output, min_val, max_val, num_bins_pow2 = tensors

    if block_size is None:
        block_size = ninetoothed.block_size()

    # input: (N, )
    # output: (bins, )

    input_tiled = input.flatten().tile((block_size, )) # (N // block_size), dtype=(block_size, )

    output_expand = output.unsqueeze(0).expand((input_tiled.shape[0], -1)) # (N // block_size, bins)
    output_tiled = output_expand.tile((1, -1)).squeeze(1) # (N // block_size, ), dtype=(1, bins)
    output_tiled.dtype = output_tiled.dtype.squeeze(0) # dtype=(bins, )

    return input_tiled, output_tiled, min_val, max_val, num_bins_pow2

def application_manual_histogram(input, output, min_val, max_val, num_bins_pow2):
    """手动实现直方图计算。
    
    摩尔线程 GPU 内置的 histogram 函数不能正确计算柱状图，
    因此使用 ntl.arange 和 ntl.where 手动实现。
    """
    # input: (block_size,)
    # output: (bins,)
    n_out_bins = output.shape[0]

    # 只需要 [min_val, max_val]
    mask = (input >= min_val) & (input <= max_val)

    # 标准化为 [0, n_out_bins)
    input_scaled = (input - min_val) / (max_val - min_val) * n_out_bins

    # histogram 需要整数 bin 索引
    input_indices = ntl.cast(input_scaled, ntl.int32)

    # max_val 应该该落在最后一个 bin 中
    input_indices = ntl.minimum(input_indices, n_out_bins - 1)

    # 将超出范围的索引设为 -1，使其不会被计入直方图
    input_indices = ntl.where(mask, input_indices, -1)

    # 初始化直方图张量
    local_hist = ntl.zeros((num_bins_pow2,), dtype=output.dtype)
    
    # 逐 bin 计数：对每个 bin，用 where 统计匹配的元素个数
    # 由于摩尔线程不支持动态索引 histogram，因此只能手动实现
    for bin_idx in range(num_bins_pow2):
        bin_idx_tensor = ntl.cast(bin_idx, ntl.int32)
        match_mask = (input_indices == bin_idx_tensor)
        count = ntl.sum(match_mask.to(output.dtype))
        idx = ntl.arange(0, num_bins_pow2)
        update_mask = (idx == bin_idx_tensor)
        local_hist = ntl.where(update_mask, count, local_hist)

    # 只需要前 n_out_bins 个 bin
    valid_mask = ntl.arange(0, num_bins_pow2) < n_out_bins
    local_hist = local_hist.to(output.dtype)
    ntl.atomic_add(output.data_ptr() + output.offsets(),
                   local_hist,
                   mask=valid_mask)


def application_builtin_histogram(input, output, min_val, max_val, num_bins_pow2):
    # input: (block_size,)
    # output: (bins,)
    n_out_bins = output.shape[0]

    # 只需要 [min_val, max_val]
    mask = (input >= min_val) & (input <= max_val)

    # 标准化为 [0, n_out_bins)
    input_scaled = (input - min_val) / (max_val - min_val) * n_out_bins

    # histogram 需要整数 bin 索引
    input_indices = ntl.cast(input_scaled, ntl.int32)

    # max_val 应该该落在最后一个 bin 中
    input_indices = ntl.minimum(input_indices, n_out_bins - 1)

    # 将超出范围的索引设为 -1，使其不会被计入直方图
    # 因为在 triton 3.5.0 版本才引入的 masked histogram
    input_indices = ntl.where(mask, input_indices, -1)

    local_hist = ntl.histogram(input_indices,
                               num_bins=num_bins_pow2)  # shape: (num_bins_pow2,)

    # 只需要前 n_out_bins 个 bin
    valid_mask = ntl.arange(0, num_bins_pow2) < n_out_bins
    local_hist = local_hist.to(output.dtype)
    ntl.atomic_add(output.data_ptr() + output.offsets(),
                   local_hist,
                   mask=valid_mask)



def premake_builtin(dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (
        Tensor(1, dtype=dtype, other=float("inf"), shape_options={"constexpr": True}),  # input
        Tensor(1, dtype=dtype, shape_options={"constexpr": True}),  # output
        Tensor(0, dtype=dtype),  # min
        Tensor(0, dtype=dtype),  # max
        Tensor(0, dtype=int, constexpr=True), # num_bins_pow2
    )

    return arrangement_, application_builtin_histogram, tensors


def premake_manual(dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (
        Tensor(1, dtype=dtype, other=float("inf"), shape_options={"constexpr": True}),  # input
        Tensor(1, dtype=dtype, shape_options={"constexpr": True}),  # output
        Tensor(0, dtype=dtype),  # min
        Tensor(0, dtype=dtype),  # max
        Tensor(0, dtype=int, constexpr=True), # num_bins_pow2
    )

    return arrangement_, application_manual_histogram, tensors

