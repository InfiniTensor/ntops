import torch

import ntops
from ntops.torch.utils import _cached_make


def bitwise_left_shift(input, other, *, out=None):
    # Check if we need to handle non-contiguous inplace operation
    is_inplace_input = out is not None and out.data_ptr() == input.data_ptr()

    if out is None:
        out = torch.empty_like(input)

    # 处理非连续张量的原地操作特殊情况：
    # 当 out 和 input 是同一个张量（原地操作）且 input 具有非标准 strides（非连续）时，
    # ninetoothed 框架中的 element_wise.arrangement 函数使用 flatten() 会丢失内存布局信息，
    # 导致 GPU kernel 无法正确将结果写回到具有特殊 strides 的原始张量中。
    # 解决方案是先将输入转换为连续张量进行计算，然后使用 copy_() 将结果复制回原始张量，
    # copy_() 方法会正确处理目标张量的 strides，确保数据被写入到正确的内存位置。
    if is_inplace_input and not input.is_contiguous():
        input_contig = input.contiguous()
        other_contig = other.contiguous() if not other.is_contiguous() else other
        out_contig = torch.empty_like(input_contig)

        kernel = _cached_make(ntops.kernels.bitwise_left_shift.premake, input.ndim)
        kernel(input_contig, other_contig, out_contig)

        out.copy_(out_contig)
    else:
        kernel = _cached_make(ntops.kernels.bitwise_left_shift.premake, input.ndim)
        kernel(input, other, out)

    return out
