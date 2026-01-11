import torch
import ntops
from ntops.torch.utils import _cached_make

def logical_not(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    # 判定是否为“非连续内存的原地操作”
    is_inplace = (out.data_ptr() == input.data_ptr())
    is_strided = (not input.is_contiguous())

    if is_inplace and is_strided:
        # 创建一个连续的临时 Tensor
        # 使用 torch.empty 确保它是连续的，避免继承 input 的非连续 stride
        temp_out = torch.empty(input.shape, dtype=input.dtype, device=input.device)
        
        kernel = _cached_make(ntops.kernels.logical_not.premake, input.ndim)
        
        # 1. 读非连续 input -> 写连续 temp_out (安全)
        kernel(input, temp_out)
        
        # 2. 复制回原处 (PyTorch 会自动处理 stride 转换)
        out.copy_(temp_out)
    else:
        kernel = _cached_make(ntops.kernels.logical_not.premake, input.ndim)
        kernel(input, out)
    
    return out