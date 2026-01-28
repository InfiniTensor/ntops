import torch
import ntops
from ntops.torch.utils import _cached_make

def select_scatter(input, src, dim, index):
    ndim = input.ndim
    if dim < 0: dim += ndim
    
    dim_size = input.shape[dim]
    dim_size_pow2 = 1 << (dim_size - 1).bit_length()
    
    src_expanded = src.unsqueeze(dim)
    output = torch.empty_like(input)
    block_size = 1024

    kernel = _cached_make(
        ntops.kernels.select_scatter.premake, 
        ndim, dim, int(index), int(dim_size_pow2),
        input.dtype, block_size
    )

    kernel(input, src_expanded, output, int(index), int(dim_size_pow2))
    return output