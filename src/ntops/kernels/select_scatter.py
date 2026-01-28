import functools
import ninetoothed.language as ntl
from ninetoothed import Tensor


def arrangement(input, src, output, index, dim_size_pow2, dim, block_size):
    ndim = input.ndim
    if dim < 0: dim += ndim
    non_target_dims = tuple(i for i in range(ndim) if i != dim)
    
    def _arrangement(t):
        return t.permute(non_target_dims + (dim,)).flatten(end_dim=-1)

    # (Remaining, Dim_Size)
    input_arranged = _arrangement(input).tile((block_size, -1)).squeeze(1)
    src_arranged = _arrangement(src).tile((block_size, -1)).squeeze(1)
    output_arranged = _arrangement(output).tile((block_size, -1)).squeeze(1)

    return input_arranged, src_arranged, output_arranged, index, dim_size_pow2

def application(input, src, output, target_index, dim_size_pow2):
    col_indices = ntl.arange(0, dim_size_pow2)
    
    col_indices = ntl.expand_dims(col_indices, 0)
    col_indices = ntl.broadcast_to(col_indices, (input.shape[0], dim_size_pow2))
    
    actual_dim_size = input.shape[1]
    
    match_mask = (col_indices == ntl.cast(target_index, ntl.int32))
    valid_mask = col_indices < ntl.cast(actual_dim_size, ntl.int32)
    
    final_mask = match_mask & valid_mask
    
    output = ntl.where(final_mask, ntl.cast(src, output.dtype), ntl.cast(input, output.dtype))

def premake(ndim, dim, index, dim_size_pow2, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, dim=dim, block_size=block_size)
    
    tensors = (
        Tensor(ndim, dtype=dtype, shape_options={"constexpr": True}),
        Tensor(ndim, dtype=dtype, shape_options={"constexpr": True}),
        Tensor(ndim, dtype=dtype, shape_options={"constexpr": True}),
        Tensor(0, constexpr=True, value=index),
        Tensor(0, constexpr=True, value=dim_size_pow2),
    )
    return arrangement_, application, tensors