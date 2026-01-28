import functools
import ninetoothed.language as ntl
from ninetoothed import Tensor

def arrangement(input, output, dim_size, dim, block_size):
    ndim = input.ndim
    if dim < 0: dim = ndim + dim
    
    tile_shape = [1] * ndim
    tile_shape[dim] = block_size
    
    in_t = input.tile(tuple(tile_shape))
    out_t = output.tile(tuple(tile_shape))
    
    for _ in range(ndim - 1):

        in_t.dtype = in_t.dtype.squeeze(0 if dim != 0 else 1)
        out_t.dtype = out_t.dtype.squeeze(0 if dim != 0 else 1)

        if dim > 0:
            dim -= 1

    return in_t, out_t, dim_size

def application(input, output, dim_size):
    half = dim_size // 2

    for i in range(half):
        a = ntl.cast(input[i], ntl.float32)
        b = ntl.cast(input[i + half], ntl.float32)
        
        res = a * ntl.sigmoid(b)
        
        output[i] = ntl.cast(res, output.dtype)

def premake(ndim, dim, dim_size, dtype=None, block_size=None):

    arrangement_ = functools.partial(arrangement, dim=dim, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype, shape_options={"constexpr": True}),
        Tensor(ndim, dtype=dtype, shape_options={"constexpr": True}),
        Tensor(0, constexpr=True, value=dim_size),
    )

    return arrangement_, application, tensors