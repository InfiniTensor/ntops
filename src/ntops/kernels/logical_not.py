import functools
import ninetoothed.language as ntl
from ninetoothed import Tensor
from ntops.kernels.element_wise import arrangement

def application(input, output):
    val_in = input
    val_bool = ntl.cast(val_in, ntl.int1)
    result_bool = ~val_bool
    val_out = ntl.cast(result_bool, output.dtype)
    
    # 4. 赋值给输出
    output = val_out

def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)
    tensors = (Tensor(ndim, dtype=dtype), Tensor(ndim, dtype=dtype))
    return arrangement_, application, tensors