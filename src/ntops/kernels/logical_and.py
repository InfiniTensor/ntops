import functools
import ninetoothed.language as ntl
from ninetoothed import Tensor
from ntops.kernels.element_wise import arrangement

def application(input1, input2, output):
    # 获取输入的数据类型
    dtype = input1.dtype
    val1 = ntl.cast(input1, ntl.int1)
    val2 = ntl.cast(input2, ntl.int1)
    result = val1 & val2
    output = ntl.cast(result, dtype)

def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)
    tensors = (Tensor(ndim, dtype=dtype), Tensor(ndim, dtype=dtype), Tensor(ndim, dtype=dtype))
    
    return arrangement_, application, tensors