"""调试: trace DSL - 1D diagonal + element_wise + ntl.sum"""
import torch, sys; sys.path.insert(0, 'src')
import ninetoothed.language as ntl
from ninetoothed import Tensor, make
from ntops.kernels.element_wise import arrangement

def application(input, output):
    sum_val = ntl.sum(input)
    output = sum_val + ntl.cast(0, ntl.float32)

def premake(ndim, dtype=None, block_size=None):
    arr = functools.partial(arrangement, block_size=block_size)
    tensors = (Tensor(ndim, dtype=dtype), Tensor(1, dtype=dtype))
    return arr, application, tensors

import functools
N = 5
x = torch.arange(N, dtype=torch.float32, device='cuda')
out = torch.zeros(1, device='cuda')

k = make(arrangement, application, (Tensor(1), Tensor(1)))
k(x, out)
print(f"sum: {out.item()}")
print(f"expected: {x.sum().item()}")
print(f"correct: {abs(out.item() - x.sum().item()) < 1e-5}")
