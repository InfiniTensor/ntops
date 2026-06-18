# 测试T-1-3这个组的所有算子

import torch
import sys
sys.path.insert(0, 'src')
import ntops
import inspect

print('=== Testing triu_indices first (no other ops) ===')
print('sig:', inspect.signature(ntops.torch.triu_indices))
r = ntops.torch.triu_indices(5, device='cuda')
ref = torch.triu_indices(5, 5, device='cuda')
assert torch.equal(r, ref), 'triu_indices alone failed'
print('triu_indices alone: OK')

print('\n=== Now test all together ===')
x = torch.randn(5, 5, device='cuda')

# tril
out = ntops.torch.tril(x)
ref = torch.tril(x)
assert torch.equal(out, ref), 'tril failed'
print('tril: OK')

# triu  
out = ntops.torch.triu(x)
ref = torch.triu(x)
assert torch.equal(out, ref), 'triu failed'
print('triu: OK')

# trace
out = ntops.torch.trace(x)
ref = torch.trace(x)
assert torch.allclose(out, ref), 'trace failed'
print('trace: OK')

# outer
a = torch.randn(5, device='cuda')
b = torch.randn(7, device='cuda')
out = ntops.torch.outer(a, b)
ref = torch.outer(a, b)
assert torch.allclose(out, ref), 'outer failed'
print('outer: OK')

# triu_indices again
print('sig after other ops:', inspect.signature(ntops.torch.triu_indices))
r = ntops.torch.triu_indices(5, device='cuda')
ref = torch.triu_indices(5, 5, device='cuda')
assert torch.equal(r, ref), 'triu_indices after tril failed'
print('triu_indices after tril: OK')

print('\nALL PASSED')
