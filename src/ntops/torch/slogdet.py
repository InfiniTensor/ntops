import math

import torch

import ntops
from ntops.torch.utils import _cached_make

# Pure-ninetoothed slogdet (DRAFT v0): single-block in-kernel Gaussian
# elimination, no torch.linalg delegation. See kernels/slogdet.py for the
# algorithm and limits (small n only; one program per matrix; no pivoting yet).
#
# NOT YET GPU-VERIFIED — wire in, run tests/test_slogdet.py on a GPU, iterate.


def _next_pow2(x):
    return 1 << (x - 1).bit_length()


def slogdet(A, *, out=None):
    assert A.ndim >= 2 and A.shape[-1] == A.shape[-2], (
        "`slogdet` requires square (batched) matrices."
    )

    n = A.shape[-1]
    batch_shape = A.shape[:-2]
    batch = math.prod(batch_shape) if batch_shape else 1

    # Accumulate in fp32. Pad to a power-of-2 N so the in-kernel `arange(N)`
    # (needed for the masks) is legal. Pad with an identity block so the matrix
    # becomes [[A, 0], [0, I]] (det unchanged): the kernel loops to N and the
    # padded pivots (=1) contribute log|1|=0, sign*1.
    a = A.reshape(batch, n, n).to(torch.float32)
    pad = _next_pow2(n)

    if pad != n:
        padded = torch.zeros((batch, pad, pad), dtype=torch.float32, device=A.device)
        diag = torch.arange(pad, device=A.device)
        padded[:, diag, diag] = 1.0
        padded[:, :n, :n] = a
        a = padded

    sign = torch.empty((batch,), dtype=torch.float32, device=A.device)
    logabsdet = torch.empty((batch,), dtype=torch.float32, device=A.device)

    kernel = _cached_make(ntops.kernels.slogdet.premake)
    kernel(a, sign, logabsdet, n=pad)

    out_shape = batch_shape if batch_shape else ()
    sign = sign.reshape(out_shape)
    logabsdet = logabsdet.reshape(out_shape)

    return torch.return_types.linalg_slogdet((sign, logabsdet))
