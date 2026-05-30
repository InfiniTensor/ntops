"""Pure-ninetoothed slogdet via single-block Gaussian elimination.

DRAFT (no pivoting). One program per (batched) matrix; the whole N*N matrix is
loaded into one tile and an unrolled `for k in range(N)` loop runs LU in-block.
Data-dependent addressing is avoided entirely: every row/column/entry is
extracted with a one-hot masked reduction, so all addressing stays affine.

The matrix dimension is referenced as `input.shape[0]` (a constexpr from the
Symbol-`n` tiling), never as a bare Symbol — a bare Symbol Name in the body is
treated as a tensor load by the codegen. The wrapper pads the n*n matrix to
N = next_pow2(n) with an identity block (`[[A, 0], [0, I]]`, det unchanged), so
padded pivots are 1 and contribute log|1|=0, sign*1.

Limits: matrix must fit one block (n <= ~64-128); no intra-matrix parallelism
(only batch parallelizes), so this targets *many small* matrices.
"""

import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor


def arrangement(input, sign, logabsdet, n, block_size=None):
    # One program per batch element: each owns the full (n, n) matrix and two
    # scalar outputs. `n` (constexpr Symbol) makes the tile dim a constexpr.
    input_arranged = input.tile((1, n, n))
    input_arranged.dtype = input_arranged.dtype.squeeze(0)

    sign_arranged = sign.tile((1,))
    sign_arranged.dtype = sign_arranged.dtype.squeeze(0)

    logabsdet_arranged = logabsdet.tile((1,))
    logabsdet_arranged.dtype = logabsdet_arranged.dtype.squeeze(0)

    return input_arranged, sign_arranged, logabsdet_arranged


def application(input, sign, logabsdet):
    # input: (N, N) for one matrix, N a power of 2. Reference the dim only via
    # `input.shape[0]` (constexpr), never a bare Symbol.
    row = ntl.arange(0, input.shape[0])
    col = ntl.arange(0, input.shape[0])
    row_c = ntl.expand_dims(row, 1)  # (N, 1)
    col_r = ntl.expand_dims(col, 0)  # (1, N)

    a = ntl.cast(input, ntl.float32)
    logabs = ntl.cast(0.0, ntl.float32)
    sgn = ntl.cast(1.0, ntl.float32)

    for k in range(input.shape[0]):  # constexpr -> unrolled
        is_k_row = row_c == k  # (N, 1)
        is_k_col = col_r == k  # (1, N)

        # pivot = a[k, k] via one-hot reduction (no data-dependent addressing).
        pivot = ntl.sum(ntl.sum(ntl.where(is_k_row & is_k_col, a, 0.0), axis=1), axis=0)

        logabs += ntl.log(ntl.abs(pivot))
        sgn = sgn * ntl.where(pivot > 0, 1.0, ntl.where(pivot < 0, -1.0, 0.0))

        col_k = ntl.sum(ntl.where(is_k_col, a, 0.0), axis=1)  # (N,)
        row_k = ntl.sum(ntl.where(is_k_row, a, 0.0), axis=0)  # (N,)

        # multipliers for rows below k; guard a zero pivot so `a` stays finite
        # (the log term already drove logabs to -inf, sgn to 0).
        m = ntl.where((row > k) & (pivot != 0.0), col_k / pivot, 0.0)  # (N,)
        m_c = ntl.expand_dims(m, 1)  # (N, 1)
        row_k_r = ntl.expand_dims(row_k, 0)  # (1, N)

        a = a - m_c * row_k_r  # rank-1 update, only rows > k change

    sign = sgn  # noqa: F841
    logabsdet = logabs  # noqa: F841


def premake(dtype=None, block_size=None):
    n = Symbol("n", constexpr=True)

    tensors = (
        Tensor(3, dtype=dtype, other=0.0),  # input (B, n, n), padded
        Tensor(1, dtype=ninetoothed.float32),  # sign (B,)
        Tensor(1, dtype=ninetoothed.float32),  # logabsdet (B,)
    )

    arrangement_ = functools.partial(arrangement, n=n, block_size=block_size)

    return arrangement_, application, tensors
