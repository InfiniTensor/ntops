"""scatter_add via data-dependent atomic scatter (O(N), like torch).

DRAFT — needs GPU iteration. The one-hot version was O(T*K) and lost / OOM'd, so
this uses `ntl.atomic_add` into data-dependent output positions (histc's
primitive, PR #60), extended to a *data-dependent* target out_ptr + (r*T + idx).

The wrapper moves `dim` to the last axis and flattens to output (R, T) (a clone
of input, the accumulation base) and src/index (R, K). One program owns a block
of flat source elements p; row r = p // K; it atomic-adds src[p] into
output[r, index[p]].

GRID NOTE: ninetoothed sets grid = prod(first arg's shape). `src` is first so the
grid = number of source blocks (O(N)). `output` is passed through un-arranged
(data_ptr() needs a source tensor); if it were first the grid would be R*T — a
huge oversized grid that kills perf and faults out of bounds.
"""

import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


def arrangement(src, index, output, t, k, n, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    src_arranged = src.flatten().tile((block_size,))
    index_arranged = index.flatten().tile((block_size,))

    return src_arranged, index_arranged, output, t, k, n


def application(src, index, output, t, k, n):
    p = src.offsets()  # flat source positions of this block
    row = p // k
    target = row * t + index  # data-dependent flat offset into output
    mask = p < n  # exclude tile padding past the real source length

    ntl.atomic_add(output.data_ptr() + target, src, mask=mask)


def premake(block_size=None):
    tensors = (
        Tensor(2, other=0),  # src (R, K)
        Tensor(2, other=0),  # index (R, K)
        # output (R, T): clone of input, accumulated in place. constexpr shape so
        # the autotuner sees concrete sizes (it can't bound an un-arranged source).
        Tensor(2, shape_options={"constexpr": True}),
        Tensor(0, dtype=int, constexpr=True),  # t = T (output row length)
        Tensor(0, dtype=int, constexpr=True),  # k = K (src/index row length)
        Tensor(0, dtype=int, constexpr=True),  # n = R*K (real source length)
    )

    arrangement_ = functools.partial(arrangement, block_size=block_size)

    return arrangement_, application, tensors
