"""multilabel_margin_loss via one-hot gather + masked reduction.

DRAFT — needs GPU iteration. One program per sample (row). Per sample:

    raw = sum_{j in valid targets} sum_{i not a target} max(0, 1 - (x[t_j] - x[i]))

where valid targets are the contiguous non-negative prefix of `target`. The
gather x[target[j]] and the "is i a target" test are done with one-hot masked
reductions (affine addressing). The wrapper divides by C and reduces over N.

Padding to C = next_pow2: x -> -inf (so padded classes contribute max(0, -inf)
= 0 and are never targets), target -> -1 (excluded from the valid prefix).
"""

import functools

import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor


def arrangement(x, target, output, c, block_size=None):
    x_arranged = x.tile((1, c))
    x_arranged.dtype = x_arranged.dtype.squeeze(0)

    target_arranged = target.tile((1, c))
    target_arranged.dtype = target_arranged.dtype.squeeze(0)

    output_arranged = output.tile((1,))
    output_arranged.dtype = output_arranged.dtype.squeeze(0)

    return x_arranged, target_arranged, output_arranged


def application(x, target, output):
    # x, target: (C,) for one sample (C a power of 2; padded x=-inf, target=-1).
    cls = ntl.arange(0, x.shape[0])  # class indices
    x_f = ntl.cast(x, ntl.float32)

    # valid target positions = contiguous non-negative prefix (before first -1).
    neg_pos = ntl.where(target == -1, cls, x.shape[0])
    first_neg = ntl.min(neg_pos)
    valid_j = cls < first_neg  # (Cj,)

    tgt_col = ntl.expand_dims(target, 1)  # (Cj, 1)
    cls_row = ntl.expand_dims(cls, 0)  # (1, Ci)
    onehot = tgt_col == cls_row  # (Cj, Ci): target[j] == i
    x_row = ntl.expand_dims(x_f, 0)  # (1, Ci)

    # xt[j] = x[target[j]]
    xt = ntl.sum(ntl.where(onehot, x_row, 0.0), axis=1)  # (Cj,)

    valid_col = ntl.expand_dims(valid_j, 1)  # (Cj, 1)
    # in_T_count[i] = #valid j with target[j] == i  -> i is a target iff > 0
    in_t_count = ntl.sum(ntl.where(valid_col & onehot, 1.0, 0.0), axis=0)  # (Ci,)
    not_in_t = ntl.expand_dims(in_t_count == 0, 0)  # (1, Ci)

    xt_col = ntl.expand_dims(xt, 1)  # (Cj, 1)
    margin = 1.0 - (xt_col - x_row)  # (Cj, Ci)
    term = ntl.where(margin > 0, margin, 0.0)  # relu

    mask = valid_col & not_in_t  # (Cj, Ci)
    output = ntl.sum(ntl.sum(ntl.where(mask, term, 0.0), axis=1), axis=0)  # noqa: F841


def premake(block_size=None):
    c = Symbol("c", constexpr=True)

    tensors = (
        Tensor(2, other=float("-inf")),  # x (N, C), pad -inf
        Tensor(2, other=-1),  # target (N, C), pad -1
        Tensor(1),  # output (N,) raw per-sample loss (pre /C)
    )

    arrangement_ = functools.partial(arrangement, c=c, block_size=block_size)

    return arrangement_, application, tensors
