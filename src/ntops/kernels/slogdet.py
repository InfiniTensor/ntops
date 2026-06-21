import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


def arrangement(input, sign, logabsdet, block_size=None):
    # block_size 必须是 2 的幂。
    # 例如 matrix_size=3 时，用 4x4 tile。
    input = input.tile((block_size, block_size))

    sign = sign.tile((1, 1))
    logabsdet = logabsdet.tile((1, 1))

    return input, sign, logabsdet


def _abs_f32(x):
    zero = ntl.cast(0.0, ntl.float32)
    return ntl.where(x < zero, -x, x)


def _sign_f32(x):
    zero = ntl.cast(0.0, ntl.float32)
    one = ntl.cast(1.0, ntl.float32)
    minus_one = ntl.cast(-1.0, ntl.float32)

    return ntl.where(x > zero, one, ntl.where(x < zero, minus_one, zero))


def application(input, sign, logabsdet):
    dtype = ntl.float32

    zero = ntl.cast(0.0, dtype)
    one = ntl.cast(1.0, dtype)
    minus_one = ntl.cast(-1.0, dtype)
    neg_inf = ntl.cast(float("-inf"), dtype)

    # input 是 block_size x block_size block。
    # 对于 3x3，实际是 4x4 block，越界位置由 Tensor(other=0.0) 填充。
    a = ntl.cast(input, dtype)

    row_idx = ntl.cast(input.offsets(-2), ntl.int32)
    col_idx = ntl.cast(input.offsets(-1), ntl.int32)

    rows = row_idx[:, None]
    cols = col_idx[None, :]

    # 真实矩阵大小，不是 block_size。
    # 例如 3x3 输入时，input.source.shape[-2] == 3。
    n_i32 = ntl.cast(input.source.shape[-2], ntl.int32)

    det_sign = one
    log_abs_det = zero
    singular = zero != zero

    # 只循环真实矩阵大小。
    for k in range(input.source.shape[-2]):
        k_i32 = ntl.cast(k, ntl.int32)

        # 取第 k 列：不用 a[:, k]，用 mask + sum。
        col_k = ntl.sum(
            ntl.where(cols == k_i32, a, zero),
            1,
        )

        col_abs = _abs_f32(col_k)

        valid_rows = (row_idx >= k_i32) & (row_idx < n_i32)

        masked_abs = ntl.where(valid_rows, col_abs, minus_one)
        pivot_abs = ntl.max(masked_abs)

        is_zero_pivot = pivot_abs == zero
        singular = singular | is_zero_pivot

        pivot_mask = (masked_abs == pivot_abs) & valid_rows

        pivot_is_k = (
            ntl.sum(
                ntl.where(pivot_mask & (row_idx == k_i32), one, zero)
            )
            > zero
        )

        det_sign = ntl.where(pivot_is_k, det_sign, -det_sign)

        # 第 k 行
        row_k = ntl.sum(
            ntl.where(rows == k_i32, a, zero),
            0,
        )

        # pivot 行
        pivot_row = ntl.sum(
            ntl.where(pivot_mask[:, None], a, zero),
            0,
        )

        # 交换第 k 行和 pivot 行。
        a = ntl.where(
            rows == k_i32,
            pivot_row[None, :],
            ntl.where(
                pivot_mask[:, None],
                row_k[None, :],
                a,
            ),
        )

        # pivot = a[k, k]
        pivot = ntl.sum(
            ntl.where((rows == k_i32) & (cols == k_i32), a, zero)
        )

        pivot_abs_after_swap = _abs_f32(pivot)
        pivot_sign = _sign_f32(pivot)

        det_sign = det_sign * pivot_sign

        safe_pivot_abs = ntl.where(
            pivot_abs_after_swap == zero,
            one,
            pivot_abs_after_swap,
        )

        log_abs_det = log_abs_det + ntl.log(safe_pivot_abs)

        safe_pivot = ntl.where(
            pivot_abs_after_swap == zero,
            one,
            pivot,
        )

        row_k_after_swap = ntl.sum(
            ntl.where(rows == k_i32, a, zero),
            0,
        )

        col_k_after_swap = ntl.sum(
            ntl.where(cols == k_i32, a, zero),
            1,
        )

        factor = col_k_after_swap / safe_pivot
        update = a - factor[:, None] * row_k_after_swap[None, :]

        # 只更新真实矩阵范围内的 trailing submatrix。
        update_mask = (
            (rows > k_i32)
            & (cols > k_i32)
            & (rows < n_i32)
            & (cols < n_i32)
        )

        a = ntl.where(update_mask, update, a)

    final_sign = ntl.where(singular, zero, det_sign)
    final_logabsdet = ntl.where(singular, neg_inf, log_abs_det)

    sign[0, 0] = final_sign  # noqa: F841
    logabsdet[0, 0] = final_logabsdet  # noqa: F841


def premake(ndim, matrix_size, block_size, dtype=None):
    arrangement_ = functools.partial(
        arrangement,
        block_size=block_size,
    )

    input_tensor = Tensor(
        ndim,
        shape=(matrix_size, matrix_size),
        dtype=dtype,
        other=0.0,
        shape_options=(
            {"constexpr": True, "upper_bound": 16},
            {"constexpr": True, "upper_bound": 16},
        ),
    )

    sign_tensor = Tensor(
        2,
        shape=(1, 1),
        dtype=dtype,
        shape_options=(
            {"constexpr": True, "upper_bound": 1},
            {"constexpr": True, "upper_bound": 1},
        ),
    )

    logabsdet_tensor = Tensor(
        2,
        shape=(1, 1),
        dtype=dtype,
        shape_options=(
            {"constexpr": True, "upper_bound": 1},
            {"constexpr": True, "upper_bound": 1},
        ),
    )

    tensors = (
        input_tensor,
        sign_tensor,
        logabsdet_tensor,
    )

    return arrangement_, application, tensors