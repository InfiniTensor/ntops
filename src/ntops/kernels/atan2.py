import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def application(input, other, output):
    y = ntl.cast(input, ntl.float32)
    x = ntl.cast(other, ntl.float32)

    # 常量定义
    PI = 3.1415927410125732
    HALF_PI = 1.5707963705062866

    abs_y = ntl.where(y < 0, -y, y)
    abs_x = ntl.where(x < 0, -x, x)

    swap_xy = abs_y > abs_x

    num = ntl.where(swap_xy, abs_x, abs_y)
    den = ntl.where(swap_xy, abs_y, abs_x)

    # 防除零
    den_safe = ntl.where(den == 0.0, 1.0, den)
    z = num / den_safe
    z_sq = z * z

    # 多项式逼近
    c0 = 0.9998660
    c1 = -0.3302995
    c2 = 0.1801410
    c3 = -0.0851330
    c4 = 0.0208351

    poly_res = z * (c0 + z_sq * (c1 + z_sq * (c2 + z_sq * (c3 + z_sq * c4))))

    theta = ntl.where(swap_xy, HALF_PI - poly_res, poly_res)

    res = theta
    res = ntl.where(x < 0, PI - theta, res)
    res = ntl.where(y < 0, -res, res)
    res = ntl.where((x == 0.0) & (y == 0.0), 0.0, res)

    output = ntl.cast(res, output.dtype)


def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
        Tensor(ndim, dtype=dtype),
    )

    return arrangement_, application, tensors
