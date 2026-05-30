"""fractional_max_pool3d — fully fused (method Z). 3D analog of the 2d kernel."""

import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


def arrangement(
    output, input, samples, alpha_d, alpha_h, alpha_w,
    d, h, w, od, oh, ow, kd, kh, kw, m, block_size=None,
):
    if block_size is None:
        block_size = ninetoothed.block_size()

    output_arranged = output.flatten().tile((block_size,))

    return (
        output_arranged, input, samples, alpha_d, alpha_h, alpha_w,
        d, h, w, od, oh, ow, kd, kh, kw, m,
    )


def application(
    output, input, samples, alpha_d, alpha_h, alpha_w,
    d, h, w, od, oh, ow, kd, kh, kw, m,
):
    p = output.offsets()
    mask = p < m

    ohw = oh * ow
    odhw = od * ohw
    nc = p // odhw
    rem = p - nc * odhw
    od_i = rem // ohw
    rem2 = rem - od_i * ohw
    oh_i = rem2 // ow
    ow_i = rem2 - oh_i * ow

    # samples is (N*C, 3): [0] -> D, [1] -> H, [2] -> W.
    sample_d = ntl.cast(ntl.load(samples.data_ptr() + nc * 3 + 0, mask=mask, other=0.0), ntl.float32)
    sample_h = ntl.cast(ntl.load(samples.data_ptr() + nc * 3 + 1, mask=mask, other=0.0), ntl.float32)
    sample_w = ntl.cast(ntl.load(samples.data_ptr() + nc * 3 + 2, mask=mask, other=0.0), ntl.float32)

    od_f = ntl.cast(od_i, ntl.float32)
    oh_f = ntl.cast(oh_i, ntl.float32)
    ow_f = ntl.cast(ow_i, ntl.float32)

    sd = ntl.cast(ntl.floor((od_f + sample_d) * alpha_d) - ntl.floor(sample_d * alpha_d), ntl.int32)
    sh = ntl.cast(ntl.floor((oh_f + sample_h) * alpha_h) - ntl.floor(sample_h * alpha_h), ntl.int32)
    sw = ntl.cast(ntl.floor((ow_f + sample_w) * alpha_w) - ntl.floor(sample_w * alpha_w), ntl.int32)
    sd = ntl.where(od_i == od - 1, d - kd, sd)
    sh = ntl.where(oh_i == oh - 1, h - kh, sh)
    sw = ntl.where(ow_i == ow - 1, w - kw, sw)

    base = ((nc * d + sd) * h + sh) * w + sw

    acc = ntl.load(input.data_ptr() + base, mask=mask, other=float("-inf"))
    for dd in range(kd):
        for dh in range(kh):
            for dw in range(kw):
                acc = ntl.maximum(
                    acc,
                    ntl.load(
                        input.data_ptr() + base + (dd * h * w + dh * w + dw),
                        mask=mask,
                        other=float("-inf"),
                    ),
                )

    output = acc  # noqa: F841


def premake(block_size=None):
    tensors = (
        Tensor(1),  # output (M,)
        Tensor(1, shape_options={"constexpr": True}),  # input flat (source)
        Tensor(1, shape_options={"constexpr": True}),  # samples flat (N*C*3) (source)
        Tensor(0, constexpr=True),  # alpha_d
        Tensor(0, constexpr=True),  # alpha_h
        Tensor(0, constexpr=True),  # alpha_w
        Tensor(0, dtype=int, constexpr=True),  # d
        Tensor(0, dtype=int, constexpr=True),  # h
        Tensor(0, dtype=int, constexpr=True),  # w
        Tensor(0, dtype=int, constexpr=True),  # od
        Tensor(0, dtype=int, constexpr=True),  # oh
        Tensor(0, dtype=int, constexpr=True),  # ow
        Tensor(0, dtype=int, constexpr=True),  # kd
        Tensor(0, dtype=int, constexpr=True),  # kh
        Tensor(0, dtype=int, constexpr=True),  # kw
        Tensor(0, dtype=int, constexpr=True),  # m
    )

    arrangement_ = functools.partial(arrangement, block_size=block_size)

    return arrangement_, application, tensors
