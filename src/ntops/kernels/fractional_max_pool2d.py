"""fractional_max_pool2d — fully fused (method Z).

Everything is in one kernel: from the raw per-(n,c) random samples, each program
computes the window start (PyTorch's interval formula), the input base offset,
then gathers the kH*kW window via data-dependent loads and maxes. The host does
no per-element work (no interval tensors, no base_offset materialization), so the
whole op is a single launch — competitive with torch's fused kernel.

Earlier methods: B (torch gather + ninetoothed max) was ~0.01x (materialized
windows); C (host base_offset + in-kernel gather) had a fast kernel but ~16 host
torch ops dominated (~0.37ms fixed). Z removes the host glue entirely.
"""

import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


def arrangement(output, input, samples, alpha_h, alpha_w, h, w, oh, ow, kh, kw, m, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    output_arranged = output.flatten().tile((block_size,))

    return output_arranged, input, samples, alpha_h, alpha_w, h, w, oh, ow, kh, kw, m


def application(output, input, samples, alpha_h, alpha_w, h, w, oh, ow, kh, kw, m):
    p = output.offsets()  # flat output index per element
    mask = p < m

    ohw = oh * ow
    nc = p // ohw
    rem = p - nc * ohw
    oh_i = rem // ow
    ow_i = rem - oh_i * ow

    # samples is (N*C, 2): [..., 1] -> H sample, [..., 0] -> W sample.
    sample_h = ntl.cast(
        ntl.load(samples.data_ptr() + nc * 2 + 1, mask=mask, other=0.0), ntl.float32
    )
    sample_w = ntl.cast(
        ntl.load(samples.data_ptr() + nc * 2 + 0, mask=mask, other=0.0), ntl.float32
    )

    oh_f = ntl.cast(oh_i, ntl.float32)
    ow_f = ntl.cast(ow_i, ntl.float32)

    # start = floor((idx + u)*alpha) - floor(u*alpha); last index forced to in-pool.
    sh = ntl.cast(
        ntl.floor((oh_f + sample_h) * alpha_h) - ntl.floor(sample_h * alpha_h), ntl.int32
    )
    sw = ntl.cast(
        ntl.floor((ow_f + sample_w) * alpha_w) - ntl.floor(sample_w * alpha_w), ntl.int32
    )
    sh = ntl.where(oh_i == oh - 1, h - kh, sh)
    sw = ntl.where(ow_i == ow - 1, w - kw, sw)

    base = (nc * h + sh) * w + sw

    acc = ntl.load(input.data_ptr() + base, mask=mask, other=float("-inf"))
    for dh in range(kh):
        for dw in range(kw):
            acc = ntl.maximum(
                acc,
                ntl.load(
                    input.data_ptr() + base + (dh * w + dw), mask=mask, other=float("-inf")
                ),
            )

    output = acc  # noqa: F841


def premake(block_size=None):
    tensors = (
        Tensor(1),  # output (M,)
        Tensor(1, shape_options={"constexpr": True}),  # input flat (source)
        Tensor(1, shape_options={"constexpr": True}),  # samples flat (N*C*2) (source)
        Tensor(0, constexpr=True),  # alpha_h
        Tensor(0, constexpr=True),  # alpha_w
        Tensor(0, dtype=int, constexpr=True),  # h
        Tensor(0, dtype=int, constexpr=True),  # w
        Tensor(0, dtype=int, constexpr=True),  # oh
        Tensor(0, dtype=int, constexpr=True),  # ow
        Tensor(0, dtype=int, constexpr=True),  # kh
        Tensor(0, dtype=int, constexpr=True),  # kw
        Tensor(0, dtype=int, constexpr=True),  # m
    )

    arrangement_ = functools.partial(arrangement, block_size=block_size)

    return arrangement_, application, tensors
