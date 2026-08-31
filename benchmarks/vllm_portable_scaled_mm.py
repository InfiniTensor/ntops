"""Portable benchmark-only adaptation of vLLM block-scaled FP8 GEMM.

The kernel structure and default launch configuration follow vLLM's
``w8a8_triton_block_scaled_mm`` implementation in ``fp8_utils.py`` (Apache-2.0).
The only semantic-neutral portability addition is an explicit FP8 operand cast
for vendor Triton toolchains whose native FP8 ``tl.dot`` is unavailable or
incorrect.  This module is an experimental baseline and is not imported by the
ntops operator implementation.
"""

import importlib.metadata

import torch
import triton
import triton.language as tl


FP8_DOT_NATIVE = 0
FP8_DOT_FLOAT16 = 1
FP8_DOT_BFLOAT16 = 2


def _fp8_dot_fallback():
    version = importlib.metadata.version("triton").lower()

    if "+corex." in version:
        return FP8_DOT_BFLOAT16
    if ".dtk" in version or "+das." in version:
        return FP8_DOT_FLOAT16
    return FP8_DOT_NATIVE


_EFFECTIVE_FP8_DOT_FALLBACK = _fp8_dot_fallback()


def fp8_dot_fallback_name():
    return {
        FP8_DOT_NATIVE: "native",
        FP8_DOT_FLOAT16: "float16",
        FP8_DOT_BFLOAT16: "bfloat16",
    }[_EFFECTIVE_FP8_DOT_FALLBACK]


@triton.jit
def _w8a8_portable_block_scaled_mm(
    A,
    B,
    C,
    As,
    Bs,
    M,
    N,
    K,
    group_n,
    group_k,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_As_m,
    stride_As_k,
    stride_Bs_k,
    stride_Bs_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    FP8_DOT_FALLBACK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    as_ptrs = As + offs_am * stride_As_m
    offs_bsn = offs_bn // group_n
    bs_ptrs = Bs + offs_bsn * stride_Bs_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptrs,
            mask=offs_k[None, :] < K - k * BLOCK_SIZE_K,
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
            other=0.0,
        )

        if FP8_DOT_FALLBACK == 1:
            a = a.to(tl.float16)
            b = b.to(tl.float16)
        elif FP8_DOT_FALLBACK == 2:
            a = a.to(tl.bfloat16)
            b = b.to(tl.bfloat16)

        k_start = k * BLOCK_SIZE_K
        offs_ks = k_start // group_k
        a_s = tl.load(as_ptrs + offs_ks * stride_As_k)
        b_s = tl.load(bs_ptrs + offs_ks * stride_Bs_k)

        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def w8a8_portable_block_scaled_mm(
    A,
    B,
    As,
    Bs,
    block_size,
    output_dtype=torch.float16,
    config=None,
):
    assert len(block_size) == 2
    block_n, block_k = block_size
    assert A.shape[-1] == B.shape[-1]
    assert A.shape[:-1] == As.shape[:-1] and A.is_contiguous()
    assert triton.cdiv(A.shape[-1], block_k) == As.shape[-1]

    m = A.numel() // A.shape[-1]
    n, k = B.shape
    assert B.ndim == 2 and Bs.ndim == 2
    assert triton.cdiv(n, block_n) == Bs.shape[0]
    assert triton.cdiv(k, block_k) == Bs.shape[1]

    output = A.new_empty(A.shape[:-1] + (n,), dtype=output_dtype)
    launch_config = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": block_n,
        "BLOCK_SIZE_K": block_k,
        "GROUP_SIZE_M": 32,
        "num_warps": 4,
        "num_stages": 2,
    }
    if config is not None:
        launch_config.update(config)
    launch_config["FP8_DOT_FALLBACK"] = _EFFECTIVE_FP8_DOT_FALLBACK

    def grid(meta):
        return (
            triton.cdiv(m, meta["BLOCK_SIZE_M"])
            * triton.cdiv(n, meta["BLOCK_SIZE_N"]),
        )

    _w8a8_portable_block_scaled_mm[grid](
        A,
        B,
        output,
        As,
        Bs,
        m,
        n,
        k,
        block_n,
        block_k,
        A.stride(-2),
        A.stride(-1),
        B.stride(1),
        B.stride(0),
        output.stride(-2),
        output.stride(-1),
        As.stride(-2),
        As.stride(-1),
        Bs.stride(1),
        Bs.stride(0),
        **launch_config,
    )
    return output


__all__ = ["fp8_dot_fallback_name", "w8a8_portable_block_scaled_mm"]
