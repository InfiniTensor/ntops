import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


def arrangement(input, random_samples, output, indices,
                _kH, _kW, _Hi, _Wi, _Ho, _Wo, _C, _alpha_h, _alpha_w, _N,
                block_size=None):
    """Output-driven arrangement for fractional max pooling 2D.

    input, random_samples: pass through as source tensors for .data_ptr()/.stride().
    output, indices: flattened and tiled for parallel output.
    Remaining: constexpr params passed through.
    """
    if block_size is None:
        block_size = ninetoothed.block_size()

    out_arr = output.flatten().tile((block_size,))
    idx_arr = indices.flatten().tile((block_size,))

    return (input, random_samples, out_arr, idx_arr,
            _kH, _kW, _Hi, _Wi, _Ho, _Wo,
            _C, _alpha_h, _alpha_w, _N)


def application(input, random_samples, output, indices,
                kH, kW, Hi, Wi, Ho, Wo, C, alpha_h, alpha_w, N):
    """Vectorized: each lane computes one output element.

    Uses per-dimension offsets from the arranged output tensor to decode
    (n, c, oh, ow).  Loads random_samples via pointer arithmetic,
    computes pool_start with the PyTorch 2.12 CUDA formula, scans the
    input window, and writes max value + flat spatial index.
    """
    n = output.offsets(0)
    c = output.offsets(1)
    oh = output.offsets(2)
    ow = output.offsets(3)

    # Padding lane mask: filter out lanes beyond tensor bounds.
    valid = (n < N) & (c < C) & (oh < Ho) & (ow < Wo)

    # --- Load random samples via pointer arithmetic ---
    rs_ptr = random_samples.data_ptr()
    rs_str_n = random_samples.stride(0)
    rs_str_c = random_samples.stride(1)
    rs_str_s = random_samples.stride(2)

    rs_base = rs_ptr + n * rs_str_n + c * rs_str_c
    w_sample = ntl.load(rs_base, mask=valid, other=ntl.cast(0, ntl.float32))
    h_sample = ntl.load(rs_base + rs_str_s, mask=valid, other=ntl.cast(0, ntl.float32))

    # --- Compute pool start positions ---
    # PyTorch 2.12 CUDA formula.
    h_start = ntl.where(
        oh == Ho - 1,
        Hi - kH,
        ntl.cast((ntl.cast(oh, ntl.float32) + h_sample) * alpha_h, ntl.int32)
        - ntl.cast(h_sample * alpha_h, ntl.int32),
    )
    w_start = ntl.where(
        ow == Wo - 1,
        Wi - kW,
        ntl.cast((ntl.cast(ow, ntl.float32) + w_sample) * alpha_w, ntl.int32)
        - ntl.cast(w_sample * alpha_w, ntl.int32),
    )

    # --- Max over kH × kW window ---
    in_ptr = input.data_ptr()
    str_n = input.stride(0)
    str_c = input.stride(1)
    str_h = input.stride(2)
    str_w = input.stride(3)

    window_base = in_ptr + n * str_n + c * str_c + h_start * str_h + w_start * str_w

    max_val = ntl.load(window_base, mask=valid, other=ntl.cast(float("-inf"), ntl.float32))
    max_idx = h_start * Wi + w_start

    for kh in range(kH):
        for kw in range(kW):
            ptr = window_base + kh * str_h + kw * str_w
            val = ntl.load(ptr, mask=valid, other=ntl.cast(float("-inf"), ntl.float32))
            # PyTorch 2.12 semantics: val > maxVal || isnan(val)
            better = (val > max_val) | (val != val)
            max_val = ntl.where(better, val, max_val)
            max_idx = ntl.where(
                better,
                (h_start + kh) * Wi + (w_start + kw),
                max_idx,
            )

    output = max_val  # noqa: F841
    indices = max_idx  # noqa: F841


def premake(kH, kW, H_in, W_in, H_out, W_out, C, alpha_h, alpha_w,
            N, dtype=None, block_size=None):
    """Create kernel factory for fractional max pooling 2D."""
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    tensors = (
        Tensor(4, shape=(N, C, H_in, W_in), dtype=dtype, other=float("-inf")),
        Tensor(3, shape=(N, C, 2), dtype=ninetoothed.float32),
        Tensor(4, shape=(N, C, H_out, W_out), dtype=dtype),
        Tensor(4, shape=(N, C, H_out, W_out), dtype=ninetoothed.int64),
        Tensor(0, constexpr=True, value=kH),
        Tensor(0, constexpr=True, value=kW),
        Tensor(0, constexpr=True, value=H_in),
        Tensor(0, constexpr=True, value=W_in),
        Tensor(0, constexpr=True, value=H_out),
        Tensor(0, constexpr=True, value=W_out),
        Tensor(0, constexpr=True, value=C),
        Tensor(0, constexpr=True, value=alpha_h),
        Tensor(0, constexpr=True, value=alpha_w),
        Tensor(0, constexpr=True, value=N),
    )

    return arrangement_, application, tensors


# ---- 3D fractional max pooling ------------------------------------------------


def arrangement_3d(input, random_samples, output, indices,
                   _kD, _kH, _kW, _Di, _Hi, _Wi, _Do, _Ho, _Wo, _C,
                   _alpha_d, _alpha_h, _alpha_w, _N,
                   block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()
    out_arr = output.flatten().tile((block_size,))
    idx_arr = indices.flatten().tile((block_size,))
    return (input, random_samples, out_arr, idx_arr,
            _kD, _kH, _kW, _Di, _Hi, _Wi, _Do, _Ho, _Wo, _C,
            _alpha_d, _alpha_h, _alpha_w, _N)


def application_3d(input, random_samples, output, indices,
                   kD, kH, kW, Di, Hi, Wi, Do, Ho, Wo, C,
                   alpha_d, alpha_h, alpha_w, N):
    """Vectorized fractional max pool 3D."""
    n = output.offsets(0)
    c = output.offsets(1)
    od = output.offsets(2)
    oh = output.offsets(3)
    ow = output.offsets(4)

    # Padding lane mask.
    valid = (n < N) & (c < C) & (od < Do) & (oh < Ho) & (ow < Wo)

    # Load random samples via pointer arithmetic.
    rs_ptr = random_samples.data_ptr()
    rs_str_n = random_samples.stride(0)
    rs_str_c = random_samples.stride(1)
    rs_str_s = random_samples.stride(2)

    rs_base = rs_ptr + n * rs_str_n + c * rs_str_c
    d_sample = ntl.load(rs_base, mask=valid, other=ntl.cast(0, ntl.float32))
    h_sample = ntl.load(rs_base + rs_str_s, mask=valid, other=ntl.cast(0, ntl.float32))
    w_sample = ntl.load(rs_base + 2 * rs_str_s, mask=valid, other=ntl.cast(0, ntl.float32))

    # Compute pool start positions.
    d_start = ntl.where(
        od == Do - 1, Di - kD,
        ntl.cast((ntl.cast(od, ntl.float32) + d_sample) * alpha_d, ntl.int32)
        - ntl.cast(d_sample * alpha_d, ntl.int32),
    )
    h_start = ntl.where(
        oh == Ho - 1, Hi - kH,
        ntl.cast((ntl.cast(oh, ntl.float32) + h_sample) * alpha_h, ntl.int32)
        - ntl.cast(h_sample * alpha_h, ntl.int32),
    )
    w_start = ntl.where(
        ow == Wo - 1, Wi - kW,
        ntl.cast((ntl.cast(ow, ntl.float32) + w_sample) * alpha_w, ntl.int32)
        - ntl.cast(w_sample * alpha_w, ntl.int32),
    )

    # Max over kD × kH × kW window.
    in_ptr = input.data_ptr()
    str_n = input.stride(0)
    str_c = input.stride(1)
    str_d = input.stride(2)
    str_h = input.stride(3)
    str_w = input.stride(4)

    plane_size = Hi * Wi
    window_base = (in_ptr + n * str_n + c * str_c
                   + d_start * str_d + h_start * str_h + w_start * str_w)

    max_val = ntl.load(window_base, mask=valid, other=ntl.cast(float("-inf"), ntl.float32))
    max_idx = d_start * plane_size + h_start * Wi + w_start

    for kd in range(kD):
        for kh in range(kH):
            for kw in range(kW):
                ptr = window_base + kd * str_d + kh * str_h + kw * str_w
                val = ntl.load(ptr, mask=valid, other=ntl.cast(float("-inf"), ntl.float32))
                better = (val > max_val) | (val != val)
                max_val = ntl.where(better, val, max_val)
                max_idx = ntl.where(
                    better,
                    (d_start + kd) * plane_size + (h_start + kh) * Wi + (w_start + kw),
                    max_idx,
                )

    output = max_val  # noqa: F841
    indices = max_idx  # noqa: F841


def premake_3d(kD, kH, kW, D_in, H_in, W_in, D_out, H_out, W_out, C,
               alpha_d, alpha_h, alpha_w, N, dtype=None, block_size=None):
    """Create kernel factory for fractional max pooling 3D."""
    arrangement_ = functools.partial(arrangement_3d, block_size=block_size)

    tensors = (
        Tensor(5, shape=(N, C, D_in, H_in, W_in), dtype=dtype, other=float("-inf")),
        Tensor(3, shape=(N, C, 3), dtype=ninetoothed.float32),
        Tensor(5, shape=(N, C, D_out, H_out, W_out), dtype=dtype),
        Tensor(5, shape=(N, C, D_out, H_out, W_out), dtype=ninetoothed.int64),
        Tensor(0, constexpr=True, value=kD),
        Tensor(0, constexpr=True, value=kH),
        Tensor(0, constexpr=True, value=kW),
        Tensor(0, constexpr=True, value=D_in),
        Tensor(0, constexpr=True, value=H_in),
        Tensor(0, constexpr=True, value=W_in),
        Tensor(0, constexpr=True, value=D_out),
        Tensor(0, constexpr=True, value=H_out),
        Tensor(0, constexpr=True, value=W_out),
        Tensor(0, constexpr=True, value=C),
        Tensor(0, constexpr=True, value=alpha_d),
        Tensor(0, constexpr=True, value=alpha_h),
        Tensor(0, constexpr=True, value=alpha_w),
        Tensor(0, constexpr=True, value=N),
    )

    return arrangement_, application_3d, tensors
