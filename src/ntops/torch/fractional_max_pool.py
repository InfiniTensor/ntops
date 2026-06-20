import torch

import ntops
from ntops.torch.utils import _cached_make


def fractional_max_pool2d(
    input, kernel_size, output_size=None, output_ratio=None,
    return_indices=False, _random_samples=None,
):
    """Fractional max pooling 2D — NineToothed kernel.

    The kernel computes pool start positions from random_samples
    internally.  Everything runs on-device; no CPU pre-computation
    of window positions.

    Equivalent to torch.nn.functional.fractional_max_pool2d.
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    kH, kW = kernel_size

    # Validate output_size and output_ratio are mutually exclusive.
    if output_size is None and output_ratio is None:
        raise ValueError(
            "fractional_max_pool2d requires either output_size or output_ratio"
        )
    if output_size is not None and output_ratio is not None:
        raise ValueError(
            "fractional_max_pool2d accepts only one of output_size, output_ratio"
        )

    # Handle unbatched (C, H, W) → (1, C, H, W).
    unbatched = input.dim() == 3
    if unbatched:
        input = input.unsqueeze(0)

    N, C, H_in, W_in = input.shape

    if output_size is None:
        assert output_ratio is not None
        if isinstance(output_ratio, (int, float)):
            output_ratio = (output_ratio, output_ratio)
        H_out = int(H_in * output_ratio[0])
        W_out = int(W_in * output_ratio[1])
    else:
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        H_out, W_out = output_size

    # Pre-compute alpha values (float division, done once on CPU).
    alpha_h = (H_in - kH) / (H_out - 1) if H_out > 1 else 0.0
    alpha_w = (W_in - kW) / (W_out - 1) if W_out > 1 else 0.0

    if _random_samples is None:
        _random_samples = torch.rand(N, C, 2, device=input.device)
    elif unbatched and _random_samples.dim() == 2:
        _random_samples = _random_samples.unsqueeze(0)

    output = torch.empty(N, C, H_out, W_out, dtype=input.dtype, device=input.device)
    indices = torch.empty(N, C, H_out, W_out, dtype=torch.int64, device=input.device)

    kernel = _cached_make(
        ntops.kernels.fractional_max_pool.premake,
        kH=kH, kW=kW,
        H_in=H_in, W_in=W_in, H_out=H_out, W_out=W_out,
        C=C, N=N, alpha_h=alpha_h, alpha_w=alpha_w,
        block_size=64, max_num_configs=1,
    )

    kernel(input, _random_samples, output, indices,
           kH, kW, H_in, W_in, H_out, W_out, C, alpha_h, alpha_w, N)

    if unbatched:
        output = output.squeeze(0)
        indices = indices.squeeze(0)

    if return_indices:
        return output, indices
    return output


def fractional_max_pool3d(
    input, kernel_size, output_size=None, output_ratio=None,
    return_indices=False, _random_samples=None,
):
    """Fractional max pooling 3D — NineToothed kernel.

    Equivalent to torch.nn.functional.fractional_max_pool3d.
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    kD, kH, kW = kernel_size

    if output_size is None and output_ratio is None:
        raise ValueError(
            "fractional_max_pool3d requires either output_size or output_ratio"
        )
    if output_size is not None and output_ratio is not None:
        raise ValueError(
            "fractional_max_pool3d accepts only one of output_size, output_ratio"
        )

    N, C, D_in, H_in, W_in = input.shape

    if output_size is None:
        assert output_ratio is not None
        if isinstance(output_ratio, (int, float)):
            output_ratio = (output_ratio, output_ratio, output_ratio)
        D_out = int(D_in * output_ratio[0])
        H_out = int(H_in * output_ratio[1])
        W_out = int(W_in * output_ratio[2])
    else:
        if isinstance(output_size, int):
            output_size = (output_size, output_size, output_size)
        D_out, H_out, W_out = output_size

    alpha_d = (D_in - kD) / (D_out - 1) if D_out > 1 else 0.0
    alpha_h = (H_in - kH) / (H_out - 1) if H_out > 1 else 0.0
    alpha_w = (W_in - kW) / (W_out - 1) if W_out > 1 else 0.0

    if _random_samples is None:
        _random_samples = torch.rand(N, C, 3, device=input.device)

    output = torch.empty(N, C, D_out, H_out, W_out, dtype=input.dtype, device=input.device)
    indices = torch.empty(N, C, D_out, H_out, W_out, dtype=torch.int64, device=input.device)

    kernel = _cached_make(
        ntops.kernels.fractional_max_pool.premake_3d,
        kD=kD, kH=kH, kW=kW,
        D_in=D_in, H_in=H_in, W_in=W_in,
        D_out=D_out, H_out=H_out, W_out=W_out,
        C=C, N=N, alpha_d=alpha_d, alpha_h=alpha_h, alpha_w=alpha_w,
        block_size=64, max_num_configs=1,
    )

    kernel(input, _random_samples, output, indices,
           kD, kH, kW, D_in, H_in, W_in, D_out, H_out, W_out, C,
           alpha_d, alpha_h, alpha_w, N)

    if return_indices:
        return output, indices
    return output
