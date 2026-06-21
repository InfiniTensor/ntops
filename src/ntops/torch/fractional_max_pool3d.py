import torch

import ntops
from ntops.torch.utils import _cached_make, _device_key

_CONFIGS = {
    "nvidia": (4, 256),
    "iluvatar": (8, 512),
    "metax": (4, 256),
    "default": (4, 256),
}


def fractional_max_pool3d(
    input,
    kernel_size,
    output_size=None,
    output_ratio=None,
    return_indices=False,
    _random_samples=None,
):
    assert not return_indices, "`return_indices` is not supported yet."

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    kernel_d, kernel_h, kernel_w = kernel_size

    input = input.contiguous()
    n, c, d, h, w = input.shape

    if output_size is not None:
        if isinstance(output_size, int):
            output_size = (output_size, output_size, output_size)
        out_d, out_h, out_w = output_size
    else:
        assert output_ratio is not None, "Either output_size or output_ratio is required."
        if isinstance(output_ratio, (int, float)):
            output_ratio = (output_ratio, output_ratio, output_ratio)
        out_d = int(d * output_ratio[0])
        out_h = int(h * output_ratio[1])
        out_w = int(w * output_ratio[2])

    if _random_samples is None:
        _random_samples = torch.rand(n, c, 3, dtype=input.dtype, device=input.device)

    alpha_d = float(d - kernel_d) / (out_d - 1) if out_d > 1 else 0.0
    alpha_h = float(h - kernel_h) / (out_h - 1) if out_h > 1 else 0.0
    alpha_w = float(w - kernel_w) / (out_w - 1) if out_w > 1 else 0.0

    m = n * c * out_d * out_h * out_w
    output = torch.empty((m,), dtype=input.dtype, device=input.device)

    num_warps, block_size = _CONFIGS[_device_key()]
    kernel = _cached_make(
        ntops.kernels.fractional_max_pool3d.premake,
        block_size=block_size,
        num_warps=num_warps,
        num_stages=1,
        max_num_configs=1,
    )
    kernel(
        output,
        input.reshape(-1),
        _random_samples.reshape(-1),
        alpha_d,
        alpha_h,
        alpha_w,
        d,
        h,
        w,
        out_d,
        out_h,
        out_w,
        kernel_d,
        kernel_h,
        kernel_w,
        m,
    )

    return output.reshape(n, c, out_d, out_h, out_w)
