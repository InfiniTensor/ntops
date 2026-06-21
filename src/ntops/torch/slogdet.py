import torch

import ntops
from ntops.torch.utils import _cached_make


def _next_power_of_2(x):
    return 1 << (int(x) - 1).bit_length()


def _to_scalar(tensor):
    # PyTorch Tensor 路径
    try:
        return tensor[0, 0]
    except Exception:
        pass

    # infinicore Tensor 路径：优先尝试 reshape 成 0-d
    try:
        return tensor.reshape(())
    except Exception:
        pass

    try:
        return torch.reshape(tensor, ())
    except Exception:
        pass

    # 如果没有 reshape，就尝试 squeeze
    try:
        return tensor.squeeze()
    except Exception:
        pass

    try:
        return torch.squeeze(tensor)
    except Exception:
        pass

    # 最后兜底：返回 1x1 buffer
    return tensor


def slogdet(input):
    if input.ndim != 2:
        raise NotImplementedError("ntops slogdet currently supports 2D matrices only")

    if input.shape[0] != input.shape[1]:
        raise RuntimeError("slogdet expects a square matrix")

    matrix_size = int(input.shape[0])
    block_size = _next_power_of_2(matrix_size)

    sign_buffer = torch.empty((1, 1), dtype=input.dtype, device=input.device)
    logabsdet_buffer = torch.empty((1, 1), dtype=input.dtype, device=input.device)

    kernel = _cached_make(
        ntops.kernels.slogdet.premake,
        input.ndim,
        matrix_size,
        block_size,
    )

    kernel(input, sign_buffer, logabsdet_buffer)

    return _to_scalar(sign_buffer), _to_scalar(logabsdet_buffer)