import torch
import math
import ntops
from ntops.torch.utils import _cached_make


def next_power_of_2(n):
    if n == 0:
        return 1
    return 1 << (n - 1).bit_length()


def get_optimal_block_size(dim_size):
    target_size = next_power_of_2(dim_size)

    if target_size > 1024:
        target_size = 1024

    if target_size < 32:
        target_size = 32

    return target_size


def cosine_embedding_loss(x1, x2, y, margin=0.0, reduction="mean"):
    embedding_dim = x1.shape[-1]
    dims = len(x1.shape)
    xshape = list(x1.shape)
    output = torch.empty(xshape[:-1], dtype=x1.dtype, device=x1.device)
    kernel_loss = _cached_make(
        ntops.kernels.cosine_embedding_loss.cosine_embedding_loss_premake,
        dims=dims,
        embedding_dim=embedding_dim,
        block_size=16,
    )
    kernel_loss(x1, x2, y, margin, output)

    if reduction == "none":
        return output
    else:
        cur = output
        block_size = get_optimal_block_size(cur.numel())
        while cur.numel() > 1:
            cur_output_len = math.ceil(cur.numel() / block_size)
            cur_output = torch.empty(
                (cur_output_len,), dtype=cur.dtype, device=cur.device
            )
            kernel_sum = _cached_make(
                ntops.kernels.cosine_embedding_loss.reduce_sum_premake,
                cur.ndim,
                cur.dtype,
                block_size,
            )
            kernel_sum(cur, cur_output)
            cur = cur_output
        res = cur.view(())
        res = cur.unsqueeze(0)
        if reduction == "mean":
            mean_out = torch.empty_like(res)
            kernel_div = _cached_make(
                ntops.kernels.cosine_embedding_loss.div_premake, res.ndim
            )
            other = output.numel()
            kernel_div(res, other, mean_out)
            mean_out = mean_out.view(())
            return mean_out
        else:
            res = res.view(())
            return res
