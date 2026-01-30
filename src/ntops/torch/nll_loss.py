
import math
import torch
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

def nll_loss(input, target, weight=None, ignore_index=-100, reduction='mean'):
    N, C = input.shape
    C_pow2 = 1 << (C - 1).bit_length()
    device = input.device
    dtype = input.dtype
    
    def iterative_reduce(current, name=""):
        step = 0
        while current.numel() > 1:
            block_size = get_optimal_block_size(current.numel())
            output_len = math.ceil(current.numel() / block_size)
            output = torch.empty((output_len,), dtype=dtype, device=device)
            
            kernel_reduce = _cached_make(
                ntops.kernels.nll_loss.premake_reduce,
                dtype,
                block_size
            )
            kernel_reduce(current, output)
            current = output
            step += 1
        return current

    tmp_loss_sum = torch.zeros((N,), dtype=dtype, device=device)
    tmp_weight_sum = torch.zeros((N,), dtype=dtype, device=device)

    if weight is None:
        dummy_weight = torch.empty_like(target)
        has_weight = False
    else:
        dummy_weight = weight.contiguous()
        has_weight = True
    
    kernel_loss = _cached_make(
        ntops.kernels.nll_loss.premake_loss,
        int(ignore_index),
        C,
        C_pow2,
        has_weight,
        dtype
    )
    kernel_loss(input, target, tmp_loss_sum, tmp_weight_sum, dummy_weight, int(ignore_index), C, C_pow2, has_weight)
    
    if reduction == 'none':
        return tmp_loss_sum

    final_loss_tensor = iterative_reduce(tmp_loss_sum, "Loss")
    final_weight_tensor = iterative_reduce(tmp_weight_sum, "Weight")
    
    loss_val = final_loss_tensor.view(())
    
    if reduction == 'sum':
        return loss_val
    
    elif reduction == 'mean':
        final_output = torch.empty((1,), dtype=dtype, device=device)
        kernel_div = _cached_make(
            ntops.kernels.nll_loss.premake_div,
            dtype
        )
        kernel_div(final_loss_tensor, final_weight_tensor, final_output)

        return final_output
