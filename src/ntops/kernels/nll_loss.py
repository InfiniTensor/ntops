import functools
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

def arrangement_loss(input, target, out_loss, out_weight, weight, ignore_index, C_val_Tensor, C_pow2_Tensor, has_weight_Tensor, C_pow2):
    input_t = input.tile((1, C_pow2)) 
    target_t = target.tile((1,))
    out_l_t = out_loss.tile((1,))
    out_w_t = out_weight.tile((1,))
    
    weight_t = weight.tile((C_pow2,)).expand((input_t.shape[0],))

    return input_t, target_t, out_l_t, out_w_t, weight_t, ignore_index, C_val_Tensor, C_pow2_Tensor, has_weight_Tensor

def application_loss(input, target, out_loss, out_weight, weight, ignore_index, C_val, C_pow2, has_weight):
    ii = ntl.cast(ignore_index, ntl.int32)
    C_num = ntl.cast(C_val, ntl.int32)
    
    col_indices = ntl.arange(0, C_pow2)
    t_val = ntl.cast(target, ntl.int32)

    match_mask = (t_val == col_indices) & (col_indices < C_num)
    is_ignore = (t_val == ii) | (t_val < 0) | (t_val >= C_num)

    input_f32 = ntl.cast(input, ntl.float32)
    prob_vec = ntl.where(match_mask, input_f32, 0.0)
    selected_prob = ntl.sum(prob_vec)

    if has_weight == True:
        weight_f32 = ntl.cast(weight, ntl.float32)
        weight_vec = ntl.where(match_mask, weight_f32, 0.0)
        selected_weight = ntl.sum(weight_vec)
    else:
        selected_weight = 1.0

    loss_val = 0.0 - (selected_prob * selected_weight)

    out_loss = ntl.where(is_ignore, 0.0, loss_val)
    out_weight = ntl.where(is_ignore, 0.0, selected_weight)


def premake_loss(ignore_index, C_val, C_pow2, has_weight, dtype=None):
    arrangement_ = functools.partial(arrangement_loss, C_pow2=C_pow2)

    tensors = (
        Tensor(2, dtype=dtype, other=0.0),      # input
        Tensor(1, dtype=ninetoothed.int64),     # target
        Tensor(1, dtype=dtype),                 # out_loss
        Tensor(1, dtype=dtype),                 # out_weight
        Tensor(1, dtype=dtype, other=0.0),      # weight
        Tensor(0, constexpr=True, value=ignore_index),
        Tensor(0, constexpr=True, value=C_val),
        Tensor(0, constexpr=True, value=C_pow2),
        Tensor(0, constexpr=True, value=has_weight),
    )
    return arrangement_, application_loss, tensors

def arrangement_reduce(input, output, block_size):
    input_t = input.tile((block_size,))
    output_t = output.tile((1,))
    return input_t, output_t

def application_reduce(input, output):
    accumulator = 0.0
    for i in range(input.shape[0]):
        accumulator += ntl.cast(input[i], ntl.float32)
    output[0] = ntl.cast(accumulator, output.dtype)

def premake_reduce(dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement_reduce, block_size=block_size)
    tensors = (
        Tensor(1, dtype=dtype, shape_options={"constexpr": True}),      # input
        Tensor(1, dtype=dtype, shape_options={"constexpr": True}),      # output
    )
    
    return arrangement_, application_reduce, tensors

def arrangement_div(loss_sum, weight_sum, output):
    return loss_sum.tile((1,)), weight_sum.tile((1,)), output.tile((1,))

def application_div(loss_sum, weight_sum, output):
    l = ntl.cast(loss_sum[0], ntl.float32)
    w = ntl.cast(weight_sum[0], ntl.float32)
    res = ntl.where(w > 0.0, l / w, 0.0)
    output[0] = ntl.cast(res, output.dtype)

def premake_div(dtype=None):
    arrangement_ = functools.partial(arrangement_div)
    tensors = (
        Tensor(1, dtype=dtype),      # loss_sum
        Tensor(1, dtype=dtype),      # weight_sum
        Tensor(1, dtype=dtype),      # output
    )
    return arrangement_, application_div, tensors
