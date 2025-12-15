
import functools

from ninetoothed import Tensor
import ninetoothed.language as ntl

def arrangement(input, output, index, T, S, T_pow2, S_pow2, dim, block_size=None):
    non_target_dim = tuple(i for i in range(input.ndim) if i != dim)
    input = input.permute(non_target_dim + (dim,))
    input = input.flatten(end_dim=-1) # shape: (..., T)

    output = output.permute(non_target_dim + (dim,))
    output = output.flatten(end_dim=-1) # shape: (..., S)

    # input: (..., T)
    # output: (..., S)
    # index: (S,)
    input_tiled = input.tile((block_size, T_pow2)).squeeze(1) # shape: (..., ), dtype=(block_size, T_pow2)
    output_tiled = output.tile((block_size, S_pow2)).squeeze(1) # shape: (..., ), dtype=(block_size, S_pow2)

    index_expand = index.unsqueeze(0).expand((input_tiled.shape[0], -1)) # shape: (..., S)
    index_expand = index_expand.tile((1, S_pow2)).squeeze(1) # shape: (..., ), dtype=(1, S_pow2)

    return input_tiled, output_tiled, index_expand, T, S

# def application(input, output, index):
#     # input: (block_size, T)
#     # output: (block_size, S)
#     # index: (1, S)
#     # 使用 gather 实现 index_select
#     # Triton 3.0.0 不支持 gather 操作，因此在摩尔线程中无法使用
#     # 这里仅作为参考
#     index_expand = ntl.broadcast_to(index, (input.shape[0], index.shape[1]))
#     # index_expand: (block_size, S)
#     output = ntl.gather(input, index, axis=1)

def application(input, output, index, T, S):
    # input: (block_size, T_pow2)
    # output: (block_size, S_pow2)
    # index: (1, S_pow2)

    # 使用 T_pow2 满足 arange 的 2 次幂要求
    col_indices = ntl.arange(0, input.shape[1])  # shape: (T_pow2,)

    # 添加维度并广播到 (block_size, S, T_pow2)
    col_indices = ntl.expand_dims(col_indices, 0)  # shape: (1, T_pow2)
    col_indices = ntl.expand_dims(col_indices, 0)  # shape: (1, 1, T_pow2)
    col_indices = ntl.broadcast_to(col_indices, (input.shape[0], output.shape[1], input.shape[1]))

    # 扩展 input 到 (block_size, S, T_pow2)
    input_expanded = ntl.expand_dims(input, 1)  # shape: (block_size, 1, T_pow2)
    input_expanded = ntl.broadcast_to(input_expanded, (input.shape[0], output.shape[1], input.shape[1]))

    # 扩展 index 到 (block_size, S, T_pow2)
    index_expanded = ntl.expand_dims(index, 2)  # shape: (block_size, S, 1)
    index_expanded = ntl.broadcast_to(index_expanded, (input.shape[0], output.shape[1], input.shape[1]))

    # 仅在有效列范围内匹配，超出原始 T 的部分屏蔽
    col_valid = col_indices < input.shape[1]
    match_mask = (col_indices == index_expanded)
    mask = ntl.where(col_valid, match_mask, False)

    # 使用 where 选择对应的值
    selected = ntl.where(mask, input_expanded, 0.0)  # shape: (block_size, S, T_pow2)

    # 对最后一个维度求和得到结果
    result = ntl.sum(selected, axis=2)  # shape: (block_size, S)

    # 写回输出
    output = result

def premake(ndim, dim, T_pow2, S_pow2, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, dim=dim, T_pow2=T_pow2, S_pow2=S_pow2, block_size=block_size)

    tensors = (
        Tensor(ndim, dtype=dtype, other=0, shape_options={'constexpr': True}),
        Tensor(ndim, dtype=dtype, other=0, shape_options={'constexpr': True}),
        Tensor(1, dtype=int, shape_options={'constexpr': True}),
        Tensor(0, dtype=int, constexpr=True),  # T
        Tensor(0, dtype=int, constexpr=True),  # S
    )

    return arrangement_, application, tensors
