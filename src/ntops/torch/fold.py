import torch

import ntops
from ntops.torch.utils import _cached_make

def fold(input, output_size, kernel_size, dilation=1, padding=0, stride=1):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    # 记录原始输入维度
    input_was_2d = input.ndim == 2
    if input_was_2d:
        input = input.view((1, input.shape[0], input.shape[1]))

    N, Ckk, L = input.shape
    H_out, W_out = output_size
    K_h, K_w = kernel_size
    D_h, D_w = dilation
    P_h, P_w = padding
    S_h, S_w = stride

    # 验证和计算 L
    C = Ckk // (K_h * K_w)
    if C * K_h * K_w != Ckk:
        raise ValueError(f"Input channel dimension {Ckk} is not divisible by kernel size product {K_h * K_w}")

    L_h = (H_out + 2 * P_h - (D_h * (K_h - 1) + 1)) // S_h + 1
    L_w = (W_out + 2 * P_w - (D_w * (K_w - 1) + 1)) // S_w + 1
    if L != L_h * L_w:
        raise ValueError(f"Input L {L} != computed L_h*L_w {L_h * L_w}")

    # 创建带 padding 的输出张量
    out_padded_h = H_out + 2 * P_h
    out_padded_w = W_out + 2 * P_w
    out = torch.empty(
        (N, C, out_padded_h, out_padded_w),
        dtype=input.dtype,
        device=input.device
    )
    torch.nn.init.zeros_(out)

    # 创建并调用 kernel
    block_size = 128
    L_pow2 = 1 << (L - 1).bit_length()
    kernel = _cached_make(
        ntops.kernels.fold.premake,
        L_pow2,
        kernel_size[0],
        kernel_size[1],
        stride[0],
        stride[1],
        dilation[0],
        dilation[1],
        padding[0],
        padding[1],
        dtype=input.dtype,
        block_size=block_size
    )
    kernel(input, out, L)

    # 移除 padding
    result = out
    if P_h > 0 or P_w > 0:
        # 目前不支持直接切片，只能用 narrow 实现
        result = torch.narrow(result, 2, P_h, H_out)
        result = torch.narrow(result, 3, P_w, W_out)

    # 由于 ninetoothed 框架下难以实现原地 padding 的操作，因此这里创建新张量
    # 创建新张量接收结果，确保内存连续
    output = torch.empty(
        (N, C, H_out, W_out),
        dtype=input.dtype,
        device=input.device)
    torch.nn.init.zeros_(output)
    torch.add(output, result, out=output)

    if input_was_2d:
        output = output.view((output.shape[1], output.shape[2], output.shape[3]))

    return output