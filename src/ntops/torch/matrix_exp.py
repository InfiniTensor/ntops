import torch
import ntops
from ntops.torch.utils import _cached_make, _get_matmul_input_precision


# def matrix_exp(input):
#     output = torch.empty_like(input)

#     c = [
#         1.0,
#         12.0,
#         66.0,
#         220.0,
#         495.0,
#         792.0,
#         924.0
#     ]

#     N_6 = torch.empty_like(input)
#     temp_N_6 = torch.empty_like(input)
#     D_6 = torch.empty_like(input)
#     temp_D_6 = torch.empty_like(input)
#     temp_up1 = torch.empty_like(input)
#     temp_up2 = torch.empty_like(input)
#     temp_down1 = torch.empty_like(input)
#     temp_down2 = torch.empty_like(input)
#     I = torch.eye(input.shape[-1], device=input.device, dtype=input.dtype).expand_as(input)


#     mm_kernel = _cached_make(ntops.kernels.mm.premake)
#     addmm_kernel = _cached_make(ntops.kernels.addmm.premake)
        
#     for i in range(7):
#         if i == 0:
#             N_6.copy_(I)
#             D_6.copy_(I)
#             temp_up1.copy_(I)
#             temp_down1.copy_(I)
#         else:
#             mm_kernel(input, N_6, temp_N_6, _get_matmul_input_precision())
#             N_6 = temp_N_6
#             addmm_kernel(temp_up1, I, N_6, 1.0, c[i], temp_up2, _get_matmul_input_precision())
#             temp_up1 = temp_up2

#             mm_kernel(input, D_6, temp_D_6, _get_matmul_input_precision())
#             D_6 = temp_D_6
#             addmm_kernel(temp_down1, I, D_6, 1.0, c[i] if i % 2 == 0 else -c[i], temp_down2, _get_matmul_input_precision())
#             temp_down1 = temp_down2
    
#     # div_kernel = _cached_make(ntops.kernels.div.premake, input.ndim, None)
#     # div_kernel(temp_up1, temp_down1, output)
#     torch.linalg.solve(temp_down1, temp_up1, out=output)
#     return output

def matrix_exp(A):
    """
    计算矩阵指数 exp(A) 使用 10 阶泰勒级数 + 缩放-平方方法
    
    exp(A) = I + A + A²/2! + A³/3! + ... + A¹⁰/10!
    
    缩放: A_scaled = A / 2^s, 其中 s = max(0, ceil(log2(||A||/theta)))
    平方还原: exp(A) = (exp(A_scaled))^(2^s)
    """
    original_dtype = A.dtype
    device = A.device
    
    if A.dtype in (torch.float16, torch.bfloat16):
        A = A.float()
    
    if A.ndim == 2:
        if A.shape[0] != A.shape[1]:
            raise RuntimeError(f"matrix_exp requires square matrix, got shape {A.shape}")
        batch_mode = False
        n = A.shape[0]
    elif A.ndim == 3:
        if A.shape[1] != A.shape[2]:
            raise RuntimeError(f"matrix_exp requires square matrices in batch, got shape {A.shape}")
        batch_mode = True
        batch_size = A.shape[0]
        n = A.shape[1]
    else:
        raise RuntimeError(f"matrix_exp expects 2D or 3D input, got ndim={A.ndim}")
    
    # ===== 缩放（reduce scaling norm） =====
    if batch_mode:
        # 对每个 batch 计算范数
        norm_A = torch.linalg.norm(A, ord=1, dim=(1, 2)).max()  # 取 batch 中的最大范数
    else:
        norm_A = torch.linalg.norm(A, ord=1)
    
    theta = 2.2  # 10 阶泰勒的最优阈值
    norm_val = norm_A.item()
    if norm_val <= theta:
        s_val = 0
    else:
        import math
        s_val = max(0, math.ceil(math.log2(norm_val / theta)))
    
    # 缩放矩阵
    A_scaled = A / (2.0 ** s_val)
    
    # ===== 初始化 =====
    dtype = A.dtype
    
    if batch_mode:
        I = torch.eye(n, dtype=dtype, device=device).unsqueeze(0).expand(batch_size, n, n)
    else:
        I = torch.eye(n, dtype=dtype, device=device)
    
    # ===== 10 阶泰勒级数计算 =====
    # exp(A_scaled) = sum_{k=0}^{10} A_scaled^k / k!
    
    result = I.clone()  # 初始值为 I (k=0 项)
    power_k = I.clone()  # 当前 A_scaled^k
    temp = I.clone()
    
    # 泰勒系数 1/k!
    factorial = [1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0, 362880.0, 3628800.0]
    for k in range(1, 11):
        power_k = ntops.torch.matmul(power_k, A_scaled)
        
        # result += power_k / k!
        coeff = 1.0 / factorial[k]
        # result = result + coeff * power_k
        if A.ndim == 2:
            result = ntops.torch.addmm(result, I, power_k, beta=1.0, alpha=coeff)
        else:
            # batch 模式下逐 batch 计算 addmm
            for b in range(batch_size):
                result[b] = ntops.torch.addmm(result[b], I[b], power_k[b], beta=1.0, alpha=coeff)
    
    # ===== 平方还原 exp(A) = (exp(A_scaled))^(2^s) =====
    for _ in range(s_val):
        temp = result
        result = ntops.torch.matmul(result, temp)
    
    # ===== 转换回原始类型 =====
    result = result.to(original_dtype)
    
    return result
