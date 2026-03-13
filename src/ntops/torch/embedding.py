import torch
import numpy as np
import ctypes
import ntops
from ntops.torch.utils import _cached_make


def _get_dtype_info(dtype):
    dtype_str = str(dtype)
    for prefix in ("infinicore.", "torch."):
        if dtype_str.startswith(prefix):
            dtype_str = dtype_str[len(prefix) :]
            break
    info = _DTYPE_MAP.get(dtype_str)
    if info is None:
        raise RuntimeError(f"Unsupported dtype: {dtype}")
    return info


def _is_cpu(tensor):
    device_str = str(tensor.device).lower()
    return "cpu" in device_str


def _get_cpu_device():
    return torch.device("cpu", 0)


_DTYPE_MAP = {
    "float16": (np.float16, 2),
    "float32": (np.float32, 4),
    "float64": (np.float64, 8),
    "bfloat16": (None, 2),
    "int8": (np.int8, 1),
    "int16": (np.int16, 2),
    "int32": (np.int32, 4),
    "int64": (np.int64, 8),
    "uint8": (np.uint8, 1),
}


def _bf16_to_fp32(bf16_uint16_arr):
    u32 = bf16_uint16_arr.astype(np.uint32) << 16
    return u32.view(np.float32)


def _fp32_to_bf16(fp32_arr):
    u32 = fp32_arr.view(np.uint32)
    u16 = (u32 >> 16).astype(np.uint16)
    return u16


def _tensor_to_numpy(tensor):
    tensor = tensor.contiguous()
    _, elem_bytes = _get_dtype_info(tensor.dtype)
    numel = tensor.numel()
    total_bytes = numel * elem_bytes
    shape = list(tensor.shape)

    dtype_str = str(tensor.dtype)
    for prefix in ("infinicore.", "torch."):
        if dtype_str.startswith(prefix):
            dtype_str = dtype_str[len(prefix) :]
            break
    is_bf16 = dtype_str == "bfloat16"
    np_dtype = _DTYPE_MAP[dtype_str][0]

    if _is_cpu(tensor):
        ptr = tensor.data_ptr()
        buf = (ctypes.c_byte * total_bytes).from_address(ptr)
        if is_bf16:
            raw = np.frombuffer(buf, dtype=np.uint16).copy()
            np_arr = _bf16_to_fp32(raw)
        else:
            np_arr = np.frombuffer(buf, dtype=np_dtype).copy()
        return np_arr.reshape(shape)
    else:
        cpu_dev = _get_cpu_device()
        cpu_tensor = torch.empty(shape, dtype=tensor.dtype, device=cpu_dev)
        cpu_tensor.copy_(tensor)
        ptr = cpu_tensor.data_ptr()
        buf = (ctypes.c_byte * total_bytes).from_address(ptr)
        if is_bf16:
            raw = np.frombuffer(buf, dtype=np.uint16).copy()
            np_arr = _bf16_to_fp32(raw)
        else:
            np_arr = np.frombuffer(buf, dtype=np_dtype).copy()
        return np_arr.reshape(shape)


def _numpy_to_tensor(np_arr, device, orig_dtype=None):
    dtype_str = ""
    if orig_dtype is not None:
        dtype_str = str(orig_dtype)
        for prefix in ("infinicore.", "torch."):
            if dtype_str.startswith(prefix):
                dtype_str = dtype_str[len(prefix) :]
                break

    shape = list(np_arr.shape)
    cpu_dev = _get_cpu_device()

    if dtype_str == "bfloat16":
        bf16_raw = _fp32_to_bf16(np_arr.astype(np.float32))
        cpu_result = torch.empty(shape, dtype=orig_dtype, device=cpu_dev)
        ptr = cpu_result.data_ptr()
        ctypes.memmove(ptr, bf16_raw.ctypes.data, bf16_raw.nbytes)
        if _is_cpu_device(device):
            return cpu_result
        else:
            result = torch.empty(shape, dtype=orig_dtype, device=device)
            result.copy_(cpu_result)
            return result
    else:
        cpu_tensor = torch.from_numpy(np_arr.copy())
        if _is_cpu_device(device):
            return cpu_tensor
        else:
            return cpu_tensor.to(device)


def _is_cpu_device(device):
    return "cpu" in str(device).lower()


def _unique(input, sorted=True, return_inverse=False, return_counts=False):
    orig_device = input.device
    np_input = _tensor_to_numpy(input).reshape(-1)

    np_results = np.unique(
        np_input,
        return_inverse=return_inverse,
        return_counts=return_counts,
    )

    if return_inverse and return_counts:
        np_unique, np_inverse, np_counts = np_results
    elif return_inverse:
        np_unique, np_inverse = np_results
    elif return_counts:
        np_unique, np_counts = np_results
    else:
        np_unique = np_results

    if not sorted:
        first_indices = np.array(
            [np.nonzero(np_input == val)[0][0] for val in np_unique]
        )
        order = np.argsort(first_indices)
        np_unique = np_unique[order]
        if return_inverse:
            inv_order = np.empty_like(order)
            inv_order[order] = np.arange(len(order))
            np_inverse = inv_order[np_inverse]
        if return_counts:
            np_counts = np_counts[order]

    unique_tensor = _numpy_to_tensor(np_unique, orig_device)

    result = (unique_tensor,)
    if return_inverse:
        result += (_numpy_to_tensor(np_inverse.astype(np.int64), orig_device),)
    if return_counts:
        result += (_numpy_to_tensor(np_counts.astype(np.int64), orig_device),)

    return result if len(result) > 1 else result[0]


def _gather_by_indices(temp_out, inverse_indices, out_flat):
    """
    temp_out:         [M, D] tensor
    inverse_indices:  [N]    tensor (int64)
    out_flat:         [N, D] tensor
    """
    np_temp = _tensor_to_numpy(temp_out)  # shape: [M, D]
    np_inv = _tensor_to_numpy(inverse_indices)  # shape: [N]
    np_inv = np_inv.astype(np.int64)
    np_result = np_temp[np_inv]  # shape: [N, D]
    result_tensor = _numpy_to_tensor(
        np_result, out_flat.device, orig_dtype=out_flat.dtype
    )
    out_flat.copy_(result_tensor)


def embedding(input, weight, out=None, max_norm=None, norm_type=2.0):
    if out is None:
        out_shape = list(input.shape) + [weight.shape[1]]
        out = torch.empty(out_shape, dtype=weight.dtype, device=input.device)

    # kernel = _cached_make(ntops.kernels.embedding.premake, input.dim(), weight.shape[0], weight.shape[1], block_size_m=4, block_size_n=4)
    # kernel(input, weight, out, max_norm, norm_type)

    # Find unique indices to reduce redundant computations, then map back
    # to original output positions. This is especially beneficial when input
    # contains many repeated indices. Otherwise, data races in parallelism will cause errors.
    unique_indices, inverse_indices = _unique(
        input.view([input.numel()]), return_inverse=True
    )
    temp_out = torch.empty(
        [unique_indices.shape[0], weight.shape[1]],
        dtype=weight.dtype,
        device=input.device,
    )
    if max_norm is None:
        kernel = _cached_make(
            ntops.kernels.embedding.premake_without_norm,
            len(unique_indices.shape),
            dtype=weight.dtype,
            block_size_m=4,
            block_size_n=4,
        )
        kernel(unique_indices, weight, temp_out)
    else:
        kernel = _cached_make(
            ntops.kernels.embedding.premake,
            len(unique_indices.shape),
            embedding_dim=weight.shape[1],
            dtype=weight.dtype,
            block_size_m=4,
            block_size_n=4,
        )
        kernel(unique_indices, weight, temp_out, max_norm, norm_type)

    # out_flat = out.view(-1, weight.shape[1])
    out_flat = out.view([input.numel(), weight.shape[1]])
    # out_flat[:] = temp_out[inverse_indices]
    _gather_by_indices(temp_out, inverse_indices, out_flat)
    return out
