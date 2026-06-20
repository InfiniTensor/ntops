import torch

import ntops
from ntops.torch.utils import _cached_make


def scatter_add(input, dim, index, src):
    """Scatter add along a dimension.

    Equivalent to input.scatter_add_(dim, index, src) but non-inplace.

    Uses a NineToothed kernel with ntl.atomic_add for scatter.
    Fixed block_size=128 and max_num_configs=1 are required because
    autotuning warmup benchmarks would repeatedly add to the same
    output, corrupting the atomic accumulation.
    """
    _validate_scatter_args(input, dim, index, src)

    # Normalize dim.
    if dim < 0:
        dim = input.ndim + dim

    if index.numel() == 0:
        return input.clone()

    # Initialize output as a clone of input — this is fresh every call,
    # which is essential because atomic kernels are not idempotent.
    # clone() is always contiguous, so strides match shape semantics.
    output = input.clone()

    # Slice src to match index shape if src is larger.
    region = tuple(slice(0, s) for s in index.shape)
    src_sliced = src[region].contiguous()

    # Use output strides (correct for any layout, even if clone were non-contiguous).
    strides = output.stride()

    flat_src = src_sliced.reshape(-1)
    flat_index = torch.zeros(index.numel(), dtype=torch.int64, device=input.device)

    for d in range(input.ndim):
        if d == dim:
            flat_index += index.reshape(-1).to(torch.int64) * strides[d]
        else:
            coord_d = (
                torch.arange(index.shape[d], device=input.device)
                .reshape([-1 if i == d else 1 for i in range(index.ndim)])
                .expand_as(index)
                .reshape(-1)
            )
            flat_index += coord_d * strides[d]

    # Use fixed block_size=128 (not auto-tuned) and max_num_configs=1
    # to prevent autotuning warmup from corrupting atomic output.
    kernel = _cached_make(
        ntops.kernels.scatter_add.premake, block_size=128, max_num_configs=1
    )

    kernel(flat_index, flat_src, output)

    return output


def _validate_scatter_args(input, dim, index, src):
    """Validate arguments match PyTorch scatter_add semantics."""
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"input must be a Tensor, got {type(input).__name__}")
    if not isinstance(index, torch.Tensor):
        raise TypeError(f"index must be a Tensor, got {type(index).__name__}")
    if not isinstance(src, torch.Tensor):
        raise TypeError(f"src must be a Tensor, got {type(src).__name__}")

    ndim = input.ndim
    if dim < -ndim or dim >= ndim:
        raise IndexError(
            f"dim {dim} out of range for input of {ndim} dimensions"
        )

    if index.dtype != torch.int64:
        raise TypeError(
            f"index dtype must be torch.int64 (long), got {index.dtype}"
        )

    if index.ndim != ndim:
        raise ValueError(
            f"index ndim ({index.ndim}) must match input ndim ({ndim})"
        )
    if src.ndim != ndim:
        raise ValueError(
            f"src ndim ({src.ndim}) must match input ndim ({ndim})"
        )

    if src.device != input.device or index.device != input.device:
        raise ValueError(
            "input, index, and src must be on the same device"
        )

    if src.dtype != input.dtype:
        raise TypeError(
            f"src dtype ({src.dtype}) must match input dtype ({input.dtype})"
        )

    for d in range(ndim):
        if d != dim and index.shape[d] > input.shape[d]:
            raise ValueError(
                f"index.shape[{d}] ({index.shape[d]}) must be <= "
                f"input.shape[{d}] ({input.shape[d]})"
            )
        if index.shape[d] > src.shape[d]:
            raise ValueError(
                f"index.shape[{d}] ({index.shape[d]}) must be <= "
                f"src.shape[{d}] ({src.shape[d]})"
            )

    if index.numel() > 0:
        min_val = index.min().item()
        max_val = index.max().item()
        normalized_dim = dim if dim >= 0 else dim + ndim
        if min_val < 0 or max_val >= input.shape[normalized_dim]:
            raise IndexError(
                f"index values must be in [0, {input.shape[normalized_dim]}), "
                f"got [{min_val}, {max_val}]"
            )
