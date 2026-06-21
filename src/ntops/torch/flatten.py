import torch


def flatten(x, start_dim=0):
    """
    Flatten a tensor from start_dim onward.

    This is a wrapper around PyTorch's flatten function for compatibility.

    Args:
        x: Input tensor
        start_dim: First dimension to flatten (default: 0)
                  All dimensions from start_dim onward will be flattened
                  into a single dimension.

    Returns:
        A flattened tensor with the same data (view operation)

    Examples:
        >>> x = torch.randn(2, 3, 4)
        >>> flatten(x, start_dim=1).shape  # (2, 12)
        >>> flatten(x, start_dim=0).shape  # (24,)
        >>> flatten(x, start_dim=2).shape  # (2, 3, 4)
    """
    # Handle start_dim >= ndim case (return copy, like NumPy behavior)
    if start_dim >= x.ndim:
        return x.clone()

    return torch.flatten(x, start_dim=start_dim)
