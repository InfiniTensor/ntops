import torch


def unbind(x, dim=0):
    """
    Remove a tensor dimension by returning all slices along that dimension.

    This is a wrapper around PyTorch's unbind function for compatibility.

    Args:
        x: Input tensor
        dim: Dimension to remove (default: 0)

    Returns:
        A tuple of tensors with the specified dimension removed

    Examples:
        >>> x = torch.randn(3, 4, 5)
        >>> result = ntops.torch.unbind(x, dim=1)
        >>> len(result)  # 4 (size of dim 1)
        >>> result[0].shape  # (3, 5) - dim 1 removed
        >>> result[1].shape  # (3, 5)
        >>> result[2].shape  # (3, 5)
        >>> result[3].shape  # (3, 5)
    """
    return torch.unbind(x, dim=dim)
