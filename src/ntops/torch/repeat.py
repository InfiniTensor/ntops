import torch


def repeat(x, repeats):
    """
    Repeat a tensor along specified dimensions.

    This is a wrapper around PyTorch's repeat function for compatibility.

    Args:
        x: Input tensor
        repeats: List/tuple of repeat counts for each dimension

    Returns:
        A tensor with repeated elements

    Raises:
        ValueError: If repeats length doesn't match tensor dimensions

    Examples:
        >>> x = torch.tensor([[1, 2], [3, 4]])
        >>> ntops.torch.repeat(x, (2, 3))
        tensor([[1, 2, 1, 2, 1, 2],
                [3, 4, 3, 4, 3, 4],
                [1, 2, 1, 2, 1, 2],
                [3, 4, 3, 4, 3, 4]])
        >>> # Shape (2, 2) -> (4, 6): repeated 2x along dim 0, 3x along dim 1
    """
    # Validate repeats length
    if len(repeats) != x.ndim:
        raise ValueError(
            f"repeats length ({len(repeats)}) must match tensor dimensions ({x.ndim})"
        )

    return x.repeat(*repeats)
