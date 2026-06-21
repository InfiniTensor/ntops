import torch


def narrow(x, dim, start, length):
    """
    Return a narrow slice of the input tensor along the given dimension.

    This is a view operation (zero-copy), equivalent to slicing.

    Args:
        x: Input tensor
        dim: Dimension along which to narrow
        start: Starting index
        length: Number of elements to select

    Returns:
        A view of the input tensor narrowed along dim.

    Examples:
        >>> x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> narrow(x, 0, 0, 2)
        tensor([[1, 2, 3],
                [4, 5, 6]])
        >>> narrow(x, 1, 1, 2)
        tensor([[2, 3],
                [5, 6],
                [8, 9]])
    """
    return torch.narrow(x, dim=dim, start=start, length=length)
