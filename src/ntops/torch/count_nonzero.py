import torch


def count_nonzero(x, dim=None, keepdim=False):
    """
    Count the number of non-zero elements in a tensor.

    Args:
        x: Input tensor
        dim: Dimension along which to count. If None, counts all elements.
        keepdim: Whether to keep the reduced dimension (default: False)

    Returns:
        If dim is None: scalar tensor with total count.
        If dim is specified: tensor with counts along that dimension.

    Examples:
        >>> x = torch.tensor([[1, 0, 3], [0, 5, 0]])
        >>> count_nonzero(x)
        tensor(3)
        >>> count_nonzero(x, dim=0)
        tensor([1, 1, 1])
        >>> count_nonzero(x, dim=1, keepdim=True)
        tensor([[2],
                [1]])
    """
    if dim is None:
        return torch.count_nonzero(x)

    result = torch.count_nonzero(x, dim=dim)
    if keepdim:
        result = result.unsqueeze(dim)
    return result
