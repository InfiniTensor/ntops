import torch


def column_stack(tensors):
    """
    Stack 1D tensors as columns into a 2D tensor, or stack N-D tensors
    along the second-to-last dimension.

    Equivalent to torch.column_stack.

    Args:
        tensors: A sequence of tensors. All tensors must have the same
                 shape along all dimensions except the columns dimension.
                 1D tensors of length N are treated as (N, 1) before stacking.

    Returns:
        A stacked tensor.

    Raises:
        RuntimeError: If the sequence of tensors is empty.

    Examples:
        >>> a = torch.tensor([1, 2, 3])
        >>> b = torch.tensor([4, 5, 6])
        >>> column_stack((a, b))
        tensor([[1, 4],
                [2, 5],
                [3, 6]])

        >>> a = torch.randn(2, 3)
        >>> b = torch.randn(2, 4)
        >>> column_stack((a, b)).shape
        torch.Size([2, 7])
    """
    return torch.column_stack(tensors)
