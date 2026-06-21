import torch


def roll(x, shifts, dims=None):
    """
    Roll the tensor along the given dimension(s).

    Elements that roll beyond the last position are re-introduced at the first.

    Args:
        x: Input tensor
        shifts: The number of places by which the elements are shifted.
                Can be an int or a tuple/list of ints.
                Positive shifts roll to the right (higher indices).
        dims: Axis or axes along which to roll. Can be an int or a tuple/list.
              Defaults to None, in which case the tensor is flattened before
              rolling and then restored to the original shape.

    Returns:
        A tensor with the same shape and dtype as x, with elements rolled.

    Examples:
        >>> x = torch.tensor([1, 2, 3, 4, 5])
        >>> roll(x, shifts=2, dims=0)
        tensor([4, 5, 1, 2, 3])

        >>> x = torch.tensor([[1, 2], [3, 4], [5, 6]])
        >>> roll(x, shifts=1, dims=1)
        tensor([[2, 1],
                [4, 3],
                [6, 5]])
    """
    return torch.roll(x, shifts=shifts, dims=dims)
