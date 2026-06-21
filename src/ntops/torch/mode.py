import torch


def mode(x, dim, keepdim=False):
    """
    Compute the mode (most frequent value) along the given dimension.

    Returns a tuple (values, counts) where:
    - values: the mode value(s) with the same dtype as the input
    - counts: the number of occurrences of the mode value(s) (int64)

    Tie-breaking: when multiple values share the maximum frequency,
    torch.mode returns the first-encountered value. The returned value
    is always a valid mode (its count equlas the maximum count).

    Args:
        x: Input tensor
        dim: Dimension along which to compute the mode
        keepdim: If True, the output tensors retain the reduced dimension
                 as size 1 (default: False)

    Returns:
        A tuple (values, counts) where both have the same shape except
        the reduced dimension is removed (or kept as 1 if keepdim=True).

    Examples:
        >>> x = torch.tensor([1, 2, 2, 3, 3, 3])
        >>> mode(x, dim=0)
        (tensor(3), tensor(3))

        >>> x = torch.tensor([[1, 2, 2], [3, 3, 3], [1, 1, 2]])
        >>> mode(x, dim=0)  # column-wise
        (tensor([1, 1, 2]), tensor([2, 1, 2]))

        >>> mode(x, dim=1)  # row-wise
        (tensor([2, 3, 1]), tensor([2, 3, 2]))
    """
    m = torch.mode(x, dim=dim, keepdim=keepdim)

    # Compute the count of each mode value by comparing against the input
    if keepdim:
        mode_vals = m.values
    else:
        mode_vals = m.values.unsqueeze(dim)

    counts = (x == mode_vals).sum(dim=dim, keepdim=keepdim).to(torch.int64)

    return m.values, counts
