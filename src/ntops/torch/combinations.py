import torch


def combinations(x, r):
    """
    Generate all combinations of r elements from the 1D input tensor.

    Returns combinations in lexicographic order as rows of a 2D tensor.

    Args:
        x: 1D input tensor of length n
        r: Number of elements in each combination

    Returns:
        2D tensor of shape (C(n, r), r) where C(n, r) = n! / (r! * (n-r)!)

    Examples:
        >>> x = torch.tensor([1, 2, 3, 4])
        >>> combinations(x, 2)
        tensor([[1, 2],
                [1, 3],
                [1, 4],
                [2, 3],
                [2, 4],
                [3, 4]])

        >>> combinations(x, 5)  # r > n returns empty
        tensor([], size=(0, 5))
    """
    if x.ndim != 1:
        raise ValueError(f"Input must be 1D, got {x.ndim}D tensor")
    if r < 0:
        raise ValueError(f"r must be non-negative, got {r}")

    n = x.shape[0]
    if r > n:
        return torch.empty(0, r, dtype=x.dtype, device=x.device)

    return torch.combinations(x, r=r)
