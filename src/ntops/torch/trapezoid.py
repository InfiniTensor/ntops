import torch


def trapezoid(y, x=None, dim=-1):
    """
    Integrate along the given dimension using the composite trapezoidal rule.

    Computes: sum((x[i+1] - x[i]) * (y[i] + y[i+1]) / 2) along dim.

    Args:
        y: Input tensor to integrate
        x: Optional 1D coordinate tensor. If None, uses unit spacing (dx=1).
           Must have the same length as y.shape[dim].
        dim: Dimension along which to integrate (default: -1)

    Returns:
        Tensor with the integrated values. The integration dimension is removed.

    Examples:
        >>> y = torch.tensor([1, 2, 3])
        >>> trapezoid(y)
        tensor(4.)  # (1+2)/2 + (2+3)/2 = 1.5 + 2.5 = 4.0

        >>> y = torch.tensor([[1, 2, 3], [4, 5, 6]])
        >>> trapezoid(y, dim=1)
        tensor([4., 10.])  # row 0: 4.0, row 1: 10.0
    """
    if x is None:
        return torch.trapezoid(y, dx=1, dim=dim)
    return torch.trapezoid(y, x=x, dim=dim)
