import torch


def corrcoef(x):
    """
    Compute the Pearson correlation coefficient matrix.

    Each row of x is a variable, each column is an observation.

    Args:
        x: 2D input tensor of shape (N_vars, N_obs)

    Returns:
        2D tensor of shape (N_vars, N_vars) with correlation coefficients.
        Diagonal elements are 1.0.

    Examples:
        >>> x = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
        >>> corrcoef(x)
        tensor([[1., 1.],
                [1., 1.]])
    """
    return torch.corrcoef(x)
