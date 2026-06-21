import torch


def eye(n, m=None, *, dtype=None, device=None):
    """
    Create a 2D tensor with ones on the diagonal and zeros elsewhere.

    This is a wrapper around PyTorch's eye function for compatibility.

    Args:
        n: Number of rows
        m: Number of columns (defaults to n if not provided)
        dtype: Data type of the output tensor (defaults to float32)
        device: Device to place the output on

    Returns:
        A 2D tensor of shape (n, m) with ones on the diagonal
    """
    if dtype is None:
        dtype = torch.float32

    # Handle default m value
    if m is None:
        m = n

    # Validate inputs
    if n < 0 or m < 0:
        raise ValueError(f"n and m must be non-negative, got n={n}, m={m}")

    # Use PyTorch's built-in eye function
    return torch.eye(n, m=m, dtype=dtype, device=device)
