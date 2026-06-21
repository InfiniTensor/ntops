import torch


def chunk(x, chunks, dim=0):
    """
    Split a tensor into a specific number of chunks along a given dimension.

    This is a wrapper around PyTorch's split function for compatibility.

    Args:
        x: Input tensor
        chunks: Number of chunks to split into
        dim: Dimension to split along (default: 0)

    Returns:
        A list of tensors along the specified dimension

    Examples:
        >>> x = torch.randn(10, 5)
        >>> chunks = ntops.torch.chunk(x, chunks=3, dim=0)
        >>> len(chunks)  # 3
        >>> chunks[0].shape  # (4, 5) - first chunk gets 4 elements
        >>> chunks[1].shape  # (3, 5)
        >>> chunks[2].shape  # (3, 5)
        >>> 4 + 3 + 3 == 10  # True
    """
    # PyTorch's split takes chunk_sizes (list of ints) or single chunk_size
    # We need to compute the sizes to match NumPy's chunk behavior

    size = x.shape[dim]
    chunk_size = size // chunks
    rem = size % chunks

    # Build chunk sizes: first `rem` chunks get chunk_size + 1, rest get chunk_size
    chunk_sizes = [chunk_size + 1 if i < rem else chunk_size for i in range(chunks)]

    # Use torch.split with computed sizes
    return torch.split(x, chunk_sizes, dim=dim)
