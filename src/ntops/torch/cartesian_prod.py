import functools
import torch


def cartesian_prod(*tensors):
    """
    Compute the Cartesian product of the input tensors.

    Each input tensor is flattened to 1D, then the Cartesian product
    of all flattened tensors is computed. The result is a 2D tensor
    where each row is one combination from the product.

    Args:
        *tensors: Input tensors of any shape. Multi-dimensional tensors
                  are flattened before computing the product.

    Returns:
        A 2D tensor of shape (N, K) where N is the product of all
        flattened sizes and K is the number of input tensors.

    Examples:
        >>> a = torch.tensor([1, 2])
        >>> b = torch.tensor([3, 4, 5])
        >>> cartesian_prod(a, b)
        tensor([[1, 3],
                [1, 4],
                [1, 5],
                [2, 3],
                [2, 4],
                [2, 5]])

        >>> x = torch.tensor([[1, 2], [3, 4]])  # flattened to [1,2,3,4]
        >>> y = torch.tensor([5, 6])
        >>> cartesian_prod(x, y)
        tensor([[1, 5],
                [1, 6],
                [2, 5],
                [2, 6],
                [3, 5],
                [3, 6],
                [4, 5],
                [4, 6]])
    """
    flat_tensors = [x.flatten() for x in tensors]

    # Cast all inputs to a common dtype when types differ
    dtypes = {t.dtype for t in flat_tensors}
    if len(dtypes) > 1:
        common_dtype = functools.reduce(torch.promote_types, dtypes)
        flat_tensors = [t.to(common_dtype) for t in flat_tensors]

    result = torch.cartesian_prod(*flat_tensors)

    # torch.cartesian_prod returns a 1D tensor for single input,
    # but the CPU reference returns a 2D column vector (N, 1)
    if len(tensors) == 1:
        result = result.unsqueeze(1)

    return result
