import torch


def meshgrid(*xs, indexing="xy"):
    """
    Create coordinate grids from 1D coordinate vectors.

    Given N 1D tensors, returns N N-D tensors where each output i is the
    input x_i broadcast to a common shape. The output tensors are views
    (strided), so no data is copied.

    Args:
        *xs: 1D tensors representing coordinate values along each dimension.
        indexing: 'xy' (default) or 'ij'.
                  - 'ij': output[i] varies along axis i (matrix convention).
                  - 'xy': output[0] varies along axis 1 (columns),
                          output[1] varies along axis 0 (rows).
                          Swaps first two outputs compared to 'ij' when ndim >= 2.

    Returns:
        A list of N tensors, each of shape (len(x_0), len(x_1), ..., len(x_{N-1})).

    Examples:
        >>> x = torch.tensor([1, 2, 3])
        >>> y = torch.tensor([4, 5, 6, 7])
        >>> gx, gy = meshgrid(x, y, indexing='ij')
        >>> gx
        tensor([[1, 1, 1, 1],
                [2, 2, 2, 2],
                [3, 3, 3, 3]])
        >>> gy
        tensor([[4, 5, 6, 7],
                [4, 5, 6, 7],
                [4, 5, 6, 7]])

        >>> gx, gy = meshgrid(x, y, indexing='xy')
        >>> gx  # y broadcast (varies along axis 1)
        tensor([[4, 5, 6, 7],
                [4, 5, 6, 7],
                [4, 5, 6, 7]])
        >>> gy  # x broadcast (varies along axis 0)
        tensor([[1, 1, 1, 1],
                [2, 2, 2, 2],
                [3, 3, 3, 3]])
    """
    ndim = len(xs)

    # Reshape each input to have size -1 at its position and 1 elsewhere
    shapes = []
    for i in range(ndim):
        shp = [1] * ndim
        shp[i] = -1
        shapes.append(shp)

    grids = [x.reshape(shp) for x, shp in zip(xs, shapes)]

    # Broadcast all grids to a common shape
    out = list(torch.broadcast_tensors(*grids))

    # For 'xy' indexing, swap the first two outputs (cartesian convention)
    if indexing == "xy" and ndim >= 2:
        out[0], out[1] = out[1], out[0]

    return out
