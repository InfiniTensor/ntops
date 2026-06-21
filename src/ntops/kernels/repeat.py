import functools

from ninetoothed import Symbol, Tensor


def arrangement_dim0(input, output, expand_size):
    """Repeat each element of input (M,) → (M, expand_size) → flatten (M*E,).

    Uses tile instead of expand to avoid Symbol upper_bound issues.
    """
    # Tile to add new dimension and repeat elements: (M,) → (M, E)
    input_2d = input.tile((-1, expand_size))
    input_flat = input_2d.flatten()
    output_arr = output.tile((-1,))
    return input_flat, output_arr


def arrangement_dim1(input, output, expand_size):
    """Tile whole input (N,) → (expand_size, N) → flatten (E*N,)."""
    input_2d = input.tile((expand_size, -1))
    input_flat = input_2d.flatten()
    output_arr = output.tile((-1,))
    return input_flat, output_arr


def application(input, output):
    output = input  # noqa: F841


def premake(expand_dim, expand_size, dtype=None, block_size=None):
    if expand_dim == 0:
        arrangement_fn = functools.partial(arrangement_dim0, expand_size=expand_size)
    else:
        arrangement_fn = functools.partial(arrangement_dim1, expand_size=expand_size)

    tensors = (Tensor(1, dtype=dtype), Tensor(1, dtype=dtype))

    return arrangement_fn, application, tensors
