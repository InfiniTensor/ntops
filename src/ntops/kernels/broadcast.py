import functools

from ninetoothed import Symbol, Tensor


def arrangement_dim0(input_1d, output_2d, N):
    """Broadcast input (M,) to (M, N) along dim=0.
    
    output[i, j] = input[i] for all j.
    """
    N_sym = Symbol("N", constexpr=True) if not isinstance(N, Symbol) else N

    input_arranged = input_1d.tile((-1, 1))
    input_arranged = input_arranged.expand((-1, N_sym))

    output_arranged = output_2d.tile((-1, -1))

    return input_arranged, output_arranged


def arrangement_dim1(input_1d, output_2d, M):
    """Broadcast input (N,) to (M, N) along dim=1.
    
    output[i, j] = input[j] for all i.
    """
    M_sym = Symbol("M", constexpr=True) if not isinstance(M, Symbol) else M

    input_arranged = input_1d.tile((1, -1))
    input_arranged = input_arranged.expand((M_sym, -1))

    output_arranged = output_2d.tile((-1, -1))

    return input_arranged, output_arranged


def application(input_1d, output_2d):
    output_2d = input_1d  # noqa: F841


def premake(expand_dim, other_size, dtype=None, block_size=None):
    """Create a broadcast premake function.

    Args:
        expand_dim: 0 to expand along rows, 1 to expand along columns.
        other_size: the size of the other (expanded-to) dimension.
    """
    if expand_dim == 0:
        arrangement_fn = functools.partial(arrangement_dim0, N=other_size)
    elif expand_dim == 1:
        arrangement_fn = functools.partial(arrangement_dim1, M=other_size)
    else:
        raise ValueError(f"expand_dim must be 0 or 1, got {expand_dim}")

    tensors = (Tensor(1, dtype=dtype), Tensor(2, dtype=dtype))

    return arrangement_fn, application, tensors
