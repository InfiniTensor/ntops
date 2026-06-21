import functools
import hashlib
import linecache

import ninetoothed
from ninetoothed import Tensor
import ninetoothed.language as ntl


def _normalize_dims(dim, ndim):
    if dim is None:
        return tuple(range(ndim))

    if isinstance(dim, int):
        dims = (dim,)
    else:
        dims = tuple(dim)

    normalized = []
    for d in dims:
        d = int(d)
        if d < 0:
            d += ndim
        if d < 0 or d >= ndim:
            raise IndexError("dim out of range")
        if d in normalized:
            raise ValueError("dim contains duplicate values")
        normalized.append(d)

    return tuple(normalized)


def _output_shape(input_shape, reduce_dims):
    return tuple(
        size
        for axis, size in enumerate(input_shape)
        if axis not in reduce_dims
    )


def arrangement(input, output, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    return input, output


def _index_expr(vars_):
    if len(vars_) == 1:
        return vars_[0]
    return ", ".join(vars_)


def _indent(level):
    return "    " * level


def _make_application(input_shape, reduce_dims):
    ndim = len(input_shape)
    input_shape = tuple(int(x) for x in input_shape)
    reduce_dims = tuple(int(x) for x in reduce_dims)
    keep_dims = tuple(axis for axis in range(ndim) if axis not in reduce_dims)

    axis_vars = [f"i{axis}" for axis in range(ndim)]

    lines = []
    lines.append("def application(input, output):")

    level = 1

    # 外层循环：非归约维度，对应 output 的每个元素
    for axis in keep_dims:
        lines.append(f"{_indent(level)}for i{axis} in range({input_shape[axis]}):")
        level += 1

    lines.append(f"{_indent(level)}acc = ntl.zeros((), dtype=ntl.int64)")

    # 内层循环：归约维度
    for axis in reduce_dims:
        lines.append(f"{_indent(level)}for i{axis} in range({input_shape[axis]}):")
        level += 1

    input_index = _index_expr(axis_vars)

    if len(reduce_dims) == 0:
        # dim=() 这种情况：不归约，每个元素输出 0/1
        lines.append(
            f"{_indent(level)}acc = ntl.where(input[{input_index}] != 0, 1, 0).to(ntl.int64)"
        )
    else:
        lines.append(
            f"{_indent(level)}acc += ntl.where(input[{input_index}] != 0, 1, 0).to(ntl.int64)"
        )

    # 写 output
    if len(keep_dims) == 0:
        output_index = "0"
    else:
        output_index = _index_expr([f"i{axis}" for axis in keep_dims])

    # 回到外层循环之后写 output
    write_level = 1 + len(keep_dims)
    lines.append(f"{_indent(write_level)}output[{output_index}] = acc  # noqa: F841")

    source = "\n".join(lines) + "\n"

    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()
    filename = f"<ntops_count_nonzero_{digest}>"
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(True),
        filename,
    )

    namespace = {
        "ntl": ntl,
    }
    code = compile(source, filename, "exec")
    exec(code, namespace)

    return namespace["application"]


def premake(
    input_shape,
    dim=None,
    dtype=None,
    block_size=None,
):
    input_shape = tuple(int(x) for x in input_shape)
    ndim = len(input_shape)

    reduce_dims = _normalize_dims(dim, ndim)
    output_shape = _output_shape(input_shape, reduce_dims)

    # ninetoothed 对 0-dim output pointer 处理不稳定，所以 scalar 用 shape (1,)。
    actual_output_shape = output_shape if len(output_shape) > 0 else (1,)
    output_ndim = len(actual_output_shape)

    arrangement_ = functools.partial(
        arrangement,
        block_size=block_size,
    )

    application = _make_application(
        input_shape=input_shape,
        reduce_dims=reduce_dims,
    )

    input = Tensor(ndim, dtype=dtype)
    output = Tensor(output_ndim, dtype=ninetoothed.int64)

    input.shape = input_shape
    output.shape = actual_output_shape

    tensors = (
        input,
        output,
    )

    return arrangement_, application, tensors