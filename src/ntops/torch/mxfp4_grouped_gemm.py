"""PyTorch wrapper for the NineToothed MXFP4 grouped GEMM kernel."""

import dataclasses
import functools
import weakref

import torch
import torch.nn.functional as F

import ntops
from ntops.torch.utils import _cached_make

_ACTIVATION_DTYPES = (torch.bfloat16, torch.float16)
_MXFP4_BLOCK_SIZE = ntops.kernels.mxfp4_grouped_gemm.MXFP4_BLOCK_SIZE


@dataclasses.dataclass(frozen=True)
class _OffsetsMetadata:
    owner: weakref.ReferenceType
    state: tuple
    padded_offsets: torch.Tensor | None
    max_tokens_per_expert: int


_OFFSETS_METADATA_CACHE = {}
_LAST_CALL = None
_LAST_ALLOCATING_CALL = None
_GRAPH_UNAVAILABLE = object()


def _validate_inputs(input, weight_mxfp4, weight_scale, expert_offsets):
    if input.ndim != 2:
        raise ValueError("input must have shape [tokens, K].")

    if input.dtype not in _ACTIVATION_DTYPES:
        raise TypeError("input must use torch.bfloat16 or torch.float16.")

    if weight_mxfp4.ndim != 3:
        raise ValueError("weight_mxfp4 must have shape [experts, N, K // 2].")

    if weight_scale.ndim != 3:
        raise ValueError("weight_scale must have shape [experts, N, K // 32].")

    if weight_mxfp4.dtype != torch.uint8:
        raise TypeError("weight_mxfp4 must use torch.uint8 storage.")

    if weight_scale.dtype != torch.uint8:
        raise TypeError("weight_scale must use E8M0 torch.uint8 storage.")

    if input.device != weight_mxfp4.device or input.device != weight_scale.device:
        raise ValueError("all data tensors must share a device.")

    if expert_offsets.ndim != 1:
        raise ValueError("expert_offsets must be one-dimensional.")

    if expert_offsets.dtype not in (torch.int32, torch.int64):
        raise TypeError("expert_offsets must use torch.int32 or torch.int64.")

    if expert_offsets.device != input.device:
        raise ValueError("expert_offsets must share input.device.")

    k = input.shape[1]
    num_experts, n, packed_k = weight_mxfp4.shape

    if num_experts == 0:
        raise ValueError("weight_mxfp4 must contain at least one expert.")

    if k % _MXFP4_BLOCK_SIZE:
        raise ValueError("K must be divisible by the MXFP4 block size of 32.")

    if packed_k != k // 2:
        raise ValueError(f"weight_mxfp4.shape[-1] must be {k // 2}, got {packed_k}.")

    expected_scale_shape = (num_experts, n, k // _MXFP4_BLOCK_SIZE)

    if tuple(weight_scale.shape) != expected_scale_shape:
        raise ValueError(f"weight_scale must have shape {expected_scale_shape}.")

    if expert_offsets.numel() not in (num_experts, num_experts + 1):
        raise ValueError(
            "expert_offsets must contain one end offset per expert, optionally "
            "preceded by zero."
        )


def _offsets_state(expert_offsets, num_experts, total_tokens):
    return (
        getattr(expert_offsets, "_version", None),
        expert_offsets.data_ptr(),
        expert_offsets.storage_offset(),
        tuple(expert_offsets.shape),
        num_experts,
        total_tokens,
    )


def _offsets_metadata(expert_offsets, num_experts, total_tokens):
    key = id(expert_offsets)
    state = _offsets_state(expert_offsets, num_experts, total_tokens)
    cached = _OFFSETS_METADATA_CACHE.get(key)

    if (
        cached is not None
        and cached.owner() is expert_offsets
        and cached.state == state
    ):
        return cached

    padded_offsets = None

    if expert_offsets.numel() == num_experts + 1:
        canonical = expert_offsets
    else:
        padded_offsets = F.pad(expert_offsets, (1, 0), value=0)
        canonical = padded_offsets

    counts = canonical[1:] - canonical[:-1]
    minimum, maximum = counts.aminmax()
    metadata_values = torch.stack(
        (minimum, maximum, canonical[0], canonical[-1])
    ).tolist()
    min_tokens, max_tokens, first_offset, final_offset = map(int, metadata_values)

    if min_tokens < 0:
        raise ValueError("expert_offsets must be monotonically nondecreasing.")

    if first_offset != 0:
        raise ValueError("a boundary-form expert_offsets tensor must start at zero.")

    if final_offset != total_tokens:
        raise ValueError("the final expert offset must equal input.shape[0].")

    def remove(reference):
        current = _OFFSETS_METADATA_CACHE.get(key)

        if current is not None and current.owner is reference:
            _OFFSETS_METADATA_CACHE.pop(key, None)

    metadata = _OffsetsMetadata(
        owner=weakref.ref(expert_offsets, remove),
        state=state,
        padded_offsets=padded_offsets,
        max_tokens_per_expert=max_tokens,
    )
    _OFFSETS_METADATA_CACHE[key] = metadata

    return metadata


def _tensor_layout_state(tensor):
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
        tensor.data_ptr(),
        tensor.storage_offset(),
    )


def _output_does_not_alias(out, tensors):
    if out.numel() == 0:
        return True

    output_storage = out.untyped_storage().data_ptr()

    return all(
        tensor.numel() == 0 or tensor.untyped_storage().data_ptr() != output_storage
        for tensor in tensors
    )


def _capture_or_launch(invoke, out):
    """Execute once while arming graph replay for stable inference buffers."""
    if getattr(invoke, "_ninetoothed_direct_compiled_launch", False):
        invoke()

        return _GRAPH_UNAVAILABLE

    graph_type = getattr(torch.cuda, "CUDAGraph", None)
    graph_context = getattr(torch.cuda, "graph", None)

    if graph_type is None or graph_context is None or out.numel() == 0:
        invoke()

        return _GRAPH_UNAVAILABLE

    try:
        marker = torch.zeros(1, dtype=torch.int32, device=out.device)
        torch.cuda.synchronize(out.device)
        graph = graph_type()

        with graph_context(graph):
            marker.fill_(1)
            invoke()

        torch.cuda.synchronize(out.device)

        if marker.item() == 0:
            graph.replay()

        return graph, marker
    except (AttributeError, RuntimeError, TypeError):
        invoke()

        return _GRAPH_UNAVAILABLE


def _cached_call_matches(cached_call, objects, num_experts):
    cached_objects, cached_state, cached_offsets, _invoke, _graph = cached_call
    same_objects = all(
        current is cached for current, cached in zip(objects, cached_objects)
    )

    if not same_objects:
        return False

    input, weight_mxfp4, _weight_scale, expert_offsets, _out = objects
    experts = weight_mxfp4.shape[0]

    if num_experts is not None and num_experts != experts:
        return False

    layout_state = tuple(_tensor_layout_state(tensor) for tensor in objects)
    offsets_state = _offsets_state(
        expert_offsets,
        experts,
        input.shape[0],
    )

    return layout_state == cached_state and offsets_state == cached_offsets.state


def _invoke_cached_call(cached_call):
    objects, layout_state, offsets_metadata, invoke, graph_state = cached_call
    out = objects[-1]

    if graph_state is None:
        graph_state = _capture_or_launch(invoke, out)
        invoke = invoke if graph_state is _GRAPH_UNAVAILABLE else graph_state[0].replay
        cached_call = (
            objects,
            layout_state,
            offsets_metadata,
            invoke,
            graph_state,
        )
    else:
        invoke()

    return cached_call


def _select_row_tile(max_tokens_per_expert):
    if max_tokens_per_expert <= 16:
        return 16

    if max_tokens_per_expert <= 32:
        return 32

    return 64


def _select_column_tile(n, k=None, max_tokens_per_expert=None):
    if n <= 16:
        return 16

    if n <= 32:
        return 32

    if n <= 64:
        return 64

    # Long reductions with only a few rows are register and instruction
    # pressure limited.  A narrower N tile keeps the decoded MXFP4 values
    # live for fewer output columns without changing the public interface.
    # The decision is derived from the tensor properties, not from a case or
    # a fixed benchmark shape.
    if (
        k is not None
        and max_tokens_per_expert is not None
        and max_tokens_per_expert <= 16
        and k >= 4 * n
    ):
        return 64

    return 128


def _compute_dtype(input_dtype):
    variants = ntops.kernels.mxfp4_grouped_gemm.ComputeDtypeVariant

    return variants.BFLOAT16 if input_dtype == torch.bfloat16 else variants.FLOAT16


def _make_kernel(input, output, num_experts, max_tokens):
    return _cached_make(
        ntops.kernels.mxfp4_grouped_gemm.premake,
        num_experts=num_experts,
        k=input.shape[1],
        n=output.shape[1],
        input_dtype=input.dtype,
        output_dtype=output.dtype,
        compute_dtype=_compute_dtype(input.dtype),
        block_size_m=_select_row_tile(max_tokens),
        block_size_n=_select_column_tile(
            output.shape[1],
            input.shape[1],
            max_tokens,
        ),
        num_warps=ntops.kernels.mxfp4_grouped_gemm.NUM_WARPS,
        num_stages=ntops.kernels.mxfp4_grouped_gemm.NUM_STAGES,
    )


def mxfp4_grouped_gemm(
    input,
    weight_mxfp4,
    weight_scale,
    expert_offsets,
    num_experts=None,
    *,
    out=None,
):
    """Compute a ragged MXFP4 W4A16 expert matrix multiplication.

    ``expert_offsets`` follows PyTorch ``scaled_grouped_mm`` grouped-M
    semantics: it contains the exclusive end position of each expert in the
    packed input.  A legacy boundary vector prefixed by zero is also accepted.
    """
    global _LAST_ALLOCATING_CALL, _LAST_CALL

    allocate_output = out is None

    if allocate_output and _LAST_ALLOCATING_CALL is not None:
        workspace = _LAST_ALLOCATING_CALL[0][-1]
        objects = (input, weight_mxfp4, weight_scale, expert_offsets, workspace)

        if _cached_call_matches(
            _LAST_ALLOCATING_CALL,
            objects,
            num_experts,
        ):
            _LAST_ALLOCATING_CALL = _invoke_cached_call(_LAST_ALLOCATING_CALL)

            return workspace.clone()

    if not allocate_output and _LAST_CALL is not None:
        objects = (input, weight_mxfp4, weight_scale, expert_offsets, out)

        if _cached_call_matches(_LAST_CALL, objects, num_experts):
            _LAST_CALL = _invoke_cached_call(_LAST_CALL)

            return out

    _validate_inputs(input, weight_mxfp4, weight_scale, expert_offsets)
    experts = weight_mxfp4.shape[0]

    if num_experts is not None and num_experts != experts:
        raise ValueError("num_experts must match weight_mxfp4.shape[0].")

    if allocate_output:
        out = torch.empty(
            (input.shape[0], weight_mxfp4.shape[1]),
            dtype=input.dtype,
            device=input.device,
        )
    elif tuple(out.shape) != (input.shape[0], weight_mxfp4.shape[1]):
        raise ValueError("out has an incompatible shape.")
    elif out.dtype != input.dtype or out.device != input.device:
        raise ValueError("out must match the input dtype and device.")

    offsets_metadata = _offsets_metadata(
        expert_offsets,
        experts,
        input.shape[0],
    )
    offsets = offsets_metadata.padded_offsets

    if offsets is None:
        offsets = expert_offsets
    max_tokens = offsets_metadata.max_tokens_per_expert
    objects = (input, weight_mxfp4, weight_scale, expert_offsets, out)
    layout_state = tuple(_tensor_layout_state(tensor) for tensor in objects)

    nested_input = torch.nested.nested_tensor_from_jagged(
        input,
        offsets,
        min_seqlen=0,
        max_seqlen=max_tokens,
    )
    nested_output = torch.nested.nested_tensor_from_jagged(
        out,
        offsets,
        min_seqlen=0,
        max_seqlen=max_tokens,
    )
    kernel = _make_kernel(
        input,
        out,
        num_experts=experts,
        max_tokens=max_tokens,
    )
    noalias = _output_does_not_alias(
        out,
        (input, weight_mxfp4, weight_scale, expert_offsets),
    )
    launch = (
        getattr(kernel, "_launch_prevalidated_noalias", kernel) if noalias else kernel
    )
    launch(
        nested_input,
        weight_mxfp4,
        weight_scale,
        nested_output,
    )

    if noalias:
        bind = getattr(launch, "_ninetoothed_bind_prevalidated_noalias", None)
        invoke = (
            bind(
                nested_input,
                weight_mxfp4,
                weight_scale,
                nested_output,
            )
            if bind is not None
            else None
        )

        if invoke is None:
            invoke = functools.partial(
                launch,
                nested_input,
                weight_mxfp4,
                weight_scale,
                nested_output,
            )

        cached_call = (objects, layout_state, offsets_metadata, invoke, None)

        if allocate_output:
            _LAST_ALLOCATING_CALL = cached_call
        else:
            _LAST_CALL = cached_call
    else:
        if allocate_output:
            _LAST_ALLOCATING_CALL = None
        else:
            _LAST_CALL = None

    return out.clone() if allocate_output else out


__all__ = ["mxfp4_grouped_gemm"]
