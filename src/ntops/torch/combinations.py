import torch

import ntops
from ntops.torch.utils import _cached_make, _device_key

# combinations = combinatorial gather, output[k, j] = input[idx[k, j]].
#
# r == 2 (the default/common case): the index tuples are torch.triu_indices
# (O(num_comb), avoiding torch's n*n value meshgrid), and a ninetoothed kernel
# does the data-dependent gather → multi-x speedup over torch.combinations.
#
# r >= 3: torch's own algorithm (value meshgrid + triu mask). It is rarely used
# and its output C(n, r) is already huge for moderate n; a separate index pass
# for a ninetoothed gather would only add work, so we match torch here.
_CONFIGS = {
    "nvidia": (4, 256),
    "iluvatar": (8, 512),
    "metax": (4, 256),
    "default": (4, 256),
}


def combinations(input, r=2, with_replacement=False):
    if input.dim() != 1:
        raise RuntimeError(f"Expect a 1D vector, but got shape {list(input.shape)}")

    if r < 0:
        raise RuntimeError(f"Expect a non-negative number, but got {r}")

    if r == 0:
        return torch.empty(0, dtype=input.dtype, device=input.device)

    n = input.size(0)

    if r == 1:
        return input.reshape(n, 1).clone()

    if r == 2:
        offset = 0 if with_replacement else 1
        ij = torch.triu_indices(n, n, offset=offset, device=input.device)  # (2, num_comb)
        num_comb = ij.shape[1]
        index = ij.t().contiguous().reshape(-1).to(torch.int64)

        m = index.numel()
        output = torch.empty((m,), dtype=input.dtype, device=input.device)

        num_warps, block_size = _CONFIGS[_device_key()]
        kernel = _cached_make(
            ntops.kernels.combinations.premake,
            block_size=block_size,
            num_warps=num_warps,
            num_stages=1,
            max_num_configs=1,
        )
        kernel(index, output, input, m)

        return output.reshape(num_comb, 2)

    # r >= 3: torch's value-meshgrid algorithm (matches torch.combinations). The
    # triu mask compares INDICES (arange grids), not values — input values are
    # unsorted, so value comparison would pick the wrong combinations/order.
    rng = torch.arange(n, device=input.device)
    index_grids = torch.meshgrid(*([rng] * r), indexing="ij")
    mask = torch.ones(index_grids[0].shape, dtype=torch.bool, device=input.device)
    for i in range(r - 1):
        if with_replacement:
            mask = mask & (index_grids[i] <= index_grids[i + 1])
        else:
            mask = mask & (index_grids[i] < index_grids[i + 1])

    value_grids = torch.meshgrid(*([input] * r), indexing="ij")
    return torch.stack([g.masked_select(mask) for g in value_grids], dim=1)
