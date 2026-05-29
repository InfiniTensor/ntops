import torch

import ntops

# ---------------------------------------------------------------------------
# corrcoef normalizes the covariance matrix. The dominant cost is the
# covariance matmul ``C @ C.T`` (O(D^2 N)), which is delegated to the
# ninetoothed ``mm`` kernel -- the same delegation style as ``matmul -> mm``.
# The cheap O(D N) / O(D^2) glue (row mean, centering, std normalization, clamp)
# stays in torch. Output matches torch.corrcoef: a (D, D) matrix for a 2-D
# input, a scalar 1.0 for a 1-D input; integer/bool inputs promote to float.
# ---------------------------------------------------------------------------


def corrcoef(input):
    if input.dim() > 2:
        raise RuntimeError(
            f"corrcoef(): expected input to have two or fewer dimensions but got "
            f"{input.dim()}"
        )

    was_1d = input.dim() < 2
    if was_1d:
        input = input.reshape(1, -1)

    # Integer / bool inputs promote to the default floating dtype (matching torch).
    if not torch.is_floating_point(input) and not torch.is_complex(input):
        input = input.to(torch.get_default_dtype())

    _, n = input.shape

    # Center each variable (row) across its observations.
    mean = input.mean(dim=1, keepdim=True)
    centered = (input - mean).contiguous()

    # Covariance via the ninetoothed matmul (the dominant O(D^2 N) work).
    cov = ntops.torch.mm(centered, centered.t().contiguous()) / (n - 1)

    # Normalize by the outer product of the standard deviations, then clamp.
    stddev = cov.diagonal().sqrt()
    cov = cov / stddev.unsqueeze(1)
    cov = cov / stddev.unsqueeze(0)
    cov = cov.clamp(-1, 1)

    if was_1d:
        return cov.squeeze()

    return cov
