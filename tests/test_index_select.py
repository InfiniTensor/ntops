import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


@skip_if_cuda_not_available
@pytest.mark.parametrize(
    "shape",
    [
        (8, 8),
        (4, 5, 6),
        (2, 3, 4, 5),
        (10, 20),
    ],
)
def test_index_select(shape):
    """Test index_select with float32 input"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create input tensor (float32 only)
    input = torch.randn(shape, dtype=torch.float32, device=device)

    # Test index_select on different dimensions
    for dim in range(len(shape)):
        # Create random indices for selection
        num_indices = torch.randint(1, shape[dim] + 1, (1,)).item()
        indices = torch.randperm(shape[dim], device=device)[:num_indices]

        # Call ntops implementation
        ninetoothed_output = ntops.torch.index_select(input, dim, indices)

        # Call reference implementation
        reference_output = torch.index_select(input, dim, indices)

        # Compare results
        assert torch.allclose(ninetoothed_output, reference_output), (
            f"Mismatch for shape={shape}, dim={dim}, num_indices={num_indices}"
        )


# @skip_if_cuda_not_available
# def test_index_select_all_indices():
#     """Test index_select with all indices (should equal identity, float32)"""
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     shape = (3, 4, 5)
#     input = torch.randn(shape, dtype=torch.float32, device=device)
#
#     # Test with all indices in order
#     for dim in range(len(shape)):
#         indices = torch.arange(shape[dim], device=device, dtype=torch.int64)
#
#         ninetoothed_output = ntops.torch.index_select(input, dim, indices)
#         reference_output = torch.index_select(input, dim, indices)
#
#         assert torch.allclose(ninetoothed_output, reference_output), (
#             f"Mismatch with all indices for dim={dim}"
#         )
