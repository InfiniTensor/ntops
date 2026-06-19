"""
eye kernel module.

Note: Due to element_wise arrangement limitations with runtime block_size and
the arange constexpr requirement, this implementation uses PyTorch's built-in
eye function in the torch layer rather than a custom GPU kernel.

The torch.eye function is already highly optimized and handles all edge cases
correctly, making it the most practical choice for this operation.
"""


def premake(ndim, n=None, m=None, dtype=None, block_size=None):
    """
    This is a placeholder for compatibility.

    The actual implementation is in the torch layer which uses PyTorch's
    built-in eye function directly.
    """
    raise NotImplementedError(
        "eye is implemented using PyTorch's torch.eye in the torch layer. "
        "GPU kernel implementation is not provided due to element_wise "
        "arrangement constraints with runtime block_size."
    )
