import torch

import ntops
from ntops.torch.utils import _cached_make


def one_hot(input, num_classes=-1):
    if input.dtype != torch.int64:
        raise AssertionError(
            "one_hot is only applicable to index tensor of type LongTensor."
        )

    if input.numel() == 0:
        if num_classes is None or num_classes == -1:
            raise ValueError(
                "Can not infer total number of classes from empty tensor."
            )
        num_classes = int(num_classes)
        if num_classes <= 0:
            raise ValueError("`num_classes` must be positive.")
    else:
        min_value = int(input.min().item())
        if min_value < 0:
            raise ValueError("Class values must be non-negative.")

        if num_classes is None or num_classes == -1:
            num_classes = int(input.max().item()) + 1
        else:
            num_classes = int(num_classes)
            if num_classes <= 0:
                raise ValueError("`num_classes` must be positive.")
            if int(input.max().item()) >= num_classes:
                raise ValueError("Class values must be smaller than num_classes.")

    output_shape = tuple(input.shape) + (num_classes,)
    output = torch.empty(output_shape, dtype=torch.int64, device=input.device)

    kernel = _cached_make(ntops.kernels.one_hot.premake, input.ndim, num_classes)

    kernel(input, output)

    return output
