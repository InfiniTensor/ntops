import ntops


def fliplr(input):
    # ``fliplr`` is exactly ``flip`` along dim 1 (columns reversed, rows kept),
    # so it reuses the tuned flip copy kernel rather than defining its own.
    if input.ndim < 2:
        raise RuntimeError("Input must be >= 2-d.")

    return ntops.torch.flip(input, (1,))
