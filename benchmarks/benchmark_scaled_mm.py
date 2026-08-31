import argparse
import time

import torch

import ntops


SCALE_BLOCK_SIZE = 128

DEFAULT_SHAPES = (
    (1, 4096, 4096),
    (16, 4096, 4096),
    (128, 4096, 4096),
)


def _ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def _make_inputs(m, n, k):
    fp8_dtype = torch.float8_e4m3fn
    input = torch.randn((m, k), device="cuda").clamp(-2, 2).to(fp8_dtype)
    mat2 = torch.randn((n, k), device="cuda").clamp(-2, 2).to(fp8_dtype).t()
    scale_a = torch.rand(
        (m, _ceil_div(k, SCALE_BLOCK_SIZE)),
        device="cuda",
        dtype=torch.float32,
    ) + 0.25
    scale_b = torch.rand(
        (
            _ceil_div(k, SCALE_BLOCK_SIZE),
            _ceil_div(n, SCALE_BLOCK_SIZE),
        ),
        device="cuda",
        dtype=torch.float32,
    ) + 0.25

    return input, mat2, scale_a, scale_b


def _torch_eager_reference(input, mat2, scale_a, scale_b):
    m, k = input.shape
    n = mat2.shape[1]
    output = torch.zeros((m, n), dtype=torch.float32, device=input.device)

    for block in range(_ceil_div(k, SCALE_BLOCK_SIZE)):
        start = block * SCALE_BLOCK_SIZE
        end = min(start + SCALE_BLOCK_SIZE, k)
        scale_b_block = scale_b[block].repeat_interleave(SCALE_BLOCK_SIZE)[:n]
        partial = input[:, start:end].float() @ mat2[start:end, :].float()
        output += partial * scale_a[:, block, None] * scale_b_block[None, :]

    return output.to(torch.bfloat16)


def _make_rowwise_baseline(input, mat2, scale_a, scale_b):
    n = mat2.shape[1]
    blocks = []

    for block in range(scale_a.shape[1]):
        start = block * SCALE_BLOCK_SIZE
        end = start + SCALE_BLOCK_SIZE
        blocks.append(
            (
                input[:, start:end],
                mat2[start:end, :],
                scale_a[:, block : block + 1].contiguous(),
                scale_b[block]
                .repeat_interleave(SCALE_BLOCK_SIZE)[:n]
                .unsqueeze(0)
                .contiguous(),
            )
        )

    def run():
        output = torch.zeros(
            (input.shape[0], mat2.shape[1]),
            dtype=torch.float32,
            device=input.device,
        )

        for input_block, mat2_block, scale_a_block, scale_b_block in blocks:
            partial = torch._scaled_mm(
                input_block,
                mat2_block,
                scale_a_block,
                scale_b_block,
                out_dtype=torch.bfloat16,
            )
            output += partial.float()

        return output.to(torch.bfloat16)

    return run


def _make_vllm_baseline(input, mat2, scale_a, scale_b):
    try:
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            w8a8_triton_block_scaled_mm,
        )
    except (ImportError, RuntimeError) as error:
        return None, f"import failed: {error}"

    m, k = input.shape
    n = mat2.shape[1]
    num_k_blocks = _ceil_div(k, SCALE_BLOCK_SIZE)
    num_n_blocks = _ceil_div(n, SCALE_BLOCK_SIZE)

    if scale_a.shape != (m, num_k_blocks) or scale_b.shape != (
        num_k_blocks,
        num_n_blocks,
    ):
        return None, "only the 1x128/128x128 scaling layout is comparable"

    # vLLM accepts a contiguous [N, K] weight and [N/128, K/128] scales.
    # Prepare the equivalent views outside the timed region.
    weight = mat2.t().contiguous()
    weight_scale = scale_b.t().contiguous()

    def run():
        return w8a8_triton_block_scaled_mm(
            input,
            weight,
            scale_a,
            weight_scale,
            [SCALE_BLOCK_SIZE, SCALE_BLOCK_SIZE],
            torch.bfloat16,
        )

    return run, None


def _correctness_error(actual, expected, rtol=0.05, atol=0.05):
    close = torch.isclose(actual, expected, rtol=rtol, atol=atol)

    if bool(close.all()):
        return None

    mismatch = int((~close).sum().item())
    maximum = float((actual.float() - expected.float()).abs().max().item())

    return (
        f"correctness failed: {mismatch}/{close.numel()} mismatched, "
        f"max_abs={maximum:.6g}"
    )


def _short_error(error):
    return " ".join(str(error).splitlines())


def _latency_ms(function, warmup, iterations):
    for _ in range(warmup):
        function()

    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(iterations):
        function()

    torch.cuda.synchronize()

    return (time.perf_counter() - start) * 1000 / iterations


def _parse_shape(text):
    values = tuple(int(value) for value in text.lower().split("x"))

    if len(values) != 3:
        raise argparse.ArgumentTypeError("Shapes must use the MxNxK format.")

    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", action="append", type=_parse_shape)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    torch.manual_seed(0)
    shapes = tuple(args.shape) if args.shape else DEFAULT_SHAPES

    print(
        "m,n,k,ntops_ms,torch_rowwise_ms,vllm_ms,eager_reference_ms,"
        "torch_rowwise_ratio,vllm_ratio,eager_reference_ratio"
    )

    for m, n, k in shapes:
        input, mat2, scale_a, scale_b = _make_inputs(m, n, k)

        def run_ntops():
            return ntops.torch.scaled_mm(
                input,
                mat2,
                scale_a,
                scale_b,
                out_dtype=torch.bfloat16,
            )

        run_rowwise = _make_rowwise_baseline(input, mat2, scale_a, scale_b)
        run_vllm, vllm_error = _make_vllm_baseline(
            input,
            mat2,
            scale_a,
            scale_b,
        )

        actual = run_ntops()
        expected = _torch_eager_reference(input, mat2, scale_a, scale_b)
        torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.05)

        ntops_ms = _latency_ms(run_ntops, args.warmup, args.iterations)

        try:
            run_rowwise()
        except RuntimeError as error:
            print(f"torch_rowwise_unavailable={_short_error(error)}")
            rowwise_ms = None
        else:
            rowwise_ms = _latency_ms(
                run_rowwise,
                args.warmup,
                args.iterations,
            )

        if run_vllm is not None:
            try:
                vllm_output = run_vllm()
            except (AssertionError, RuntimeError) as error:
                vllm_error = f"execution failed: {_short_error(error)}"
            else:
                vllm_error = _correctness_error(vllm_output, expected)

        if vllm_error is not None:
            print(f"vllm_unavailable={vllm_error}")
            vllm_ms = None
        else:
            vllm_ms = _latency_ms(
                run_vllm,
                args.warmup,
                args.iterations,
            )

        eager_ms = _latency_ms(
            lambda: _torch_eager_reference(input, mat2, scale_a, scale_b),
            args.warmup,
            args.iterations,
        )

        rowwise_text = "nan" if rowwise_ms is None else f"{rowwise_ms:.4f}"
        vllm_text = "nan" if vllm_ms is None else f"{vllm_ms:.4f}"
        rowwise_ratio = (
            "nan" if rowwise_ms is None else f"{rowwise_ms / ntops_ms:.3f}"
        )
        vllm_ratio = "nan" if vllm_ms is None else f"{vllm_ms / ntops_ms:.3f}"
        print(
            f"{m},{n},{k},{ntops_ms:.4f},{rowwise_text},{vllm_text},"
            f"{eager_ms:.4f},{rowwise_ratio},{vllm_ratio},"
            f"{eager_ms / ntops_ms:.3f}"
        )


if __name__ == "__main__":
    main()
