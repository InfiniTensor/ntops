import functools

from ninetoothed import Tensor
import ninetoothed.language as ntl

def arrangement(*tensors, L_pow2, kernel_size_h, kernel_size_w, stride_h, stride_w, dilation_h, dilation_w, padding_h, padding_w, block_size=None):
    # input: (N, C * k_w * k_h, H_out * W_out)
    # output: (N, C, H_in, W_in)
    input, output, L_val = tensors

    # 排布 output, 使其与 input 对齐
    output = output.tile((1, 1, kernel_size_h, kernel_size_w), (1, 1, stride_h, stride_w), (1, 1, dilation_h, dilation_w))
    # => output: (N, C, H_out, W_out), dtype=(1, 1, k_h, k_w)
    output = output.ravel() # => output: (N, C, H_out, W_out, 1, 1, k_h, k_w)
    output = output.permute((0, 1, 4, 5, 6, 7, 2, 3))
    # => output: (N, C, 1, 1, k_h, k_w, H_out, W_out)
    output = output.flatten(start_dim=0, end_dim=6).flatten(start_dim=1)
    # => output: (N * C * k_h * k_w, H_out * W_out)
    output = output.tile((block_size, L_pow2)).squeeze(1)
    # => output: (... // block_size, ), dtype=(block_size, L_pow2)

    input = input.flatten(end_dim=2) # => input: (N * C * k_h * k_w, H_out * W_out)
    input = input.tile((block_size, L_pow2)).squeeze(1)
    # => input: (... // block_size), dtype=(block_size, L_pow2)

    return input, output, L_val

def application(input, output, L):
    # input: (block_size, L_pow2)
    # output: (block_size, L_pow2)
    ntl.atomic_add(output.data_ptr() + output.offsets(), input)

def premake(L_pow2, kernel_size_h, kernel_size_w, stride_h, stride_w, dilation_h, dilation_w, padding_h, padding_w, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, L_pow2=L_pow2, kernel_size_h=kernel_size_h, kernel_size_w=kernel_size_w, stride_h=stride_h, stride_w=stride_w, dilation_h=dilation_h, dilation_w=dilation_w, padding_h=padding_h, padding_w=padding_w, block_size=block_size)

    tensors = (
        Tensor(3, dtype=dtype, other=0, shape_options={'constexpr': True}),
        Tensor(4, dtype=dtype, other=0, shape_options={'constexpr': True}),
        Tensor(0, dtype=int, constexpr=True),  # L
    )

    return arrangement_, application, tensors
