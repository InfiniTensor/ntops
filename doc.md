目前，九齿（NineToothed）是一门基于 Triton 的领域特定语言（DSL），旨在进一步简化高性能计算内核的开发。它通过引入面向张量的元编程（tensor-oriented metaprogramming），抽象掉了指针算术运算和内存访问等底层细节，能够降低并行编程的门槛。九齿能够让开发者使用少量简洁的代码实现较高性能的计算内核，并且可以提高代码的可读性和可维护性。
核心概念
符号
符号这一概念，与这篇 SymPy 教程当中描写的类似。符号并不存储实际的数值，只存储符号或是符号表达式，所以允许进行一些符号化的数学运算。在九齿中，我们可以使用 Symbol 来创建一个符号。例如，在下面的代码里，我们先是创建了名为 BLOCK_SIZE_M 和 BLOCK_SIZE_N 的两个符号，之后对它们进行了乘法操作：
>>> from ninetoothed import Symbol
>>> BLOCK_SIZE_M = Symbol("BLOCK_SIZE_M")
>>> BLOCK_SIZE_M
BLOCK_SIZE_M
>>> BLOCK_SIZE_N = Symbol("BLOCK_SIZE_N")
>>> BLOCK_SIZE_N
BLOCK_SIZE_N
>>> BLOCK_SIZE_M * BLOCK_SIZE_N
BLOCK_SIZE_M * BLOCK_SIZE_N
符号张量
张量是深度学习领域的基础概念之一，如果您对张量尚不熟悉，可以参考这篇 PyTorch 的教程。九齿当中的张量，与 PyTorch 中的类似，但是并不存储实际数据，仅在 shape、strides 等成员变量中存储符号表达式。在九齿中，我们可以使用 Tensor 来创建一个张量。如下方代码所示，Tensor(2) 表示构造一个二维张量，也就是一个矩阵，而它的 shape 成员里所存储的，也都是符号，并非具体的数值：
>>> from ninetoothed import Tensor
>>> x = Tensor(2)
>>> x.shape
(ninetoothed_tensor_0_size_0, ninetoothed_tensor_0_size_1)
面向张量的元编程
得益于符号张量，我们可以对九齿中的张量进行一些编译期操作，这样的操作被称为元操作，如 tile、expand、squeeze、permute 等。例如，在这一段代码中，我们对 x 进行了 tile 操作，即将 x 分为形状为 (BLOCK_SIZE_M, BLOCK_SIZE_N) 的块：
>>> x_tiled = x.tile((BLOCK_SIZE_M, BLOCK_SIZE_N))
>>> x_tiled.shape
((ninetoothed_tensor_0_size_0 - (BLOCK_SIZE_M - 1) - 1 + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M + 1, (ninetoothed_tensor_0_size_1 - (BLOCK_SIZE_N - 1) - 1 + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N + 1)
>>> x_tiled.dtype.shape
(BLOCK_SIZE_M, BLOCK_SIZE_N)
我们注意到，x_tiled 的 dtype 也有 shape 这一成员变量。这是由于，九齿当中的张量是可以嵌套的，即一个张量的元素也可以是一个张量。也就是说，在 tile 的过程中，我们创建了一个双层的张量，其中外层张量的每一个元素，都是一个内层张量。为了方便理解，我们可以使用如下的数值化示例来进行说明：
>>> BLOCK_SIZE_M = 2
>>> BLOCK_SIZE_N = 2
>>> x = Tensor(shape=(4, 8))
>>> x_tiled = x.tile((BLOCK_SIZE_M, BLOCK_SIZE_N))
>>> x_tiled.shape
(2, 4)
>>> x_tiled.dtype.shape
(2, 2)
就像下图所示的那样，我们将一个形状为 (4, 8) 的张量 x 分成了形状为 (2, 2) 的块（内层张量），总共分成了 (2, 4) 个这样的张量（外层张量）：
[图片]
排布与应用范式
九齿引入了排布与应用（arrange-and-apply）范式，其中排布指的是如何使用元操作，对张量进行分块等排布，使得各参数张量的分块能够对齐；应用则指的是如何应用排布后的分块来完成整个算法。或者说，排布后所产生的多层张量的最外层，将会被用于并行程序的启动，而每一个并行程序所实际使用的，其实是内层张量。
接下来，让我们先通过一个简单的向量加法的例子，来理解这一范式：
import ninetoothed
from ninetoothed import Symbol, Tensor


def arrangement(lhs, rhs, output):
    BLOCK_SIZE = Symbol("BLOCK_SIZE", meta=True)

    return lhs.tile((BLOCK_SIZE,)), rhs.tile((BLOCK_SIZE,)), output.tile((BLOCK_SIZE,))


def application(lhs, rhs, output):
    output = lhs + rhs


tensors = (Tensor(1), Tensor(1), Tensor(1))
add_kernel = ninetoothed.make(arrangement, application, tensors)
在以上代码中，我们先是定义了三个张量的排布，即将三个参数张量全都分成形状为 (BLOCK_SIZE,) 的块，之后又定义了排布后张量的应用，即将 lhs 与 rhs 每一组对应的分块相加，之后写入到 output 所对应的分块当中。最后我们使用 ninetoothed.make 来将“张量”、“排布”、“应用”三个算法的组成部分进行整合，生成出可以运行的 add_kernel。需要注意的是：application 当中的 lhs、rhs、output 都是参数张量的每一组分块，而并非张量本身，如果使用上面提到的数值化示例，那 application 的参数，应当为形状为 (2, 2) 的分块，而不是形状为 (4, 8) 的原张量。
有了 add_kernel，我们就可以直接使用以下方式进行调用：
import torch

dtype = torch.float16
device = "cuda"

lhs = torch.tensor((1, 2, 3), dtype=dtype, device=device)
rhs = torch.tensor((4, 5, 6), dtype=dtype, device=device)
output = torch.empty_like(lhs)
add_kernel(lhs, rhs, output)
reference = torch.tensor((5, 7, 9), dtype=dtype, device=device)
assert torch.allclose(output, reference)
可以看到，我们在调用 add_kernel 时，并没有提供 BLOCK_SIZE 的实际取值。这是因为，在构造 BLOCK_SIZE 时，我们在 Symbol 中使用了 meta=True，这代表我们希望使用九齿所提供的配置组合来进行自动调优。如果我们希望人为提供取值（比如我们在进行调试时），我们可以使用 constexpr=True 来替代 meta=True，这样我们就可以使用以下方式传递具体的取值：
add_kernel(lhs, rhs, output, BLOCK_SIZE=1024)
索引和迭代
九齿当中的张量并不局限于双层，也可以是三层甚至更多层，但是只有排布后张量的最外层会被用于启动并行程序。换句话说，三及以上层数的张量，在应用函数里，也是层级张量，是可以被索引和迭代的。
让我们来通过一个稍微复杂一点的矩阵乘法的例子，来理解一下张量的索引和迭代，并进一步体会一下排布与应用范式：
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor


def arrangement(lhs, rhs, output):
    BLOCK_SIZE_M = Symbol("BLOCK_SIZE_M", meta=True)
    BLOCK_SIZE_N = Symbol("BLOCK_SIZE_N", meta=True)
    BLOCK_SIZE_K = Symbol("BLOCK_SIZE_K", meta=True)

    output_arranged = output.tile((BLOCK_SIZE_M, BLOCK_SIZE_N))

    lhs_arranged = lhs.tile((BLOCK_SIZE_M, BLOCK_SIZE_K))
    lhs_arranged = lhs_arranged.tile((1, -1))
    lhs_arranged = lhs_arranged.expand((-1, output_arranged.shape[1]))
    lhs_arranged.dtype = lhs_arranged.dtype.squeeze(0)

    rhs_arranged = rhs.tile((BLOCK_SIZE_K, BLOCK_SIZE_N))
    rhs_arranged = rhs_arranged.tile((-1, 1))
    rhs_arranged = rhs_arranged.expand((output_arranged.shape[0], -1))
    rhs_arranged.dtype = rhs_arranged.dtype.squeeze(1)

    return lhs_arranged, rhs_arranged, output_arranged


def application(lhs, rhs, output):
    accumulator = ntl.zeros(output.shape, dtype=ntl.float32)
    for k in range(lhs.shape[0]):
        accumulator += ntl.dot(lhs[k], rhs[k])
    output = accumulator.to(ntl.float16)


tensors = (Tensor(2), Tensor(2), Tensor(2))
matmul_kernel = ninetoothed.make(arrangement, application, tensors)
可以看出，矩阵乘法的张量排布，要比向量加法复杂不少。为了辅助理解，以下是一张该分块算法的图示：
[图片]
在代码中，我们首先定义了 BLOCK_SIZE_M、BLOCK_SIZE_N、BLOCK_SIZE_K 三个符号，用于表示分块的形状。具体来讲，我们先将 output 矩阵 tile 成形状为 (BLOCK_SIZE_M, BLOCK_SIZE_N) 的块，将 lhs 矩阵 tile 成形状为 (BLOCK_SIZE_M, BLOCK_SIZE_K) 的块，并将 rhs 矩阵 tile 成形状为 (BLOCK_SIZE_K, BLOCK_SIZE_N) 的块：
[图片]
[图片]
[图片]
我们注意到，只进行分块对于矩阵乘法是不足的。按照上面的算法图示，output 当中的每一个分块，对应的是 lhs 的一行分块，与 rhs 的一列分块，所以我们还需要对 lhs 和 rhs 进行进一步的 tile，也就是将 lhs 的每一行 tile 在一起，和将 rhs 的每一列 tile 在一起：
[图片]
[图片]
但是这还并不是全部。还记得在进行张量排布时，我们最终需要做到什么嘛？没错，是使得各参数张量的分块能够对齐。再结合九齿的工作原理，排布后张量的最外层将被用于启动并行程序，我们可以引申出一条重要的结论：各参数张量排布后的最外层应当具有相同的形状。很明显，目前我们的三个张量的最外层，形状并不相同，这往往说明我们的排布并不正确，或者尚未完成。通过图示我们可以知道，我们需要将 lhs 的每一行分块，与 rhs 的每一列分块对齐，这一点我们可以通过广播来做到，也就是将 lhs 沿着横向 expand，将 rhs 沿着竖向 expand，均 expand 至与 output 有同样的形状：
[图片]
[图片]
至此，三个参数张量排布后的最外层，便具有了相同的形状。实际上，排布阶段可以在此停止，我们已经可以据此写出 application 函数，但是我们发现，刚才所分成的 lhs 的行分块和 rhs 的列分块是二维的，并且具有 (1, ...) 和 (..., 1) 这样形式的形状。也就是说，如果不进行其它操作，那么我们访问行分块和列分块的方式就得是 lhs[0, k] 和 rhs[k, 0]，如果我们想要依靠 lhs 找到 k 的范围，那就需要通过 lhs.shape[1]。但是我们知道，大小为 1 的维度，在这种情况下完全可以被去掉，这就是为什么我们在最后加入了 squeeze 操作。这样，我们在访问行分块和列分块时就可以使用 lhs[k] 和 rhs[k]，寻找 k 的范围时也可以使用 lhs.shape[0] 了。
现在让我们来看 application 函数。在函数体当中，我们先定义了一个 accumulator，用于累加中间结果，之后就迭代了对应好的 lhs 的行块和 rhs 的列块，并且把他们相乘的结果累加到了 accumulator 当中，最后再将 accumulator 放到了对应的 output 的分块当中。由于参数张量被分成的每一块都被执行了这样的操作，因此对于整体而言，矩阵乘法就完成了。
与向量加法相同，在定义好 arrangement 和 application 后，我们可以使用 ninetoothed.make 对它们进行整合，从而形成一个可以运行的 matmul_kernel。我们可以使用以下方式对其进行调用：
import torch

dtype = torch.float16
device = "cuda"

lhs = torch.tensor(((1, 2), (3, 4)), dtype=dtype, device=device)
rhs = torch.tensor(((5, 6), (7, 8)), dtype=dtype, device=device)
output = torch.empty((lhs.shape[0], rhs.shape[1]), dtype=dtype, device=device)
matmul_kernel(lhs, rhs, output)
reference = torch.tensor(((19, 22), (43, 50)), dtype=dtype, device=device)
assert torch.allclose(output, reference)
这些就是九齿当中最核心的几个概念。
