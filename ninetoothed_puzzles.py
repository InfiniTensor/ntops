# %% [markdown]
# # NineToothed Puzzles
#
# 九齿是一门张量级的深度学习领域特定语言（DSL），主要用途为开发计算内核。与 Triton 等传统并行编程语言相比，其通过引入面向张量的元编程，抽象掉了指针算术运算等底层细节，在保持性能与 Triton 相当的同时，提高了代码的可读性与可维护性。
#
# 该笔记将带你由浅入深地学习和掌握九齿的使用方法。

# %% [markdown]
# ## 依赖
#
# 首先，请确保 `ninetoothed` 等依赖已经安装好了。一般可以使用以下代码块进行安装，但是由于大家的环境可能各不相同，包管理工具也可能各不相同，所以以下代码块仅供参考，实际使用时请按需修改为合适的版本。

# %%
# %pip install ninetoothed
# %pip install ninetoothed[debugging]
# %pip install ninetoothed[visualization]

# %% [markdown]
# 如果可以成功运行以下代码块，那就说明我们的环境已经准备好了。

# %%
import ninetoothed
import ninetoothed.language as ntl
import numpy as np
import torch
from ninetoothed import Symbol, Tensor
from ninetoothed.visualization import visualize

assert torch.cuda.is_available(), "CUDA is not available."

# %% [markdown]
# ## Quickstart
#
# 既然说九齿是一门张量级的 DSL，那么就让我们先来创建一个[张量](https://ninetoothed.org/python_api/tensor.html)看看。注：以下内容默认大家对张量有一个基本的了解，如果没有的话可以先参考[这篇 PyTorch 的教程](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)。

# %%
x = Tensor(shape=(2, 3))

# %% [markdown]
# 恭喜，我们刚刚成功创建了一个形状为 `(2, 3)` 的张量 `x`。让我们先试着调用它的一个方法：[`Tensor.eval`](https://ninetoothed.org/python_api/generated/ninetoothed.Tensor.eval.html#ninetoothed.Tensor.eval)。

# %%
x.eval()

# %% [markdown]
# 在九齿当中，我们可以使用 `Tensor.eval` 对一个张量进行求值，从而可以将其打印出来。你或许会有疑问：为什么不能直接打印 `x`，还要调用 `Tensor.eval`？为了解答这个问题，让我们先来看看下面这个代码块。

# %%
x = Tensor(2)

# %% [markdown]
# 我们会发现上面这行代码也可以顺利地完成。可是 `2` 在这里是什么意思呢？答案是维度数。以上代码也可以写成 `x = Tensor(ndim=2)`，换句话说，就是创建了一个矩阵（二维张量）`x`。如果大家对 PyTorch 或者 NumPy 很熟悉，就一定会好奇，单纯传递了维度数，怎么就能够创建出一个张量呢？正常情况下，不都得起码传递一个形状嘛？这是因为：九齿当中的张量是符号张量，其不存储实际数据，仅在 `Tensor.shape` 等成员变量中存储符号表达式。

# %%
x.shape

# %% [markdown]
# 从以上代码块的输出我们可以看到 `x.shape` 的每个元素都是九齿生成的符号，这也印证了刚才所说的“九齿当中的张量是符号张量”。正因为如此，所以我们不能直接打印 `x`，而是需要先调用 `Tensor.eval`，而所谓求值，其实就是希望将一个不易可视化的符号张量，转化为一个更加看得见、摸得着的数值张量（目前的默认输出是 `numpy.ndarray`），其中每个元素都表示一个索引。

# %%
x.eval({x: Tensor(shape=(2, 3))})

# %% [markdown]
# 大家肯定注意到了我们这次调用加入了参数 `Tensor(shape=(2, 3))`，这是因为在进行求值时，我们必须提供数值化所需的全部信息。由于 `x` 的形状是未知的，所以我们必须要手动传值，确保每个符号都有对应的数值代入，才可以完成这一过程。

# %%
subs = {x: Tensor(shape=(8, 8))}
x.eval(subs)

# %% [markdown]
# 根据代入的数值不同，`Tensor.eval` 的结果自然也可以是不同的。

# %%
x_substituted = x.subs(subs)
x_substituted, x_substituted.shape

# %% [markdown]
# 九齿当中还提供 [`Tensor.subs`](https://ninetoothed.org/python_api/generated/ninetoothed.Tensor.subs.html#ninetoothed.Tensor.subs) 函数，可以将一个九齿张量数值化，注意 `Tensor.subs` 的输出与 `Tensor.eval` 不同，`Tensor.eval` 的输出是一个非嵌套数值张量，里面存储着索引数据，而 `Tensor.subs` 的输出仍然是九齿张量，只不过其中如 `shape` 等属性中的符号会被替换为数值。

# %%
x_substituted.eval()

# %% [markdown]
# 单独提供 `Tensor.subs` 的好处就是可以方便使用九齿当中一些专门给九齿数值化张量使用的工具，如 `visualize` 等，这些在后文当中也会逐步介绍。由于 `x_substituted` 已经是数值化了的，所以后面的 `Tensor.eval` 中就不需要再传入 `subs` 了。

# %% [markdown]
# 很好，我们现在掌握了可以将九齿当中本来相对抽象的符号张量转化为比较具象的数值张量的方法，接下来就可以尝试对张量进行一些操作了。就让我们从 [`Tensor.tile`](https://ninetoothed.org/python_api/generated/ninetoothed.Tensor.tile.html#ninetoothed.Tensor.tile) 开始吧，这也是九齿当中最为核心的操作。

# %%
x_tiled = x.tile((4, 4))

# %% [markdown]
# 上面这行代码的意思是：将 `x` 分成每一块大小为 `(4, 4)` 的块。如果按照上面 `x` 的形状为 `(8, 8)` 的话，按理说就可以分成 `(2, 2)` 块。

# %%
x_tiled_substituted = x_tiled.subs(subs)
x_tiled_substituted.eval()

# %% [markdown]
# 可以看到求值后的张量的维度数从原先的 `2` 变成了 `4`，这是因为 `Tensor.tile` 操作会生成嵌套张量。用上面的例子来说，结果就是一个双层的张量，其中外层的形状为 `(2, 2)`，内层的形状为 `(4, 4)`，所以求值后的形状就是 `(2, 2, 4, 4)`。为了能够更清晰地理解嵌套张量，我们可以使用 [`visualize`](https://ninetoothed.org/python_api/visualization.html#ninetoothed.visualization.visualize) 来可视化一个张量。

# %%
visualize(x_tiled_substituted)

# %% [markdown]
# 如图所示，该张量是一个双层张量，其中外层的形状为 `(2, 2)`，内层的形状为 `(4, 4)`。也可以说，外层的每个元素，都是一个 `(4, 4)` 的内层张量。

# %%
x_tiled.dtype

# %% [markdown]
# 我们可以通过 `Tensor.dtype` 来访问内层张量，因为嵌套张量的元素类型已经不仅仅局限于 `float`、`int` 等，也可以是 `Tensor`。这样的好处是我们可以方便地对其中某一层进行操作，比如对 `x_tiled` 的内层 `x_tiled.dtype` 进行 [`flatten`](https://ninetoothed.org/python_api/generated/ninetoothed.Tensor.flatten.html#ninetoothed.Tensor.flatten) 操作。

# %%
x_tiled.dtype = x_tiled.dtype.flatten()
x_tiled_substituted = x_tiled.subs(subs)
visualize(x_tiled_substituted)
x_tiled_substituted.eval()

# %% [markdown]
# 如果我们集中注意力，会发现上述代码中调用 `Tensor.tile` 时传入的是 `(4, 4)`，也就是确定的数值，这看起来没什么问题，但是在九齿面向张量的元编程中，却的确有些格格不入。实际上，我们可以定义[符号](https://ninetoothed.org/python_api/symbol.html)来进行操作，而非必须使用具体的数值。

# %%
block_size_m = Symbol("block_size_m", constexpr=True)
block_size_n = Symbol("block_size_n", constexpr=True)

x_tiled = x.tile((block_size_m, block_size_n))

subs |= {block_size_m: 4, block_size_n: 4}

x_tiled_substituted = x_tiled.subs(subs)
visualize(x_tiled_substituted)
x_tiled_substituted.eval()

# %% [markdown]
# 在九齿中，我们可以通过 [`Symbol`](https://ninetoothed.org/python_api/symbol.html#symbol) 来定义符号（先不用管 `constexpr=True`，后面会讲），从而使用符号来参与张量操作。但是需要注意，由于我们引入了符号，所以在 `subs` 中我们就需要再加上 `block_size_m` 和 `block_size_n` 的取值，才能保证代入的完整性。

# %% [markdown]
# 很好，我们现在学会了如何对张量进行操作。我们把一系列这样的操作，称之为排布。既然可以对一个张量进行排布，那当然也可以对多个张量进行排布。

# %%
block_size = Symbol("block_size", constexpr=True)


def arrangement(x, y, z, block_size=block_size):
    return x.tile((block_size,)), y.tile((block_size,)), z.tile((block_size,))


# %% [markdown]
# 在以上的函数中，我们便分别对三个参数张量 `x`、`y`、`z` 进行了排布。

# %%
x = Tensor(1)
y = Tensor(1)
z = Tensor(1)

x_arranged, y_arranged, z_arranged = arrangement(x, y, z)

shape = (8,)
subs = {
    x: Tensor(shape=shape),
    y: Tensor(shape=shape),
    z: Tensor(shape=shape),
    block_size: 4,
}

x_arranged_substituted = x_arranged.subs(subs)
y_arranged_substituted = y_arranged.subs(subs)
z_arranged_substituted = z_arranged.subs(subs)

print(x_arranged_substituted.shape, x_arranged_substituted.dtype.shape)
print(y_arranged_substituted.shape, y_arranged_substituted.dtype.shape)
print(z_arranged_substituted.shape, z_arranged_substituted.dtype.shape)

visualize(x_arranged_substituted)
visualize(y_arranged_substituted)
visualize(z_arranged_substituted)

x_arranged_evaluated = x_arranged_substituted.eval()
y_arranged_evaluated = y_arranged_substituted.eval()
z_arranged_evaluated = z_arranged_substituted.eval()

print(x_arranged_evaluated)
print(y_arranged_evaluated)
print(z_arranged_evaluated)

# %% [markdown]
# 不难看出，如果输入张量的形状都为 `(8,)`，`block_size` 为 `4`，则输出张量均为双层张量，其中外层形状为 `(2,)`，内层形状为 `(4,)`。通过观察发现，我们可以通过排布将一个源张量变换为一个多层的嵌套张量。同理，我们也可以通过排布将多个源张量变换为多个多层的嵌套张量。
#
# 九齿的运行机制也由此而生：**九齿会根据各个参数张量排布后的最外层张量的形状启动程序实例，并把次外层张量映射到这些程序实例上。**
#
# 所以，如果按照上述的排布，九齿就会启动 `2` 个程序实例，并把排布后的 `x`、`y`、`z` 的最外层张量的每个元素，也就是次外层张量，与这 `2` 个程序实例一一对应。

# %%
print("Program instance 0:")
print("x:", x_arranged_evaluated[0])
print("y:", y_arranged_evaluated[0])
print("z:", z_arranged_evaluated[0])

print("-" * 32)

print("Program instance 1:")
print("x:", x_arranged_evaluated[1])
print("y:", y_arranged_evaluated[1])
print("z:", z_arranged_evaluated[1])

# %% [markdown]
# 从以上输出不难看出，`x`、`y`、`z` 三个张量在程序实例 `0` 上分得的都是 `[0 1 2 3]`，在程序实例 `1` 上分得的则都是 `[4 5 6 7]`。这里我们就可以更清晰地明白 `Tensor.eval` 后张量中存储索引的用途：建立张量排布前后元素的对应关系。我们通过以上的打印方式，可以看到每个程序实例上，各个参数张量的次外层张量的对应关系。我们也可以按照以下方式打印，这样就可以清晰地看出一个张量的元素在各个程序实例上的分布。总而言之，熟练地使用 `Tensor.eval` 和 `Tensor.subs` 等工具，非常有助于对九齿的理解和使用。

# %%
print("x:", x.eval(subs))
print("x at program instance 0:", x_arranged_evaluated[0])
print("x at program instance 1:", x_arranged_evaluated[1])


# %% [markdown]
# 这就是九齿编程模型的并行原理。但是光有排布还不够，因为虽然我们已经将参数张量分到了各个程序实例上，但是每一个程序实例要做什么，我们还没有定义。在九齿中，我们可以通过定义应用函数来告诉九齿每个程序实例需要做什么。


# %%
def application(x, y, z):
    z = x + y  # noqa: F841


# %% [markdown]
# 上面的应用函数代码逻辑很简单，就是把 `x` 和 `y` 相加，并把结果放入 `z` 中。但是需要注意的是：应用函数的参数，是参数张量排布后的最外层张量的元素，也就是次外层张量，而不是张量本身。也就是说，如果套用上面的假设，这里的 `x`、`y`、`z` 都是指长度为 `4` 的块，而不是长度为 `8` 的原本的张量。
#
# 很好，我们现在有了一个排布函数 `arrangement` 和一个应用函数 `application`，接下来就可以将它们整合，从而形成一个完整可运行的计算内核。

# %%
kernel = ninetoothed.make(arrangement, application, (Tensor(1), Tensor(1), Tensor(1)))

# %% [markdown]
# 这段代码的意思就是说，我想要按照 `arrangement` 函数对三个一维张量，也就是向量，进行排布，并按照 `application` 函数应用排布后的张量，最终形成一个计算内核 `kernel`。我们把这样构造计算内核的范式，称之为排布与应用范式。
#
# 我们可以如下所示对 `kernel` 进行调用：

# %%
size = 240620
device = "cuda"

x = torch.randn(size, device=device)
y = torch.randn(size, device=device)
z = torch.empty_like(x)

kernel(x, y, z, block_size=64)

print(x)
print(y)
print(z)

reference = x + y

print(reference)

assert torch.allclose(z, reference)

# %% [markdown]
# 我们不难发现，上面实现出的 `kernel`，其实就是一个向量加法的计算内核。所以说，使用九齿实现向量加法，实际只需要以下几行即可。

# %%
block_size = Symbol("block_size", constexpr=True)


def arrangement(x, y, z, block_size=block_size):
    return x.tile((block_size,)), y.tile((block_size,)), z.tile((block_size,))


def application(x, y, z):
    z = x + y  # noqa: F841


kernel = ninetoothed.make(arrangement, application, (Tensor(1), Tensor(1), Tensor(1)))

# %% [markdown]
# 现在让我们来看一下 `Symbol` 中的 `constexpr=True`。对 C++ 熟悉的小伙伴应该对 `constexpr` 不陌生，其在 C++ 中表示编译时常量，这也是 `constexpr` 在九齿中的含义。换句话说，`Symbol("block_size", constexpr=True)` 表示我们希望创建一个编译时确定取值的符号。之所以如此，是因为九齿当前继承了 Triton 的一个约束：最内层张量的形状需要在编译时是确定的。这就是为什么之前调用 `kernel` 时我们传递了 `block_size=64`：因为 JIT 编译时需要知道 `block_size`。我们还可以交由九齿来选择具体的取值，就像 Triton 当中有 `triton.autotune` 一样，九齿也提供自动调优功能。

# %%
block_size = Symbol(
    "block_size", meta=True, lower_bound=32, upper_bound=128, power_of_two=True
)

# %% [markdown]
# `meta=True` 的意思就是，我希望创建一个元符号，即将该符号的具体取值交由九齿决定，当然我们也需要提供一些信息，比如这个符号的取值范围，以及它是否为 2 的幂之类的。

# %% [markdown]
# 由于 block size 的创建太高频，几乎是所有计算内核必要的，所以九齿提供一个很好用的函数 `ninetoothed.block_size`，专门用来定义 block size，使用它时九齿将自动选择合适的配置进行自动调优。

# %%
block_size = ninetoothed.block_size()


# %% [markdown]
# 让我们尝试使用它来定义和运行计算内核，这次调用 `kernel` 时便不再需要传递 `block_size` 了，因为我们已经告诉了九齿，我们希望通过自动调优来找到合适的 `block_size` 取值。
#
# 注：由于自动调优需要时间，所以在后面的部分我们会统一使用 `constexpr` 符号。事实上，在刚开始开发和调试某一计算内核时，也建议先使用 `constexpr`，一方面加快原型验证的速度，一方面有确定的取值也有助于调试。可以在调试完成后需要性能时再打开自动调优，就好比 Debug 和 Release 模式一样。


# %%
def arrangement(x, y, z, block_size=block_size):
    return x.tile((block_size,)), y.tile((block_size,)), z.tile((block_size,))


def application(x, y, z):
    z = x + y  # noqa: F841


kernel = ninetoothed.make(arrangement, application, (Tensor(1), Tensor(1), Tensor(1)))

size = 240620
device = "cuda"

x = torch.randn(size, device=device)
y = torch.randn(size, device=device)
z = torch.empty_like(x)

kernel(x, y, z)

print(x)
print(y)
print(z)

reference = x + y

print(reference)

assert torch.allclose(z, reference)

# %% [markdown]
# 在向量加法中，参数张量经过排布变成了双层的张量，但是九齿当中的张量并不局限于双层，也可以是三层甚至更多层。

# %%
x = Tensor(2)

x_arranged = x.tile((1, block_size))
x_arranged = x_arranged.tile((1, -1))

subs = {x: Tensor(shape=(4, 8)), block_size: 4}

x_arranged_substituted = x_arranged.subs(subs)
visualize(x_arranged_substituted)
x_arranged_evaluated = x_arranged_substituted.eval()
x_arranged_evaluated

# %% [markdown]
# 可以看出，以上代码构造出了一个三层的张量。具体而言，上述代码先是将 `x` `tile` 成了形状为 `(1, block_size)` 的若干块，也就是每一行若干块，之后又将每一行分块 `tile` 在了一起（跟很多 PyTorch 函数一样，`-1` 在 `tile` 中表示维度原本的大小）。这里需要注意一点：只有排布后的最外层会被用于启动程序实例。换句话说，三及以上层的张量，在应用函数里，也是层级张量，是可以被索引和迭代的。

# %%
for pid, index in enumerate(np.ndindex(x_arranged_substituted.shape)):
    print(f"x at program instance {pid}:")
    for idx in np.ndindex(x_arranged_substituted.dtype.shape):
        print(x_arranged_evaluated[index][idx])

# %% [markdown]
# 从上面的输出我们可以看出，每个程序实例中都得进一步迭代，才能得到最内层形状为 `(1, 4)` 的张量。为了进一步理解，让我们基于上述排布，完整地实现一个计算内核看看。

# %%
block_size = Symbol("block_size", constexpr=True)


def arrangement(x, y, block_size=block_size):
    x_arranged = x.tile((1, block_size))
    x_arranged = x_arranged.tile((1, -1))

    y_arranged = y.tile((1, 1))

    return x_arranged, y_arranged


def application(x, y):
    acc = ntl.zeros(y.shape, dtype=y.dtype)

    for i in range(x.shape[1]):
        acc += ntl.sum(x[0, i], axis=-1)

    y = acc


kernel = ninetoothed.make(arrangement, application, (Tensor(2, other=0), Tensor(2)))

# %% [markdown]
# 用自然语言来描述的话，上述计算内核做的事情就是把 `x` 的每一行分块，并且对每个分块求和，再将求和结果累加在一起，然后存入 `y` 对应的分块中。换句话说，就是把 `x` 每行的和存入 `y` 中。大家可能会想，那不是 `tile` 一次就行了，何必搞两次，还要 `for` 一下，这是因为 `block_size` 是有大小限制的，不可以无限大，所以如果 `x` 的一行特别长，就可能会超出限制，这个时候就需要把一行分为多块。事实上如果已知输入 `x` 的列数不可能过大，那么只 `tile` 一次再直接 `ntl.sum` 是完全可以的。顺带一提，这里提供的求和只是为了方便大家理解应用函数中的迭代，并不一定是很高效的实现。
#
# 大家可能注意到了第一个 `Tensor` 中的 `other=0`，这个指的是越界情况下的取值。那什么情况下可能会越界呢？比如如果 `x` 的列数无法被 `block_size` 整除，这时就会有一个程序实例所处理的分块实际上是越过了边界的。之所以这里设置为了 `0`，是因为我们想要求和。具体情况需要具体分析。比如如果我们希望求最大值，那可能就需要设为 `float("-inf")` 了。
#
# 有些小伙伴可能会对 `application` 中的 `ntl.zeros` 和 `ntl.sum` 感兴趣，希望知道什么函数在应用函数内可以使用，什么不可以，那么这里就可以参考 [Triton 的文档](https://triton-lang.org/main/python-api/triton.language.html)，基本上 `triton.language` 中有的，都可以通过 `ninetoothed.language` 来调用，这就像 C++ 中的 `<cstdlib>` 和 C 中的 `<stdlib>`，或者说 C++ 中的 `std::size_t` 和 C 中的 `size_t` 一样。当然，九齿也会添加额外的东西，比如 `Tensor.shape`、`Tensor.dtype`、`Tensor.offsets` 等。
#
# 好，那接下来，让我们运行以下代码来验证一下刚刚定义的计算内核。

# %%
m = 240
n = 620

x = torch.randn(m, n, device=device)
y = torch.empty((m, 1), device=device)

kernel(x, y, block_size=64)

print(x)
print(y)

reference = torch.sum(x, dim=-1, keepdim=True)

print(reference)

assert torch.allclose(y, reference, atol=1e-5)

# %% [markdown]
# ## 致谢
#
# 本项目受到了 [Triton-Puzzles](https://github.com/srush/Triton-Puzzles) 的启发。