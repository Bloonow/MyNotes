[toc]

# Autograd Mechanics

本节内容若无特殊说明，均在`torch.autograd`模块的命名空间中。

本文将概述autograd（自动梯度，automatical gradient）是如何工作和记录操作的，这并非必须了解的内容，但理解该机制有助于编写高效性能的程序。

## 1. How autograd encodes the history

autograd是一个反向自动微分（reverse automatic differentiation）系统。从概念上讲，autograd记录一个图（graph），又称**计算图（compute graph）**，**反向传播图（backward graph）**，它记录创建数据的所有操作，生成一个有向无环图，**叶（leaves）**是输入张量，**根（root）**是输出张量。从根到叶追踪（trace）该图，就可以使用链式法则（chain rule）自动计算梯度。

内部实现上，autograd将这个图表示为函数对象（torch.autograd.Function）的图（实际上是表达式），可通过apply()来计算评估此图的结果。在正向forward时，autograd执行所请求的计算（requested computations），同时构建一个表示函数的图，这些函数是用于计算梯度的，即每个torch.Tensor的grad_fn属性（torch.autograd.graph.Node），它是图的一个入口点（entry point）。当forward完成后，在反向backward中，评估此图（evaluate graph）以计算梯度。

需要注意是，图在每次迭代中都是从头重新创建的，这正是能够使用任意Python控制语句的原因，即可以在每次迭代中改变图的整体形状和大小。在启动训练之前，不必对所有可能的路径进行编码（encode），在运行中所途径的路径，即是所进行微分的部分。也即，PyTorch的计算图是动态的（dynamic）。

对于操作$y=f(x_1,x_2,\cdots,x_n)$来说，节点y的`grad_fn`属性其实就是它的梯度函数，即
$$
\mathtt{gran\_fn} = \nabla y = \nabla f = (\frac{\part f}{\part x_1}, \frac{\part f}{\part x_2}, \cdots, \frac{\part f}{\part x_n})^\top
$$
其中y和x都可以是标量、矢量、矩阵、多维张量，当y不是标量时，是y的每个元素都对自变量求梯度。

### 1.1 Saved tensors

有些操作需要在forward时保存中间结果，以便用于backward。例如，函数$x\mapsto x^2$保存输入x以用来计算梯度。

在自定义`torch.autograd.Function`函数时，在forward期间可以使用save_for_backward()方法保存张量，并在向后backward期间使用saved_tensors检索它们。参阅[Extending PyTorch](https://pytorch.org/docs/stable/notes/extending.html)以了解更多信息。

对于PyTorch预定义的操作，如torch.pow()，张量会根据需要自动保存。可以访问grad_fn的前缀为_saved的属性来查看grad_fn保存了哪些张量。

在内部，为防止循环引用，在保存时会将张量打包（pack），在读取时会将其解包到另一个张量对象中。一个张量是否会被打包到一个不同的张量对象中，取决于该张量是否为它自己的grad_fn的输出。

通常来说，一个张量grad_fn的输出不会是它自己，这时可用grad_fn._saved_self属性访问Function的（第一个）自变量。例如，对于$y=x^2$来说，它的梯度为$\nabla y = 2x$，张量y的grad_fn的输出2x不是它自己，有如下

```python
x = torch.tensor([2., 3.], requires_grad=True)
y = x ** 2
itself = y.grad_fn._saved_self
print(itself, itself is x, itself.equal(x))
"""
tensor([2., 3.], requires_grad=True) True True
"""
```

而当一个张量grad_fn的输出是它自己时，这时可用grad_fn._saved_result属性访问Function的输出。例如，对于$y=e^x$来说，它的梯度为$\nabla y=e^x$，张量y的grad_fn的输出还是它自己，为避免循环引用，读取时会将其解包到另一个张量对象中，有如下

```python
x = torch.tensor([2., 3.], requires_grad=True)
y = torch.exp(x)
result = y.grad_fn._saved_result
print(result, result is y, result.equal(y))
"""
tensor([ 7.3891, 20.0855], grad_fn=<ExpBackward0>) False True
"""
```

可以使用一些hook方法，如pack_hook()和unpack_hook()，来控制打包/解包的行为，参阅[Hooks for saved tensors](https://pytorch.org/docs/stable/notes/autograd.html#saved-tensors-hooks-doc)。

## 2. Gradients for non-differentiable functions

只有当每个元素的函数都可微时，使用自动微分（Automatic Differentiation）的autograd才是有效的。不幸的是许多实际使用的函数都没有该性质，如relu和sqrt在0处的微分。为尽量减少不可微函数的影响，通过依次应用下述规则来定义函数的梯度。对于函数$y=f(x)$来说，有

1. 如果函数可微，且当前点存在梯度，即使用当前点的梯度。
2. 如果函数在某邻域上为凸函数（convex），使用最小范数的次梯度（sub-gradient），它是最速下降方向。
3. 如果函数在某邻域上为凹函数（concave），使用最小范数的超梯度（super-gradient），考虑$-f(x)$并应用上面第2点。
4. 如果函数有界（defined），则通过连续性（continuity）定义当前点的梯度，注意它可能是$\inf$，例如sqrt(0)的梯度。如果可能有多个值，则任意选择一个。
5. 如果函数无界（not defined），例如sqrt(-1)、log(-1)和多数输入为NaN的函数，那么可能会用任意值作为梯度，也有可能抛出错误，但不确定。多数函数会使用NaN作为梯度，但处于性能考虑，有效函数会使用其他值，如log(-1)。
6. 如果函数是非确定的映射（not a deterministic mapping），例如非数值函数（not a mathematical function），它是不可微的。在backward期间，该函数如果要在no_grad环境外对某个张量求导，则会抛出错误。

## 3. Locally disabling gradient computation

有多种方法可以在局部地禁止梯度计算。

要在整个代码块中禁用梯度，可以使用诸如no-grad模式或inference模式之类的上下文管理器。要在更细粒度（fine-grained）禁用子图的梯度计算，可在Tensor域设置张量的requires_grad属性。

除了讨论上述机制外，还有一种评估模式，即nn.Module.eval()，但它不是用于禁用梯度计算的方法。

### 3.1 Setting requires_grad

`requires_grad`是一个标识（flag），通常默认为False，而在被包装到nn.Parameter中时默认为True，这允许在更细粒度（fine-grained）禁止子图的梯度计算。它在forward和backward中都有效。

在forward过程中，一个操作只有在其输入张量中至少有一个需要梯度时才会记录在backward图中。在backward期间，只有requires_grad=True的叶张量才会将梯度累积（accumulate）到它们的`grad`属性中。

需要注意的是，虽然每个张量都有requires_grad标识，但设置标识只有对叶张量（没有grad_fn属性）才有意义，如nn.Module的模型参数（Parameter）。非叶张量（有grad_fn属性）是具有backward图的张量，故需要它们的梯度作为中间结果，以计算叶张量的梯度。由此，所有非叶张量都自动设为requires_grad=True。

设置requires_grad应该是控制模型的哪些部分需要梯度计算的主要方法，例如，在模型微调期间冻结模型预训练的部分。要冻结模型的某些部分，只需对不想更新的参数调用requires_grad_(False)方法。如上所述，由于使用这些参数作为输入的计算不会被记录在前向forward中，因此不会在后向backward中更新它们的grad字段，因为它们不是后向backward图的一部分。

此外，requires_grad也可以在Module级别使用nn.Module.requires\_grad\_()方法进行设置，它对Module的所有参数生效（模型的参数变量默认情况下requires_grad=True）。

### 3.2 Grad Modes

除了设置requires_grad之外，还选择三种grad模式，它们可以影响autograd内部如何处理PyTorch中的计算，包括：grad模式（默认），no-grad模式，inference模式，所有这些模式都可以通过上下文管理器和装饰器进行切换。

### 3.3 Grad Mode (Default Mode)

Grad Mode (Default Mode)是没有启用其他模式时默认采用的模式。它是requires_grad标识唯一生效的模式，在其他两种模式中requires_grad总是被重写为False。

### 3.4 No-grad Mode

在no-grad模式下，表示计算的所有输入都不需梯度。即使有require_grad=True的输入，计算也不会记录在backward图中。

当需要执行不该被autograd记录的操作，但仍要在grad模式下使用这些计算的输出时，可启用no-grad模式。使用上下文管理器可以方便的在代码块或函数作用域禁用梯度。

例如，在编写优化器optimizer时，no-grad模式可能很有用。当执行训练更新时，希望在不被autograd记录更新的情况下就地更新参数。但还需要在下一个前向forward中使用更新后的参数进行grad模式的计算。

torch.nn.init模块在初始化参数时也依赖于no-grad模式，避免在就地更新初始化参数时进行autograd跟踪。

### 3.5 Inference Mode

inference模式是no-grad模式的极端版本。inference模式下的计算不会记录在后向backward图中，但启用inference模式将允许PyTorch进一步加快模型的速度。在inference模式中创建的张量，不能用于退出inference模式后的由autograd记录的计算中。

当执行无需在backward图中记录的计算时，并且不打算在由autograd记录的任何计算中使用inference模式中创建的张量，可启用inference模式。

建议在代码中不需要autograd追踪的部分（如数据处理和模型评估）尝试inference模式如果您在启用inference模式后遇到错误，请确保没有在退出推理模式后，在由autograd记录的计算中使用inference模式中创建的张量。

### 3.6 Evaluation Mode

evaluation模式不是局部禁用梯度计算的机制，但这里仍对它进行相关介绍。

在功能上，nn.Module.eval()或等价的nn.Module.train(False)启用evaluation模式，它与no-grad模式和inference模式完全正交（orthogonal）。方法model.eval()如何影响模型取决于特定的Module以及这些Module是否定义了training模式的特殊行为。

如果模型依赖于nn.Dropout和nn.BatchNorm等模块，则需要调用model.eval()和model.train()切换不同的行为模式。BatchNorm根据训练过程的不同可能会有不同的行为，例如，为了避免进行验证集验证时，更新于训练过程求得的全局指数平滑的统计学信息。

## 4. In-place operations with autograd

在autograd中支持就地操作是一件困难的事情，在大多数情况下不建议使用它们。autograd采用积极的缓冲区释放和重用策略，这使得它非常高效，很少有就地操作显著降低内存使用量的情况。除非在巨大的内存负载压力下运行，否则尽量永远都不要使用就地操作。

限制就地操作适用性的主要原因有两个：

1. 就地操作可能会覆盖计算梯度所需的值。
2. 每个就地操作都需要实现重写计算图的方法。非就地（out-of-place）版本的操作只是分配新的对象并保持对旧图的引用；而就地操作则需要更改Function所有输入的创建者。这可能很棘手，特别是当有许多张量引用相同的存储时（例如通过索引或转置创建的），如果任何其他张量引用了修改后的输入的storage，就地操作/函数将会引发错误。

### 4.1 In-place correctness checks

每个张量都保持一个版本计数器（version counter），任何操作中每次被标记为脏（dirty）时，该版本计数器都会增加（incremente）。当一个Function保存任何用于后向backward的张量时，也会保存包含它们的版本计数器。访问self.saved_tensors时会进行检查，如果它大于保存的值，则会引发错误。这可确保，如果在使用就地函数时没有看到任何错误，便可确定计算出的梯度是正确的。

## 5. Multithreaded Autograd

autograd引擎负责运行计算backward传递所需的所有反向操作。本节将描述多线程环境中的细节。

用户可以用多线程训练模型，并不阻塞地进行并发backward计算，如下示例。

```python
# Define a train function to be used in different threads
def train_fn():
    print(threading.current_thread().getName())
    x = torch.ones([5, 5], requires_grad=True)
    y = (x + 3) * (x + 4) * 0.5
    y.sum().backward()
    # optimizer update

# User write their own threading code to drive the train_fn
threads = []
for _ in range(4):
    p = threading.Thread(target=train_fn, args=())
    p.start()
    threads.append(p)
for p in threads:
    p.join()
```

### 5.1 Concurrency on CPU

当通过Python或C++ API在CPU上多线程运行backward()或grad()时，执行期间会并发backward，而不是按特定顺序执行所有backward调用。

### 5.2 Non-determinism

如果多个线程并发地调用backward()，并且共享输入时，会产生不确定性。

这是因为模型参数是自动跨线程共享的，因此，多个线程可能会在梯度累积（gradient accumulation）过程中访问并尝试累积相同的grad属性。这在技术上是不安全的，而且可能会导致竞争条件，结果可能是无效的。

开发具有共享参数的多线程模型，应该牢记线程模型，并应该理解上述问题。函数式API torch.autograd.grad()可以用来代替backward()来计算梯度，以避免不确定性。

### 5.3 Graph retaining

如果autograd图的一部分在线程之间共享，即forward第一部分单线程运行，第二部分多线程运行，那么图的第一部分是共享的。在这种情况下，在同一图上执行grad()或backward()的不同线程可能会在一个线程运行时破坏图，而另一个线程将崩溃。

autograd将向用户抛出错误，类似于“在没有retain_graph=True的情况下调用两次backward()”，并让用户知道应该使用retain_graph=True。

### 5.4 Thread Safety on Autograd Node

由于autograd允许调用者线程驱动其backward执行，以实现潜在的并行性，因此确保CPU上的共享GraphTask一部分/全部的并行backward()调用的线程安全是很重要的

因为Python的全局解释器锁（Global Interpreter Lock，GIL），自定义autograd.Function自动就是线程安全的。

对于内嵌C++的autograd节点（例如AccumulateGrad,CopySlices）和自定义autograd::Function，autograd引擎使用线程互斥锁（thread mutex lock）来确保autograd节点上可能的读/写状态的线程安全。

### 5.5 No thread safety on C++ hooks

autograd依赖于用户编写线程安全的C++ hook方法。如果想在多线程环境中正确地应用hook，需要应用适当的线程锁，以确保hook方法是线程安全的。

## 6. Autograd for Complex Numbers

当需要复数函数求微分等操作时，已有的优化器Optimizer是可直接使用的。关于具体的数学定义和PyTorch请查看官方文档，这里从略。

# Double Backward With Custom Functions

有时在反向传播图中运行两次或多次backward很有用，例如要计算高阶梯度。不过，要支持重复反向传播（double backward）需要对autograd有一定了解。支持单次backward后向操作的torch.autograd.Function函数，不一定支持两次backward后向操作。本教程将展示如何编写一个支持重复backward的自定义torch.autograd.Function函数，并指出一些需要注意的地方。

对于double backward重复反向传播最直观的示例是，在训练时，一个模块可能会输出多个结果，根据不同的模型输出，可以计算得到多个不同的loss损失值，它们对模型权重更新都有贡献，在多个loss损失值上都调用backward()方法传播梯度，那么对于多个loss所共用的模型层，就会调用多次backward()过程，从而导致double backward重复反向传播。

当编写一个需要支持多次backward向后遍历的自定义torch.autograd.Function函数时，重要的是要知道，在自定义Function函数中执行的操作，何时会被autograd记录，而何时不会被记录；最重要的是，清楚ctx.save_for_backward()如何与之起作用。

自定义Function函数隐式地以两种方式影响grad梯度模式：

- 在forward期间，autograd不会记录在forward函数中执行的任何操作的任何图。当forward函数完成后，自定义Function函数的backward函数会成为forward函数所输出张量的grad_fn属性。
- 在backward期间，如果指定crate_graph为True，则autograd会记录计算图以用于计算backward传递。

接下来，为理解ctx.save_for_backward()是如何与上述代码交互的，此处列举几个示例。

## 1. Saving the Inputs

考虑最简单的平方函数$y=x^2$，它保存一个input输入张量以用于backward过程。当autograd能记录backward传递过程中的操作时，重复backward可自动工作，故当将一个input输入张量保存起来用于backward时，通常无需担心。因为如果input是其他任意需要梯度的张量的函数（即input不是叶张量），则输入input应该持有grad_fn属性。这允许梯度正确传播。

```python
class SquareFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x ** 2

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output * 2 * x

if __name__ == '__main__':
    x = torch.rand((3, 3), dtype=torch.double, requires_grad=True)
    # Use double precision because finite differencing method magnifies errors
    c1 = torch.autograd.gradcheck(SquareFunction.apply, x)
    # Use gradcheck to verify second-order derivatives
    c2 = torch.autograd.gradgradcheck(SquareFunction.apply, x)
    print(c1, c2)
"""
True True
"""
```

可以使用torchviz将计算图可视化，如下所示。可以指定torchviz.make_dot()函数的show_attrs参数和show_saved参数指定为True，以显示更多信息。

```python
import torchviz

x = torch.tensor(3.14, requires_grad=True)
out = SquareFunction.apply(x)
grad_x, = torch.autograd.grad(out, x, create_graph=True)
print(x, out, grad_x, sep='\n')
"""
tensor(3.1400, requires_grad=True) 
tensor(9.8596, grad_fn=<SquareFunctionBackward>)
tensor(6.2800, grad_fn=<MulBackward0>)
"""

DIS = torchviz.make_dot((x, out, grad_x), {'x': x, 'out': out, 'grad_x': grad_x})
DIS.format = 'png'
DIS.view()
```

<img src="PyTorch Autograd.assets/y=x^2.png" style="zoom: 80%;" />

可以看到，输出out对于变量x的梯度，是关于x的函数$\mathrm d{out}/\mathrm dx=2x$，并且被成功构建。

## 2. Saving the Outputs

与前一个示例稍有不同的是$y=e^x$，它保存输出output而不是输入input以用于backward过程，机制是类似的，因为输出也关联到grad_fn属性。

```python
class ExpFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ret = torch.exp(x)
        ctx.save_for_backward(ret)
        return ret

    @staticmethod
    def backward(ctx, grad_output):
        ret, = ctx.saved_tensors
        return grad_output * ret
```

```python
x = torch.tensor(3.14, requires_grad=True)
out = ExpFunction.apply(x)
grad_x, = torch.autograd.grad(out, x, create_graph=True)
print(x, out, grad_x, sep='\n')
"""
tensor(3.1400, requires_grad=True)
tensor(23.1039, grad_fn=<ExpFunctionBackward>)
tensor(23.1039, grad_fn=<MulBackward0>)
"""

DIS = torchviz.make_dot((x, out, grad_x), {'x': x, 'out': out, 'grad_x': grad_x})
DIS.format = 'png'
DIS.view()
```

<img src="PyTorch Autograd.assets/y=e^x.png" style="zoom:80%;" />

## 3. Saving Intermediate Results

更棘手的情况是要保存forward过程的中间临时张量，此处以双曲正弦函数为例
$$
\begin{align}
y  &= \sinh(x) = \frac{e^x - e^{-x}}{2} \\
y' &= \frac{e^x + e^{-x}}{2}
\end{align}
$$
由微分公式可以看到，在backward过程中，会重用exp(x)和exp(-x)这两个中间结果。

注意，在forward()过程中，通过ctx.save_for_backward()保存的张量tmp，在backward()过程中通过ctx.saved_tensors将张量tmp取出时，**其requires_grad属性与保存前一致**；而若将张量tmp同时作为forward()的返回值进行返回，即如return out,tmp之类的形式，则在backward()中再将其从ctx取出时，其requires_grad属性就变为True。

我们知道，通过默认torch操作得到的张量其默认requires_grad属性为False，故对于需要在backward()过程中用到中间张量的情况，若只是用ctx保存，则其requires_grad属性为False，那么它不会被记录在反向传播计算图中，则会造成计算图的断裂。故对于需要用ctx保存的中间张量，还需要将其作为forward()的返回值进行返回，保证其在backward()过程中从ctx取出后的requires_grad属性为True。

也就是说，中间张量不应该只调用ctx.save_for_backward()保存并在backward过程中使用，因为forward是在无梯度模式下执行的，如果在backward中只使用forward保存的中间张量来计算梯度，则backward梯度图将不包括计算中间张量的操作，这会导致不正确的梯度。

```python
class SinhFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        expx = torch.exp(x)
        expnegx = torch.exp(-x)
        ctx.save_for_backward(expx, expnegx)
        return (expx - expnegx) / 2, expx, expnegx

    @staticmethod
    def backward(ctx, grad_output, grad_expx, grad_expnegx):
        expx, expnegx = ctx.saved_tensors
        grad_input = grad_output * (expx + expnegx) / 2
        grad_input += grad_expx * expx
        grad_input -= grad_expnegx * expnegx
        return grad_input

def sinh(x):
    return SinhFunction.apply(x)[0]
```

```python
x = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
out = torch.sum(sinh(x))
grad_x, = torch.autograd.grad(out, x, create_graph=True)
print(x, out, grad_x, sep='\n')
"""
tensor([[1., 2.], [3., 4.]], requires_grad=True)
tensor(42.1099, grad_fn=<SumBackward0>)
tensor([[ 1.5431,  3.7622], [10.0677, 27.3082]], grad_fn=<SubBackward0>)
"""

DIS = torchviz.make_dot((x, out, grad_x), {'x': x, 'out': out, 'grad_x': grad_x})
DIS.format = 'png'
DIS.view()
```

<img src="PyTorch Autograd.assets/y=e^x-e^{-x}div2.png" style="zoom:80%;" />

## 4. Saving Intermediate Results: What not to do

此处列举一个错误的示例，仍然是以双曲正弦函数为例，若不将中间结果作为forward输出返回时，如下示例。

```python
class SinhBadFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        expx = torch.exp(x)
        expnegx = torch.exp(-x)
        ctx.save_for_backward(expx, expnegx)
        return (expx - expnegx) / 2

    @staticmethod
    def backward(ctx, grad_output):
        expx, expnegx = ctx.saved_tensors
        grad_input = grad_output * (expx + expnegx) / 2
        return grad_input
```

```python
x = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
out = torch.sum(SinhBadFunction.apply(x))
grad_x, = torch.autograd.grad(out, x, create_graph=True)

DIS = torchviz.make_dot((x, out, grad_x), {'x': x, 'out': out, 'grad_x': grad_x})
DIS.format = 'png'
DIS.view()
```

<img src="PyTorch Autograd.assets/y=e^x-e^{-x}div2 Bad.png" style="zoom:80%;" />

可以看到，grad_x甚至不会有backward图，它不是backward图的一部分，因为它只是exp和expnegx函数，而它们（exp和expnegx）不需要grad梯度。

## 5. When backward is not Tracked

最后一个示例，当autograd可能根本无法追踪某个Function函数的backward过程中的梯度时。需要注意的是，对于下降到C++层面的操作算子，若仅仅使用了Torch在C++层面提供的ATen库函数API接口，而没有进行直接对张量底层的一维存储数据进行操作，则PyTorch的torch.autograd机制仍能自动跟踪。

假设有一个自定义操作算子y=f(x)，实现为MyFunction函数类，其正向传播和反向传播分别为f_forward()函数和f_backward()函数，如下所示。

```python
# Assume these two functions are implemented in C++.
def f_forward(x): ...           # return y
def f_backward(grad_y, x): ...  # return grad_x
```

假设，这两个函数都是使用非PyTorch库（如SciPy或NumPy），或使用C++扩展编写的，所以若要构建反向传播图，torch.autograd无法跟踪f_backward()过程内部计算过程。所以，我们将f_backward(grad_y,x)包装到一个HelpFunction的forward()过程中，即包装到HelpFunction.forward(grad_y,x)函数中，而该HelpFunction函数是能够被autograd跟踪的。

那么，对于HelpFunction来说，我们还要实现它的backward()过程，它要实现HelpFunction.forward()的输出对输入的梯度，因为HelpFunction.forward()是对f_backward()的包装，所以，HelpFunction.backward()要实现f_backward()的输出对输入的梯度，也即，对于运算grad_x=f_backward(grad_y,x)来说，求grad_x关于grad_y和x的梯度。

因为grad_x是HelpFunction.forward()的输出，所以HelpFunction.backward()实际接受参数的是grad_x的梯度，即grad_grad_x，也即是，HelpFunction.backward()需要由梯度grad_grad_x，求经过运算f_backward()的，对于grad_y和x的梯度。

为避免混淆，此处取f_backward()别名为help_fb()，如下所示。

```python
# The implemention of help_fb(a, b) is same to f_backward(grad_y, x).
# a = grad_y
# b = x
# c = grad_x
def help_fb(a, b): ...  # return c
```

那么需要对于计算c=help_fb(a,b)，由c的梯度grad_c，求关于a和b的梯度grad_a和grad_b，即$\nabla_c\dfrac{\part c}{\part a}$和$\nabla_c\dfrac{\part c}{\part b}$，可将之实现为两个函数，然后由HelpFunction.backward()调用，如下所示。

```python
def cal_grad_a(grad_c, a, b): ...  # return grad_a
def cal_grad_b(grad_c, a, b): ...  # return grad_b
```

这两个函数也可由C++/CUDA等扩展实现，而无需进一步跟踪。至此，HelpFunction.backward()使用cal_grad_a()和cal_grad_b()函数，实现了HelpFunction.forward()的输出对输入的梯度，并返回即可。

此处以求立方操作$y=x^3$为例，其实现如下所示。

```python
# Assume these four functions are implemented in C++.
def cube_forward(x: Tensor):
    return x**3
def cube_backward(grad_cube: Tensor, x: Tensor):
    return grad_cube * 3 * x**2
def cal_d_grad_cube(grad_help: Tensor, x: Tensor):
    return grad_help * 3 * x**2
def cal_d_x(grad_help: Tensor, grad_cube: Tensor, x: Tensor):
    return grad_help * grad_cube * 6 * x

class CubeHelpFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grad_cube: Tensor, x: Tensor):
        ctx.save_for_backward(grad_cube, x)
        return cube_backward(grad_cube, x)
    @staticmethod
    def backward(ctx, grad_help: Tensor):
        grad_cube, x = ctx.saved_tensors
        d_grad_cube = cal_d_grad_cube(grad_help, x)
        d_x = cal_d_x(grad_help, grad_cube, x)
        return d_grad_cube, d_x

class CubeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor):
        ctx.save_for_backward(x)
        return cube_forward(x)
    @staticmethod
    def backward(ctx, grad_out):
        x, = ctx.saved_tensors
        # return cube_backward(grad_out, x)
        return CubeHelpFunction.apply(grad_out, x)

def cube(x: Tensor):
    return CubeFunction.apply(x)
```

```python
x = torch.randn((3, 3), requires_grad=True, dtype=torch.double)
c1 = torch.autograd.gradcheck(CubeFunction.apply, x)
c2 = torch.autograd.gradgradcheck(CubeFunction.apply, x)
print(c1, c2)
"""
True True
"""
```

```python
x = torch.tensor([[1., 2., 3., 4.]], requires_grad=True)
out = torch.sum(CubeFunction.apply(x))
grad_x, = torch.autograd.grad(out, x, create_graph=True)
print(x, out, grad_x, sep='\n')
"""
tensor([[1., 2., 3., 4.]], requires_grad=True)
tensor(100., grad_fn=<SumBackward0>)
tensor([[ 3., 12., 27., 48.]], grad_fn=<CubeHelpFunctionBackward>)
"""

DIS = torchviz.make_dot((x, out, grad_x), {'x': x, 'out': out, 'grad_x': grad_x})
DIS.format = 'png'
DIS.view()
```

<img src="PyTorch Autograd.assets/y=x^3.png" style="zoom:80%;" />

总而言之，重复反向传递（double backward）是否适用于自定义的torch.autograd.Function函数，仅仅取决于autograd是否可以跟踪backward的反向传递过程。

本文前两个示例中，可以直接使用double backward；而在后两个示例中，演示了如何在原本无法跟踪的情况下跟踪backward反向函数。

# torch.autograd

本节内容若无特殊说明，均在`torch.autograd`模块的命名空间中。

该模块提供了一些类和函数，实现了任意标量值函数（scalar valued functions）的自动微分（automatic differentiation）。不推荐修改本模块内容，用户只需声明Tensor张量并指定requires_grad=True，即可实现自动微分。目前，只支持浮点型和复数型张量的自动微分。

```python
def backward(tensors, grad_tensors=None, retain_graph=None, create_graph=False,
             grad_variables=None, inputs=None) -> None
```

计算给定tensors张量对图的叶张量的梯度（之和），并将梯度累积到叶张量的grad属性之中。利用链式法则对图形进行微分。

如果tensor张量是非标量函数所得的值，且它需要梯度，则将计算雅可比向量积（Jacobian-vector product）。在这种情况下，该函数还需要指定grad_tensor，它应该是一个匹配长度的序列。grad_tensor应是雅可比向量积中的“向量”，通常是tensor每个元素对相应叶张量的预计算梯度。对于所有不需要梯度的张量来说，None是一个可接受的值。

backward()函数在叶张量中累积梯度，需要在调用之前将张量的grad属性设置为0或None。关于累积梯度的内存布局，将在后面介绍。

注意，在create_graph=True的情况下使用backward()函数，将在模型参数与其梯度之间创建一个循环引用，这可能会导致内存泄漏。推荐在创建图时使用autograd.grad()来避免这种情况。如果必须使用这个函数，确保在使用后将参数的grad字段重置为None，以打破循环并避免泄漏。

当提供了inputs张量且不是叶张量节点时，当前的实现将调用它的grad_fn，即使不是严格需要它来获得叶张量的梯度。

```python
def grad(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False,
         only_inputs=True, allow_unused=False, is_grads_batched=False) -> Tuple[torch.Tensor, ...]
```

计算给定outputs张量对图的inputs张量的梯度（之和），并将返回所计算的梯度。需要注意，该函数不会将梯度累积到inputs的grad属性之中。

如果output张量是非标量函数所得的值，且它需要梯度，则将计算雅可比向量积。在这种情况下，该函数还需要指定grad_output，它应该是一个匹配长度的序列。grad_output应是雅可比向量积中的“向量”，通常是output每个元素对相应input张量的预计算梯度。对于所有不需要梯度的张量来说，None是一个可接受的值。

上述这种情况，通常出现在链式求导的中间步骤中，例如$\mathbf x\mapsto\mathbf y\mapsto z$，其中$\mathbf x$是输入张量，$\mathbf y$是中间结果的张量，$z$是经过损失函数后的标量。设$\mathbf x\in\R^n,\mathbf y\in\R^m,z\in\R$，若输入张量和中间结果张量不是列向量（而是多维张量）时，内部实现会将其view(-1)看作列向量，并无实际影响。若要求$z$对$\mathbf x$的梯度，则需要计算雅可比向量积
$$
\begin{align}
\nabla_\mathbf x z &= \frac{\part z}{\part\mathbf x} = \frac{\part\mathbf y}{\part\mathbf x}\frac{\part z}{\part\mathbf y} = \mathbf J^\top\mathbf v \\
&= \begin{bmatrix} 
\frac{\part y_1}{\part x_1} & \frac{\part y_2}{\part x_1} & \cdots & \frac{\part y_m}{\part x_1} \\
\frac{\part y_1}{\part x_2} & \frac{\part y_2}{\part x_2} & \cdots & \frac{\part y_m}{\part x_2} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\part y_1}{\part x_n} & \frac{\part y_2}{\part x_n} & \cdots & \frac{\part y_m}{\part x_n}
\end{bmatrix} \begin{bmatrix}
\frac{\part z}{\part y_1} \\ \frac{\part z}{\part y_2} \\ \vdots \\ \frac{\part z}{\part y_m}
\end{bmatrix} = \begin{bmatrix}
\frac{\part z}{\part x_1} \\ \frac{\part z}{\part x_2} \\ \vdots \\ \frac{\part z}{\part x_n}
\end{bmatrix}
\end{align}
$$
其中，$\mathbf J$是$\mathbf{y}$对于$\mathbf{x}$的雅可比矩阵，$\mathbf v$即是grad_output参数，所谓雅可比向量积中的“向量”，当其实际上是张量时，内部实现会通过view(-1)将其看作向量。

如下，一个手动链式求导的例子。

```python
x = torch.tensor([2., 3., 4.], requires_grad=True)
y = torch.pow(x, 2)
z = torch.sum(y)
dz_dy = torch.autograd.grad(z, y, retain_graph=True)[0]
dz_dx = torch.autograd.grad(y, x, grad_outputs=dz_dy, retain_graph=True)[0]
g = torch.autograd.grad(z, x)[0]
what(dz_dy, dz_dx, g)
"""
name: dz_dy | type: <class 'torch.Tensor'> | shape: torch.Size([3]) | value: tensor([1., 1., 1.])
name: dz_dx | type: <class 'torch.Tensor'> | shape: torch.Size([3]) | value: tensor([4., 6., 8.])
name: g     | type: <class 'torch.Tensor'> | shape: torch.Size([3]) | value: tensor([4., 6., 8.])
"""
```

此外，若通过grad()直接计算张量对张量的微分，会抛出错误，给grad_output参数传递全为1的张量，即可使用grad()求张量对张量的微分，如下一个例子。

```python
x = torch.tensor([[2., 3.], [4., 5.]], requires_grad=True)
y = torch.matmul(x.t(), x)
g = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(x))[0]
print(g)
"""
tensor([[10., 10.], [18., 18.]])
"""
```

## 1. Gradient layouts

### 1.1 Default radient layouts

当非稀疏参数param在torch.autograd.backward()或torch.Tensor.backward()期间接收到非稀疏梯度时，其属性param.grad按如下方式累积。

如果param.grad初始化为None，

1. 若param是非重叠密集型的存储布局，则会根据param的跨度（strides）来创建grad，因此grad与param的布局相匹配。
2. 否则，grad以行主序连续跨度（rowmajor-contiguous strides）的存储布局被创建。

如果param已有非稀疏的grad属性，

3. 如果create_graph=False，则backward()会就地累积梯度new_grad到grad属性，并保持其strides跨度。
4. 如果create_graph=True，则backward()会用grad+new_grad替换张量原来的grad属性，并尝试匹配原来的strides跨度，但不保证。

默认行为可以获得最佳性能，即在第一次backward()调用之前初始化grad为None，让其根据1或2创建存储布局，并在之后根据3或4保留布局。

调用model.zero_grad()或optimizer.zero_grad()不会影响grad属性的布局。实际上，在每次累积迭代之前，将所有grad属性置为None，如

```python
for it in iterations:
    for param in model.parameters():
        param.grad = None
    loss.backward()
```

```python
for it in iterations:
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
```

这样，grad每次都根据1或2重新创建，这是model.zero_grad()或optimizer.zero_grad()的一种有效替代方案，对于某些网络模来说，可以提高其性能。

### 1.2 Manual gradient layouts

如果需要手动控制grad的strides跨度，可以在第一次backward()之前，将param.grad指定为具有特定stride跨度布局的全零张量torch.zeros(layout=stride)即可，并且永远不要将其重置为None。

## 2. Tensor with autograd

本小节内容在`torch`模块的命名空间中，列举的是torch.Tensor类的一些属性和方法。

```python
torch.Tensor.grad
```

该属性默认为None，并在第一次调用backward()计算self的梯度时成为一个张量。

```python
torch.Tensor.requires_grad
```

如果需要为该张量计算梯度，则为True，否则为False。

```python
torch.Tensor.is_leaf
```

按照惯例，所有requires_grad为False的张量都是叶张量。

```python
class Tensor(torch._C._TensorBase):
	def backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None)
```

计算当前张量的梯度。

```python
class Tensor(torch._C._TensorBase):
    def detach(self) -> Tensor
    def detach_(self) -> Tensor
```

detach()方法返回一个与当前图分离的新张量。detach_()方法返回一个与当前图分离的新张量，并使其成为叶节点。

## 3. Custom Functions

### 3.1 torch.autograd.Function

```python
class Function(with_metaclass(FunctionMeta, _C._FunctionBase, FunctionCtx, _HookMixin)):
    def __init__(self, *args, **kwargs)
    def __call__(self, *args, **kwargs)
    def apply(self, *args)
    @staticmethod
    def forward(ctx, *args, **kwargs) -> Any
    @staticmethod
    def backward(ctx, *grad_outputs) -> Any
    @staticmethod
    def jvp(ctx, *grad_inputs) -> Any  # alias to forward
    @staticmethod
    def vjp(ctx, *grad_inputs) -> Any  # alias to backward
```

用于创建自定义函数torch.autograd.Function的基类，即创建自定义操作算子operator。

创建一个自定义autograd.Function，需要继承这个类并实现静态方法forward()和backward()。若要使用自定义算子op，请调用类方法apply()，而不要直接调用forward()。

为确保正确性和最佳性能，请确保在ctx上调用正确的方法，并使用torch.autograd.gradcheck()验证backward()函数。如下一个例子所示。

```python
class Exp(Function):
    @staticmethod
    def forward(ctx, inputs):
        result = inputs.exp()
        ctx.save_for_backward(result)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        return grad_output * result
    
# Use it by calling the apply method
output = Exp.apply(inputs)
```

参考文档[Extending torch.autograd](https://pytorch.org/docs/stable/notes/extending.html#extending-autograd)以了解更多使用细节。

```python
class Function:
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any
```

forward()函数应被所有子类覆盖。有两种定义forward()的方法。

第一种是结合forward()和上下文ctx，如下示例。

```python
class Function:
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any
```

它必须接受上下文ctx作为第一个参数，后面跟着任意数量的参数（张量或其他类型）。见[Combined or separate forward() and setup_context()](https://pytorch.org/docs/stable/notes/extending.html#combining-forward-context)了解更多信息。

第二种是分离forward()和上下文ctx，如下示例。

```python
class Function:
    @staticmethod
    def forward(*args: Any, **kwargs: Any) -> Any

    @staticmethod
    def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None
```

在这种方法中，forward()不再接受ctx参数。还需覆盖torch.autograd.Function.setup_context()的静态方法来设置并处理ctx对象，该方法的output参数是forward()的输出，inputs参数是forward()的输入组成的元组。见[Extending torch.autograd](https://pytorch.org/docs/stable/notes/extending.html#extending-autograd)了解更多信息。

上下文ctx可用于存储任意数据，然后在backward期间检索这些数据。如果张量打算用于backward()或jvp()，应使用ctx.save_for_backward()保存。注意，张量不应该直接存储在ctx上，尽管为了向后兼容，目前没有强制这样做。

```python
class Function:
    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any
```

backward()函数定义在backward自动微分过程中，对自定义操作求微分的公式，backward()函数（也即vjp()函数）。该函数应被所有子类覆盖。

backward()必须接受上下文ctx作为第一个参数，若干个grad_outputs张量，其数目应与forward()函数返回值outputs的个数相同，若forward()函数返回的是非张量，则传入None作为grad_outputs参数。

对于backward()函数来说，它应返回与forward()函数的inputs张量相同数量的输出。backward()的每个参数都是给定outputs张量的梯度，每个返回值都应该是相应inputs张量的梯度。如果某个input不是一个张量，或者是一个不需要梯度的张量，则可以将None作为该input的梯度。

上下文ctx可用于检索forward期间保存的张量。它还有一个布尔元组的ctx.needs_input_grad属性，表示每个input是否需要梯度。例如，在backward()若检查到ctx.needs_input_grad[0]==True，则forward()的第一个input需要根据grad_output计算梯度。

### 3.2 Context method mixins

在创建自定义函数torch.autograd.Function时，torch.autograd.function.FunctionCtx上下文类的对象ctx可以使用以下方法。

```python
class FunctionCtx:
    def save_for_backward(self, *tensors: torch.Tensor)
    def save_for_forward(self, *tensors: torch.Tensor)
    def mark_dirty(self, *args: torch.Tensor)
    def mark_shared_storage(self, *pairs)
    def mark_non_differentiable(self, *args: torch.Tensor)
    def set_materialize_grads(self, value: bool)
```

mark_dirty()函数用于标记给定张量在就地操作中被修改，它只能从forward()方法内部调用，且至多调用一次，并且所有参数都应该是inputs。对每个在forward()调用中被修改的张量，都应该调用这个函数，以确保PyTorch检查的正确性。在修改之前还是之后调用mark_dirty()函数无关紧要。如下所示。

```python
class Inplace(Function):
    @staticmethod
    def forward(ctx, x):
        x_npy = x.numpy()  # x_npy shares storage with x
        x_npy += 1
        ctx.mark_dirty(x)
        return x
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        return grad_output

a = torch.tensor(1., requires_grad=True).clone()
b = a * a
# This would lead to wrong gradients! 
# But the engine would not know unless we mark_dirty().
Inplace.apply(a)
# RuntimeError: one of the variables needed for gradient computation 
# has been modified by an inplace operation.
b.backward()
```

save_for_backward()函数用于保存给定的张量，以在backward()函数中使用。它只能从forward()方法内部调用，且至多调用一次，而且只能用张量作为参数。

所有打算在backward中使用的张量都应该用save_for_backward保存，而不是直接在ctx上保存，以防止不正确的梯度和内存泄漏。save_for_backward()支持应用保存的张量hook方法。详见之后的torch.autograd.graph.saved_tensors_hooks()部分。

在backward()中，可以通过ctx.saved_tensors属性访问保存的张量。在将它们返回给用户之前，会进行检查，以确保没有就地操作修改了其内容。如下例子。

```python
class Func(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, y: torch.Tensor, z: int):
        w = x * z
        out = x * y + y * z + w * y
        ctx.save_for_backward(x, y, w, out)
        ctx.z = z  # z is not a tensor
        return out
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        x, y, w, out = ctx.saved_tensors
        z = ctx.z
        gx = grad_out * (y + y * z)
        gy = grad_out * (x + z + w)
        gz = None
        return gx, gy, gz

a = torch.tensor(1., requires_grad=True)
b = torch.tensor(2., requires_grad=True)
c = 4
d = Func.apply(a, b, c)
```

请注意，如果save_for_backward()需要为backward()保存既不是forward()输入也不是输出的中间张量，则可能需要支持重复后向（double backward），可以根据backward()过程中的输入重新计算中间节点，或者将中间节点作为自定义Function的输出返回。更多细节请参阅[double backward tutorial](https://pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html)文档。

不支持重复backward的自定义Function应该用@once_differentiable修饰其backward()方法，这样执行重复backward就会引发错误。

## 4. Numerical gradient checking

```python
def gradcheck(
    func: Callable, inputs, *, eps=1e-6, atol=1e-5, rtol=1e-3,
    raise_exception=True, check_sparse_nnz=False, nondet_tol=0.0,
    check_undefined_grad=True, check_grad_dtypes=False, check_batched_grad=False,
    check_batched_forward_grad=False, check_forward_ad=False, check_backward_ad=True,
    fast_mode=False,
) -> bool
```

检查inputs张量的“有限差分梯度”与“解析梯度”，张量是浮点型或复数型，且其requires_grad=True。使用torch.allclose()来检查数值（numerical）梯度和解析（analytical）梯度。

注意，默认值是为双精度input设计的。如果input精度较低，例如FloatTensor，则此检查可能会失败。gradcheck在不可微的点上评估时可能会失败，因为通过有限差分数值计算的梯度可能与解析计算的梯度不同（不一定因为两者都是不正确的）。

```python
def gradgradcheck(
    func: Callable, inputs, grad_outputs=None, *, eps=1e-6, atol=1e-5, rtol=1e-3,
    gen_non_contig_grad_outputs=False, raise_exception=True, nondet_tol=0.0,
    check_undefined_grad=True, check_grad_dtypes=False, check_batched_grad=False,
    check_fwd_over_rev=False, check_rev_over_rev=True, fast_mode=False,
) -> bool
```

检查inputs张量和grad_outputs张量的“有限差分梯度”与“解析梯度”，张量是浮点型或复数型，且其requires_grad=True。使用torch.allclose()来检查数值（numerical）梯度和解析（analytical）梯度。

该函数检查通过grad_outputs梯度进行的反向传播（backpropagate）是否正确。

## 5. Anomaly detection

```python
class detect_anomaly:
    def __init__(self, check_nan=True) -> None
    def __enter__(self) -> None
    def __exit__(self, *args) -> None
```

可为autograd引擎启用异常检测的上下文管理器。

在启用检测的情况下运行forward，将允许backward打印导致backward失败的forward操作的回溯（traceback）信息。如果check_nan=True，任何生成NaN的backward计算都将引发错误。

注意，此模式应仅在调试时启用，因为不同的测试将降低程序执行速度。如下一个例子所示。

```python
class MyFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp):
        return inp.clone()
    @staticmethod
    def backward(ctx, g0):
        # Error during the backward pass
        raise RuntimeError("Some error in backward")
        return g0.clone()

with torch.autograd.detect_anomaly():
    inp = torch.rand([10, 10], requires_grad=True)
    out = MyFunc.apply(inp).sum()
    out.backward()
```

```python
class set_detect_anomaly:
    def __init__(self, mode, check_nan=True) -> None
    def __enter__(self) -> None
    def __exit__(self, *args) -> None
```

设置autograd引擎异常检测开关的上下文管理器。set_detect_anomaly将根据mode参数是否为True来启用或禁用autograd异常检测。它可以用作上下文管理器，也可以用作函数。有关异常检测行为的详细信息，请参见上面的detect_anomaly。

## 6. Autograd graph

本节内容若无特殊说明，均在`torch.autograd.graph`模块的命名空间中。

autograd暴露了一些方法，允许在backward期间检查计算图并干预一些行为。

如果启用了grad_mode模式，且至少有一个输入张量需要梯度，那么若某个torch.Tensor张量是autograd所记录操作的输出，则该Tensor的grad_fn属性将是一个torch.autograd.graph.Node对象，否则为None。

```python
class Node:
    def name(self) -> str
    def next_functions(self) -> Tuple[Tuple[Optional['Node'], int], ...]
    def metadata(self) -> dict
```

有些操作需要在forward遍历时保存中间结果，以便用于backward遍历。这些中间结果可以保存在grad_fn的属性中。

还可以使用hook函数定义如何打包/解包这些保存的张量。一种常见的应用是将这些中间结果保存到磁盘或CPU，而不是将它们留在GPU上，来换取GPU内存。如果模型在评估期间适用GPU，这尤其有用。具体详见[Hooks for saved tensors](https://pytorch.org/docs/stable/notes/autograd.html#saved-tensors-hooks-doc)。

```python
class saved_tensors_hooks:
    def __init__(self, pack_hook: Callable, unpack_hook: Callable)
    def __enter__(self)
    def __exit__(self, *args)
```

上下文管理器，为保存的张量设置一对打包/解包hook函数。使用此上下文管理器可以定义在保存之前如何打包操作的中间结果，以及在检索时如何解包。

在该上下文中，每当操作保存一个用于backward的张量时，都将调用pack_hook函数，这包括使用save_for_backward()保存的中间结果，以及由PyTorch定义的操作记录的结果。pack_hook的输出将存储在计算图中，而不是原始张量中。

当需要访问保存的张量时，即在执行torch.Tensor.backward()或torch.autograd.grad()时，都将调用unpack_hook函数。它接受pack_hook返回的打包对象作为参数，并应该返回一个与原始张量内容相同的张量。即unpack_hook(pack_hook(t))返回的结果应当完全等于t，无论是value,size,dtype还是device。

```python
class MyExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        print('### MyExp.forward()')
        inp = args[0]
        out = inp.exp()
        ctx.save_for_backward(out)
        return out
    @staticmethod
    def backward(ctx, *grad_outputs):
        print('### MyExp.backward()')
        grad_output = grad_outputs[0]
        out = ctx.saved_tensors[0]
        return out * grad_output

class SelfDeletingTempFile:
    def __init__(self):
        self.path = os.path.join('G:/', str(uuid.uuid4()))
    def __del__(self):
        os.remove(self.path)

def my_pack_hook(tensor):
    print('### my_pack_hook()')
    temp_file = SelfDeletingTempFile()
    torch.save(tensor, temp_file.path)
    return temp_file
def my_unpack_hook(temp_file):
    print('### my_unpack_hook()')
    return torch.load(temp_file.path)

x = torch.tensor([2., 3., 4.], requires_grad=True)
with torch.autograd.graph.saved_tensors_hooks(my_pack_hook, my_unpack_hook):
    y = MyExp.apply(x)
y = y.sum()
y.backward()
print(x.grad)
"""
### MyExp.forward()
### my_pack_hook()
### MyExp.backward()
### my_unpack_hook()
tensor([ 7.3891, 20.0855, 54.5981])
"""
```

```python
class save_on_cpu(saved_tensors_hooks):
    def __init__(self, pin_memory=False):
        def pack_to_cpu(tensor):
            if not pin_memory:
                return (tensor.device, tensor.cpu())
            packed_tensor = torch.empty(
                tensor.size(), dtype=tensor.dtype, layout=tensor.layout, 
                pin_memory=(torch.cuda.is_available() and not tensor.is_sparse))
            packed_tensor.copy_(tensor)
            return (tensor.device, packed_tensor)
        
        def unpack_from_cpu(packed_tensor):
            device, tensor = packed_tensor
            return tensor.to(device, non_blocking=pin_memory)
        
        super().__init__(pack_to_cpu, unpack_from_cpu)
```

上下文管理器，在forward期间将张量将存储到CPU内存上，在backward期间再从CPU内存中检索。

在该上下文中，在forward期间保存在图中的中间结果将被移动到CPU内存，然后在backward需要时复制回原设备。如果图已经在CPU上，则不会执行张量复制。

此上下文管理器可以用来权衡GPU内存占用和计算，例如，模型在训练期间不适合GPU内存时。

对于上述一个例子，使用save_on_cpu的示例如下。

```python
x = torch.tensor([2., 3., 4.], device='cuda', requires_grad=True)
with torch.autograd.graph.save_on_cpu():
    y = MyExp.apply(x)
y = y.sum()
y.backward()
print(x.grad)
"""
tensor([ 7.3891, 20.0855, 54.5981], device='cuda:0')
"""
```

## 7. Profiler

本节内容若无特殊说明，均在`torch.autograd.profiler`模块的命名空间中。

autograd包含一个分析器（profiler），可以检查模型中不同算子（operators）的成本（cost），包括CPU和GPU两种情况。

目前有三种实现模式，使用profile()，只使用CPU进行分析；使用emit_nvtx()，注册CPU和GPU活动，基于nvprof英伟达性能分析器；使用emit_itt()，基于VTune英特尔性能分析器。

```python
class profile:
    def __init__(self, enabled=True, *, use_cuda=False, record_shapes=False,
            with_flops=False, profile_memory=False, with_stack=False,
            with_modules=False, use_kineto=False, use_cpu=True)
    def __enter__(self)
    def __exit__(self, exc_type, exc_val, exc_tb)
    def table(self, sort_by=None, row_limit=100, max_src_column_width=75, header=None, top_level_events_only=False)
    def key_averages(self, group_by_input_shape=False, group_by_stack_n=0)
    def total_average(self)
    def export_chrome_trace(self, path)
```

上下文管理器，管理autograd分析器状态并保存结果概要（summary）。在底层，它只是记录以C++执行的函数事件，并将这些事件暴露给Python。可以将任何代码包装到该上下文中，但它只会报告PyTorch函数的运行时间。注意，profiler是线程局部的，并会自动传播（propagate）到异步任务中。

```python
with torch.autograd.profiler.profile(use_cuda=True, profile_memory=True) as prof:
    for samples, targets in data_loader:
        outputs = model(samples)
        loss = loss_fn(outputs, targets)
        # Back Propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
# print(prof.key_averages())
print(prof.key_averages().table(sort_by='self_cpu_time_total'))
```

```python
class emit_nvtx:
    def __init__(self, enabled=True, record_shapes=False)
class emit_itt:
    def __init__(self, enabled=True, record_shapes=False)
```

