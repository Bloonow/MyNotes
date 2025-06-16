[toc]

# Automatic Mixed Precision

本节内容若无特殊说明，均在`torch.amp`模块、`torch.cpu.amp`模块、`torch.cuda.amp`模块的命名空间中。

torch.amp提供了混合精度的方便使用方法，其中一些操作使用torch.float32精度的浮点数据类型，而其他一些操作使用更低精度的torch.float16或者torch.bfloat16浮点数据类型。有些操作，例如矩阵乘法与卷积操作，在低精度模式下的执行速度会快很多；而其他操作，例如归约，通常需要float32的动态精度范围。混合精度试图将每个操作匹配到其合适的数据类型，以减少运行时间和内存占用。

> FP16采用1bit表示符号，5bit表示指数，10bit表示尾数。BF16采用1bit表示符合，8bit表示指数，7bit表示尾数。

通常，使用torch.float16进行自动混合精确训练（automatic mixed precision training）通常需要结合torch.amp.autocast()上下文管理器和torch.cuda.amp.GradScaler类一起使用，不过它们也是模块化的，如果需要也可以单独使用。

对于CPU设备和GPU设备来说，上下文管理器torch.amp.autocast()分别提供了相应的API接口。torch.amp.autocast('cuda')等价于torch.cuda.amp.autocast()，而torch.amp.autocast('cpu')等价于torch.cpu.amp.autocast()，需要注意的是，目前在CPU设备上只支持torch.bfloat16的低精度运算。

## 1. Autocasting

```python
class autocast:
    def __init__(self, device_type: str, dtype=None, enabled=True, cache_enabled=None):
    def __enter__(self)
    def __exit__(self, exc_type, exc_val, exc_tb)
```

torch.amp.autocast类的实例是一个上下文管理器或装饰器，允许其作用范围内的操作（operations）以混合精度的模式运行。当进入torch.amp.autocast上下文管理器的作用范围时，张量可以是任意数据类型，在torch.amp.autocast的作用范围内，不应对模型或输入张量调用half()或bfloat16()方法。

自动精度转换torch.amp.autocast的作用范围内只包含forward正向过程和loss_fn损失计算过程即可，而无需包含backward反向过程。因为backward反向过程会使用与forward正向过程相同的数据类型，而forward正向过程已经自动使用低精度计算，则backward反向过程也会自动使用低精度计算。如下所示。

```python
model = MyNet().to('cuda')
optimizer = torch.optim.SGD(model.parameters(), lr=1.e-3)
loss_fn = torch.nn.MSELoss()

for input, target in zip(inputs, targets):
    optimizer.zero_grad()
    with torch.amp.autocast(device_type='cuda'):
        output = model(input)
        loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
```

torch.amp.autocast也可以用作装饰器，例如，用于装饰模型的forward方法，如下所示。

```python
class MyNet(torch.nn.Module):
    @torch.amp.autocast(device_type='cuda')
    def forward(self, X):
        return self.layers(X)
```

在自动精度转换的作用域中产生的浮点张量可能是float16数据类型，如果在退出自动精度转换作用域后，还需要使用这些低精度数据，那么可能会导致数据类型不匹配的错误，这时需要手动将其转换为所需的类型。如果在自动精度转换作用域中产生的浮点张量的数据类型已经与所需精度相匹配，那么这种转换是空操作（no-op），不会产生额外的开销。如下示例。

```python
A32 = torch.rand([128, 128], device='cuda')
B32 = torch.rand([128, 128], device='cuda')
with torch.amp.autocast(device_type='cuda'):
    C16 = torch.matmul(A32, B32)
    D16 = torch.matmul(A32, C16)
E32 = torch.matmul(A32, D16.float())
```

此外，torch.amp.autocast上下文管理器是可以嵌套的，这允许用户对程序的不同部分执行更细致的控制，注意在禁用自动精度转换的作用域内，需要手动确保所执行操作的操作数的精度数据类型是相匹配的。

需要注意的是，torch.amp.autocast上下文管理器的生命周期与可见范围是一个线程本地的（thread-local），如果希望在新的线程中也使用自动精度转换，则需要在该线程中也调用torch.amp.autocast上下文管理器或装饰器。当一个进程使用多个GPU设备时，这可能会影响到torch.nn.DataParallel和torch.nn.parallel.DistributedDataParallel的行为，详见后续的讨论。

此外，在torch.cuda.amp模块中还存在torch.cuda.amp.custom_fwd与torch.cuda.amp.custom_bwd两个装饰器，用于自定义torch.autograd.Function函数的情形，具体的使用方式见后续介绍。

## 2. Gradient Scaling

如果某个操作的forward正向过程具有float16精度类型的输入，那么该操作的backward反向过程将产生float16精度类型的梯度。若梯度的数值较小，可能无法使用float16浮点数表示，这些较小的值将被刷新为零，从而产生下溢（underflow）现象，这会导致相应参数的更新梯度被丢失。

为防止参数的更新梯度丢失，梯度缩放（Gradient Scale）会将网络模型的损失值乘以一个缩放因子，并在缩放后的损失值上调用backward反向过程。在backward反向过程中传播的梯度会被以相同的因子进行缩放。换句话说，这样产生的梯度具有更大的幅值，所以它们不会被截断为零。而且，在优化器更新模型参数之前，会对每个参数的.grad梯度进行缩放，这样缩放因子就不会干扰学习率。

注意，自动混合精度和fp16精度类型可能不适用于每个模型。例如，大多数使用bf16精度类型的预训练模型，无法在最大值为65504的fp16精度类型范围内运行，这回导致梯度溢出而不是下溢。在这种情况下，缩放因子可能会减小到1.0以下，以尝试将梯度的值缩放到fp16动态范围内可表示的数字。虽然期望缩放因子的值总是大于1.0，但GradScaler并不能保证保持性能，如果在使用自动混合精度或fp16精度时遇到损失值或梯度为NaN，需要用户验证模型是兼容的。

```python
# torch.cuda.amp.GradScaler
class GradScaler:
    _scale: Optional[torch.Tensor]
    def __init__(self, init_scale=2.0**16, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000, enabled=True)
```

```python
# torch.cuda.amp.GradScaler
class GradScaler:
    def scale(self, outputs)
    def unscale_(self, optimizer)
```

函数scale()将outputs张量或张量列表乘以\_scale缩放因子，返回缩放后的结果。

函数unscale\_()将优化器optimizer所持有的梯度张量除以\_scale缩放因子，已取消对梯度的缩放，而且，unscale\_()可能会使用稀疏梯度替换掉模型参数原来的.grad属性。注意，该方法unscale\_()不会导致CPU与GPU同步。

该unscale\_()函数是可选的，适用于在backward()反向过程和optimizer.step()之间，需要检查或修改梯度的情况。而且，如果没有显式调用unscale\_()方法，则在GradScaler.step()过程中，会自动根据缩放因子对梯度撤销缩放。如下使用示例。

```python
grad_scaler.scale(loss).backward()
grad_scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.99)
grad_scaler.step(optimizer)
grad_scaler.update()
```

注意，函数unscale\_(optimizer)应在为该优化器optimizer分配的所有参数梯度都已经累积之后，且在该优化器的optimizer.step()步骤之前，调用至多一次。针对给定的同一个优化器optimizer的step()之前调用多次unscale\_()会引发RuntimeError错误。

```python
# torch.cuda.amp.GradScaler
class GradScaler:
    def step(self, optimizer, *args, **kwargs)
```

函数step()执行以下两个操作。注意，该方法step()可能会导致CPU与GPU同步。

1. 在内部调用unscale\_(optimizer)方法，获得撤销缩放的梯度，若用户已经手动调用过则不会再次执行。
2. 检查是否存在INF与NaN梯度，若不存在非法梯度，则调用优化器的optimizer.step()方法，使用无缩放的梯度更新模型权重。若存在INF或NaN梯度，则跳过不执行optimizer.step()方法，以免破坏模型权重参数。

函数GradScaler.step()中的args与kwargs参数会原封不动地传递给优化器的optimizer.step()方法，且GradScaler.step()的会原封不动地返回optimizer.step()的返回值。

```python
# torch.cuda.amp.GradScaler
class GradScaler:
    def update(self, new_scale=None)
```

函数update()更新GradScaler的缩放因子，更新规则如下所述。如果跳过了任何optimizer.step()方法，则会乘以backoff_factori使缩放因子减少，如果连续进行了growth_interval个迭代而无跳过，则会乘以growth_factor使缩放因子增大。也可传入new_scale参数来手动设置缩放因子的值。

函数update()只应该在每轮迭代结束时调用，也即在所有优化器都执行GradScaler.step(optimizer)之后调用。注意，该方法update()不会导致CPU与GPU同步。

# CUDA Automatic Mixed Precision

## 1. Typical Mixed Precision Training

```py
model = MyNet().to('cuda')
optimizer = torch.optim.SGD(model.parameters(), lr=1.e-3)
grad_scaler = torch.cuda.amp.GradScaler()
for input, target in zip(inputs, targets):
    optimizer.zero_grad()
    with torch.amp.autocast(device_type='cuda'):
        pred = model(input)
        loss = loss_fn(pred, target)
    grad_scaler.scale(loss).backward()
    grad_scaler.step(optimizer)
    grad_scaler.update()
```

## 2. Working with Unscaled Gradients Clipping

所有由GradScaler.scale(loss).backward()产生的梯度都会被缩放，如果想在backward()和step()之间修改或检查模型参数的梯度，需要手动调用GradScaler.unscale\_()对梯度撤销缩放。如下所示。

例如，在对梯度进行裁剪之前调用GradScaler.unscale\_(optimizer)方法，可以让用户像往常一样裁剪未缩放的梯度。

```python
model = MyNet().to('cuda')
optimizer = torch.optim.SGD(model.parameters(), lr=1.e-3)
grad_scaler = torch.cuda.amp.GradScaler()
for input, target in zip(inputs, targets):
    optimizer.zero_grad()
    with torch.amp.autocast(device_type='cuda'):
        pred = model(input)
        loss = loss_fn(pred, target)
    grad_scaler.scale(loss).backward()
    grad_scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.99)
    grad_scaler.step(optimizer)
    grad_scaler.update()
```

## 3. Working with Scaled Gradients

### 3.1 Gradient accumulation

首先明确一点，深度学习模型的训练是一轮一轮迭代的，一次迭代是指forward,loss_fn,backward,step四个过程的迭代，通常情况下一次迭代处理一个batch的数据。而梯度累积则是指迭代处理多个batch的数据，也即将多个batch数据所产生的梯度累积起来，并只执行一次step模型权重参数的更新。

梯度累积在一个大小为$\text{batch\_per\_iter}\times\text{iters\_to\_accumulate}\times\text{num\_process}$的有效批次上累计梯度。且GradScaler.scale缩放因子应该根据有效批次的大小进行校准（调整），这意味着进行INF/NaN检查，若发现INF/NaN梯度则应该跳过step过程，而且缩放因子的更新应该以有效批次的大小为粒度。

此外，梯度应该保持缩放，而缩放因子应该保持不变，同时给定有效批次数据的梯度应该是累积的。如果在有效批次的梯度累积完成之前，对梯度进行了unscale\_()撤销缩放操作或更改了缩放因子，则下一个批次数据所产生的梯度在累积时，将是未缩放的或是被被不同缩放因子缩放的，这将导致之后无法正确恢复累积梯度的未缩放梯度。

因此，如果想使用unscale\_()撤销梯度缩放（如裁剪未缩放的梯度），需要在调用unscale\_()之后继续调用step()操作，将已经累积的梯度更新到模型权重参数上，以免影响后续的梯度累积。此外需要注意的是，只有在迭代结束时为完整的有效批次数据调用step之后才能调用GradScaler.update()更新缩放因子。

```python
model = MyNet().to('cuda')
optimizer = torch.optim.SGD(model.parameters(), lr=1.e-3)
grad_scaler = torch.cuda.amp.GradScaler()
for idx, (input, target) in enumerate(zip(inputs, targets)):
    optimizer.zero_grad()
    with torch.amp.autocast(device_type='cuda'):
        pred = model(input)
        loss = loss_fn(pred, target)
        loss = loss / iters_to_accumulate
    grad_scaler.scale(loss).backward()  # 累积梯度
    if (idx + 1) % iters_to_accumulate == 0:
        # 已累积完整的有效批次的梯度，可在此处对梯度进行检查或裁剪
        # grad_scaler.unscale_(optimizer)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.99)
        grad_scaler.step(optimizer)
        grad_scaler.update()
```

### 3.2 Gradient penalty

一个典型的梯度惩罚实现，通常使用torch.autograd.grad()函数来创建模型权重参数的梯度，并由这些梯度来创建惩罚项，然后将惩罚项添加到损失值当中。

如下一个示例展示了普通的L2惩罚，其没有使用自动精度转换torch.amp.autocast和梯度缩放torch.cuda.amp.GradScaler，如下所示。

```python
model = MyNet().to('cuda')
optimizer = torch.optim.SGD(model.parameters(), lr=1.e-3)
for input, target in zip(inputs, targets):
    optimizer.zero_grad()
    pred = model(input)
    loss = loss_fn(pred, target)
    # 手动创建梯度，并计算梯度的惩罚项
    param_grads = torch.autograd.grad(outputs=loss, inputs=model.parameters(), create_graph=True)
    grad_norm = 0
    for grad in param_grads:
        grad_norm += grad.pow(2).sum()
    grad_norm = grad_norm.sqrt()
    # 计算带有梯度惩罚项的损失函数值
    loss = loss + grad_norm
    loss.backward()
    optimizer.step()
```

当使用自动精度转换torch.amp.autocast或梯度缩放torch.cuda.amp.GradScaler时，在实现梯度惩罚时，传递给torch.autograd.grad()方法的outputs损失值应当是经过缩放的（防止梯度减益为零），由此生成经过缩放的梯度。然后对这些手动创建的缩放的梯度执行撤销缩放（unscale），注意这些手动梯度不属于任何优化器，因而需要手动对其进行逆缩放，而不能使用GradScaler.unscale\_()方法，然后使用撤销缩放的梯度创建惩罚项。

此外需要注意的是，梯度惩罚项的计算属于正向传递的一部分，因此该计算过程也应该包含到autocast等上下文管理器当中。

如下一个示例展示了普通的L2惩罚，并使用自动精度转换torch.amp.autocast和梯度缩放torch.cuda.amp.GradScaler，如下所示。

```python
model = MyNet().to('cuda')
optimizer = torch.optim.SGD(model.parameters(), lr=1.e-3)
grad_scaler = torch.cuda.amp.GradScaler()
for input, target in zip(inputs, targets):
    optimizer.zero_grad()
    with torch.amp.autocast(device_type='cuda'):
        pred = model(input)
        loss = loss_fn(pred, target)
    # 使用缩放的损失值（防止梯度减益为零），来手动创建梯度
    param_grads = torch.autograd.grad(outputs=grad_scaler.scale(loss), inputs=model.parameters(), create_graph=True)
    # 对缩放的梯度执行撤销缩放操作，
    # 因为这些手动创建的梯度不属于任何优化器，故需要手动对其逆缩放，而不能执行GradScaler.unscale_()
    inv_scale = 1. / grad_scaler.get_scale()
    param_grads = [inv_scale * g for g in param_grads]
    # 根据手动梯度计算惩罚项
    with torch.amp.autocast(device_type='cuda'):
        grad_norm = 0
        for grad in param_grads:
            grad_norm += grad.pow(2).sum()
        grad_norm = grad_norm.sqrt()
        # 计算带有梯度惩罚项的损失函数值
        loss = loss + grad_norm
    # 执行backward反向过程，并更新模型参数
    grad_scaler.scale(loss).backward()
    grad_scaler.step(optimizer)
    grad_scaler.update()
```

## 4. Working with Multiple Models, Losses, and Optimizers

如果所涉及的深度学习模型具有多个损失函数与损失值，必须对它们分别调用GradScaler.scale(loss)缩放；如果存在多个优化器，则需要为它们分别调用GradScaler.step()方法，如果需要也分别调用GradScaler.unscale\_()方法。在最终迭代结束时，调用一次GradScaler.update()更新即可。

```python
model1 = MyNet().to('cuda')
model2 = MyNet().to('cuda')
optimizer1 = torch.optim.SGD(model1.parameters(), lr=1.e-3)
optimizer2 = torch.optim.SGD(model2.parameters(), lr=1.e-3)
grad_scaler = torch.cuda.amp.GradScaler()
for input, target in zip(inputs, targets):
    optimizer1.zero_grad()
    optimizer2.zero_grad()
    with torch.amp.autocast(device_type='cuda'):
        pred1 = model1(input)
        pred2 = model2(input)
        loss1 = loss_fn(torch.exp(pred1 + pred2), target)
        loss2 = loss_fn(torch.exp(pred1 - pred2), target)
    # 因为两个损失值的共享了相同的模型部分，故第一次进行backward反向过程传播梯度时，需要保留计算图
    grad_scaler.scale(loss1).backward(retain_graph=True)
    grad_scaler.scale(loss2).backward()
    # 可选是否进行梯度检查与裁剪
    # grad_scaler.unscale_(optimizer1)
    # torch.nn.utils.clip_grad.clip_grad_norm_(optimizer1, max_norm=0.99)
    grad_scaler.step(optimizer1)
    grad_scaler.step(optimizer2)
    grad_scaler.update()
```

每个优化器都会检查INF/NaN梯度，并独立地决定是否跳过step()过程，这可能会使得其中一个优化器跳过step过程，而另一个优化器没有跳过step过程。由于跳过step的情况很少发生，这应该不会阻止模型收敛。

## 5. Working with Multiple GPUs

此处讨论的情况只会影响torch.amp.autocast上下文管理器的使用，而不会影响torch.cuda.amp.GradScaler的使用方式。此处的讨论是对前述的补充。

如前所述，torch.amp.autocast上下文管理器的生命周期与可见范围是一个线程本地的（thread-local），如果希望在新的线程中也使用自动精度转换，则需要在该线程中也调用torch.amp.autocast上下文管理器或装饰器。当一个进程使用多个GPU设备时，这可能会影响到torch.nn.DataParallel和torch.nn.parallel.DistributedDataParallel的行为。

### 5.1 Single process with DataParallel

使用torch.nn.DataParallel进行数据并行训练，它会生成多个线程（thread）来在每个GPU设备上执行forward正向传递，为了正确使用具有线程本地（thread-local）属性的torch.amp.autocast上下文管理器，可按照如下方式所示。

```python
model = MyNet().to('cuda')
dp_model = nn.DataParallel(model)  # 构建数据并行模型，其生成多个线程

# 在主线程中设置 autocast 自动精度转换上下文管理器
with torch.amp.autocast(device_type='cuda'):
    # dp_model 的内部线程也会使用 autocast 自动精度转换
    pred = dp_model(input)
    loss = loss_fn(pred, target)
```

### 5.2 One GPU per process with DistributedDataParallel

torch.nn.parallel.DistributedDataParallel的文档建议，为获得最佳性能，每个进程使用一个GPU，在这种情况下，DistributedDataParallel不会在内部产生线程，因此torch.amp.autocast和torch.cuda.amp.GradScaler的使用不会受到影响。

### 5.3 Multiple GPUs per process with DistributedDataParallel

此时torch.nn.parallel.DistributedDataParallel可能会产生分支线程来在每个设备上运行forward正向过程，就像torch.nn.DataParallel一样。此时可以将torch.amp.autocast上下文管理器作为模型forward过程中的一部分，以确保它在每个线程中都正确启用。

# Autocast and Custom Autograd Functions

如果深度学习模型使用了自定义的torch.autograd.Function函数，且这些函数符合以下情况，则需要配置torch.amp.autocast自动精度转换的兼容性。

- 自定义Function接受多个浮点张量输入，或者
- 自定义Function对原来满足自动精度转换的操作（详见[Autocast Op Reference](https://pytorch.org/docs/stable/amp.html#autocast-op-reference)）进行了封装，或者
- 自定义Function实现了自定的CUDA扩展，但只提供了某个特定浮点精度的实现，而未提供其他精度的实现。

使用所有情况的是，如果自定义Function函数对自动精度转换的float16低精度不支持，一个安全的备用方案是禁用自动转换，并在任何发生错误的地方强制执行float32精度类型的函数。

如果可以修改自定义Function函数的实现，更好的解决方案是使用torch.cuda.amp.custom_fwd和torch.cuda.amp.custom_bwd装饰器，如下示例所示。

## 1. Functions with multiple inputs or autocastable ops

将无参数的torch.cuda.amp.custom_fwd和torch.cuda.amp.custom_bwd装饰器分别用于forward正向过程和backward反向过程，可以确保当forward在某个精度下执行时，backward可以在相同的精度下执行，从而避免精度类型不匹配错误。

```python
class MyMM(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return torch.mm(a, b)
    
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad):
        a, b = ctx.saved_tensors
        return torch.mm(grad, b.t()), torch.mm(a.t(), grad)
    
with torch.amp.autocast(device_type='cuda'):
    output = MyMM.apply(input1, input2)
```

## 2. Functions that need a particular dtype

若自定义Function函数只实现了某个特定精度的版本，例如只实现了torch.float32精度类型的版本。则使用torch.cuda.amp.custom_fwd装饰器修饰该自定义Function函数的forward过程，并使用cast_inputs参数指定其所接受的输入精度为torch.float32数据类型。那么，当在自动精度转换torch.amp.autocast的作用域内使用了该自定义Function函数，装饰器会自动将该自定义Function函数所接受的输入数据转换为torch.float32数据类型，并且在forward过程和backward过程中禁用torch.amp.autocast自动精度转换。

```python
class MyFloat32GeMM(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, A, B):
        ctx.save_for_backward(A, B)
        return myext_sgemm(A, B)
    
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, G):
        A, B = ctx.saved_tensors
        return myext_sgemm(G, B.t()), myext_sgemm(A.t(), G)
```

这样就能在任意位置调用MyFloat32GeMM自定义函数了，而无需手动禁用torch.amp.autocast自动精度转换的作用范围。
