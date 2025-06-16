[toc]

# PyTorch Benchmark

本节内容若无特殊说明，均在`torch.utils.benchmark`模块的命名空间中。该模块提供比较代码性能的API接口。

当进行基准测试时，PyTorch提供了许多选项，包括Python内置的`timeit`模块。然而，对PyTorch代码进行基准测试有许多要注意的事项，例如，需要管理线程数量和同步CUDA设备等。

本教程演示如何使用PyTorch的benchmark模块来避免常见错误，同时使其更容易比较不同代码的性能，为基准测试生成输入等。

## 1. Defining functions to benchmark

首先实现求批向量点积的两种方法，以用于评估。

```python
# at myfunpack vdot.py
import torch

def b_vec_dot_v1(a, b):
    return torch.sum(torch.mul(a, b), -1)

def b_vec_dot_v2(a, b):
    ret = torch.bmm(torch.reshape(a, (-1, 1, a.shape[-1])), 
                    torch.reshape(b, (-1, b.shape[-1], 1)))
    return torch.flatten(ret, -3)

if __name__ == '__main__':
    x = torch.randn(10000, 64)
    print(torch.allclose(b_vec_dot_v1(x, x), b_vec_dot_v2(x, x)))
```

## 2. Benchmarking with timeit.Timer

可以使用Python内置的timeit模块对代码进行基准测试。

```python
import torch
import timeit

x = torch.randn(10000, 64)

t1 = timeit.Timer(
    stmt='b_vec_dot_v1(x, x)',
    setup='from myfunpack.vdot import b_vec_dot_v1',
    globals={'x': x})

t2 = timeit.Timer(
    stmt='b_vec_dot_v2(x, x)',
    setup='from myfunpack.vdot import b_vec_dot_v2',
    globals={'x': x})

print(f'b_vec_dot_v1(x, x): {t1.timeit(100) / 100 * 1e6:>5.1f} us')
print(f'b_vec_dot_v2(x, x): {t2.timeit(100) / 100 * 1e6:>5.1f} us')
"""
b_vec_dot_v1(x, x):  99.2 us
b_vec_dot_v2(x, x):  70.7 us
"""
```

其中，timeit.Timer()类的构造参数setup，指定要从哪个包中导入测试函数；timeit.Timer.timeit(num)函数中的num参数指定要对测试函数重复执行的次数。另一个更全面的示例如下所示。

```python
def square_func(x):
    return x ** 2

def time_it():
    x = torch.randn(size=[8, 8, 8, 8], dtype=torch.double)
    square_m = benchmark.Timer(
        stmt='square_func(x)',
        globals={
            '__name__': __name__,
            'square_func': square_func,
            'x': x
        }
    ).blocked_autorange()
    print(f'square_m Mean:   {square_m.mean * 1e6:6.2f} us')

time_it()
```

可以看到，在benchmark.Timer的构造参数globals中传入了\_\_name\_\_变量和要测试的函数对象。

## 3. Benchmarking with torch.utils.benchmark.Timer

PyTorch基准模块benchmark与Python内置的timeit模块拥有相似的API接口，不过它的默认值使其更容易和更安全地用于PyTorch代码的基准测试。

```python
import torch
import torch.utils.benchmark as benchmark

x = torch.randn(10000, 64)

t1 = benchmark.Timer(
    stmt='b_vec_dot_v1(x, x)',
    setup='from myfunpack.vdot import b_vec_dot_v1',
    globals={'x': x})

t2 = benchmark.Timer(
    stmt='b_vec_dot_v2(x, x)',
    setup='from myfunpack.vdot import b_vec_dot_v2',
    globals={'x': x})

print(t1.timeit(100))
print(t2.timeit(100))
"""
<torch.utils.benchmark.utils.common.Measurement object at 0x7efd0c790550>
b_vec_dot_v1(x, x)
setup: from myfunpack.vdot import b_vec_dot_v1
  150.14 us
  1 measurement, 100 runs , 1 thread
<torch.utils.benchmark.utils.common.Measurement object at 0x7efd0c790e80>
b_vec_dot_v2(x, x)
setup: from myfunpack.vdot import b_vec_dot_v2
  293.15 us
  1 measurement, 100 runs , 1 thread
"""
```

PyTorch基准模块benchmark与Python内置的timeit模块拥有相似的API接口，但也有一些重要的差异。benchmark.Timer.timeit(num)返回每次运行的时间，而不是像timeit.Timer.timeit(num)那样返回总运行时间。PyTorch基准模块benchmark还提供了格式化字符串表示，用于打印结果。另一个重要的区别是，PyTorch基准模块benchmark默认在单线程中运行，可以用num_threads参数改变线程的数量。

torch.utils.benchmark.Timer接受几个额外的构造参数，包括label、sub_label、description、env，它们会更改返回的测量对象的\_\_repr\_\_，并用于对结果进行分组。

```python
import torch
import torch.utils.benchmark as benchmark

x = torch.randn(10000, 64)

num_threads = torch.get_num_threads()
print(f'Benchmarking on {num_threads} threads')

t1 = benchmark.Timer(
    stmt='b_vec_dot_v1(x, x)',
    setup='from myfunpack.vdot import b_vec_dot_v1',
    globals={'x': x},
    num_threads=num_threads,
    label='Multithreaded batch dot',
    sub_label='Implemented using mul and sum')

t2 = benchmark.Timer(
    stmt='b_vec_dot_v2(x, x)',
    setup='from myfunpack.vdot import b_vec_dot_v2',
    globals={'x': x},
    num_threads=num_threads,
    label='Multithreaded batch dot',
    sub_label='Implemented using bmm')

print(t1.timeit(100))
print(t2.timeit(100))
"""
Benchmarking on 12 threads
<torch.utils.benchmark.utils.common.Measurement object at 0x7f6b7715d940>
Multithreaded batch dot: Implemented using mul and sum
setup: from myfunpack.vdot import b_vec_dot_v1
  47.93 us
  1 measurement, 100 runs , 12 threads
<torch.utils.benchmark.utils.common.Measurement object at 0x7f6b7715d550>
Multithreaded batch dot: Implemented using bmm
setup: from myfunpack.vdot import b_vec_dot_v2
  57.06 us
  1 measurement, 100 runs , 12 threads
"""
```

在所有线程可用的情况下运行benchmark会得到与timeit模块类似的结果。更重要的是，哪个版本更快取决于运行代码的线程数量。所以，使用代表真实用例的线程设置对代码进行基准测试很重要。

此外，在GPU上进行基准测试时需要同步CPU和CUDA，如下所示，分别使用了Python自带的timeit模块与PyTorch提供的benchmark模块。

```python
import torch
import timeit

x = torch.randn(10000, 64).to(device='cuda')

t1 = timeit.Timer(
    stmt='b_vec_dot_v1(x, x)',
    setup='from myfunpack.vdot import b_vec_dot_v1',
    globals={'x': x})

t2 = timeit.Timer(
    stmt='b_vec_dot_v2(x, x)',
    setup='from myfunpack.vdot import b_vec_dot_v2',
    globals={'x': x})

# Run each twice to show difference before/after warm-up
print(f'b_vec_dot_v1(x, x): {t1.timeit(100) / 100 * 1e6:>5.1f} us')
print(f'b_vec_dot_v1(x, x): {t1.timeit(100) / 100 * 1e6:>5.1f} us')
print(f'b_vec_dot_v2(x, x): {t2.timeit(100) / 100 * 1e6:>5.1f} us')
print(f'b_vec_dot_v2(x, x): {t2.timeit(100) / 100 * 1e6:>5.1f} us')
"""
b_vec_dot_v1(x, x):  10.0 us
b_vec_dot_v1(x, x):   5.9 us
b_vec_dot_v2(x, x):  14.5 us
b_vec_dot_v2(x, x):   6.6 us
"""
```

```python
import torch
import torch.utils.benchmark as benchmark

x = torch.randn(10000, 64).to(device='cuda')

t1 = benchmark.Timer(
    stmt='b_vec_dot_v1(x, x)',
    setup='from myfunpack.vdot import b_vec_dot_v1',
    globals={'x': x})

t2 = benchmark.Timer(
    stmt='b_vec_dot_v2(x, x)',
    setup='from myfunpack.vdot import b_vec_dot_v2',
    globals={'x': x})

# Run only once since benchmark module does warm-up for us
print(t1.timeit(100))
print(t2.timeit(100))
"""
<torch.utils.benchmark.utils.common.Measurement object at 0x7fd2fc4cf6d0>
b_vec_dot_v1(x, x)
setup: from myfunpack.vdot import b_vec_dot_v1
  29.08 us
  1 measurement, 100 runs , 1 thread
<torch.utils.benchmark.utils.common.Measurement object at 0x7fd2fc4cf8e0>
b_vec_dot_v2(x, x)
setup: from myfunpack.vdot import b_vec_dot_v2
  29.57 us
  1 measurement, 100 runs , 1 thread
"""
```

可以看到，使用timeit模块的bmm版本的第一次运行比第二次运行花费的时间长得多。这是因为bmm调用cuBLAS，而cuBLAS需要在第一次调用时加载，这需要一些时间。所以在基准测试之前进行热身（warm-up）是重要的，不过，PyTorch的基准模块benchmark负责这项工作。

timeit和基准模块benchmark之间的结果差异是因为timeit模块没有同步CUDA，因此只计时包括启动内核的时间。

## 4. Benchmarking with Blocked Autorange

除timeit.Timer.timeit(num)外，还可以使用timeit.Timer.autorange()方法进行测量，该方法会自动决定要对测试函数重复执行的次数。这是一个方便函数，反复调用timeit(num)，其中num自动从序列{1,2,5,10,20,50,...}依次选择，直到使总时间大于等于0.2秒，返回最终的结果(num,total_time)，其中num是对测试函数反复执行的次数，total_time是执行num次所用的总时间。

timeit.Timer.autorange()会进行一次至少0.2秒的连续测量，而torch.utils.benchmark.blocked_autorange则会进行多次连续测量，其总时间至少为0.2秒（可以通过min_run_time参数更改），并且满足“计时开销只占总体测量时间一小部分”的条件。要实现这一点，首先要不断增加每个循环的运行次数，直到运行时间远远大于测量开销（这也用作预热warm-up），然后进行测量，直到达到目标时间。这样做的好处是浪费的数据少，可以通过统计来估计测量结果的可靠性。

```python
import torch
import torch.utils.benchmark as benchmark

x = torch.randn(10000, 64).to(device='cuda')

t1 = benchmark.Timer(
    stmt='b_vec_dot_v1(x, x)',
    setup='from myfunpack.vdot import b_vec_dot_v1',
    globals={'x': x})

t2 = benchmark.Timer(
    stmt='b_vec_dot_v2(x, x)',
    setup='from myfunpack.vdot import b_vec_dot_v2',
    globals={'x': x})

m1 = t1.blocked_autorange()
m2 = t2.blocked_autorange()

print(m1)
print(m2)
print(f'm1 Mean:   {m1.mean * 1e6:6.2f} us')
print(f'm1 Median: {m1.mean * 1e6:6.2f} us')
"""
<torch.utils.benchmark.utils.common.Measurement object at 0x7fd257f4c520>
b_vec_dot_v1(x, x)
setup: from myfunpack.vdot import b_vec_dot_v1
  Median: 7.93 us
  3 measurements, 10000 runs per measurement, 1 thread
<torch.utils.benchmark.utils.common.Measurement object at 0x7fd257f4c6d0>
b_vec_dot_v2(x, x)
setup: from myfunpack.vdot import b_vec_dot_v2
  Median: 14.23 us
  2 measurements, 10000 runs per measurement, 1 thread
m1 Mean:     8.00 us
m1 Median:   8.00 us
"""
```

## 5. Comparing benchmark results

在实践中，经常需要尝试各种输入规模的组合以及不同数量的线程。torch.utils.benchmark.Compare类有助于在一个格式化的表中显示许多测量的结果，它使用前述的注解（label、sub_label、num_threads、description等）对表进行分组和组织。

```python
import torch
import torch.utils.benchmark as benchmark

results = []
for batch_size in (64, 128):
    for dim in (512, 1024):
        label = 'Batched vector dot product'
        sub_label = f'[{batch_size}, {dim}]'
        x = torch.rand((batch_size, dim))

        for num_threads in (1, 4, 8):
            results.append(benchmark.Timer(
                stmt='b_vec_dot_v1(x, x)',
                setup='from myfunpack.vdot import b_vec_dot_v1',
                globals={'x': x},
                num_threads=num_threads,
                label=label,
                sub_label=sub_label,
                description='mul sum',
            ).blocked_autorange(min_run_time=1.))
            results.append(benchmark.Timer(
                stmt='b_vec_dot_v2(x, x)',
                setup='from myfunpack.vdot import b_vec_dot_v2',
                globals={'x': x},
                num_threads=num_threads,
                label=label,
                sub_label=sub_label,
                description='bmm',
            ).blocked_autorange(min_run_time=1.))

compare = benchmark.Compare(results=results)
compare.print()
"""
[----- Batched vector dot product ----]
                   |  mul sum  |   bmm 
1 threads: ----------------------------
      [64, 512]    |     5.0   |   69.9
      [64, 1024]   |    18.2   |  136.9
      [128, 512]   |     8.6   |  137.0
      [128, 1024]  |    19.7   |  269.1
4 threads: ----------------------------
      [64, 512]    |    13.2   |   54.2
      [64, 1024]   |    15.0   |   95.4
      [128, 512]   |    26.6   |   83.7
      [128, 1024]  |    14.6   |   85.4
8 threads: ----------------------------
      [64, 512]    |    18.4   |   90.0
      [64, 1024]   |    25.4   |  162.4
      [128, 512]   |    18.3   |   78.1
      [128, 1024]  |    52.8   |  157.9

Times are in microseconds (us).
"""
```

benchmark.Compare还提供了更改表格式的函数，如下所示。

```python
compare.trim_significant_figures()
compare.colorize()
compare.print()
```

此外，还可以直接将benchmark.Compare输出到文件，其自己实现了\_\_str\_\_()方法，如下所示。

```python
with open('timeuse.txt', encoding='utf-8', mode='w+') as f:
    f.write(str(compare))
```

## 6. Saving and Loading benchmark results

测量结果可以由pickle模块序列化写入磁盘。这使得进行A/B测试更方便，可以从两个不同的环境中收集测量数据，对它们进行序列化存储，然后将它们加载到一个环境中。Timer需要一个env构造参数，以便这样的A/B测试可以无缝地工作。

# Profiling your PyTorch Module

本节内容若无特殊说明，均在`torch.profiler`模块的命名空间中。

PyTorch的分析器（Profiler）是一个工具，它实现为torch.profiler.profile上下文管理器类，提供各种API接口，允许在训练和推断期间收集性能指标，能够识别代码中各种PyTorch操作的时间开销和内存占用，结果可以作为表格打印或以JSON跟踪文件（trace file）返回。Profiler上下文管理器API可以用于了解哪些模型操作符是最耗时的，并检查它们的输入形状和堆栈跟踪，研究设备的内核活动并可视化执行跟踪（execution trace）。

分析器支持多线程模型，分析器与操作在同一个线程中运行，但它也会分析可能在另一个线程中运行的子操作符（子线程中的子操作）；而同时运行的分析器将被限制在它们自己的线程中，以防止结果混淆。

注意，使用Profiler分析器会产生一些开销，并且最好仅用于调查代码，如果要对运行时进行基准测试，请记住删除它。

> 早期的分析器API位于torch.autograd.profiler模块，将被弃用。

## 1. Using profiler

分析器对于识别模型中的性能瓶颈非常有用，此处列举一个示例，构建了一个自定义模块来执行两个子任务，一是对输入张量进行线性变换，二是根据变换结果获得掩码张量的索引。

使用torch.profiler.record_function(name='label')上下文管理器，可以将其作用域的代码包装在单独的子任务中，在分析器的输出中，子任务的所有操作的性能指标，将显示在其相应的标签下。

```python
class MyModel(torch.nn.Module):
    def __init__(self, in_feats, out_feats, bias=True):
        super(MyModel, self).__init__()
        self.linear = torch.nn.Linear(in_feats, out_feats, bias)

    def forward(self, input, mask: torch.Tensor):
        with torch.profiler.record_function('linear pass'):
            x = self.linear(input)

        with torch.profiler.record_function('mask indices'):
            threshold = torch.sum(x, dim=1).mean().item()
            hi_idx = torch.argwhere(mask > threshold)

        return x, hi_idx
```

接下来，初始化模型与随机张量，并使用torch.profiler.profile()来统计性能指标。

最后，可以打印分析器的结果，使用torch.profiler.profile.key_averages()方法获得事件列表（EventList）以用于打印，列表内元素为FunctionEventAvg事件对象，该方法按操作符的名称来聚合结果。此外，也可通过设置group_by_input_shape=True参数，或设置group_by_stack_n=N参数，以按输入形状或堆栈跟踪事件聚合结果。按输入形状分组有助于识别模型所使用的张量形状。

```python
hidden_feats = 512
model = MyModel(in_feats=hidden_feats, out_feats=8).to('cuda')
input = torch.rand(size=[128, hidden_feats]).to('cuda')
mask = torch.rand(size=[hidden_feats, hidden_feats, hidden_feats], dtype=torch.double).to('cuda')

model(input, mask)  # for warmming-up

with torch.profiler.profile(with_stack=True, profile_memory=True) as prof:
    out, idx = model(input, mask)

print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=10))
```

```
### Some columns are omitted ###
---------------  ------------  ------------  ------------  ------------
           Name      Self CPU     CPU total  Self CPU Mem    # of Calls
---------------  ------------  ------------  ------------  ------------
  aten::nonzero     375.844ms     375.868ms       3.00 Gb             1
       aten::gt      74.316ms      74.316ms     128.00 Mb             1
   mask indices      12.614ms     462.912ms    -128.00 Mb             1
    aten::addmm     140.000us     152.000us       4.00 Kb             1
    linear pass      92.000us     281.000us           0 b             1
      aten::sum      41.000us      47.000us         512 b             2
    aten::empty      19.000us      19.000us           0 b             1
     aten::div_      17.000us      37.000us          -4 b             1
        aten::t      16.000us      27.000us           0 b             1
     aten::mean      16.000us      61.000us           4 b             1
---------------  ------------  ------------  ------------  ------------
Self CPU time total: 463.193ms
```

注意，若要统计每个操作算子的输入张量的形状，需要为torch.profiler.profile()指定record_shapes=True参数，同时在打印时为profiler.key_averages()指定group_by_input_shape=True参数，二者缺一不可。

另外，如果不使用torch.profiler.profile的with语句格式，也可按如下方式使用。

```python
prof = torch.profiler.profile(with_stack=True, profile_memory=True)
prof.start()
for _ in range(10):
    out, idx = model(input, mask)
    prof.step()
prof.stop()
```

## 2. Visualizing profile results

分析结果可以使用torch.profiler.profile.export_chrome_trace()方法导出为.json跟踪文件，如下所示。

```python
with torch.profiler.profile() as prof:
    out, idx = model(input, mask)
prof.export_chrome_trace(path='model_trace.json')
```

打开Chrome内核的浏览器，在地址栏键入chrome://tracing后回车，打开浏览器自带的[Perfetto](https://ui.perfetto.dev/)分析工具，点击Load加载按钮，选择导出的model_trace.json文件即可查看分析结果。

可以将执行时间（self_cpu_time_total指标、self_cuda_time_total指标）和堆栈跟踪（需要指定with_stack=True参数）可视化为火焰图（FlameGraph），使用torch.profiler.profile.export_stacks()导出原始数据，如下所示。

```python
with torch.profiler.profile() as prof:
    out, idx = model(input, mask)
prof.export_stacks(path='model_stacks.txt', metric='self_cpu_time_total')
```

然后使用[FlameGraph tool](https://github.com/brendangregg/FlameGraph)工具可视化火焰图，生成.svg图像，如下所示。

```shell
git clone https://github.com/brendangregg/FlameGraph
cd FlameGraph
./flamegraph.pl --title "PyTorch Time" --countname "us." model_stacks.txt > perf_viz.svg
```

# torch.profiler

本节内容若无特殊说明，均在`torch.profiler`模块的命名空间中。

```python
class _KinetoProfile:
    def __init__(self, *, activities: Optional[Iterable[ProfilerActivity]] = None, record_shapes=False, profile_memory=False, 
                 with_stack=False, with_flops=False, with_modules=False,
                 experimental_config: Optional[_ExperimentalConfig] = None): ...
    def key_averages(self, group_by_input_shape: bool = False, group_by_stack_n: int = 0): ...
    def export_chrome_trace(self, path: str): ...
    def export_stacks(self, path: str, metric: str = "self_cpu_time_total"): ...
```

该torch.profiler._KinetoProfile类是对torch.autograd.profiler.profile类的封装，抽象出底层API接口。

- activities参数，表示要参与分析的活动组（activity groups）的列表，其元素值取枚举类torch.profiler.ProfilerActivity的值，可取ProfilerActivity.CPU或ProfilerActivity.CUDA枚举值，默认情况（若CUDA可用）下，该参数取值为[CPU,CUDA]的列表。
- record_shapes参数(bool)，是否保存操作算子输入形状的信息。
- profile_memory参数(bool)，是否跟踪张量内存的分配与释放，即是否跟踪内存占用变化。
- with_stack参数(bool)，是否记录操作算子的源信息（算子所在文件和行号）。
- with_flops参数(bool)，是否用公式来估算某些特定操作算子的FLOPS计算能力（例如矩阵乘、二维卷积等）。
- with_modules参数(bool)，是否记录操作算子调用栈所对应的模块层次结构（包括函数名）。例如，如果模块A的forward()调用模块B的forward()，其中包含一个aten::add算子，那么aten::add算子的模块层次结构是A.B。注意，目前这种支持只存在于TorchScript模型中，而不存在于eager模式的模型中。
- experimental_config参数，一系列用于实验的参数。

注意，此API是实验性的，未来可能会更改。此外，启用形状和栈跟踪会导致额外的开销，当指定record_shapes=True时，分析器将临时保存对张量的引用，这可能会进一步妨碍某些依赖于引用计数的优化，并引入额外的张量拷贝。

```python
class _KinetoProfile:
    def key_averages(group_by_input_shape=False, group_by_stack_n=0) -> torch.autograd.profiler.EventList: ...
```

对键上的所有函数事件求平均值，可根据操作符名称、输入形状、堆栈跟踪信息、对它们进行分组。若group_by_input_shape参数为True，则根据(event_name, input_shapes)进行分组而不仅仅是根据event_name进行分组。group_by_stack_n参数，指定按前n个堆栈跟踪条目分组。

可以看到，该key_averages()方法返回一个torch.autograd.profiler.EventList事件列表对象，它有一个table()方法最为常用，其语法格式如下。

```python
class EventList(list):
    def table(self, sort_by=None, row_limit=100, max_src_column_width=75, max_name_column_width=55,
              max_shapes_column_width=80, header=None, top_level_events_only=False) -> str: ...
```

可以看到，该torch.autograd.profiler.EventList.table()方法返回一个易于阅读的格式化为表格形式的字符串，以用于打印。

- sort_by参数(str)，用于对表格中的条目进行排序，默认情况下，表格中条目的顺序与其加入表格的顺序一致。可指定字符串的有效取值为：cpu_time，cuda_time，cpu_time_total，cuda_time_total，self_cpu_time_total，self_cuda_time_total，cpu_memory_usage，cuda_memory_usage，self_cpu_memory_usage，self_cuda_memory_usage，count。
- top_level_events_only参数(bool)，决定所显示事件的层级。若为True，则分析器只显示处于Python顶层调用的事件，例如lstm事件，add事件，其他函数事件等，而忽略诸如底层CPU/CUDA算子一类的内嵌事件，以提高可读性。

需要知道cpu_time和self_cpu_time之间的区别，首先一个操作算子可以调用其他的操作算子，self_cpu_time只统计操作自己的执行时间，而不包括其调用的子操作的时间；cpu_time统计操作总的执行时间，也包括其子操作的时间。

```python
class _KinetoProfile:
    def export_chrome_trace(self, path: str): ...
    def export_stacks(self, path: str, metric: str = "self_cpu_time_total"): ...
```

export_chrome_trace()方法以Chrome JSON格式导出收集到的跟踪信息。export_stacks()方法以适用于可视化的格式保存堆栈跟踪信息。

```python
class profile(_KinetoProfile):
    def __init__(self, *, activities: Optional[Iterable[ProfilerActivity]] = None,
        schedule: Optional[Callable[[int], ProfilerAction]] = None, on_trace_ready: Optional[Callable[..., Any]] = None,
        record_shapes=False, profile_memory=False, with_stack=False, with_flops=False, with_modules=False,
        experimental_config: Optional[_ExperimentalConfig] = None,
        use_cuda: Optional[bool] = None  # deprecated
    ): ...
```

用作上下文管理器的分析器torch.profiler.profile类。其中一些参数与其父类用法一致，注意use_cuda参数已弃用，使用activities参数代替。

- schedule参数(Callable)，可调用的函数，其接受一个int类型的step作为参数，并返回ProfilerAction值，以指定在每一步执行的分析器操作。
- on_trace_ready参数(Callable)，可调用函数，在每一步当schedule函数返回ProfilerAction.RECORD_AND_SAVE时，该回调函数被调用。

```python
def schedule(*, wait: int, warmup: int, active: int, repeat: int = 0, skip_first: int = 0) -> Callable: ...
def tensorboard_trace_handler(dir_name: str, worker_name: Optional[str] = None, use_gzip: bool = False): ...
```

使用torch.profiler.schedule()方法来生成可调用的schedule，以传入作为torch.profiler.profile类的构造参数。该方法是有用的，对于长时间的训练情景，它允许用户获得训练过程的不同迭代的多个跟踪信息（traces）。默认情况下，它简单地连续记录上下文环境中的所有事件。

使用schedule.tensorboard_trace_handler()方法来生成可调用的on_trace_ready，以传入作为torch.profiler.profile类的构造参数。默认情况下，该方法生成的可调用对象，会与TensorBoard交互，会将上下文中的事件记录到可被TensorBoard识别的文件中。该方法的dir_name参数指定TensorBoard的logdir日志目录，在分析结束后，即可在dir_name目录下找到TensorBoard可用的文件，在该目录下使用tensorboard --logdir=${dir_name}即可启用TensorBoard可视化分析。

当然，也可自定义profiler.profile分析器的schedule参数和on_trace_ready参数，而不使用上述两个schedule()和tensorboard_trace_handler()辅助函数。使用torch.profiler.profile分析器的schedule参数，on_trace_ready参数，和step()函数，一个示例如下所示。

```python
def trace_handler(profiler: torch.profiler.profile):
    """
    trace_handler is called every time a new trace becomes available
    """
    print(profiler.key_averages().table(sort_by='self_cuda_time_total', row_limit=-1))
    # profiler.export_chrome_trace('./tmp/test_trace_' + str(profiler.step_num) + '.json')

if __name__ == '__main__':
    """ 
    We use schedule argument with (wait=1, warmup=1, active=2, repeat=1), and this means profiler will:
    [wait   = 1]  skip the first step/iteration,
    [warmup = 1]  start warming up on the second,
    [active = 2]  record the third and the forth iterations, 
                      after which the trace will become available and on_trace_ready (when set) is called;
    [repeat = 1]  the cycle repeats starting with the next step.
    """
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
        on_trace_ready=trace_handler,
        # on_trace_ready=torch.profiler.tensorboard_trace_handler('./tblog'),
    ) as prof:
        for iter_idx in range(1000):
            code_to_profile(iter_idx)
            prof.step()  # send a signal to the profiler that the next iteration has started
```

# PyTorch Profiler With TensorBoard

本节展示了如何使用PyTorch Profiler的TensorBoard插件检测模型的性能瓶颈。

自PyTorch 1.8以来，更新后的分析器API既能够记录CPU端的操作，也能够记录在GPU端运行的CUDA核函数，且分析器能够将所记录的信息在TensorBoard插件中可视化，并提供性能瓶颈的分析。

## 1. Profile model and export TensorBoard log

此节使用一个简单的ResNet模型为例，展示如何使用TensorBoard插件分析模型性能。

采用torchvision提供的标准CIFAR10数据集，并使用torchvision.transforms将其变成所需格式，然后使用DataLoader加载。接下来，创建ResNet模型，损失函数，优化器对象，并将模型和损失函数移动到GPU端。如下所示。

```python
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=32, shuffle=True)

model = torchvision.models.resnet18(weights='IMAGENET1K_V1').to('cuda')
criterion = torch.nn.CrossEntropyLoss().to('cuda')
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
model.train()
```

定义训练一个batch数据的方法，如下所示。

```python
def train_one_batch(model, batch_data, criterion, optimizer, device='cuda'):
    inputs, lables = batch_data[0].to(device), batch_data[1].to(device)
    outputs = model(inputs)
    loss = criterion(outputs, lables)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

使用torch.profiler.profile分析模型性能，并将之导出为用于TensorBoard的日志，如下所示。

```python
torch.cuda.empty_cache()
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(dir_name='./tblog'),
    record_shapes=True, profile_memory=True,
) as prof:
    for step, batch_data in enumerate(train_loader):
        if step >= (1 + 1 + 3) * 2:
            break
        train_one_batch(model, batch_data, criterion, optimizer, device='cuda')
        prof.step()  # Need to call this at the end of each step to notify profiler of steps' boundary.
```

执行代码，分析结果会保存到上述指定的'./tblog'目录下。

## 2. Use TensorBoard to view resluts

首先安装PyTorch Profiler的TensorBoard插件，如下所示。

```shell
pip install torch_tb_profiler
```

然后在终端cd到源码目录，启动TensorBoard，并指定分析结果所在的目录，如下所示。

```shell
tensorboard --logdir=./tblog
```

```
I0831 09:40:32.463490 11848 plugin.py:429] Monitor runs begin
I0831 09:40:32.463867 11848 plugin.py:444] Find run directory D:\CodePython\T\tblog
I0831 09:40:32.464867  6092 plugin.py:493] Load run tblog
I0831 09:40:32.472933  6092 loader.py:57] started all processing
Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
TensorBoard 2.10.1 at http://localhost:6006/ (Press CTRL+C to quit)
I0831 09:40:33.688163  6092 plugin.py:497] Run tblog loaded
I0831 09:40:33.688163  8040 plugin.py:467] Add run tblog
```

在浏览器打开TensorBoard服务所返回的地址，即打开http://localhost:6006/，即可看到TensorBoard界面。

可在左侧控制面板中的Views下拉菜单，选择不同的可视化视图。其中Overview显示总的概述，Operator显示操作算子，GPU Kernel显示GPU核函数执行情况，Trace显示.json堆栈跟踪分析，Memory显示CPU内存与GPU内存占用，Module显示模型个操作执行时间。有时，在概述Overview页面的底部，会显示Performance Recommendation性能建议。如果进行分布式训练或推理，还可选择Distribute视图。目前该插件支持以NCCL/GLOO为后端分析DDP的分布式视图。

此外，若在TensroBoard日志目录下存在多条记录，还可通过Workers下拉菜单，选择不同的记录进行可视化。
