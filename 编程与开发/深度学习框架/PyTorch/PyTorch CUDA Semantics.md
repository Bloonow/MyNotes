[toc]

# CUDA Semantics

本节内容若无特殊说明，均在`torch.cuda`模块的命名空间中。

可以使用torch.cuda来设置或执行CUDA操作，它会跟踪所选的当前GPU设备，所创建的CUDA张量默认分配到当前GPU设备上。可使用torch.cuda.device()上下文管理器在作用范围内切换当前GPU设备，或使用torch.cuda.set_device()方法指定默认GPU设备。需要注意，一旦张量在设备上分配，就可以直接对其进行操作而无需考虑所选择的设备，且操作结果始终放置在相同设备上。

默认情况下，除torch.Tensor.copy_()或其他具有类似复制功能如torch.Tensor.to()、torch.Tensor.cuda()之类的方法以外，是不允许进行跨GPU设备对张量进行操作的。除非启用端到端内存访问（peer-to-peer memory access），也即统一内存编址，否则任何跨GPU设备的张量操作都将引发错误。

## 1. CUDA Compute Precision

### 1.1 TensorFloat-32 (TF32) on Ampere devices

从PyTorch 1.7开始，有一个名为allow_tf32的新标志（flag），该标志在PyTorch 1.7到PyTorch 1.11中默认为True，在PyTorch 1.12及以后版本中默认为False。该标志控制PyTorch是否允许使用TensorFloat32（TF32）的张量核心（Tensor Cores）来计算矩阵乘法操作和卷积操作。

> 张量核心（Tensor Cores）是NVIDIA GPU自Ampere架构以来引入的，专门加速矩阵乘法计算的特殊功能硬件单元，旨在提供更高的矩阵乘法性能。

TF32张量核心能够在PyTorch的torch.float32张量类型上实现性能更好的矩阵乘法操作和卷积操作，这是通过将数据舍入到10位尾数进行计算，并使用FP32精度积累结果，以保持FP32精度的动态范围。

注意，矩阵乘法操作和卷积操作是分开进行控制的，可通过如下标志进行设置，如下所示。

```python
# The flag below controls whether to allow TF32 on matmul.
torch.backends.cuda.matmul.allow_tf32 = True
# The flag below controls whether to allow TF32 on cuDNN (convolutions).
torch.backends.cudnn.allow_tf32 = True
```

如果需要在C++代码中控制是否启用TF32张量核心加速，可使用如下代码进行设置。

```c++
torch::globalContext().setAllowTF32CuBLAS(true);
torch::globalContext().setAllowTF32CuDNN(true);
```

需要注意的是，除直接的torch.matmul()和torch.conv2d()之类的方法本身外，凡是在实现中使用到矩阵乘法和卷积的模块也会收到影响，例如nn.Linear、nn.Conv、cdist、dot、affine_grid、grid_sample、log_softmax、GRU、LSTM等。

### 1.2 Reduced Precision Reduction in FP16/BF16 GEMMs

FP16 GEMMs在可能会中间计算过程中使用更低的精度（例如采用FP16而不是FP32）来完成归约（reduction），这在某些特定的工作负责（例如K维度特别大的情况）或特定的GPU架构上，会取得更高的性能。但可能会面临潜在的数值精度溢出问题。

可通过如下标志设置是否启用FP16低精度计算，如下所示。

```python
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
```

```c++
torch::globalContext().setAllowFP16ReductionCuBLAS(true);
```

对于BF16 GEMMs也存在类似的标志，如下所示。

```python
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
```

```c++
torch::globalContext().setAllowBF16ReductionCuBLAS(true);
```

需注意的是，对于BF16其默认标志设置为True，如果观察到工作负载中的数值不稳定，可以将其设为False。

## 2. Asynchronous Execution

默认情况下，GPU操作是异步的，当调用一个使用GPU的函数时，操作会被加入到特定的设备中，但不一定会在稍后执行。这允许用户并行执行更多的计算，包括在CPU或其他GPU上的操作。一般来说，异步计算的效果对调用者是不可见的，因为(1)每个设备按照它们排序的操作队列来顺序执行，(2)PyTorch在CPU和GPU之间或两个GPU之间复制数据时自动执行必要的同步。

可以通过将环境变量CUDA_LAUNCH_BLOCKING设置为1来强制同步计算。当GPU会引发错误时，这可以便于调试，因为在异步执行中，这样的错误直到操作实际执行后才会报告，因此堆栈跟踪不会显示它是在哪里被请求的。

异步计算的一个结果是，没有同步的时间测量是不准确的，要获得精确的测量结果，应该在待测试代码前后调用torch.cuda.synchronize()手动同步，或者使用torch.cuda.Event时间来记录时间，如下所示。

```python
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()

""" Run some things here """

end_event.record()
torch.cuda.synchronize()  # Wait for the events to be recorded
elapsed_time_ms = start_event.elapsed_time(end_event)
```

此外，需要注意的是，一些函数如torch.Tensor.to()和torch.Tensor.copy\_()是隐式同步的，即CPU端会阻塞等待GPU端完成才会继续执行，但它们支持显式的non\_blocking参数，这使得CPU端可以避免非必要的同步。

### 2.1 CUDA Streams

一个CUDA Stream流是属于某个特定GPU设备的CUDA操作的线性队列，默认情况下，每个GPU设备使用其自己的默认CUDA流，而无需显式创建CUDA流。每个CUDA流中的操作会按照提交顺序序列化执行，但不同流中的操作可以以任何相对顺序并发执行，除非显式地使用同步函数，例如torch.cuda.synchronize()函数或torch.cuda.Stream.wait_stream()函数。

需要注意的是，当CUDA上下文采用的CUDA流是默认流时，PyTorch会在数据移动时自动执行必要的同步。然而，当使用非默认流时，用户有责任确保适当的同步，例如下述示例代码是不正确的。

```python
s = torch.cuda.Stream()  # Create a new stream
A = torch.empty([100, 100], device='cuda').normal_(0.0, 1.0)
with torch.cuda.stream(s):
    B = torch.sum(A)  # sum() may start execution before normal_() finishes
```

### 2.2 Stream semantics of backward passes

训练中，backward反向过程中的CUDA操作，与其相对应的forward正向过程中的CUDA操作，会在相同的CUDA流上执行。也即，forward中的CUDA操作在某一个CUDA流上执行，其在backward中对应的CUDA操作也会在相同的CUDA流上执行。如果在forward过程中使用了多个CUDA流并行执行，那么backward过程中也会自动使用多个相同的CUDA流并行执行。

对于反向传播来说，一次backward过程调用与它周围操作之间的语义，与其他任何操作的语义都是相同的。即使如前所述的forward在多个CUDA流上执行的情况，backward反向过程也会在其内部插入同步来确保这一点。更具体地说，当调用torch.autograd.backward()、torch.autograd.grad()、torch.Tensor.backward()时，以下三个行为“(1)可选的初始梯度构造、(2)调用backward反向传递过程、(3)使用梯度”与其他任意一组操作具有相同的CUDA流语义。

```python
s = torch.cuda.Stream()

# Safe, grads are used in the same stream context as backward()
with torch.cuda.stream(s):
    loss.backward()
    use_grads()

# Unsafe
with torch.cuda.stream(s):
    loss.backward()
use_grads()

# Safe, with synchronization
with torch.cuda.stream(s):
    loss.backward()
torch.cuda.current_stream().wait_stream(s)
use grads

# Safe, populating initial grad and invoking backward are in the same stream context
with torch.cuda.stream(s):
    loss.backward(gradient=torch.ones_like(loss))

# Unsafe, populating initial_grad and invoking backward are in different stream contexts, without synchronization
initial_grad = torch.ones_like(loss)
with torch.cuda.stream(s):
    loss.backward(gradient=initial_grad)

# Safe, with synchronization
initial_grad = torch.ones_like(loss)
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    initial_grad.record_stream(s)
    loss.backward(gradient=initial_grad)
```

## 3. Memory Management

PyTorch使用缓存内存分配器（caching memory allocator）来加速内存分配，这允许在不同步设备的情况下快速释放内存。但是，由分配器管理的未使用内存仍然会在nvidia-smi中显示，就像其仍在使用一样。

可以使用torch.cuda.memory_allocated()和torch.cuda.max_memory_allocated()来监控张量占用的内存，并使用torch.cuda.memory_reserved()和torch.cuda.max_memory_reserved()来监控缓存分配器管理的内存总量。调用torch.cuda.empty_cache()将释放PyTorch中所有未使用的缓存内存，以便其他GPU应用程序使用这些内存，不过，张量已经实际占用的GPU内存将不会被释放。

### 3.1 Using custom memory allocators for CUDA

可通过C/C++扩展将缓存内存分配器定义为简单的函数，并将它们编译为共享库，下面的代码展示了一个基本的分配器，它能够跟踪所有的内存操作。

```c++
// my_alloc.cc
// g++ my_alloc.cc -o my_alloc.so -I /usr/local/cuda/include -shared -fPIC
#include <sys/types.h>
#include <iostream>
#include <cuda_runtime_api.h>

extern "C" {
void* my_malloc(size_t size, int device, cudaStream_t stream) {
    void *ptr;
    cudaMalloc(&ptr, size);
    std::cout << "alloc " << ptr << " " << size << std::endl;
    return ptr;
}
void my_free(void* ptr, size_t size, int device, cudaStream_t stream) {
    cudaFree(ptr);
    std::cout << "free " << ptr << " " << stream << std::endl;
}
}
```

在Python中通过torch.cuda.memory.CUDAPluggableAllocator()方法使用自定义内存分配器，用户负责提供.so库文件的路径，以及相应的alloc与free函数的签名名称，如下所示。

```python
import torch

# Load the allocator
new_alloc = torch.cuda.memory.CUDAPluggableAllocator('my_alloc.so', 'my_malloc', 'my_free')
# Swap the current allocator
torch.cuda.memory.change_current_allocator(new_alloc)
# This will allocate memory in the device using the new allocator
b = torch.rand([3, 4], device='cuda')
```

需要注意，使用torch.cuda.memory.CUDAPluggableAllocator()指定自定义内存分配器的操作，应该在PyTorch程序最开始处，在所有张量内存分配操作之前执行，否则会因为默认内容分配器已经实例化，而无法替换为自定义的内存分配器。

### 3.2 Use pinned memory buffers

注意，如果过度使用固定内存，当RAM不足时可能会导致严重的问题，应该意识到固定通常是一个昂贵的操作。

当数据源位于主机的固定内存时，也即不可分页内存（non-pageable memory）或称为固定页内存（page-locked memory）时，主机到设备（Host to GPU）的内存拷贝会快得多。CPU张量暴露了一个torch.Tensor.pin_memory()方法，该方法返回原始张量对象的一个副本，它的数据会被存放在一个固定页内存中。

一旦固定了张量，就可以使用异步GPU拷贝，也即指定torch.Tensor.to()和torch.Tensor.copy\_()方法的non_blocking=True参数，这可以用于将数据传输与计算重叠。此外，可以通过将torch.utils.data.DataLoader的构造参数pin_memory指定为True，以使得DataLoader将加载的张量数据存储在固定页内存中。

## 4. Just-in-Time Compilation

PyTorch在CUDA张量上执行操作时，会即时编译一些操作，例如torch.special.zeta等操作。

这种编译可能很耗时（取决于硬件和软件），并且对于单个操作符可能会发生多次即时编译，因为许多PyTorch操作符实际上是根据输入规模等不同情况，从各种Kernels中选择使用哪一个，每个内核都必须编译一次。该编译在每个进程中发生一次，而如果使用了内核缓存（kernel cache），则只发生一次。

默认情况下，如果定义了环境变量XDG_CACHE_HOME，则PyTorch会在\$XDG_CACHE_HOME/torch/kernels路径下创建内核缓存，否则会在\$HOME/.cache/torch/kernels路径下创建内核缓存。注意，在Windows平台上暂不支持内核缓存。

可以通过环境变量USE_PYTORCH_KERNEL_CACHE控制内核缓存的行为，若环境变量USE_PYTORCH_KERNEL_CACHE设为0，则不适用内核缓存，若设置为某个路径，则使用该路径作为内核缓存，而不使用默认路径。

# torch.cuda

本节内容若无特殊说明，均在`torch.cuda`模块的命名空间中。

该软件包增加了对CUDA张量类型的支持，这些张量实现了与CPU张量相同的功能，但它们利用GPU进行计算。可使用torch.cuda.is_available()来确定系统是否支持CUDA加速。

```python
def is_available() -> bool
def can_device_access_peer(device: _device_t, peer_device: _device_t) -> bool
```

函数is_available()返回一个布尔值，表示CUDA是否可用。函数can_device_access_peer()检查两个GPU设备之间是否可以进行直接对等访问。

```python
def device_count() -> int
def current_device() -> int
def set_device(device: _device_t) -> None
```

函数device_count()用于返回可用GPU设备的数量。函数current_device()用于获取当前所使用的GPU设备编号。函数set_device()用于设置当前所使用的GPU设备，不推荐使用该函数，而应该使用CUDA_VISIBLE_DEVICES环境变量。

```python
class device:
    def __init__(self, device: Any)
    def __enter__(self)
    def __exit__(self, type: Any, value: Any, traceback: Any)
class device_of(device):
    def __init__(self, obj):
```

用于指定所选设备的上下文管理器，其中device_of可以指定上下文设备为与obj相同的设备。

```python
def get_device_name(device: Optional[_device_t] = None) -> str
def get_device_capability(device: Optional[_device_t] = None) -> Tuple[int, int]
def get_device_properties(device: _device_t) -> _CudaDeviceProperties
```

获取GPU设备的名称、计算能力、属性信息。

```python
def utilization(device: Optional[Union[Device, int]] = None) -> int
def memory_usage(device: Optional[Union[Device, int]] = None) -> int
def temperature(device: Optional[Union[Device, int]] = None) -> int
def power_draw(device: Optional[Union[Device, int]] = None) -> int
def clock_rate(device: Optional[Union[Device, int]] = None) -> int
```

这几个函数所返回的信息都是由nvidia-smi给出的过去采样周期内的。函数utilization()返回一个或多个Kernel在GPU上执行的时间百分比。函数memory_usage()返回全局设备内存被读取或写入的时间百分比。函数temperature()返回GPU传感器的平均温度，以摄氏度为单位。函数power_draw()返回返回GPU传感器的平均功耗，以毫瓦（mW）为单位。函数clock_rate()返回GPU的SM的时钟频率速度，以赫兹（Hz）为单位。

```python
def current_blas_handle()
```

该函数返回指向当前cuBLAS句柄的cublasHandle_t指针。

```python
def get_arch_list() -> List[str]
def get_gencode_flags() -> str
```

函数get_arch_list()返回该PyTorch库是针对哪些CUDA架构所编译的。函数get_gencode_flags()返回该PyTorch库在编译时所使用的NVCC标志。

```python
def current_stream(device: Optional[_device_t] = None) -> Stream
def default_stream(device: Optional[_device_t] = None) -> Stream
def set_stream(stream: Stream)
```

函数current_stream()返回给定设备的当前使用的CUDA流。函数default_stream()返回给定设备的默认使用的CUDA流。函数set_stream()设置当前使用的CUDA流，推荐使用CUDA流上下文管理器，而不是该函数。

```python
class StreamContext:
    def __init__(self, stream: Optional["torch.cuda.Stream"])
    def __enter__(self)
    def __exit__(self, type: Any, value: Any, traceback: Any)
def stream(stream: Optional["torch.cuda.Stream"]) -> StreamContext
```

封装给定stream流的上下文管理器，所有在其上下文中CUDA Kernels操作将会提交到指定的stream流排队。

```python
def synchronize(device: _device_t = None) -> None
```

该函数等待指定设备上所有CUDA流的所有Kernel操作完成，用于在给定GPU设备上全局同步。

```python
def ipc_collect()
```

该函数用于在CUDA进程间通信（Inter-Process Communication）后强制收集GPU内存。检查是否可以从内存中清除任何发送的CUDA张量。如果没有激活的计数器，将强制关闭用于引用计数的共享内存文件。可用于“生产者进程停止主动发送张量并想要释放未使用内存”的情况。

## 1. Random Seed

本节内容若无特殊说明，均在`torch.cuda.random`模块的命名空间中。

```python
def manual_seed(seed: int) -> None
def manual_seed_all(seed: int) -> None
```

函数manual_seed()用于设置当前GPU的随机数种子，如果CUDA不可用，则该函数无任何作用。函数manual_seed_all()用于设置所有GPU的随机数种子。

```python
def seed() -> None
def seed_all() -> None
```

函数seed()用于将当前GPU的随机数种子设置为当前的随机数。函数seed_all()用于将所有GPU的随机数种子设置为当前的随机数。

```python
def initial_seed() -> int
```

该函数返回当前GPU的当前随机数种子。

## 2. Streams and Events

本节内容若无特殊说明，均在`torch.cuda.streams`模块的命名空间中。

```python
class Stream(torch._C._CudaStreamBase):
    def __new__(cls, device=None, priority=0, **kwargs)
```

该类是对一个CUDA流（cudaStream_t）的封装，一个CUDA流是属于特定GPU设备的线性执行序列，独立于其他流。其中，参数device指定特定GPU设备，参数priority指定当前CUDA流的优先级，取值为负数和0，使用负数表示更高的优先级，默认采用0优先级。

```python
class Stream(torch._C._CudaStreamBase):
    def query(self)
    def synchronize(self)
    def wait_stream(self, stream)
```

函数query()用于检查提交的所有工作是否已经完成。函数synchronize()用于等待流中的所有内核完成，它是对cudaStreamSynchronize()函数的封装。函数wait_stream()与另一个流同步，用于让所有提交到当前流中的未来工作，等待给定stream流执行完成。

```python
class Stream(torch._C._CudaStreamBase):
    def record_event(self, event=None) -> Event
    def wait_event(self, event)
```

函数record_event()用于记录当前流中的事件，并返回一个torch.cuda.streams.Event对象。函数wait_event()用于让所有提交到当前流中的未来工作，等待给定event事件完成。

```python
class Event(torch._C._CudaEventBase):
    def __new__(cls, enable_timing=False, blocking=False, interprocess=False)
```

该类是对一个CUDA事件（cudaEvent_t）的封装，一个CUDA事件一个同步标记（synchronization marker），可用于监控设备的进度、精确测量时间以及同步CUDA流。在事件首次记录或导出到另一个进程时，底层CUDA事件将延迟初始化。创建后，只有同一设备上的CUDA流才能记录该事件。不过，任何设备上的流都可以等待事件。

```python
class Event(torch._C._CudaEventBase):
    def query(self)
    def synchronize(self)
```

函数query()返回当前事件捕获的所有工作是否已经完成。函数synchronize()用于等待该事件中当前捕获的所有工作完成，这将阻塞CPU线程继续运行，直到事件完成。

```python
class Event(torch._C._CudaEventBase):
    def record(self, stream=None)
    def wait(self, stream=None)
```

函数record()用于记录（捕获）给定流stream中的事件，CUDA流的设备必须与事件的设备匹配。函数wait()用于让所有提交到给定stream流中的未来工作，等待当前事件完成。

```python
class Event(torch._C._CudaEventBase):
    def elapsed_time(self, end_event)
```

该函数用于计算两个事件执行record()记录的时刻，之间所经过的时间，以毫秒为单位。

```python
class ExternalStream(Stream):
    def __new__(cls, stream_ptr, device=None, **kwargs):
```

该类是对一个CUDA流的封装，区别在于该CUDA流是由外部其他程序或库分配的，以便于跨进程数据交换以及多个库之间的交互。注意，PyTorch不管理该外部CUDA流的生命周期，用户有责任在使用这个类时保持所引用CUDA流的活动状态，并负责其销毁时机。

## 3. Communication Collectives

本节内容若无特殊说明，均在`torch.cuda.comm`模块的命名空间中。


```python
def broadcast(tensor, devices=None, *, out=None)
def broadcast_coalesced(tensors, devices, buffer_size=10485760)
```

函数broadcast()将张量广播到指定的GPU设备，必须指定devices参数和out参数中的一个。函数broadcast_coalesced()将张量序列广播到指定的GPU设备，小的张量首先会合并到缓冲区中，以减少同步的次数。

```python
def reduce_add(inputs, destination=None)
def reduce_add_coalesced(inputs, destination=None, buffer_size=10485760)
```

函数reduce_add()将来自多个GPU设备的张量做求和归约，所有输入张量都应具有匹配的形状、dtype类型和存储布局。函数reduce_add_coalesced()将来自多个GPU设备的张量序列做求和归约，小的张量首先会合并到缓冲区中，以减少同步的次数。

```python
def scatter(tensor, devices=None, chunk_sizes=None, dim=0, streams=None, *, out=None)
```

函数scatter()将张量散射到多个GPU设备上，该函数在dim所指定的张量维度上，将其以chunk_sizes为大小划分若干份，散射到相对应的GPU设备上。其中，参数chunk_sizes的长度应与GPU设备的数目相匹配，且其值之和应为张量在dim维度轴上的维数。注意，必须指定devices参数和out参数中的一个，如果指定了out参数，则根据out参数推断每个GPU设备所散射到的张量大小，且此时不能指定chunk_sizes参数。

```python
def gather(tensors, dim=0, destination=None, *, out=None)
```

函数gather()将来自多个GPU设备的张量进行聚集，该函数在dim所指定的张量维度上，将多个GPU设备上的张量聚集起来。注意，如果指定了out参数，则不能指定destination参数。

## 4. Memory Management

本节内容若无特殊说明，均在`torch.cuda.memory`模块的命名空间中。

```python
def memory_stats(device: Union[Device, int] = None) -> Dict[str, Any]
def reset_peak_memory_stats(device: Union[Device, int] = None) -> None
```

函数memory_stats()返回给定设备的CUDA内存分配器统计信息的字典，该函数的返回值是一个统计量的字典，每个字典都是非负整数。函数reset_peak_memory_stats()重置CUDA内存分配器跟踪的峰值（peak）统计量。

```python
def memory_allocated(device: Union[Device, int] = None) -> int
def max_memory_allocated(device: Union[Device, int] = None) -> int
def reset_max_memory_allocated(device: Union[Device, int] = None) -> None
```

函数memory_allocated()返回在给定GPU设备上，张量当前占用的GPU内存，以字节为单位。函数max_memory_allocated()返回自程序开始以来，张量占用的峰值内存，以字节为单位。可使用reset_max_memory_allocated()重置跟踪起点，例如，这两个函数可以测量训练循环中每次迭代时的峰值内存使用情况。

```python
def memory_reserved(device: Union[Device, int] = None) -> int
def max_memory_reserved(device: Union[Device, int] = None) -> int
def reset_max_memory_cached(device: Union[Device, int] = None) -> None
```

函数memory_reserved()返回在给定GPU设备上，由缓存分配器管理的（预留的）GPU内存，以字节为单位。函数max_memory_reserved()返回自程序开始以来，缓存分配器管理的（预留的）最大GPU内存，以字节为单位。可使用reset_max_memory_cached()重置跟踪起点。

```python
def mem_get_info(device: Union[Device, int] = None) -> Tuple[int, int]
def memory_summary(device: Union[Device, int] = None, abbreviated: bool = False) -> str
def list_gpu_processes(device: Union[Device, int] = None) -> str
```

函数mem_get_info()返回给定设备的全局空闲内存和全局总内存。函数memory_summary()返回给定GPU设备的当前内存分配器统计信息。函数list_gpu_processes()返回给定设备上的正在运行的进程以及它们的GPU内存使用情况。

```python
def empty_cache() -> None
```

该函数用于释放缓存分配器当前持有的所有未占用的缓存内存，以便在其他GPU应用程序中使用，并在nvidia-smi中可见。

# CUDA Graph

本节内容若无特殊说明，均在`torch.cuda.graphs`模块的命名空间中。

```python
def is_current_stream_capturing()
```

如果当前CUDA流正在进行CUDA图捕获，则返回True，否则返回False；如果当前设备上不存在CUDA上下文，则返回False而不初始化上下文。

```python
def make_graphed_callables(callables, sample_args, num_warmup_iters=3, allow_unused_input=False)
```

接受可调用对象（函数或torch.nn.Module模块）作为callables参数输入，并返回其CUDA图版本，称该语义为图捕获（graphize）。

```python
class graph:
    def __init__(self, cuda_graph, pool=None, stream=None, capture_error_mode: str = "global")
    def __enter__(self)
    def __exit__(self, exc_type, exc_value, traceback)
```

一个上下文管理器，用于将CUDA操作捕获到torch.cuda.CUDAGraph对象中，以便稍后重播（replay）。

```python
class CUDAGraph(torch._C._CUDAGraph):
    def __new__(cls)
    def capture_begin(self, pool=None, capture_error_mode="global")
    def capture_end(self)
    def replay(self)
    def reset(self)
```

该类是对一个CUDA图（cudaGraph_t）的封装，一个CUDA图指的是，一个CUDA流以及其所依赖的CUDA流上的，所执行工作的记录，主要是Kernel与其参数的记录。

PyTorch支持使用流捕获的方法，也即cudaStreamBeginCapture(stream)与cudaStreamEndCapture(stream, &graph)方法，来获构建CUDA图，这将使一个CUDA流处于捕获模式。提交到捕获流（capturing stream）上的CUDA工作并不会在GPU上实际执行，而是被记录到CUDA图当中。需要注意的是，用于捕获CUDA图的流必须是非默认的CUDA流。一旦CUDA图捕获完成，它可以在任何CUDA流上执行重播过程。

捕获后，可用启动CUDAGraph执行（launched），并且可根据需求执行多次，称为重播（replay），每次重播都使用相同的参数运行相同的Kernel。对于指针参数（例如tensor.data_ptr<>()返回的指针）来说，这意味着使用相同的内存地址。通过在每次重播之前用新的数据，填充给定指针所指向的内存地址，可以使用新的数据重新运行相同的工作。

## 1. Why CUDA Graphs?

使用CUDA图的重播来执行程序，会牺牲即时执行（eager execution）的灵活性，但会极大地降低CPU端的开销。因为CUDA图的参数和Kernel都是固定的，因此图的重播会跳过参数设置和Kernel分发的所有层，包括Python、C++和CUDA驱动程序的开销。在CUDA底层实现上，只需通过一次cudaGraphLaunch()调用即可将整个CUDA图所捕获的工作全部提交到GPU设备上执行。

如果所构建的神经网络模型是图安全（graph-safe）的，即神经网络的形状和控制流是静态的，且有可能程序性能在一定程度上受限于CPU端，则可以尝试使用CUDA图执行模式。

## 2. How to Use CUDA Graphs?

在PyTorch中，使用torch.cuda.CUDAGraph类、上下文管理器torch.cuda.graph()、和torch.cuda.make_graphed_callables()辅助方法来使用CUDA图执行模型。一旦CUDA图捕获完成，它可以在任何CUDA流上执行重播过程。

torch.cuda.graph()是一个简单通用的上下文管理器，可以在其上下文中捕获CUDA工作，而且它会自动在一个新的CUDA流上捕获CUDA操作。在捕获之前，通过运行一些预迭代流程来对要捕获的工作负载进行预热，需要注意的是，预热工作必须在其他非默认的CUDA流（side stream）上执行。

由于图在每次重播过程中，会读取和写入相同的内存地址，因此用户必须在捕获期间维持所涉及数据的输入张量与输出张量的长期活动引用。要在新的输入数据上执行图的重播过程，可将新数据复制到所捕获的输入张量的内存地址，重播图过程，然后从捕获的输出张量的内存地址中读取新输出。一个示例如下所示。

```python
# 一些需要被捕获的CUDA操作
def some_cuda_ops(static_input: Tensor) -> Tensor:
    suqared = static_input.square()
    rooted = static_input.sqrt()
    return torch.exp(suqared + rooted) + torch.exp(suqared - rooted)

# 输入张量的占位符（placeholder）
static_input = torch.empty([3, 4], device='cuda')

# 在开始捕获CUDA操作前，使用其他CUDA流预热
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        static_output = some_cuda_ops(static_input)
torch.cuda.current_stream().wait_stream(s)

# 要捕获CUDA操作的CUDA图
g = torch.cuda.CUDAGraph()
# 开始捕获CUDA图，该上下文管理器会自动创建新的CUDA流来捕获CUDA操作
with torch.cuda.graph(g):
    static_output = some_cuda_ops(static_input)

# 此时已完成捕获，可使用新的张量数据执行所捕获的CUDA操作，
# 新的张量数据 my_input 可以位于任意设备上，因为 copy_() 操作支持跨设备执行
my_input = torch.rand([3, 4], device='cuda')
static_input.copy_(my_input)
g.replay()
my_output = static_output.clone().detach()
print(my_input, my_output, sep='\n')
```

后续一些例子展示了torch.cuda.graph()和torch.cuda.make_graphed_callables()的高级用法。

此处有一些注意事项，如果一组操作operations不违反以下任何约束，则它是可捕获的。这些约束规则对于torch.cuda.graph()上下文中的以及torch.cuda.make_graphed_callables()中的所有CUDA工作负载都起作用。

违反以下任何一条都可能导致运行时错误：

- 捕获必须在非默认CUDA流上执行。这只需在使用原生的torch.cuda.CUDAGraph.capture_begin()与torch.cuda.CUDAGraph.capture_end()方法时需要注意，因为上下文管理器torch.cuda.graph()和辅助方法torch.cuda.make_graphed_callables()会自动创建一个新的CUDA流用于捕获CUDA负载。
- 一些需要GPU与CPU同步的操作是禁止被捕获的，例如torch.Tensor.item()方法。
- CUDA的随机数生成器（Random Number Generator，RNG）操作是允许的，但必须使用默认的生成器。例如，显式创建一个新的torch.Generator对象并将其作为CUDA随机数生成器操作的generator等参数是不被允许的。

违反以下任何一条都可能导致潜在的数值错误或未定义行为：

- 在一个进程（process）中，在同一时刻只能同时进行一个CUDA图的捕获。
- 在CUDA图的捕获过程中，任何不参与CUDA图捕获的其他CUDA工作负载不能在该进程（及其任何线程上）运行。
- 不能捕获CPU端上的工作负载，如果捕获的操作中包含CPU工作负载，则该工作负载将在CUDA图重播期间被忽略。
- 每次CUDA重播都读取和写入相同的（虚拟）内存地址。
- 禁止捕获（基于CPU或GPU数据的）动态控制流。
- 禁止捕获动态数据形状（shape），CUDA图加速所捕获的操作序列中的每个张量都具有固定的形状大小和布局。
- 在CUDA图捕获过程中使用多个CUDA流是允许的，但存在一些限制，详细见后续介绍。

### 2.1 Whole-network capture

如果整个深度学习网络模型都是可捕获的，那么可以捕获并重播训练过程中的一个完整迭代，如下示例。

```python
# 构造模型及其训练配置
N, D_in, H, D_out = 640, 4096, 2048, 1024
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
    torch.nn.ReLU()
).to('cuda')
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1.e-3)
lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)

# 数据占位符
static_input  = torch.empty([N, D_in],  device='cuda')
static_target = torch.empty([N, D_out], device='cuda')

# 在开始捕获CUDA操作前，使用其他CUDA流预热
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        pred = model(static_input)
        loss = loss_fn(pred, static_target)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
torch.cuda.current_stream().wait_stream(s)

# 要捕获CUDA操作的CUDAGraph图
g = torch.cuda.CUDAGraph()
# 在捕获开始之前，将模型参数的梯度设置为None，使得backward()过程会
# 使用CUDA图的私有内存池（graph's private pool）来创建模型参数的 .grad 属性
optimizer.zero_grad(set_to_none=True)
# 开始捕获CUDA图，该上下文管理器会自动创建新的CUDA流来捕获CUDA操作
with torch.cuda.graph(g):
    static_pred = model(static_input)
    static_loss = loss_fn(static_pred, static_target)
    static_loss.backward()
    optimizer.step()
    lr_scheduler.step()

# 此时已完成捕获，可使用新的张量数据执行所捕获的CUDA操作
real_inputs  = [torch.rand([N, D_in],  device='cuda') for _ in range(10)]
real_targets = [torch.rand([N, D_out], device='cuda') for _ in range(10)]
for input, target in zip(real_inputs, real_targets):
    # 使用新的数据张量填充CUDA图所使用的内存地址
    static_input.copy_(input)
    static_target.copy_(target)
    # 所捕获的该CUDA图的重播过程包括 forward, backward, step 过程
    # 甚至不用在迭代中调用 optimizer.zero_grad() 方法，
    # 因为所捕获的 backward 过程每次会就地（in-place）填充模型参数的 .grad 属性
    g.replay()
    # 执行CUDA图的重播过程后，模型参数会被正确更新，且
    # static_pred, static_loss 和 .grad 持有根据当前迭代数据所计算的结果

# 训练完成后，也可以使用CUDA图重播的方式进行推理，但推荐直接使用model模型，
# 因为所捕获的CUDA图中包含其他训练操作是推理过程中不必须的，当然也可使用新的CUDA图捕获一个推理过程
my_input = torch.rand([N, D_in],  device='cuda')
my_pred = model(my_input)
print(my_input, my_pred, sep='\n')
```

### 2.2 Partial-network capture

如果网络模型并不全部都是可捕获的，例如存在动态控制流、动态形状、CPU同步或基本的CPU端逻辑，则可以使用即时模式（eager mode）运行不安全的部分，并使用torch.cuda.make_graphed_callables()仅捕获可进行捕获的部分。

默认情况下，由make_graphed_callables()返回的可调用对象是能够感知自动梯度机制的（autograd-aware），可以在训练的循环迭代中，直接替换掉原来相应的函数或torch.nn.Module模块来使用。

函数make_graphed_callables()会在内部创建CUDAGraph对象，运行预热迭代，并根据需要维护静态输入（static inputs）和静态输出（static outputs）。这与torch.cuda.graph()上下文管理器不同，所以无需手动管理这些数据。

下面示例中展示了存在动态控制流的网络模型，依赖于数据的动态控制流意味着模型不能进行端到端的捕获，但make_graphed_callables()允许捕获其中可捕获的部分。

```python
# 构造模型及其训练配置
N, D_in, H, D_out = 640, 4096, 2048, 1024
model1 = torch.nn.Linear(D_in, H).to('cuda')
model2 = torch.nn.Linear(H, D_out).to('cuda')
model3 = torch.nn.Linear(H, D_out).to('cuda')
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(itertools.chain(model1.parameters(), model2.parameters(), model3.parameters()), lr=1.e-3)
lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)

# 数据占位符，其 requires_grad 属性必须与真实数据相匹配
static_x = torch.randn(N, D_in, device='cuda')
static_h = torch.randn(N, H, device='cuda', requires_grad=True)

# 使用make_graphed_callables()自动捕获相关可迭代对象的CUDA图版本
model1 = torch.cuda.make_graphed_callables(model1, (static_x,))
model2 = torch.cuda.make_graphed_callables(model2, (static_h,))
model3 = torch.cuda.make_graphed_callables(model3, (static_h,))

# 此时已完成捕获，可使用新的张量数据执行所捕获的CUDA操作
real_inputs  = [torch.rand([N, D_in],  device='cuda') for _ in range(10)]
real_targets = [torch.rand([N, D_out], device='cuda') for _ in range(10)]
# 整个训练过程并不是全部可捕获的
for input, target in zip(real_inputs, real_targets):
    optimizer.zero_grad(set_to_none=True)
    # model1, model2, model3 的 forward 过程的CUDA操作会根据相应的所捕捉的CUDA图来执行
    x = model1(input)
    if x.sum().item() > 0:
        x = model2(x)
    else:
        x = model3(x)
    loss = loss_fn(x, target)
    # 整个 backward 过程中，无论是执行 model2 还是执行 model3 的反向过程，
    # 都会与 model1 一起，以所捕捉的CUDA图模式执行
    loss.backward()
    optimizer.step()
    lr_scheduler.step()

# 训练完成后，使用模型
my_input = torch.rand([N, D_in],  device='cuda')
x = model1(my_input)
if x.sum().item() > 0:
    my_pred = model2(x)
else:
    my_pred = model3(x)
print(my_input, my_pred, sep='\n')
```

### 2.3 Usage with torch.cuda.amp

混合精度通常与torch.autocast上下文管理器，以及torch.cuda.amp.GradScaler类一起使用，梯度缩放GradScaler有助于防止小的梯度被减益为零。

对于一些典型的优化器，torch.cuda.amp.GradScaler.step()方法会同步CPU和GPU设备，这在图捕获期间是禁止的。为避免错误，要么使用make_graphed_callables()捕获部分网络，要么仅捕获forward,loss_fn,backward过程，而不捕获step过程。如下示例所示。

```python
# 构造模型及其训练配置
N, D_in, H, D_out = 640, 4096, 2048, 1024
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
    torch.nn.ReLU()
).to('cuda')
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1.e-3)
grad_scaler = torch.cuda.amp.GradScaler()

# 数据占位符
static_input  = torch.empty([N, D_in],  device='cuda')
static_target = torch.empty([N, D_out], device='cuda')

# 在开始捕获CUDA操作前，使用其他CUDA流预热
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=True):
            pred = model(static_input)
            loss = loss_fn(pred, static_target)
        grad_scaler.scale(loss).backward()
        grad_scaler.step(optimizer)  # 可能同步CPU与GPU设备
        grad_scaler.update()
torch.cuda.current_stream().wait_stream(s)

# 要捕获CUDA操作的CUDAGraph图
g = torch.cuda.CUDAGraph()
optimizer.zero_grad(set_to_none=True)
with torch.cuda.graph(g):
    with torch.cuda.amp.autocast(enabled=True):
        static_pred = model(static_input)
        static_loss = loss_fn(static_pred, static_target)
    grad_scaler.scale(static_loss).backward()
    # 此处不捕获可能会同步CPU与GPU设备的操作，而是在训练过程中将之以即时模式执行
    # grad_scaler.step(optimizer)
    # grad_scaler.update()

# 此时已完成捕获，可使用新的张量数据执行所捕获的CUDA操作
real_inputs  = [torch.rand([N, D_in],  device='cuda') for _ in range(10)]
real_targets = [torch.rand([N, D_out], device='cuda') for _ in range(10)]
for input, target in zip(real_inputs, real_targets):
    static_input.copy_(input)
    static_target.copy_(target)
    # 以CUDA图模式执行所捕捉的 forward, backward 操作
    g.replay()
    # 以即时模式执行可能会导致CPU与GPU同的 step 操作
    grad_scaler.step(optimizer)
    grad_scaler.update()

# 训练完成后，使用模型
my_input = torch.rand([N, D_in],  device='cuda')
my_pred = model(my_input)
print(my_input, my_pred, sep='\n')
```

### 2.4 Usage with multiple streams

若在捕获过程中，启用多个CUDA流，则捕获模式（capture mode）会自动应用到这些多个CUDA流上，但应注意的是，启用的其他多个CUDA流应该与捕获流（capturing stream）进行同步。也就是说，在捕获过程中，可以通过调用不同的CUDA流来暴露并行性，但整个流依赖的有向无环图（Directed Acyclic Graph，DAG）必须是从捕获流中分支出来，并在捕获结束时再合并入捕获流。如下所示。

```python
# 要捕获CUDA操作的CUDAGraph图
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):  # 进入CUDA图捕获模式，工作负载自动在捕获流上执行
    # 与捕获流同步，也即从捕获流中分支出来
    else_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(else_stream):
        cuda_work()
    # 再合并入捕获流
    torch.cuda.current_stream().wait_stream(else_stream)
```

### 2.5 Usage with DistributedDataParallel

当所使用的NCCL后端在2.9.6版本之前，CUDA图捕获不允许对集合通信（collectives）进行捕获，而必须使用make_graphed_callables()捕获部分网络，它会将allreduce归约操作推迟到所捕获的backward过程之外执行。

在使用DistributedDataParallel封装网络模型torch.nn.Module之前，在可捕获的网络模型上调用make_graphed_callables()进行捕获。

当NCCL版本大于等于2.9.6时，可以对使用集合通信（collectives）的整个网络模型进行捕获，但需要进行一些配置工作，如下所示。

1. 禁用DistributedDataParallel内部的异步错误处理：

   ```python
   os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '0'
   torch.distributed.init_process_group()
   ```

2. 在开始整个模型的捕获之前，在非默认的其他流上构建DistributedDataParallel模型：

   ```python
   s = torch.cuda.Stream()
   with torch.cuda.stream(s):
       model = DistributedDataParallel(model)
   ```

3. 在开始捕获前，必须以即时模式（eager mode）运行至少11次DistributedDataParallel迭代。

## 3. Graph memory management

所捕获的CUDA图在每次重播时，都会使用相同的虚拟地址。如果PyTorch将该地址的内存释放掉，则稍后的CUDA图重播可能会造成非法内存访问。如果PyTorch将内存地址重新分配给新的张量，则CUDA图的重播可能会破获这些张量的值。

因此，CUDA图所使用的虚拟地址必须保留，PyTorch缓存分配器会检测何时进行图捕获，并从CUDA图的私有内存池（private pool）中分配内存空间。一个CUDA图的私有内存池会保持激活（stays alive），直到它的CUDAGraph对象和捕获期间所创建的所有张量超出作用范围。

私有内存池会自动维护，默认情况下，分配器为每个捕获图创建一个单独的私有内存池。如果捕获了多个CUDA图，这种保守的方法能够确保多个CUDA图的重播不会破坏彼此的值，但有时可能会不必要的浪费内存。

### 3.1 Sharing memory across captures

为节省CUDA图的私有内存池的内存占用，上下文管理器torch.cuda.graph()和辅助方法torch.cuda.make_graphed_callables()允许不同捕获的CUDA图共享同一个私有内存池。如果用户明确知道多个CUDA图的重播顺序总是和其捕获顺序一样，并且不会并发地重播，那么多个CUDA图共享同一个私有内存池是安全的。

可以通过指定torch.cuda.graph()的可选的pool参数，来指定CUDA图捕获使用特定的私有内存池，这可以在多个CUDA图之间共享内存池，如下所示。

```python
g1 = torch.cuda.CUDAGraph()
g2 = torch.cuda.CUDAGraph()

with torch.cuda.graph(g1):
    static_out_1 = g1_workload(static_input_1)
with torch.cuda.graph(g2, pool=g1.pool()):
    static_out_2 = g2_workload(static_input_2)

static_input_1.copy_(real_data_1)
static_input_2.copy_(real_data_2)
g1.replay()
g2.replay()
```

若让torch.cuda.make_graphed_callables()方法使用共享的私有内存池，则可将多个可调用对象组成一个元组传给该辅助方法，这些可调用对象的顺序与实际工作负载中的顺序相同。注意，应确保这些可调用对象在工作负载中的调用顺序完全确定，且不存在并发执行，否者这些可调用对象必须分别使用make_graphed_callables()方法构造CUDA图。

