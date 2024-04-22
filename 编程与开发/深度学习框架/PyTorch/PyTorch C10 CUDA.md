# PyTorch C10

## 1. PyTorch基本架构

为实现对底层硬件（如CUDA等）的精准控制和使用，PyTorch的底层采用C++编写；为提供方便的用户API接口，PyTorch的上层模块采用Python编写。基本结构如下图所示。

<img src="PyTorch C10 CUDA.assets/PyTorch基本架构.png" style="zoom: 25%;" />

PyTorch使用C++完成对CUDA等底层硬件的对接，并高效地实现基础组件和部分算法；借助Python的原生（native）调用，将用C++实现的模块封装成接口，提供给Python代码调用，如此即可使用Python实现更多模块；在模块之上，封装更方便的API接口，提供给机器学习领域的开发者使用。

PyTorch的跨语言环境接口主要有两大部分：一是C++与原生运行环境的对接，二是Python与C++的对接。C++与原生运行环境的对接全部在ATen和C10内实现。例如，C10/CUDA的CUDAFunctions模块完成对NVIDIA CUDA Runtime API的二次封装，以支持上层更定制化的操作。Python与C++的对接层为torch._C模块，该部分接口在编译期自动生成，为Python代码提供支持。

## 2. PyTorch整体模块依赖关系

经过对代码的阅读，以及与PyTorch C++ API维护者Brain Hirsh的邮件沟通，可以了解到，PyTorch的整体模块依赖关系大致如下图所示。

<img src="PyTorch C10 CUDA.assets/PyTorch整体模块依赖关系.png" style="zoom:25%;" />

ATen代表A Tensor Library的缩写，是PyTorch最原始的Tensor基础库。C10是一个仍处于开发阶段的库，将逐步代替ATen的地位，注意到，ATen内的很多实现已经改为直接包含C10内的同类实现。例如，Tensor的实现已经完全迁移到C10内。因此，可以观察到，ATen对C10呈现出依赖关系。

ATen和C10封装了含CUDA在内的底层操作。基于ATen和C10的能力，PyTorch的开发者们在csrc模块（C Source）内使用C和C++实现autograd等更高级的功能。如Java一样，Python具有原生调用接口的功能。PyTorch的原生接口层为torch._C模块，其中暴露的所有原生功能皆由csrc模块提供。

PyTorch的更高层功能使用Python编写，并在部分功能处通过torch._C调用原生代码，以获得更好的性能和对CUDA等高级硬件的操控能力。

# C10 CUDA

PyTorch C10库的CUDA模块直接覆盖在CUDA运行环境上，为外层开发者提供基础资源管理服务。该模块由多个子模块组成，含实现了GC和类似Arena的GPU内存管理器和CUDA执行流管理器。这些子模块对CUDA Runtime API进行封装，为外层提供方便的接口，并通过对资源的复用，减少对底层API的频繁访问，以提供更好的性能。

本节将对C10/CUDA模块内的多个子模块展开分析。

## 1. CUDAException

<img src="PyTorch C10 CUDA.assets/C10 CUDAException.png" style="zoom:25%;" />

CUDAException模块对CUDA的调用提供最基础的保障功能。

C10_CUDA_CHECK负责检查运行过程调用的CUDA函数是否支持。在执行被包裹语句后，调用c10_cuda_check_implementation，得以实现功能。

c10_cuda_check_implementation通过读取CUDA Kernel Launch Registry（一个基于循环队列的日志记录器）单例，检查其是否登记错误信息，以判断是否出现过运行时错误。如果有出现，则将报错信息输出。

C10_CUDA_KERNEL_LAUNCH_CHECK用于检测CUDA是否存在未处理的错误报告。

C10_CUDA_CHECK_WARN用于包装某个运行过程。在执行被包裹语句后，会对CUDA是否出错进行检查，读取CUDA错误信息，并自动输出带文件名和代码行号的报错日志。

## 2. CUDAFunctions

<img src="PyTorch C10 CUDA.assets/C10 CUDAFunctions.png" style="zoom:25%;" />

CUDAFunctions模块对CUDA的部分接口进行封装。这些接口包含设备信息、内存拷贝和流同步。CUDA Runtime API函数的结果值返回通常采用传入引用的方式完成，且函数本身的返回值是错误码。C10库将错误码统一处理，将外层调用者希望得到的结果通过函数返回的方式传递到外层。

device_count方法最底层对接CUDA提供的cudaGetDeviceCount接口，并进行一些处理。正常情况下，该函数应该返回一个不太大（int8_t）的数字，开发者认为，不可能有安装超过255个CUDA设备的主机。但是，开发者们对这部分代码并不完全确信。因此，它们将取设备个数的实现用try语句块包裹，检查得到的结果，并在结果超过255时，抛出一个异常，提醒外部开发者将这个Issue提交给PyTorch。

device_count方法获取设备数量时，首先调用cudaGetDeviceCount接口。成功则直接返回结果；遇到错误则需要做错误处理。如果错误由主机未安装CUDA设备产生，直接返回0，表示没有CUDA设备；如果是CUDA驱动版本太低，则提醒用户更新驱动程序；如果是驱动启动失败，则提醒用户驱动启动错误；如果是其他错误，则将unknown error通过报错信息告知用户。

## 3. CUDAStream

该模块是对CUDA执行流的一层封装，借助池技术（pool technology）避免流的反复创建销毁带来的开销。为每个设备创建三个CUDA流池（stream pool），分别负责存储默认优先级流、低优先级流和高优先级流。这些流池都采用Lazy法进行初始化。

getStreamFromPool方法用于从流池获取一个暂时未被使用的流，该方法允许调用者声明希望得到的流是高优先级的还是低优先级的。对于每种优先级，为每次创建的请求按照自增方式创建一个ID。通过对ID取模，即可实现流的循环使用。

C10库的开发者希望调用getStreamFromPool方法的开发者认为自己创建了一个新的流；然而在底层，这个流是预先统一创建好的，被用来循环使用。
