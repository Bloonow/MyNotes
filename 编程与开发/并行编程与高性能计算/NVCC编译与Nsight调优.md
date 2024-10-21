# NVCC编译阶段

GPU应用程序的源文件由标准C++主机代码和GPU设备函数代码混合组成。NVCC（NVIDIA CUDA Compiler）编译套件的CUDA编译轨迹（compilation trajectory）指示如何编译GPU程序、链接运行时库、生成可执行文件，涉及对每个CUDA源文件的拆分（split）、预处理（preprocess）、编译（compilation）、合并（merge）。在编译阶段，需要将主机代码与设备函数代码分离，然后使用C++主机编译器编译主机代码，使用NVCC编译器和汇编器编译设备函数代码，然后将编译后的GPU函数作为富二进制映像（fatbinary image）嵌入到主机的目标文件（object file）中。在链接阶段，会添加特定的CUDA运行时库以支持SPMD例程调用，并提供显式GPU操作，例如GPU内存缓冲区的分配和主机GPU数据传输。

NVCC可接受一系列传统C++编译器选项，例如宏定义、头文件路径、库路径等，以及指示编译过程的选项。所有非CUDA编译步骤都会传递给C++主机编译器，并将NVCC的选项转换为适当的主机编译器的命令行选项。在所有平台上，会使用当前的执行搜索路径来寻找默认的主机编译器可执行文件（例如Linux平台上的gcc和g++，Windows平台上的cl.exe等），也可以使用NVCC的选项手动指定。

NVCC编译套件预定义了一些宏，例如\_\_NVCC\_\_在编译C/C++/CUDA源文件时定义，\_\_CUDACC\_\_在编译CUDA源文件时定义，\_\_CUDACC\_RDC\_\_在可重定位设备代码模式下编译CUDA源文件时定义。

编译阶段（compilation phase）是一个逻辑上的翻译步骤（translation step），来完成特定的编译功能，可以通过NVCC的命令行选项来选择。单个编译阶段仍然可以细分为更小的步骤，但这些更小的步骤只是该阶段的实现，它们依赖于NVCC所使用内部工具的功能，并且所有这些内部功能都可能会随着新版本的发布而改变。因此，只有编译阶段在各个版本中都是稳定的。尽管NVCC提供显式执行编译步骤的选项，但这些仅用于调试目的，不应在构建脚本中使用。

下表给出了NVCC可识别的文件后缀。

| 文件后缀            | 描述                                         |
| ------------------- | -------------------------------------------- |
| .cu                 | CUDA源文件，包含主机代码和设备代码           |
| .c, .cc, .cxx, .cpp | C/C++源文件                                  |
| .i, .ii             | C/C++源文件预处理后的文件                    |
| .ptx                | PTX中间汇编文件                              |
| .cubin              | 单GPU架构的CUDA设备代码的二进制文件          |
| .fatbin             | CUDA富二进制文件，可能包含多个PTX和CUBIN文件 |
| .o, .obj            | 目标文件                                     |
| .a, .lib            | 库文件                                       |
| .res                | 资源文件                                     |
| .so                 | 共享动态库文件                               |

值得注意的是，NVCC对待目标文件、库文件、资源文件之间没有任何区别，它只是在执行链接阶段时将这些类型的库文件传递给链接器。

下表给出了NVCC所支持的用于控制编译阶段的命令行选项。

| 编译阶段                                                     | 命令行选项             | 输入                    | 输出                   |
| ------------------------------------------------------------ | ---------------------- | ----------------------- | ---------------------- |
| 预处理CUDA源文件                                             | --cuda, -cuda          | .cu                     | .cu.cpp.ii             |
| 预处理C++源文件                                              | --preprocess, E        | .cu, .cpp               | 终端标准输出           |
| 生成PTX中间汇编，该步骤会抛弃.cu输入文件的主机代码           | --ptx                  | .cu                     | .ptx                   |
| 生成CUBIN二进制文件，该步骤会抛弃.cu输入文件的主机代码       | --cubin, -cubin        | .cu, .gpu, .ptx         | .cubin                 |
| 生成FATBIN富二进制文件                                       | --fatbin, -fatbin      | .cu, .gpu, .ptx, .cubin | .fatbin                |
| 生成目标文件                                                 | --compile, -c          | .cu, .cpp               | .o, .obj               |
| 生成含可重定位设备代码的目标文件；等价于--relocatable-device-code=true选项和--compile选项 | --device-c, -dc        | .cu, .cpp               | .o, .obj               |
| 生成含可执行设备代码的目标文件；等价于--relocatable-device-code=false选项和--compile选项 | --device-w, -dw        | .cu, .cpp               | .o, .obj               |
| 链接CUBIN等二进制文件和含可重定位设备代码的目标文件，生成含可执行设备代码的目标文件，可传给C++主机链接器 | --device-link, -dlink  | .ptx, .cubin, .fatbin   | a_dlink.o, a_dlink.obj |
| 链接含可重定位设备代码的目标文件，生成CUBIN二进制文件        | --device-link --cubin  | .o, .obj                | a_dlink.cubin          |
| 链接含可重定位设备代码的目标文件，生成FATBIN富二进制文件     | --device-link --fatbin | .o, .obj                | a_dlink.fatbin         |
| 构建库文件                                                   | --lib, -lib            | .cu                     | .a, .lib               |
| 生成可执行文件并执行，用于开发调试过程，由NVCC自动设置CUDA环境而无需手动指定 | --run, -run            | .cu                     | a.exe, a.out           |
| 生成可执行文件                                               |                        | .cu                     | a.exe, a.out           |

值得注意的是，除非手动指定编译阶段，否者NVCC将会编译并链接所有输入文件。

CUDA编译的工作原理如下所述。先对输入源文件进行预处理，对设备代码进行编译，生成PTX中间汇编代码或CUBIN二进制代码，并放置在FATBIN富二进制代码中。再次对输入程序进行预处理，对主机代码进行编译，并嵌入FATBIN富二进制代码，以将CUDA C++扩展转换为标准C++结构。然后，C++主机编译器将带有FATBIN嵌入的集成主机代码（synthesized host code）编译成一个主机目标文件。过程如下图所示。当主机程序执行设备函数代码（启动CUDA Kernel）时，CUDA运行时会检查所嵌入的FATBIN以获取当前GPU合适的FATBIN映像。

<img src="NVCC编译与Nsight调优.assets/cuda compilation trajectory.png" style="zoom: 67%;" />

此处列举一个分开编译的示例，如下所示。

```c++
// kernel.cuh
#pragma once
typedef unsigned int uint32_t;
extern __global__ void scale_kernel(float *data, const float factor, const uint32_t N);
```

```c++
// kernel.cu
#include "kernel.cuh"
__global__ void scale_kernel(float *data, const float factor, const uint32_t N) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) data[tid] = data[tid] * factor;
}
```

```c++
// main.cu
#include <stdio.h>
#include "kernel.cuh"
int main(int argc, char *argv[]) {
    float *data;
    cudaMalloc(&data, sizeof(float) * 4096);
    scale_kernel<<<(4096 + 127) / 128, 128>>>(data, 3.14f, 4096);
    cudaFree(data);
    printf("Over!\n");
    return 0;
}
```

```shell
nvcc -dc kernel.cu -o kernel.o
nvcc -dc program.cu -o program.o
nvcc -dlink kernel.o program.o -o gpu.link.o
g++ kernel.o program.o gpu.link.o -L /path/cuda/libs -l cudart -o run
```

上述编译命令等价于下述一行命令。

```shell
nvcc kernel.cu program.cu -o run 
```

# CUDA-GDB调试

CUDA-GDB调试器用于在Linux系统上调试CUDA应用程序，它是GDB调试器的扩展，通常会随CUDA Toolkit软件包一起安装。CUDA-GDB提供一个无缝的调试环境，允许在同一应用程序中同时调试CPU代码和GPU代码，允许设置断点、单步执行CUDA应用程序，还可以检查和修改硬件上运行的任何给定线程的内存和变量。

在使用NVCC编译源代码以生成可执行程序时，使用-O0选项禁用编译器优化，使用-g选项生成主机代码的调试信息，使用-G选项生成设备代码的调试信息，使用-keep选项保持编译过程的中间文件。此外，CUDA-GDB还支持调试针对特定CUDA架构编译的核函数，例如sm_75或sm_80，而且还支持调试运行时编译的内核，称为JIT（just-in-time）即时编译。默认情况下，NVCC会针对compute_52和sm_52架构生成目标对象，可使用-gencode选项指定所需架构。

使用cuda-gdb命令进行调试，可以使用可执行文件、核心转储、进程编号作为拟调试对象。使用无参数的cuda-gdb命令可进入CUDA-GDB调试环境，键入help可查看命令手册，键入help后跟命令类别查看该类命令，键入quit或q退出调试环境。

```shell
gdb [option] executable_file
```

使用--args选项，可以在可执行文件之后为程序指定命令行参数；使用--core=COREFILE选项分析核心转储文件；使用--exec=EXECFILE选项指定可执行文件；使用--pid=PID选项指定要附加到的进程；使用--directory=DIR指定源文件搜索目录。需要注意的是，cuda-gdb不支持--tui选项开启文本用户界面。

CUDA-GDB与GDB的使用方法类似，而且对于主机代码，使用GDB原有命令即可进行调试，对于设备代码，CUDA-GDB额外提供以cuda开头的命令扩展。例如，使用info cuda threads命令可以查看CUDA线程，使用cuda thread #命令可以切换CUDA线程。在CUDA-GDB调试模式下，一些CUDA专有的常用命令如下所示，更详细命令可使用help命令查看，或键入apropos命令以查询与给定字符串相匹配的命令。

| 命令                           | 描述                                                         |
| ------------------------------ | ------------------------------------------------------------ |
| info cuda devices, cuda device | 显示可见的GPU设备；切换到给定设备                            |
| info cuda sms, cuda sm         | 显示使用的SM流多处理器；切换到给定SM流多处理器               |
| info cuda kernels, cuda kernel | 显示当前正在执行的Kernel核函数；切换到给定Kernel核函数执行   |
| info cuda blocks, cuda block   | 显示线程块；切换到给定线程块，可使用一维索引n指定，也可使用三维索引[x,y,z]指定 |
| info cuda threads, cuda thread | 显示线程；切换到给定线程，可使用一维索引n指定，也可使用三维索引[x,y,z]指定 |
| info cuda warps, cuda warp     | 显示线程束；切换到给定线程束                                 |
| info cuda lanes, cuda lane     | 显示Warp中的线程；切换到Warp中给定的线程                     |
| info cuda contexts             | 显示当前的GPU上下文                                          |
| cuda grid                      | 切换到给定线程网格                                           |

在调试CUDA程序的过程中，如果程序挂起或陷入无限循环，可使用CTRL+C手动中断应用程序，此时GPU会暂停且CUDA-GDB会出现提示，用户可自行决定检查、修改、单步执行、恢复或终止程序。此功能仅限于在CUDA-GDB调试器中运行的应用程序。

需要注意的是，CUDA Kernel核函数的调试是以一个Warp线程束为单位的，即每次单步执行都是一个Warp中的32个线程单步执行。一种特殊情况是\_\_syncthreads()栅障，CUDA-GDB会在栅障之后设置隐式的临时断点，并恢复所有线程，直到命中临时断点。当设置断点时，它会强制所有驻留的GPU线程在到达相应的PC时停止在该位置。

调试设备函数时，只要设备函数不是内联的，用户就可以单步进入、执行、越过、退出，使用\_\_noinline\_\_说明符可以强制编译器不内联所修饰的函数。

# NVIDIA Nsight Compute

Nsight是NVIDIA面向开发者提供的开发工具套件，能提供深入的跟踪、调试、评测和分析，以优化跨NVIDIA GPU和CPU的复杂计算应用程序。Nsight主要包含Nsight Graphics、Nsight System、Nsight Compute三部分。在连接服务器时，可能会出现https://developer.nvidia.com/nvidia-development-tools-solutions-ERR_NVGPUCTRPERM-permission-issue-performance-counters中所提到的权限错误。

Nsight Graphics是一个用于调试、评测和分析Microsoft Windows和Linux上的图形应用程序的工具。它允许优化基于Direct3D 11，Direct3D 12，DirectX，Raytracing 1.1，OpenGL，Vulkan和KHR Vulkan Ray Tracing Extension的应程序的性能。

Nsight System给开发者一个系统级别的应用程序性能的可视化分析。所有与NVIDIA GPU相关的程序开发都可以从Nsight System开始以确定最大的优化机会。开发人员可以优化瓶颈，以便在任意数量或大小的CPU和GPU之间实现高效扩展。

Nsight Compute是一个CUDA应用程序的交互式Kernel分析器。它通过用户接口和命令行工具的形式提供了详细的性能分析度量和API调试。Nsight Compute还提供了定制化的和数据驱动的用户接口和度量集合，可以使用分析脚本对这些界面和度量集合进行扩展，以获得后处理的结果。Nsight Compute CLI提供了一个命令行分析器，其命令的可执行文件是ncu，它可以直接在命令行上打印结果或将其存储在报告文件中。

## 分析方式与指标集

用户常规启动的CUDA应用程序进程，会基于CUDA运行时库以及CUDA驱动程序执行。

！！！

使用 NVIDIA Nsight Compute 对应用程序进行性能分析时，行为会有所不同。用户在主机系统上启动 NVIDIA Nsight Compute 前端（UI 或 CLI），然后主机系统将实际应用程序作为目标系统上的新进程启动。虽然主机和目标通常是同一台机器，但目标也可能是具有可能不同操作系统的远程系统。

该工具将其测量库插入到应用程序进程中，从而使分析器能够拦截与 CUDA 用户模式驱动程序的通信。此外，当检测到内核启动时，库可以从 GPU 收集请求的性能指标。然后将结果传回前端。
