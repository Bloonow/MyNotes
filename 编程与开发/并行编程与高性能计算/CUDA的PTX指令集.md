# 基本语法

一般情况下，使用高级语言编写的程序，可以通过编译器直接转换成机器指令以在硬件上运行。但NVIDIA历代GPU产品的底层架构不尽相同，运行的机器指令也存在一定的差异，为保证软件层面的兼容性，CUDA定义了一套稳定的指令集架构和编程模型，称为PTX（Parallel Thread Execution），这是一种并行线程执行的底层虚拟机（low-level virtual machine）及其指令集（instruction set architecture）。PTX指令集是一种中间表示，它展示了GPU支持的功能，但是不同架构的GPU对这些功能的实现方式存在差异。这种虚拟让PTX代码与GPU底层硬件架构基本无关，能够跨越多种硬件，从而更具有通用性。

在需要时，PTX代码可以即时编译（Just-In-Time）得到更底层的二进制机器指令代码，这是由编译器根据GPU硬件的实际计算能力与具体架构所生成的二进制cubin目标代码，可以由机器直接执行的二进制目标代码对应的汇编称为SASS（Streaming Assembly）流汇编代码，又称为低级汇编指令（Low-Level Assembly Instruction），是特定于GPU架构的。机器指令SASS相比于PTX指令更接近GPU的底层架构，SASS指令集可以体现出特定架构的GPU对于PTX功能的实现方式。

PTX代码和SASS代码中的标识符都是区分大小写的，但预留关键字并不区分大小写，但通常PTX代码使用小写，SASS代码使用大写。

PTX程序的源文件通常是以.ptx作为后缀的文本文件，其语法具有汇编语言样式的风格。PTX语句可以是指令（instruction），也可以是指示（directive），语句以可选标签（label）开头，以分号`;`结尾，以`\n`表示换行。PTX源文件使用C风格的注释，例如`//`和`/**/`注释。使用C风格的以`#`开头的预处理指令，例如#include、#define、#if、#endif等预处理指令。源文件中所有空白字符都是等效的，并且空格将被忽略，除非它在语言中用于分隔标记。

每个PTX文件必须在开头使用.version指示PTX版本，后跟.target指示所假设的目标架构，一个示例代码如下。

```
.version 8.5
.target sm_70, sm_75, sm_86, sm_89

.reg.b32 r1, r2;
.global.f32 array[N];

_start:
mov.b32 r1, %tid.x;           // move threadIdx.x to r1
shl.b32 r1, r1, 2;            // shift left tidx by 2 bits
ld.global.b32 r2, array[r1];  // thread[tidx] gets array[tidx << 2]
add.f32 r2, r2, 0.5;          // add 0.5
```

指令语句（Instruction Statement）用于表示某个具体的功能，与机器指令一一对应，所有指令语句的集合即构成一个指令集。指令由指令操作码（instruction opcode）组成，后跟以逗号分隔的操作数（operand）列表，并以分号结尾。SASS的习惯是第一个作为目的操作数，然后是源操作数；操作数可以是寄存器变量、常量表达式、地址表达式、标签（label）名称。指令具有可选的表示预测谓词（predication）的哨兵（guard），用于控制指令是否按照条件谓词的结果执行；谓词位于可选的label之后，指令的opcode之前，通常写作`@p`或者`@!p`，其中p是声明的谓词寄存器，叹号`!`表示对谓词进行取反。

指示语句（Directive Statement）用于指示一个特定属性或特定行为，对指令起到修饰和限定的作用，或用于指示编译器的行为，指示语句以点号`.`开头。

<table>
    <caption>Directive Statement</caption>
    <tr>
        <td>.address_size</td> <td>.alias</td> <td>.align</td> <td>.branchtargets</td> <td>.callprototype</td>
        <td>.calltargets</td> <td>.common</td> <td>.const</td> <td>.entry</td>
    </tr>
    <tr>
        <td>.explicitcluster</td> <td>.extern</td> <td>.file</td> <td>.func</td> <td>.global</td> 
        <td>.loc</td> <td>.local</td> <td>.maxclusterrank</td> <td>.maxnctapersm</td>
    </tr>
    <tr>
        <td>.maxnreg</td> <td>.maxntid</td> <td>.minnctapersm</td> <td>.noreturn</td> <td>.param</td>
        <td>.pragma</td> <td>.reg</td> <td>.reqnctapercluster</td> <td>.reqntid</td>
    </tr>
    <tr>
        <td>.section</td> <td>.shared</td> <td>.sreg</td> <td>.target</td> <td>.tex</td> 
        <td>.version</td> <td>.visable</td> <td>.weak</td>
    </tr>
</table>
用户定义的标识符（Identifier）遵循扩展的C++规则，它们以字母开头，后跟零个或多个字母、数字、下划线或美元字符，或者，它们以下划线、美元字符或百分号字符开头，后跟一个或多个字母、数字、下划线或美元字符。

```
followsym:   [a-zA-Z0-9_$]
identifier:  [a-zA-Z]{followsym}* | [_$%]{followsym}+
```

许多高级语言的标识符名称遵循类似的规则，只是不允许使用百分号，而PTX允许百分号作为标识符的第一个字符，这可以用于避免名称冲突，例如，用户定义的变量名称和编译器生成的名称之间的冲突。不过，PTX预定义了一个WARP_SZ常量和少量以百分号开头的特殊寄存器，如下所述。

<table>
    <caption>Predefined Identifier</caption>
    <tr>
        <td>%clock</td> <td>%clock64</td> <td>%ctaid</td> <td>%envreg<32></td> <td>%gridid</td> <td>%laneid</td>
        <td>%lanemask_eq</td> <td>%lanemask_le</td> <td>%lanemask_lt</td> <td>%lanemask_ge</td>
    </tr>
    <tr>
        <td>%lanemask_gt</td> <td>%nctaid</td> <td>%ntid</td> <td>%nsmid</td> <td>%nwarpid</td> <td>%pm0, ..., %pm7</td>
        <td>%smid</td> <td>%tid</td> <td>%warpid</td> <td>WARP_SZ</td>
    </tr>
</table>

PTX支持整数和浮点常量以及常量表达式，这些常量可用于数据初始化，并用作指令的操作数。在PTX中，整数常量可用作谓词，零值为False，非零值为True。

假设存在一个名称为mycode.ptx的PTX模块文件，可以通过cuModuleLoad()方法和cuModuleGetFunction()方法在CUDA C++代码中使用PTX提供的内核函数。需要注意的是，此处使用的以cu前缀开头的API函数，是由CUDA的驱动API提供的，而不是由诸如cudaXXX()的运行时提供的接口，因此在编译时要使用-lcuda链接到相应的库文件。

```c++
#include <cuda.h>

int main(int argc, char *argv[]) {
    CUdevice device;
    CUcontext context;
    cuInit(0);
    cuDeviceGet(&device, 0);
    cuCtxCreate(&context, 0, device);

    CUmodule ptx_module;
    CUfunction ptx_kernel_function;
    cuModuleLoad(&ptx_module, "/path/to/mycode.ptx");
    cuModuleGetFunction(&ptx_kernel_function, ptx_module, "my_kernel");

    float* d_input;  cudaMalloc(&d_input, 128 * sizeof(float));
    float* d_output; cudaMalloc(&d_output, 128 * sizeof(float));
    // args是指向实际参数的地址的指针的数组
    void* args[2] = { &d_input, &d_output };
    cuLaunchKernel(ptx_kernel_function, 1, 1, 1, 128, 1, 1, 0, nullptr, args, nullptr);

    cuModuleUnload(ptx_module);
    cuCtxDestroy(context);
    return 0;
}
```

```shell
nvcc -arch=sm_89 -lcuda demo.cu -o run
```

# 特殊寄存器

PTX提供许多预定义的只读变量，这些变量以特殊寄存器的形式可见，其名称统一以百分号%作为前缀，可以通过mov指令或cvt指令访问。

一些用于表示各种或概念上或硬件上的标识符寄存器，用于在代码执行时判断当前资源的编号，如下所示。

| 名称                 | 预定义类型                       | 描述                        | 备注                                                         |
| -------------------- | -------------------------------- | --------------------------- | ------------------------------------------------------------ |
| %tid                 | .sreg.v4.u32 %tid;               | CTA中各个维度上的线程标识符 | 有效值为三维的%tid.x、%tid.y、%tid.z                         |
| %ntid                | .sreg.v4.u32 %ntid;              | CTA中各个维度上的线程数目   | 有效值为三维的%ntid.x、%ntid.y、%ntid.z                      |
| %laneid              | .sreg.u32 %laneid;               | Warp中的线程标识符          | 有效值为0到WARP_SZ－1                                        |
| %warpid              | .sreg.u32 %warpid;               | CTA中的Warp标识符           | 该值表示线程的位置，但由于抢占式的线程重调度，该值在执行中可能会更改 |
| %nwarpid             | .sreg.u32 %nwarpid;              | CTA中的Warp数目             |                                                              |
| %ctaid               | .sreg.v4.u32 %ctaid;             | 网格中各个维度上的CTA标识符 | 有效值为三维的%ctaid.x、%ctaid.y、%ctaid.z                   |
| %nctaid              | .sreg.v4.u32 %nctaid;            | 网格中各个维度的CTA数目     | 有效值为三维的%nctaid.x、%nctaid.y、%nctaid.z                |
| %smid                | .sreg.u32 %smid;                 | 正在执行线程的SM标识符      | 该值表示线程的位置，但由于抢占式的线程重调度，该值在执行中可能会更改；<br>不保证SM标识符是连续的，因为SM是否可用于调度执行受很多因素影响 |
| %nsmid               | .sreg.u32 %nsmid;                | 正在执行线程的SM数目        | 由于不保证SM标识符是连续的，因此%nsmid可能大于物理硬件中的SM数目 |
| %gridid              | .sreg.u64 %gridid;               | 上下文中的网格标识符        | 执行期间会启动多次核函数，该值提供上下文中的临时网格编号     |
| %is_explicit_cluster | .sreg.pred %is_explicit_cluster; | 用户是否指定以簇组启动      |                                                              |
| %clusterid           | .sreg.v4.u32 %clusterid;         | 网格中各个维度的簇组标识符  | 有效值为三维的%clusterid.x、%clusterid.y、%clusterid.z       |
| %nclusterid          | .sreg.v4.u32 %nclusterid;        | 网格中各个维度上的簇组数目  | 有效值为三维的%nclusterid.x、%nclusterid.y、%nclusterid.z    |
| %cluster_ctaid       | .sreg.v4.u32 %cluster_ctaid;     | 簇组中各个维度的CTA标识符   | 有效值为三维的%cluster_ctaid.x、%cluster_ctaid.y、%cluster_ctaid.z |
| %cluster_nctaid      | .sreg.v4.u32 %cluster_nctaid;    | 簇组中各个维度上的CTA数目   | 有效值为三维的%cluster_nctaid.x、%cluster_nctaid.y、%cluster_nctaid.z |
| %cluster_ctarank     | .sreg.u32 %cluster_ctarank;      | 簇组中平铺的CTA标识符       | 有效值为0到%cluster_nctarank－1                              |
| %cluster_nctarank    | .sreg.u32 %cluster_nctarank;     | 簇组中平铺的CTA数目         |                                                              |

一些用于表示共享内存使用信息的寄存器，如下所示。其中，预留共享内存是给NVIDIA系统软件使用的。

| 名称                        | 预定义类型                             | 描述                                                         |
| --------------------------- | -------------------------------------- | ------------------------------------------------------------ |
| %reserved_smem_offset_begin | .sreg.b32 %reserved_smem_offset_begin; | 预留共享内存区域开始处的偏移                                 |
| %reserved_smem_offset_end   | .sreg.b32 %reserved_smem_offset_end;   | 预留共享内存区域结束处的偏移                                 |
| %reserved_smem_offset_cap   | .sreg.b32 %reserved_smem_offset_cap;   | 预留共享内存区域的总容量大小                                 |
| %reserved_smem_offset_<2>   | .sreg.b32 reserved_smem_offset_<2>;    | 预留共享内存区域中的偏移                                     |
| %total_smem_size            | .sreg.u32 %total_smem_size;            | 内核的CTA使用的共享内存总大小，包括静态分配和动态分配的，不包括系统预留的 |
| %aggr_smem_size             | .sreg.u32 %aggr_smem_size;             | 内核的CTA使用的共享内存总大小，包括静态分配和动态分配的，以及系统预留的 |
| %dynamic_smem_size          | .sreg.u32 %dynamic_smem_size;          | 内核启动时一个CTA动态分配的共享内存大小                      |

一些预定义的线程掩码寄存器，使用32位无符号整数表示一个Warp中的32个线程，从高位到低位以[31:30:29:28:xxx:3:2:1:0]的形式表示对应线程，如下所示。

| 名称         | 预定义类型              | 描述                                                         |
| ------------ | ----------------------- | ------------------------------------------------------------ |
| %lanemask_eq | .sreg.u32 %lanemask_eq; | 仅当前线程的位为1，其它线程的位为0；该寄存器对不同线程具有不同的值 |
| %lanemask_le | .sreg.u32 %lanemask_le; | 仅当前线程和编号更小线程的位为1，其它线程的位为0             |
| %lanemask_lt | .sreg.u32 %lanemask_lt; | 仅编号更小线程的位为1，其它线程的位为0                       |
| %lanemask_ge | .sreg.u32 %lanemask_ge; | 仅当前线程和编号更大线程的位为1，其它线程的位为0             |
| %lanemask_gt | .sreg.u32 %lanemask_gt; | 仅编号更大线程的位为1，其它线程的位为0                       |

一些与时间计时和性能采样等相关的寄存器，如下所示。

| 名称                               | 预定义类型                      | 描述                                                   |
| ---------------------------------- | ------------------------------- | ------------------------------------------------------ |
| %clock                             | .sreg.u32 %clock;               | 时钟计数器，32位无符号整数                             |
| %clock64                           | .sreg.u64 %clock64;             | 时钟计数器，64位无符号整数                             |
| %clock_hi                          | .sreg.u32 %clock_hi;            | 时钟计数器%clock64的高32位数据                         |
| %globaltimer                       | .sreg.u64 %globaltimer;         | 全局的纳秒计数器，64位无符号整数                       |
| %globaltimer_lo                    | .sreg.u32 %globaltimer_lo;      | 全局的纳秒计数器的低32位数据                           |
| %globaltimer_hi                    | .sreg.u32 %globaltimer_hi;      | 全局的纳秒计数器的高32位数据                           |
| %pm0, %pm1, ..., %pm7              | .sreg.u32 %pm<8>;               | 性能监视计数器，目前行为未定义                         |
| %pm0_64, %pm1_64, ..., %pm7_64     | .sreg.u64 %pm<8>_64;            | 性能监视计数器，目前行为未定义                         |
| %envreg0, %envreg1, ..., %envreg31 | .sreg.b32 %envreg<32>;          | 用于捕获PTX虚拟机外部的PTX程序执行环境，由驱动程序定义 |
| %current_graph_exec;               | .sreg .u64 %current_graph_exec; | 如果执行的内核不是CUDA设备图的一部分，则此寄存器值为0  |

# 类型系统

不同架构的GPU中可用的硬件资源数目不同，但资源类型在各个架构上是通用的，这些资源在PTX中通过存储状态空间和数据类型进行抽象化，以一种概念表示。

## 存储状态空间

存储状态空间（State Space）指的是具有某种特征的存储区域，所有变量都驻留在某个存储状态空间中。存储状态空间的特征包括容量大小、可寻址性、访问速度、访问权限、线程之间的共享级别。在PTX中定义的存储状态空间如下所示。

| 名称    | 描述               | 是否可寻址 | 是否可初始化 | 可访问性 | 共享级别           |
| ------- | ------------------ | ---------- | ------------ | -------- | ------------------ |
| .reg    | 寄存器             | 否         | 否           | 可读可写 | 一个线程私有       |
| .sreg   | 特殊寄存器         | 否         | 否           | 只读     | 一个线程块之内共享 |
| .const  | 常量内存           | 是         | 是           | 只读     | 一个网格之内共享   |
| .global | 全局内存           | 是         | 是           | 可读可写 | 一个上下文之内共享 |
| .local  | 局部内存           | 是         | 否           | 可读可写 | 一个线程私有       |
| .param  | 内核函数的参数     | 是         | 否           | 只读     | 一个网格之内共享   |
| .param  | 设备函数的参数     | 受限的     | 否           | 可读可写 | 一个线程私有       |
| .shared | 共享内存           | 是         | 否           | 可读可写 | 一个簇组之内共享   |
| .tex    | 纹理内存（已弃用） | 否         | 是           | 只读     | 一个上下文之内共享 |

对于一个数据变量而言，它一定位于某个存储状态空间中，使用mov指令获得变量的地址时，该地址是该变量在该存储状态空间中的地址，即该变量相对于该存储状态空间起始位置的偏移量（字节）。PTX提供了一个运算符generic()来将某个存储状态空间中的地址转换为一个通用地址，或者在运行时使用诸如cvta.global的指令将某个存储状态空间中的地址转换为通用地址。当然，如果使用带有存储状态空间修饰符的指令，则可以直接使用相应的存储状态空间中的地址（而无需转换为通用地址）。

如果一个使用地址的指令未指明存储状态空间，则使用通用地址进行寻址操作。存储状态空间.const、.local、.shared、.param实际上都是通用地址空间中的一个窗口，每个窗口都由一个窗口基址和窗口大小的定义。除了.const、.local、.shared窗口之外，一个通用地址映射到设备的全局内存之上，参数.param窗口包含在.global窗口之中。对于一个地址，减去其所在窗口的基址，即可得到数据在相应存储状态空间中的地址。

寄存器.reg是快速存储状态空间，且数量是有限的，当超过限制时，寄存器变量将溢出到内存中，从而导致性能下降。寄存器可以是类型化的（有符号整数、无符号整数、浮点数、谓词），也可以是无类型的。寄存器变量的大小具有限制，谓词占用1位，标量可以占用8位、16位、32位、64位、128位，向量可以占用16位、32位、64位、128位。其中，8位寄存器常用于ld、st和cvt指令，或作为向量元组的元素。在加载或存储多字长（multi-word）数据时，寄存器可能需要边界对齐。

特殊寄存器.sreg是预定义的、特定于平台的专用寄存器空间，例如网格、簇组、线程块、线程等专用标识寄存器，时钟计数、性能监控寄存器等。寄存器空间与其它存储状态空间的不同之处在于，寄存器是不可寻址的，即不可以引用寄存器的地址。

常量.const是由主机初始化的只读的存储状态空间，大小限制为64KB，用于保存固定内存大小的常量数据，可通过ld.const指令访问。固定内存大小的常量变量可以指定初始值，若不指定则默认初始化为零。此外，还有一个640KB的常量内存空间，组织为10个独立的64KB区域，驱动程序可以在这些区域中分配和初始化常量缓冲区，并将指向缓冲区的指针作为内核函数的参数传递。由于这10个区域不是连续的，因此驱动程序必须确保缓冲区不会越界。

全局.global存储状态空间是上下文中所有线程都可以访问的内存，不同网格、簇组、线程块之中的线程可以使用全局内存进行通信，可通过ld.global指令、st.global指令和atom.global指令访问全局内存变量。全局内存中的变量可以指定初始值，若不指定则默认初始化为零。

局部.local存储状态空间是每个线程的私有内存，用于保留自己的数据，它通常是带缓存的标准内存。局部私有内存的大小是有限的，因为它必须基于每个线程进行分配。可通过ld.local指令和st.local指令访问局部内存变量。

参数.param存储状态空间用于：(1)将输入参数从主机传递给内核函数；(2)声明设备函数的输入参数和返回值参数；(3)声明局部范围的字节数组变量，通常用于按值传递（而非引用）将大型结构体参数给函数。需注意，内核函数的参数是在一个网格之内共享的，设备函数的参数是一个线程私有的。对于指令而言，为区分不同类型的参数，使用xxx.param::entry形式访问内核函数参数，使用xxx.param::func形式访问设备函数参数，若省略::entry或::func后缀，则会根据指令推断。

> 参数空间的位置是特定于实现的。例如，某些实现中，内核函数参数是驻留在全局内存中的，在这种情况下，参数空间和全局内存空间不提供访问保护；尽管内核参数空间的确切位置是特定于实现的，但内核参数空间窗口（kernel parameter space window）是始终包含在全局空间窗口中的。同样地，函数参数会根据ABI的函数调用约定将所传递的参数映射到寄存器或堆栈位置。因此，PTX代码不应该对.param空间变量的相对位置或顺序做出任何假设。

每个内核函数定义都包含一个可选的参数列表，这些参数是在.param存储状态空间中声明的可寻址的只读变量。可通过mov指令将内核参数的地址移动到寄存器中，生成的地址属于.param存储状态空间，可使用ld.param{::entry}指令访问这些参数变量。一个示例如下所示。

```
.entry foo (.param.b32 len, .param.b8.align 8 buffer[64]) {
    .reg.u32 %addr;
    .reg.u32 %len1;
    .reg.u32 %len2;
    .reg.f64 %data;
    
    mov.u32      %addr, len;
    ld.param.u32 %len1, [%addr];    // 使用mov和ld.param加载参数
    ld.param.u32 %len2, [len];      // 直接使用ld.param
    ld.param.f64 %data, [buffer];
}
```

内核函数参数可以表示正常的数据值，也可以保存常量内存、全局内存、局部内存、共享内存中对象的地址指针。对于地址指针，编译器和运行时系统需要一些信息，来判断哪些参数是地址指针，以及这些指针指向哪个存储状态空间。内核参数的属性指示语句用于在PTX级别提供这些信息。内核函数参数可以使用可选的.ptr属性进行声明，以指示该参数是指向内存的指针，还可以指示所指向的存储状态空间和对齐方式，如下所示。

```
.param .type .ptr .space .align N  varname
.param .type .ptr        .align N  varname
```

当使用.ptr指示参数为地址指针时，可以使用.space指定存储状态空间，可以是.const、.global、.local、.shared存储状态空间，若未指定则假定指针是指向const、global、local、shared之一的通用地址，对齐值N应是2的整数次幂（单位为字节），若未指定则假定4字节对齐。值得注意的是，可以消除.type、.ptr、.space、.align之间的空格以提高可读性。

```
.entry foo (
    .param.u32 arg1,
    .param.u32.ptr.global.align 16 arg2,
    .param.u32.ptr.const.align 8 arg3,
    .param.u32.ptr.align 16 arg4  // generic address pointer
) { ... }
```

PTX 2.0版本将.param参数空间的使用扩展到设备函数参数。最常见用途是按值传递将大型结构体给函数（不适合使用PTX寄存器传递），这种情况下，会使用参数空间中的字节数组，被调用方声明一个.param形式的参数，该参数与传递的实参具有相同的大小和对齐方式。如下所示。

```
struct MyStruct { double fp; int val; };

// pass object of type MyStruct, 8 + 4 = 12Byte
.func foo (.reg.b32 len, .param.b8.align 8 buffer[12]) {
    .reg.f64 %fp;
    .reg.s32 %val;

    ld.param.f64 %fp, [buffer];
    ld.param.s32 %val, [buffer + 8];
}
```

共享.shared存储状态空间是正在执行的一个CTA所持有的内存空间，簇组内的所有CTA的线程都可以访问，可通过ld.shared指令和st.shared指令进行访问。使用.shared::cta指示当前正在执行的CTA的共享内存窗口，使用.shared::cluster指示同簇组中其它CTA的共享内存窗口，当然.shared::cta的地址窗口也位于.shared::cluster地址窗口中；若省略后缀则默认是.shared::cta指定的内存窗口。在.shared中声明的变量是值当前CTA中的内存地址，使用mapa指令获得簇组中另一个CTA中的相应变量的.shared::cluster地址。

## 基本数据类型

在PTX代码中，基本数据类型（Fundamental Data Type）反映了目标架构所支持的本机数据类型，其中寄存器变量始终是基本类型。类型大小（type-size）修饰符同时用于变量定义和指令说明中，如下所示。

| 基本类型         | 类型说明符                   |
| ---------------- | ---------------------------- |
| unsigned integer | .u8、.u16、.u32、.u64        |
| signed integer   | .s8、.s16、.s32、.s64        |
| floating point   | .f16、.f32、.f64             |
| bits（untyped）  | .b8、.b16、.b32、.b64、.b128 |
| predicate        | .pred                        |

大多数指令都有一个或多个类型说明符，以指定指令的行为，指令会根据类型检查操作数类型和大小以确保兼容性。只要类型和大小完全相同就是兼容的，此外，无符号整数和有符号整数只要具有相同的大小，也是兼容的，bit位类型与具有相同大小的任何基本类型都是兼容的。原则上，所有变量（谓词除外）都可以仅使用bit位类型进行声明，但类型化的变量增强了程序的可读性，并允许更好的操作数类型检查。

类型.u8、.s8、.b8仅用于ld、st、cvt指令。为方便起见，ld、st、cvt指令允许操作数的类型大小比指令的类型大小更宽，因此可以使用常规宽度的寄存器加载、存储和转换更窄的值。例如，8位或16位值在加载、存储或转换为其他类型和大小时，可以直接保存在32位或64位寄存器中。

PTX中支持的基本浮点类型具有隐式的位表示形式，即用于存储指数和尾数的位数，例如，类型.f16为指数保留5位，为尾数保留10位。除基本的浮点类型之外，PTX还支持一些其它格式的浮点类型。类型bf16一共16位，8位指数，7位尾数，包含bf16数据的寄存器必须声明为.b16类型；类型e4m3一共8位，4位指数，3位尾数，包含e4m3数据的寄存器必须声明为.b8类型；类型e5m2一共8位，5位指数，2位尾数，包含e5m2数据的寄存器必须声明为.b8类型；类型tf32一共32位，范围与.f32相同，精度降低（仍然大于等于10位），数据的内部布局是基于实现的，包含tf32数据的寄存器必须声明为.b32类型。

某些PTX指令在两组输入上并行运行，并产生两个输出，此类指令可以使用以打包（packed）格式存储的数据。PTX支持将相同标量数据类型的两个值打包到一个更大的值中，打包的值被视为打包数据类型的值。打包数据类型包括.f16x2、.bf16x2、.e4m3x2、.e5m2x2类型。

## 变量的声明

在PTX中，声明一个变量（Variable）时需要描述变量的存储状态空间及其数据类型，除基本类型之外，PTX还支持简单的聚合类型，例如向量和数组。数据的所有存储状态空间都在变量声明时指定，谓词变量只能在寄存器空间中声明。变量声明指定一个名称的存储状态空间、类型、大小、可选初始值、可选的数组大小、可选的固定地址。

```
.global.u32 var1;
.global.u8  var2[4] = { 0, 0, 0, 0 };
.const.f32  var3[] = { -1.0, 1.0 };
.reg.s32    var4;
.reg.v4.f32 var5;
.reg.pred   p, q, r, s;
```

PTX支持有限长度的向量类型，任何非谓词基本类型的长度为2或4的向量都可以通过在类型前加上.v2或.v4来声明，如下所示。向量必须基于基本类型，并且它们可以驻留在寄存器空间中。但需要注意的是，向量的总长度不能超过128位，例如，不允许使用.v4.f64类型。可以使用.v4处理三元素向量，其中第四个元素提供无效的填充值，这是3D网格纹理的常见情况。

```
.shared.v2.u16 var1;  // a length-2 vector of unsigned ints
.global.v4.f32 var2;  // a length-4 vector of floats
.global.v4.b8  var3;  // a length-4 vector of bytes
```

默认情况下，向量变量与其总大小的倍数（向量长度乘以基本类型大小）对齐，以启用向量化的加载和存储指令，这些指令需要地址与访问大小的倍数对齐。

PTX提供数组声明以允许程序员预留内存空间。声明数组时，变量名称后跟维度声明，类似于C语言中的固定大小的数组声明，每个维度的大小都是一个常量表达式。当使用初始值设定项声明时，可以省略数组的第一个维度，第一个数组维度的大小由数组初始值中的元素数决定。

```
.shared.u8  var1[128];
.local.u16  var2[16][16];
.global.u32 var3[] = { 0, 1, 2, 3, 4, 5, 6, 7 };
.global.s32 var4[][2] = { {-1, 0}, {0, -1}, {1, 0}, {0, 1} };
```

声明的变量可以使用类似于C的语法指定初始值，标量采用单个值，而向量和数组采用大括号内的嵌套值列表，如下所示。目前只有.const常量存储状态空间和.global全局存储状态空间支持变量初始化，外部.extern变量不允许初始化。默认情况下，没有显式初始值的变量会被初始化为零。

```
.const.f32     var1 = 3.14;
.global.v4.f32 var2 = { 3.14, 2.71, 1.5 };       // sane to { 3.14, 2.71, 1.5, 0.0 };
.global.f32    var2[8] = { 0.33, 0.25, 0.125 };  // same to { 0.33, 0.25, 0.125, 0.0, 0.0, 0.0, 0.0, 0.0 };
.global.s32    var3[3][2] = { {1, 2}, {3} };     // same to { {1, 2}, {3, 0}, {0, 0} };
```

如果在初始值列表中出现一个变量名称，则表示的是该变量的地址（即变量在某个存储状态空间中的偏移量），这可用于静态初始化指向变量的指针。可以使用设备函数的名称作为初始化值，表示函数中第一条指令的地址，这可用于初始化间接调用的函数指针表。从PTX 3.1版本开始，内核函数名称也可以用作初始化值，例如初始化内核函数指针表，与CUDA动态并行一起使用，从而可以从GPU端启动内核函数。需要注意的是，保存地址的变量应为.u8、.u32或者.u64类型。

```
.const.u32 arr[] = { 3, 5, 7, 9, 11 };
.global.u64 addr1 = foo;               // .const address of arr, namely offset in .const space
.global.u64 addr2 = generic(foo);      // generic address of arr[0], the value is 3
.global.u64 addr3 = generic(foo) + 8;  // generic address of arr[2], the value is 7
```

对使用地址作初始值的情况，也可以使用addr＋offset的形式，表示addr添加offset字节偏移之后的地址。需要注意的是，默认情况下，这种地址是某个存储状态空间中的地址，也即相对于存储状态空间起始位置的偏移量，就像使用mov指令获得的变量地址一样。PTX提供了一个运算符generic()来将某个存储状态空间中的地址转换为一个通用地址，或者在运行时使用诸如cvta.global的指令将某个存储状态空间中的地址转换为通用地址。

在PTX中声明变量时，可以指定可寻址变量的存储地址的字节对齐，对齐方式使用紧跟在存储状态空间说明符后面的可选的.align N说明符指定，对齐值N以字节为单位，且必须是2的整数次幂。所声明的变量的存储地址将是对齐值的整数倍。对于数组而言，对齐是指整个数组的起始地址的对齐方式，而不是单个元素的地址对齐方式。标量变量和数组变量的默认对齐方式是基本类型大小的倍数；向量变量的默认对齐方式是整体向量大小的倍数。

```
// Allocate array at 4-byte aligned address. Elements are bytes.
.const.b8.align 4 arr[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
```

需要注意的是，所有访存指令都要求访问地址与访问大小的倍数对齐，访存指令的访问大小是内存中访问的总字节数。例如，ld.v4.b32的访问大小为16字节。

由于PTX支持虚拟寄存器，因此编译器前端生成大量寄存器名称是很常见的，PTX支持创建一组变量的语法，该变量具有通用的前缀字符串，以及附加的整数后缀。此语法糖可用于任何基本类型和任何存储状态空间，并且可以在前面带有对齐方式说明符，但数组变量不能以这种方式声明，也不允许设定初始值。例如，假设一个程序使用大量的寄存器，示例如下所示。

```
.reg.b32 %r<100>;  // declare %r0, %r1, ..., %r99
```

## 张量的声明

张量（Tensor）是内存中的多维矩阵结构，一个张量由维度、维数、跨步、元素类型四个部分定义。PTX支持操作张量数据的指令，包括wmma.mma.xxx指令、mma.xxx指令、wgmma.mma_async.xxx指令。PTX张量指令将全局内存中的张量数据视为多维结构，而将共享内存中的数据视为线性数据。

https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensors

# 寻址方式

对于非访存的计算指令，其描述的是ALU等计算单元的工作负载，因此指令的源操作数和目标操作数都必须全部位于.reg寄存器存储状态空间中。而对于涉及访存的指令，诸如ld、st、mov、cvt指令，会将数据从一个位置复制到另一个位置，指令ld会将数据从可寻址存储状态空间中移动到寄存器中，指令st会将数据从寄存器中移动到某个存储状态空间中，指令mov可以在寄存器之间复制数据。不同存储状态空间的操作数会影响操作的速度，寄存器最快，而设备全局内存最慢。

如果一个使用地址的指令未指明存储状态空间，则使用通用地址进行寻址操作。存储状态空间.const、.local、.shared、.param实际上都是通用地址空间中的一个窗口，每个窗口都由一个窗口基址和窗口大小的定义。除了.const、.local、.shared窗口之外，一个通用地址映射到设备的全局内存之上，参数.param窗口包含在.global窗口之中。对于一个地址，减去其所在窗口的基址，即可得到数据在相应存储状态空间中的地址。

所有访存指令都采用一个地址（address）作为操作数，指定要访问的存储状态空间中的位置，可寻址的操作数可以是如下类型之一。地址算术使用整数算术和逻辑指令执行，例如指针算术和指针比较。PTX中的所有地址和地址计算都是基于字节的，不支持C样式的指针算术。例如，对于uint32_t数组array，在C语言中，array[1]表示数组中的第2个元素，其距离首地址有sizeof(int32_t)即4个字节，而在PTX中，array[1]表示数组中的第2个字节，其距离首地址有1个字节。

| 语法形式      | 描述                                                         |
| ------------- | ------------------------------------------------------------ |
| [immAddr]     | immAddr是一个绝对地址，用32位无符号整数表示，[immAddr]为对应地址位置的数据 |
| [reg]         | 寄存器reg中存储的地址，访问对应地址处的数据，用[reg]表示；寄存器所包含的地址可以声明为bit位类型或是整数类型 |
| [reg＋immOff] | 寄存器reg中存储的地址，再加上一个32位有符号整数的（字节）偏移，访问对应地址处的数据，用[reg＋immOff]表示 |
| [var]         | 名称var是变量的地址，例如mov r0, var指令会将var地址复制给r0寄存器；[var]是变量数据，即访问变量所在地址处的数据 |
| [var＋immOff] | 变量的地址var，再加上一个32位有符号整数的（字节）偏移，访问对应地址处的数据，用[var＋immOff]表示 |
| var[immOff]   | 名称var此时被当作数组名称，表示数组的首地址，根据一个32位无符号整数的（字节）偏移，访问对应地址处的数据，用var[immOff]表示 |

地址大小可以是32位或64位，不支持128位的地址。地址可以根据需要扩展到指定的宽度，如果寄存器宽度超过目标架构的存储状态空间的地址宽度，则截断地址。需要注意的是，所有访存指令都要求访问地址与访问大小的倍数对齐，访存指令的访问大小是内存中访问的总字节数。例如，ld.v4.b32的访问大小为16字节。如果地址未正确对齐，则结果行为未定义。

mov指令可用于将变量的地址移动到一个地址指针中，地址值是变量在其存储状态空间中的偏移量，加载和存储操作在可寻址存储状态空间中的位置和寄存器之间移动数据。该语法类似于许多汇编语言中的语法，其中变量只是简单地命名称，通过将地址放置在[]方括号中来取消引用。地址表达式包括地址寄存器、变量名称、字节偏移量、立即数地址，这些表达式在编译时计算为常量地址。

```
.shared.u16    var1;
.global.v4.f32 var2;
.const.s32     var3[4] = { 0, 1, 2, 3 };
.reg.u16    r0;
.reg.v4.f32 r1;
.reg.s32    r2;
.reg.b32    r3;

ld.shared.u16    r0, [var1];
ld.global.v4.f32 r1, [var2];
ld.const.s32     r2, [var3 + 8];  // value is 2
mov.u32          r3, var3;
```

mov指令也可以使用一个函数名称作为操作数，用于将函数的地址放入寄存器中，以便在间接调用中使用。不过，可以直接使用call指令调用函数，使用bra指令和brx.idx指令调转到某个label标签处执行代码。

向量操作数由有限的指令子集支持，包括mov、ld、st、atom、red、tex指令，向量也可以作为参数传递给被调用的函数。对于向量数据而言，可以使用.x、.y、.z、.w后缀访问其中的元素，或者使用.r、.g、.b、.a后缀进行访问。向量化加载和向量化存储可用于实现更宽的加载和存储，这可能会提高内存性能。使用{}花括号可以用于展开向量。如下所示。

```
.reg.f32    a, b, c, d;
.reg.v4.f32 var1;
.global.f32 var2;

ld.global.v4.f32 var1, [var2 + 16];
mov.v4.f32       {a, b, c, d}, var1;         // a, b, c, d = var1.x, var1.y, var1.z, var1.w;
ld.global.v4.f32 {a, b, c, d}, [var2 + 32];
```

指令中的所有操作数的类型都是已知的，其类型在声明时确定，每个操作数类型必须与指令指示的类型兼容。相同大小的无符号整数和有符号整数兼容，位bit类型与具有相同大小的任何类型兼容，而对于浮点数指令，操作数的类型与大小必须一致。

对于类型转换（convert）指令而言，诸如cvt指令和cvta指令，因为其工作是从几乎任何数据类型转换为任何其他数据类型，所以转换指令会采用各种类型和大小的操作数。从低位转换为高位会进行扩展，从高位转换为低位会进行截取。例如，cvt.s32.u16指令可以将一个u16数据转换为一个s32数据。

转换指令可以指定舍入修饰符，在PTX中，有5个浮点舍入修饰符.rn、.rz、.rm、.rp，以及4个整数舍入修饰符.rni、.rzi、.rmi、.rpi。

| 浮点舍入修饰符 | 描述                                                         | 整数舍入修饰符 | 描述                                               |
| -------------- | ------------------------------------------------------------ | -------------- | -------------------------------------------------- |
| .rn            | 尾数部分的最低有效位（Least Significant Bit，LSB）舍入到最近的偶数 | .rni           | 舍入到最近的整数，若被舍弃的是中间值，则向偶数舍入 |
| .rz            | 尾数部分的LSB向零值进行舍入                                  | .rzi           | 在零值方向上进行舍入到最近的整数                   |
| .rm            | 尾数部分的LSB向负无穷大进行舍入                              | .rmi           | 舍入到负无穷大方向上的最近的整数                   |
| .rp            | 尾数部分的LSB向正无穷大进行舍入                              | .rpi           | 舍入到正无穷大方向上的最近的整数                   |

# 指令系统

指令由opcode操作码和operand操作数组成，多个操作数之间以逗号分隔，习惯是第一个作为目的操作数，然后是源操作数，如下所示。对于一些操作而言，目标操作数是可选的或者是不需要的，则可以使用下划线`_`表示的bit bucket来代替目标操作数。此外，指令还具有一个可选的谓词哨兵（predication guard），用于控制指令是否按照条件谓词的结果执行。

```
@p opcode  d, a, b, c;  // if(p == true) d = op(a, b, c);
@!p opcode d, a, b, c;  // if(p != true) d = op(a, b, c);
```

在PTX中，谓词寄存器是虚拟的，并且使用.pred作为声明时的类型说明符。谓词寄存器通常使用setp指令设置为比较指令的bool结果，setp最多可以指定两个目标寄存器，并使用`|`进行分隔。所有指令都有一个可选的guard谓词，用于控制指令的条件执行，语法是在谓词寄存器之前加上`@`或`@!`语法，如下所示。

```
.reg.pred p, q;
setp.lt.s32 p|q, a, b;  // p = (a < b); q = !(a < b);
@p add.s32 a, a, 1;     // if (p == true) a = a + 1;
```

谓词可以使用and、or、xor、not、mov指令操作。谓词和整数值之间没有直接转换，也没有直接的方法来加载或存储谓词寄存器值。但是，setp可用于从整数生成谓词，而基于谓词的selp指令可用于根据谓词的值生成整数值。 

一条指令必须指定类型大小（type-size）修饰符作为操作码的后缀，对于类型转换指令cvt和cvta则需要多个类型大小修饰符，并且这些修饰符的放置顺序与操作数顺序相同。通常，操作数的类型必须与指令的类型大小修饰符一致，或者至少兼容。相同大小的无符号整数和有符号整数兼容，位bit类型与具有相同大小的任何类型兼容，而对于浮点数指令，操作数的类型与大小必须一致。当源操作数的大小超过指令类型大小时，源数据将被截断为指令类型大小指定的适当位数。当目标操作数的大小超过指令类型大小时，目标数据将以符号（仅针对有符号整数）或零扩展到目标寄存器的大小。

为方便起见，ld、st、cvt指令允许操作数的类型大小比指令的类型大小更宽，因此可以使用常规宽度的寄存器加载、存储和转换更窄的值。例如，8位或16位值在加载、存储或转换为其他类型和大小时，可以直接保存在32位或64位寄存器中。操作数类型检查规则对于整数指令和位数据指令放宽，而浮点指令仍要求操作数类型大小完全匹配，除非操作数是位数据类型。

## 算术运算指令

整数算术指令（integer arithmetic instruction）对寄存器操作数和立即数进行操作，一些最常见的指令操作如下所示。

| 指令 | 语法                      | 语义                             | 备注                                                         |
| ---- | ------------------------- | -------------------------------- | ------------------------------------------------------------ |
| add  | add.type d, a, b;         | d = a + b;                       | .type = { .u16, .u32, .u64, .s16, .s32, .s64, .u16x2, .s16x2 }; |
| sub  | sub.type d, a, b;         | d = a - b;                       | .type = { .u16, .u32, .u64, .s16, .s32, .s64 };              |
| mul  | mul.mode.type d, a, b;    | d = a * b;                       | .mode = { .hi, .lo, .wide };<br/>.type = { .u16, .u32, .u64, .s16, .s32, .s64 }; |
| mad  | mad.mode.type d, a, b, c; | d = a * b + c;                   | .mode = { .hi, .lo, .wide };<br/>.type = { .u16, .u32, .u64, .s16, .s32, .s64 }; |
| div  | div.type d, a, b;         | d = a / b;                       | .type = { .u16, .u32, .u64, .s16, .s32, .s64 };              |
| rem  | rem.type d, a, b;         | d = a % b;                       | .type = { .u16, .u32, .u64, .s16, .s32, .s64 };              |
| neg  | neg.type d, a;            | d = -a;                          | .type = { .s16, .s32, .s64 };                                |
| abs  | abs.type d, a;            | d = \|a\|;                       | .type = { .s16, .s32, .s64 };                                |
| sad  | sad.type d, a, b, c;      | d = (a < b ? b - a : a - b) + c; | .type = { .u16, .u32, .u64, .s16, .s32, .s64 };              |
| min  | min.type d, a, b;         | d = a < b ? a : b;               | .type = { .u16, .u32, .u64, .u16x2, .s16, .s64 };            |
| max  | max.type d, a, b;         | d = a > b ? a : b;               | .type = { .u16, .u32, .u64, .u16x2, .s16, .s64 };            |

<table>
    <tr>
        <td>popc</td> <td>clz</td> <td>bfind</td> <td>fns</td> <td>letter</td> <td>bfe</td> <td>bfi</td>
        <td>sext</td> <td>bmsk</td> <td>dp4a</td> <td>dp2a</td>
    </tr>
    <tr>
        <td>add.cc</td> <td>addc</td> <td>sub.cc</td> <td>subc</td> <td>mad.cc</td> <td>madc</td>
    </tr>
</table>

浮点指令（floating-point instruction）对.f32类型和.f64类型的寄存器和立即数进行操作。其中舍入修饰符.rnd可以指定.rn、.rz、.rm、.rp四种模式之一，且有的指令具有默认舍入模式.rn，而有的指令无默认舍入模式，使用时注意分别。一些最常见的指令操作如下所示。

| 指令     | 语法                     | 语义                        | 备注                                                         |
| -------- | ------------------------ | --------------------------- | ------------------------------------------------------------ |
| testp    | testp.op.type p, a;      | p = op(a);                  | .op = { .finite, .infinite, .number, .notanumber, .normal, .subnormal };<br/>.type = { .f32, .f64 }; |
| copysign | copysign.type d, a, b;   | b.sign = a.sign;<br/>d = b; | .type = { .f32, .f64 };                                      |
| add      | add.rnd.type d, a, b;    | d = a + b;                  | .type = { .f32, .f32x2, .f64 };                              |
| sub      | sub.rnd.type d, a, b;    | d = a - b;                  | .type = { .f32, .f32x2, .f64 };                              |
| mul      | mul.rnd.type d, a, b;    | d = a * b;                  | .type = { .f32, .f32x2, .f64 };                              |
| fma      | fma.rnd.type d, a, b, c; | d = a * b + c;              | .type = { .f32, .f32x2, .f64 };                              |
| mad      | mad.rnd.type d, a, b, c; | d = a * b + c;              | .type = { .f32, .f64 };                                      |
| div      | div.mode.type d, a, b;   | d = a / b;                  | .mode = { .approx, .full, .rnd };  .rnd = { .rn, .rz, .rm, .rp };<br/>.type = { .f32, .f64 }; |
| neg      | neg.type d, a;           | d = -a;                     | .type = { .f32, .f64 };                                      |
| abs      | abs.type d, a;           | d = \|a\|;                  | .type = { .f32, .f64 };                                      |
| min      | min.type d, a, b;        | d = a < b ? a : b;          | .type = { .f32, .f64 };                                      |
| max      | max.type d, a, b;        | d = a > b ? a : b;          | .tpye = { .f32, .f64 };                                      |
| rcp      | rcp.mode.type d, a;      | d = 1 / a;                  | .mode = { .approx, .rnd };  .rnd = { .rn, .rz, .rm, .rp };<br/>.tpye = { .f32, .f64 }; |
| sqrt     | sqrt.mode.type d, a;     | d = sqrt(a);                | .mode = { .approx, .rnd };  .rnd = { .rn, .rz, .rm, .rp };<br/>.tpye = { .f32, .f64 }; |
| rsqrt    | rsqrt.approx.type d, a;  | d = 1 / sqrt(a);            | .tpye = { .f32, .f64 };                                      |
| sin      | sin.approx.f32 d, a;     | d = sin(a);                 |                                                              |
| cos      | cos.approx.f32 d, a;     | d = cos(a);                 |                                                              |
| tanh     | tanh.approx.f32 d, a;    | d = tanh(a);                |                                                              |
| lg2      | lg2.approx.f32 d, a;     | d = log~2~(a);              |                                                              |
| ex2      | ex2.approx.f32 d, a;     | d = 2 ^ a;                  |                                                              |

半精度浮点指令（half-precision floating-point instruction）对.f16类型和.bf16类型的寄存器和立即数进行操作。其中舍入修饰符几乎默认仅支持.rn模式，下面描述中将省略。一些最常见的指令操作如下所示。

| 指令 | 语法                   | 语义               | 备注                                              |
| ---- | ---------------------- | ------------------ | ------------------------------------------------- |
| add  | add.type d, a, b;      | d = a + b;         | .type = { .f16, .f16x2, .bf16, .bf16x2 };         |
| sub  | sub.tpye d, a, b;      | d = a - b;         | .type = { .f16, .f16x2, .bf16, .bf16x2 };         |
| mul  | mul.tpye d, a, b;      | d = a * b;         | .type = { .f16, .f16x2, .bf16, .bf16x2 };         |
| fma  | fma.tpye d, a, b, c;   | d = a * b + c;     | .type = { .f16, .f16x2, .bf16, .bf16x2 };         |
| neg  | neg.tpye d, a;         | d = -a;            | .type = { .f16, .f16x2, .bf16, .bf16x2 };         |
| abs  | abs.type d, a;         | d = \|a\|;         | .type = { .f16, .f16x2, .bf16, .bf16x2 };         |
| min  | min.type d, a, b;      | d = a < b ? a : b; | .type = { .f16, .f16x2, .bf16, .bf16x2 };         |
| max  | max.type d, a, b;      | d = a > b ? a : b; | .type = { .f16, .f16x2, .bf16, .bf16x2 };         |
| tanh | tanh.approx.type d, a; | d = tanh(a);       | .type = { .f16, .f16x2, .bf16, .bf16x2 };         |
| ex2  | ex2.approx.type d, a;  | d = 2 ^ a;         | .type = { .f16, .f16x2, .ftz.bf16, .ftz.bf16x2 }; |

混合精度浮点指令（mixed precision floating-point instruction）对具有不同浮点精度的数据进行作操作。其中舍入修饰符.rnd可以指定.rn、.rz、.rm、.rp四种模式之一，且默认舍入模式.rn。在操作执行之前，需要转换具有不同精度的操作数，以便指令的所有操作数都可以用一致的浮点精度表示；不同操作数使用的寄存器类型也根据指令类型的组合有所不同。

| 指令 | 语法                           | 语义                                     | 备注                       |
| ---- | ------------------------------ | ---------------------------------------- | -------------------------- |
| add  | add.rnd.f32.atype d, a, b;     | d = convert_f32(a) + b;                  | .atype = { .f16, .bf16 };  |
| sub  | sub.rnd.f32.atype d, a, b;     | d = convert_f32(a) - b;                  | .atype = { .f16, .bf16 };  |
| fma  | fma.rnd.f32.abtype d, a, b, c; | d = convert_f32(a) * convert_f32(b) + c; | .abtype = { .f16, .bf16 }; |

## 逻辑和移位指令

逻辑指令（logic instruction）和移位指令（shift instruction）基本上是无类型的或者是无符号整数类型的，只要操作数的位数大小相同，就可以对任何类型的操作数执行逐bit位元素的操作，这甚至允许对浮点数执行按位运算且不考虑符号位。此外，逻辑运算还可以对.pred谓词寄存器进行操作。

| 指令 | 语法                                      | 语义                                                         | 备注                                                         |
| ---- | ----------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| and  | and.type d, a, b;                         | d = a & b;                                                   | .type = { .pred, .b16, .b32, .b64 };                         |
| or   | or.type d, a, b;                          | d = a \| b;                                                  | .type = { .pred, .b16, .b32, .b64 };                         |
| xor  | xor.type d, a, b;                         | d = a ^ b;                                                   | .type = { .pred, .b16, .b32, .b64 };                         |
| not  | not.type d, a;                            | d = ~a;                                                      | .type = { .pred, .b16, .b32, .b64 };                         |
| cnot | cnot.type d, a;                           | d = a == 0 ? 1 : 0;                                          | .type = { .b16, .b32, .b64 };                                |
| lop3 | lop3.BoolOp.b32 d\|p, a, b, c, immLut, q; | d = get_op_table(immLut)(a, b, c);<br/>p = (d != 0) BoolOp q; | .BoolOp = { .and, .or };<br/>see PTX Document for immLut Table |
| shl  | shl.type d, a, b;                         | d = a << b;  /* fill 0 to right */                           | .type = { .b16, .b32, .b64 };                                |
| shr  | shr.type d, a, b;                         | d = a >> b;  /* fill 0 or sign-bit to left */                | .type = { .b16, .b32, .b64, .u16, .u32, .u64, .s16, .s32, .s64 }; |

shf指令用于更复杂的移位操作，它先将两个32位数据拼接成64位数据，a是低32位，b是高32位，然后进行c次移位得到结果，若左移则取高32位写入目标操作数，若右移则取低32位写入目标操作数，如下所示。其中，.mode指定为.clamp表示移位次数c最多是32次，指令为.warp表示移位次数c最多是31次。

```
shf.l.mode.b32  d, a, b, c;  // left shift
shf.r.mode.b32  d, a, b, c;  // right shift

.mode = { .clamp, .wrap };
```

```
u32 cnt = .mode == .clamp ? min(c, 32): c & 0x1f;
if (shf.direction == shf.l) {
    d = (b << cnt) | (a >> (32 - cnt));
} else if (shf.direction == shf.r) {
    d = (b << (32 - cnt)) | (a >> cnt);
}
```

## 比较和选择指令

比较指令（comparison instruction）根据两个源操作数的比较结果，对谓词寄存器或目标操作数进行赋值；而选择指令（select instruction）则根据谓词寄存器或第三个源操作数的值，选择前两个操作数之中的一个。不同类型的操作数支持不同的比较运算符，如下所述。

有符号整数具有6个比较运算符，分别是：eq、ne、lt、le、gt、ge。无符号整数具有6个比较运算符，分别是：eq、ne、lo（lower）、ls（lower-or-same）、hi（higher）、hs（higher-or-same）。位数据具有2个比较运算符，分别是：eq、ne。浮点数具有6个比较运算符，分别是：eq、ne、lt、le、gt、ge，但如果任一操作数为NaN，则比较结果为False；此外浮点数还具有6个比较运算符，分别是：equ、neu、ltu、leu、gtu、geu，如果两个操作数都是数值，则正常进行比较，若任一操作数为NaN，则比较结果为True。为测试NaN，提供两个运算符num和nan，其中，num当前仅当两个操作数都是数值时才返回True，nan当任一操作数为NaN时就返回True。

set指令比较两个数值，（可选的）将比较结果与一个谓词c做布尔运算，然后将最终结果写入到目标寄存器当中，目标寄存器是.dtype类型，如下所示。

```
set.CmpOp.dtype.stype        d, a, b;
set.CmpOp.BoolOp.dtype.stype d, a, b, {!}c;

.CmpOp  = { eq, ne, lt, le, gt, ge, lo, ls, hi, hs, equ, neu, ltu, leu, gtu, geu, num, nan };
.BoolOp = { and, or, xor };
.dtype  = { .f16, .bf16, .u32, .s32, .f32 };
.stype  = { .f16, .f16x2, .bf16, .bf16x2, .b16, .b32, .b64, .u16, .u32, .u64, .s16, .s32, .s64, .f32, .f64 };
```

```
t = (a CmpOp b) ? 1 : 0;
if (isFloat(dtype)) {
    d = BoolOp(t, c) ? 1.0 : 0;
} else {
    d = BoolOp(t, c) ? 0xffffffff : 0;
}
```

setp指令比较两个数值，（可选的）将比较结果与一个谓词c做布尔运算，然后将最终结果写入到寄存器当中，目标寄存器是.pred谓词类型，如下所示。

```
setp.CmpOp.type        p[|q], a, b;
setp.CmpOp.BoolOp.type p[|q], a, b, {!}c;

.CmpOp  = { eq, ne, lt, le, gt, ge, lo, ls, hi, hs, equ, neu, ltu, leu, gtu, geu, num, nan };
.BoolOp = { and, or, xor };
.type   = { .f16, .bf16, .b16, .b32, .b64, .u16, .u32, .u64, .s16, .s32, .s64, .f32, .f64 };
```

```
t = (a CmpOp b) ? 1 : 0;
p = BoolOp(t, c);
q = BoolOp(!t, c);
```

若setp指令的源操作数使用.f16x2和.bf16x2的形式，则指令的语法和语义有些微不同，主要体现在目标寄存器p和q不再是相反值，如下所示。

```
setp.CmpOp.f16x2        p|q, a, b;
setp.CmpOp.BoolOp.f16x2 p|q, a, b, {!}c;
```

```
t[0] = (a[0:15] CmpOp b[0:15]) ? 1 : 0;
t[1] = (a[16:31] CmpOp b[16:31]) ? 1 : 0;
p = BoolOp(t[0], c);
q = BoolOp(t[1], c);
```

slct指令根据第三个操作数的符号，选择前两个操作数之中的一个，写入到目标操作数，如下所示。

```
slct.dtype.ctype d, a, b, c;

.dtype = { .b16, .b32, .b64, .u16, .u32, .u64, .s16, .s32, .s64, .f32, .f64 };
.ctype = { .u32, .f32 };
```

```
d = c >= 0 ? a : b;
```

selp指令根据第三个谓词操作数c的值，选择前两个操作数之中的一个，写入到目标操作数，如下所示。

```
selp.type d, a, b, c;

.type = { .b16, .b32, .b64, .u16, .u32, .u64, .s16, .s32, .s64, .f32, .f64 };
```

```
d = c == 1 ? a : b;
```

## 控制流指令

控制流指令和语法（control flow instruction and syntax）用于控制PTX程序的执行流程。指令分组（instruction grouping）语法使用大符号`{}`包含一组指令，主要用于定义函数体，此外大括号还提供了一种确定变量范围的机制，使得在大括号范围内声明的任何变量在范围之外不可用。谓词执行（predicated execution）语法使用`@`符号引用一个谓词寄存器，从而可以按条件谓词是否为True来执行一条指令，带有False谓词的线程不执行任何操作。

bra指令将控制流转到一个label标签分支处继续执行，使用.uni告诉编译器该跳转是非发散的，即当前Warp的所有活动线程具有相同的条件谓词和target标签。

```
@p bra{.uni} target;  // target is a label
   bra{.uni} target;  // unconditional branch
```

```
if (p) {
    %pc = target;
}
```

brx.idx指令将控制流转到一个label标签处继续执行，该标签是从一个标签列表中根据一个索引选出来的，标签列表tlist必须是由.branchtargets指示声明的。

```
@p brx.idx{.uni} index, tlist;
   brx.idx{.uni} index, tlist;
```

```
if (p) {
    if (index < length(tlist)) {
        %pc = tlist[index];
    } else {
        %pc = undefined;
    }
}
```

call指令调用一个函数，并记录该函数的返回位置，也即下一条指令的地址，因此在执行ret指令后，可以在该点恢复执行。函数调用被假定为发散的，使用.uni后缀告诉编译器该调用是非发散的，即当前Warp中的所有活动线程都具有相同的谓词和目标函数地址。输入参数和返回值是可选的，参数可以是.const常量、.reg寄存器、.param参数状态存储空间，并且参数是按值进行传递的。

```
call{.uni} (ret-param), func_name, (param-list);         // direct call to named function, func_name is a symbol
call{.uni} (ret-param), func_ptr, (param-list), flist;   // indirect call via pointer, with full list of call targets
call{.uni} (ret-param), func_ptr, (param-list), fproto;  // indirect call via pointer, with no knowledge of call targets
```

call指令可以使用函数符号名称func_name直接调用，也可以使用函数地址func_ptr间接调用，间接调用时func_ptr必须是寄存器中保存的函数地址。间接调用需要额外的flist或fproto操作数，其中flist提供底层的调用目标的完整列表，后端可以自由优化调用约定；而在不知底层的完整列表时，使用fproto给出调用目标的通用函数原型，并且调用必须遵循ABI的调用约定。

flist操作数要么是一个数组的名称，该数组（调用表）使用函数名称列表进行初始化，要么是一个使用.calltargets指示声明的标签。在这两种情况下，func_ptr寄存器都保存调用表或.calltargets列表中一个函数的地址，并且根据flist指示的函数类型签名对操作数进行类型检查。fproto操作数是一个与.callprototype指示关联的标签，它会根据原型对调用操作数进行类型检查，代码生成将遵循ABI调用约定。

```
// example of direct call
call (%ret) func_bar (%a, %b);

.func (.reg.u32 ret) func1 (.reg.u32 a, .reg.u32 b) { ... }
.func (.reg.u32 ret) func2 (.reg.u32 a, .reg.u32 b) { ... }
.func (.reg.u32 ret) func3 (.reg.u32 a, .reg.u32 b) { ... }

// call-via-pointer using jump table
.global.u32 jmptbl[] = { func1, func2, func3 };
ld.global.u32 %r0, [jmptbl + 4];
call (%ret) %r0, (%x, %y), jmptbl;

// call-via-pointer using .calltargets directive
flist: .calltargets func1, func2, func3;
mov.u32 %r0, func2;
call (%ret) %r0, (%x, %y), flist;

// call-via-pointer using .callprototype directive
.func dispatch (.reg.u32 fptr, .reg.u32 idx) { ... }
fproto: .callprototype _ (.param.u32 _, .param.u32 _);
call %fptr, (%x, %y), fproto;
```

ret指令将程序执行的控制权返回到调用者的环境，默认假定返回是发散的，发散会挂起线程，直到所有线程都准备好返回给调用者，这允许多个发散的ret指令，使用.uni后缀执行返回是非发散的。需要注意的是，在执行ret指令之前，应将函数返回的任何值移动到返回值指定的参数变量中。

exit指令用于终止线程的执行。当线程退出时，将检查等待所有线程的栅障，以查看退出线程是否阻碍了栅障，如果退出线程阻碍了栅障，则释放栅障。

## 堆栈操作指令

堆栈操作指令（stack manipulation instruction）可用于在当前函数的堆栈帧上动态分配和释放内存。

stacksave,stackstore,alloca,

after Directive.

## 数据移动和转换指令

## 并行同步和通信指令

## Warp矩阵乘法累加指令

## 异步Warpgroup矩阵乘法累加指令

## 第五代TensorCore指令

## 其它指令（断点、视频、纹理、表面）

# 指示语句

指示语句（Directive Statement）用于指示一个特定属性或特定行为，对指令起到修饰和限定的作用，或用于指示编译器的行为，指示语句以点号.开头。

.version指示PTX ISA的版本号，.target指示目标平台的架构，每个PTX模块必须以.version指示开头，后跟.target指示目标平台的架构。此外，.address_size指示在PTX代码中使用的地址位数大小，可选值为32和64，若未指定，则默认地址大小是32位。

```
.version major.minor  // major, minor are integers
.target stringlist    // comma separated list of target specifiers, e.g. sm_10, sm_20, sm_30, sm_50, sm_60, sm_70, sm_80, ...
.address_size 64
```

## 函数定义与ABI接口

PTX并没有公开堆栈布局、函数调用约定、程序二进制接口ABI的详细信息，而是提供了更高级别的ABI抽象，以支持多个ABI实现。本节将介绍PTX所提供的ABI抽象的功能，包括函数定义、函数调用、参数传递等机制。

在PTX中，使用.entry指示声明和定义一个核函数（kernel function），使用.func指示声明和定义一个设备函数（device function），函数声明需要指定函数名称、输入参数列表、（仅设备函数的）返回值列表，函数定义需要指定函数主体。函数在调用之前必须声明或定义。

```
.entry kernel-name (param-list) {
    kernel-body;
}
.func (ret-param) func_name (param-list) {
    function-body;
}
```

在函数声明时，形式参数可以是.reg变量、.param变量，设置返回值的变量可以是.reg类型、.param类型；在函数调用时，实际参数可以是.const变量、.reg变量、.param变量，接收返回值的变量可以是.reg类型、.param类型。在使用.reg类型的变量时，既支持使用标量也支持使用向量。需要注意的是，在使用抽象ABI时，作为参数的.reg变量的大小必须至少是32位，在C++/PTX混合编程环境中，在C++中不足32位的数据，在PTX中应该提升为32位寄存器。

值得注意的是，为参数传递选择.reg空间或.param空间对参数最终是在物理寄存器中传递还是在堆栈中传递没有影响，参数到物理寄存器或堆栈位置的映射取决于ABI定义以及参数的顺序、大小和对齐方式。

此处讨论一下在函数中，使用.param参数存储状态空间的概念性方法，.param变量只能用于传递参数和接收返回值，不能他用。对于调用者，.param用于设置传递给函数的参数值，以及在调用之后接收函数返回值；对于被调用者，.param用于接收参数值，并用于向调用者设置函数返回值。在调用者代码中，设置实参的st.param指令必须紧跟在call指令之前，接收返回值的ld.param指令必须紧跟在call指令之后，不得更改任何控制流。用于参数传递的st.param和ld.param指令不能设置条件谓词，否则会启用编译器优化。

如果函数的参数或返回值是诸如大型结构体等非基本数据类型，则可以使用.param存储空间的字节数组作为参数类型或返回值类型，且实际参数必须也是类型、大小、对齐方式都匹配的.param字节数组数据，且必须在调用者的本地范围内声明.param实际参数。

一段示例的CUDA C++代码如下所示。

```c++
struct arg_t { double value; short alpha; };
struct ret_t { double value; char pad[2]; };

__device__ __noinline__ ret_t func_bar(arg_t arg, uint2 bias) {
    ret_t ret;
    ret.value = arg.value * arg.alpha + bias.x;
    ret.pad[0] = (char)(bias.y * 2);
    ret.pad[1] = (char)(bias.y * 4);
    return ret;
}

__global__ void my_kernel(const double *input, double* output, const uint32_t length) {
    if (threadIdx.x > length) return;
    arg_t arg;
    arg.value = input[threadIdx.x];
    arg.alpha = (short)(threadIdx.x);
    uint2 bias = { length / 2, length / 4 };
    ret_t ret = func_bar(arg, bias);
    output[threadIdx.x] = ret.value + ret.pad[0] + ret.pad[1];
}
```

上述CUDA C++代码的PTX代码如下所示。



























## 链接指令



## 控制流指令

