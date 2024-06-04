# Efficient GEMM in CUDA

计算矩阵乘法的三层嵌套循环可以进行分块（blocked）与分片（tiled）以使用并行编程模式来匹配硬件的并发性、内存局部性。CUTLASS将通用矩阵乘法GEMM映射到GPU设备上，并使用CUDA并行编程模型，伪代码如下所示。

```c++
for (int bn = 0; bn < gemmN; bn += blockN)  // for each blockIdx.y
for (int bm = 0; bm < gemmM; bm += blockM)  // for each blockIdx.x
for (int bk = 0; bk < gemmK; bk += blockK)  // GEMM mainloop, no unrolling; one iteration is one "stage"
    for (int wn = 0; wn < blockN; wn += warpN)  // for each warpIdx.y
    for (int wm = 0; wm < blockM; wm += warpM)  // for each warpIdx.x
    for (int wk = 0; wk < blockK; wk += warpK)  // fully unroll across blockK; one iteration is one "k Group"
        for (int mk = 0; mk < warpK; mk += mmaK)  // outer product loop, fully unroll across warpK
        for (int mn = 0; mn < warpN; mn += mmaN)  // for each threadIdx.y
        for (int mm = 0; mm < warpM; mm += mmaM)  // for each threadIdx.x
            mma_instruction(d, a, b, c);  // one single mma instruction
```

该嵌套循环在线程块、线程束、线程三个层级划分，并利用共享内存和寄存器两种存储，下图为层级划分的示意图。

<img src="CUTLASS模板库.assets/gemm-hierarchy.png" style="zoom:15%;" />

为重用在最后一级缓存中的数据，CUTLASS在cutlass/gemm/threadblock_swizzle.h头文件中定义了一些函数可以影响线程块到矩阵划分的逻辑映射。这种映射模式会在矩阵划分的二维区域上紧凑地启动线程块，以提高线程块能够几乎同时访问到设备全局内存中同一个Tile分片的可能性。称为线程块重排。

> 通常Cache缓存分为L1、L2、L3缓存，则最后一级缓存（Last Level Cache，LLC）是指速度最慢、容量最大的缓存，LLC缓存之后即是内存。

## Tiling and Epilogue

一个线程块负责计算输出矩阵的一部分，迭代加载输入矩阵的分片并对矩阵乘积进行累加。在线程块层级，从全局内存中加载数据。分块策略是性能的关键，并需要平衡多个目标。更大的线程块意味着更少的全局内存读取，从而保证DRAM带宽不是性能瓶颈；然而更大的线程块可能与问题规模不匹配。如果M或N维度较小，线程块中的一些线程可能因为已经超出问题边界而在做无效计算。如果M和N维度较小而K维度较大，这种模式可能会启动很少的计算线程，无法充分利用GPU设备的流多处理器；通过在K维度上进行线程块或线程束的划分，然后再执行归约，可以对这种问题规模的计算进行优化。在CUTLASS中，可以使用ThreadblockShape::{kM,kN,kK}指定线程块分片的尺寸，以匹配不同的硬件架构和问题规模。

一个线程束从共享内存中加载数据到寄存器并执行计算。在实现上，线程束的计算可以通过mma.sync指令或wmma指令传递到TensorCore完成计算，或通过线程划分传递给CUDA核心完成计算。为取得最高性能，对共享内存的访问应该避免bank冲突。为重用数据，应该尽可能划分更大的线程束。

一个线程负责处理特定数目的元素。因为线程无法访问其它线程的寄存器，故应该选择一种线程布局，使多条计算指令能够重用寄存器中数据；也即一个线程处理一个二维的分片结构，于是线程能够将一组独立的计算指令传递给CUDA核心计算，并计算累加的外积。SGEMM、DGEMM、HGEMM、IGEMM等通过单指令多线程SIMT指令完成计算。

上述划分完成矩阵乘法计算之后，计算结果保存在每个线程的寄存器当中。一个线程所负责的输出矩阵的部分元素的划分能取得最高的矩阵乘法计算效率，但在存储时并不能实现高效的合并访存模式。结尾（epilogue）是一个单独阶段，线程使用共享内存交换数据，以取得对设备全局内存的高效的合并访问模式；同时，这也是对计算结果进行缩放，以及应用其它逐元素操作的阶段。CUTLASS定义一些典型的结尾操作，例如线性缩放与收缩等，也支持自定义逐元素操作函数。

## Pipeline

层级划分结构使得每个CUDA线程需要占用大量的寄存器，且每个线程所持有的累加值至少要占用寄存器预算的一半以上；因此GPU设备的占用率较低，线程块、线程束、线程数目通常低于其它任务的工作负载；这会导致GPU难以隐藏内存延迟和切换线程上下文时所带来停顿间隔（stall）。为减轻内存延迟，CUTLASS使用软件流水线，也即使用双缓冲技术，以重叠线程的访存和计算，如下所示。

- 线程块层级，持有两个共享内存空间，一个用于为当前次的矩阵计算提供数据，另一个用于从设备全局内存中加载下一次主循环迭代所需的数据。
- 线程束层级，持有两个存储于寄存器的矩阵片段（fragment），一个用于传递给CUDA核心或TensorCore执行当前次的矩阵计算，另一个用于从共享内存中加载下一次Warp循环迭代所需的数据。

下图展示CUTLASS所使用的GEMM主循环流水线。

<img src="CUTLASS模板库.assets/software-pipeline.png" style="zoom: 25%;" />

## SplitK and SliceK

矩阵乘法中线程块的划分具有在O(MN)上的并行性，并独立地执行内积计算。当问题规模M,N足够大时，CUTLASS的矩阵乘法kernel能够达到最大理论计算吞吐量；而当问题规模M,N较小时，则启动的线程块数目太少难以充分利用整个GPU设备。

SplitK（reduction across Block）通过将内积计算过程中的归约操作并行化，可以启动更多的线程块并发执行，从而在线程块层级充分利用计算吞吐量。CUTLASS在问题规模的K维度上进行划分，并在每个划分上启动一组线程块执行计算，然后执行并行的归约操作。用户需要管理工作缓冲区以保存中间结果。

划分维度K的GEMM允许指定问题规模以及划分数目，并且允许维度K无法被整除的情况。例如M,N,K=128,128,4096的问题规模和SplitNum=20的划分数目，会产生20个矩阵乘法kernel，前19个计算所划分到的SplitK=4096/20=204，最后一个计算所划分到的SplitK=220，这能完整处理K维度上的计算。然后再在维度K上执行归约操作，以获得最终结果。

因为每个线程块负责blockM,blockN的输出矩阵，那么线程束的划分具有在O(blockM,blockN)上的并行性。更大的线程束分片warpM,warpN允许更好的指令并行和重用，但当问题规模M,N更小时，这会限制每个线程块所持有的线程束数目，从而导致效率降低。

SliceK（reduction across Warp）通过在blockK维度上划分线程束，能够允许一个线程块产生更多线程束并发执行。SliceK策略不仅会将blockM,blockN划分给warpM,warpN，还会将线程块的计算在blockK维度进一步划分给warpK。然后在线程块的所有线程束计算完成后，再在相关的线程束之间执行归约操作。

## Warp Specialization

从Hopper架构开始，CUTLASS 3.0引入线程束专业化的概念，即一个线程块中的线程束被分为两组，分别是生产者线程束与消费者线程束。生产者使用新架构的张量内存加速器（Tensor Memory Accelerator，TMA）将数据从设备全局内存中加载到共享内存缓冲区中，并更新该阶段所关联的栅障以通知相关消费者数据已填充；消费者等待生产者的填充信号，然后启动TensorCore的MMA操作，然后释放共享内存缓冲区，并使用新引入的Async Pipeline Class类通知生产者共享内存缓冲区已为空，以执行下一组TMA工作负载。

# Overview

> 文档可参阅https://github.com/NVIDIA/cutlass/tree/main/media/docs网址，本文章摘抄自CUTLASS 3.5.0版本。需要注意的是，CUTLASS 3.0需要CUDA 11.4及更新的版本，且GPU设备为SM70及更新的架构。

CUTLASS是CUDA Templates for Linear Algebra Subroutines and Solvers的缩写，是基于CUDA运行时的线性代数例程与求解器的C++模板库，用于实现高性能的矩阵乘法GEMM及其相关计算。除通用矩阵乘法之外，CUTLASS通过隐式GEMM算法实现高性能的卷积操作。

CUTLASS 2.x其采用与cuBLAS和cuDNN相似的层级划分（hierarchical decomposition）与数据移动策略，并将这些功能组件实现为C++模板类。通过自定义Tile大小、数据类型、算法策略，可以对并行层级划分中的不同层级的基本操作（primitive）进行定制和微调。为支持各种应用，CUTLASS提供对特定数据移动、乘法累加（multiply-accumulate）的混合精度支持，包括FP16、BF16、Tensor Float 32，FP32、FP64，以及整型和二进制数据类型。

CUTLASS 3.0引入一个新的核心库CuTe，是一个C++模板的集合，用于对线程和数据的层级多维布局进行定义和操作。CuTe提供Layout和Tensor对象，将类型、形状、内存空间、数据布局等概念结合在一起，为用户执行复杂的索引操作。CuTe的核心是层级多维布局（hierarchically multidimensional layout），可以表示线程组织的布局与数据张量的布局。在CUTLASS 3.0及之后的版本中，广泛使用CuTe以简化设计并提高代码可读性。

该存储仓库包括若干组件。在顶层include目录中提供CUTLASS模板库和CuTe模板库的头文件，应用程序编程需要将顶层include目录添加到编译器的头文件搜索路径；在顶层tools目录中提供CUTLASS Instance模板实例、CUTLASS Profiler分析器、CUTLASS Utilities额外工具；在顶层examples目录中提供使用示例；在顶层media目录中提供文档；在顶层test目录中提供测试组件。

```shell
.
├── include       # Top-level include directory. Client applications should target this path.
│   ├── cutlass   # CUTLASS Template Library, CUDA Templates for Linear Algebra Subroutines and Solvers
│   └── cute      # CuTe Template Library, CuTe Layout, layout algebra, MMA/Copy atoms, tiled MMA/Copy
├── tools
│   ├── library   # CUTLASS Instance Library, static/dynamic library containing all kernel instantiations of interest
│   ├── profiler  # CUTLASS Profiler
│   └── util      # CUTLASS Utilities
├── examples      # CUTLASS Examples
├── media         # Documentation
└── test
```

多维对象（multidimensional object）是一个统称，可以指数组（array）、矩阵（matrix）、张量（tensor）、索引空间（index space）、形状（shape）、跨步（stride）、布局（layout）等。逻辑数目（logical number）是指，在逻辑表示上，有效元素的数目。实际存储数目（physical number）是指，在内存空间中进行存储时，占用物理存储空间的实际存储的元素数目，包括有效元素和填充元素。

使用Index表示某个逻辑维度轴上的索引，使用Extent表示某个逻辑维度轴上的逻辑维数，使用Rank表示维度轴的数目，使用Size表示全部逻辑元素的数目；使用LongIndex表示在内存空间中存储位置的线性偏移，使用Capacity表示多维对象在内存中实际需要存储的元素数目，包括填充元素。

# CUTLASS Template

```shell
cutlass      # CUTLASS Template Library
├── *            # fundamental types
├── layout       # layout for matrices, tensors, and other mathematical objects in memory
├── arch         # direct exposure of architecture features (including instruction-level GEMMs)
├── gemm         # code specialized for general matrix product computations
│   ├── *            # basic types for GEMM
│   ├── device       # launches kernel(s) over a full device
│   ├── kernel       # CUDA kernel entry points
│   ├── threadblock  # CTA-level operators
│   ├── warp         # warp-level operators
│   └── thread       # thread-level operators
├── epilogue     # epilogue rearranges result to canonical layouts, and supports conversion and reduction operations
├── reduction    # bandwidth-limited reduction kernels that do not fit the "gemm" models
└── transform    # code specialized for layout, type, and domain transformations
```

## Fundamental Types

CUTLASS会额外定义一些数值类型与容器类型，而多数CUTLASS基本类型与C++标准库类型一致，并可用于主机代码和设备代码，且与设备的计算能力无关。需要注意的是，一些类型或函数在较低的架构上并不支持，例如较旧的CUDA不支持hrsqrt函数，可以在编译时使用-arch=sm_89指定目标架构。

在cutlass/numeric_types.h头文件和cutlass/complex.h头文件中，提供一些数值类型的定义，如下所示。

| 数值类型     | 字面量后缀 | 描述                                       |
| ------------ | ---------- | ------------------------------------------ |
| half_t       | _hf        | IEEE半精度浮点数；尾数10位，指数5位        |
| bfloat16_t   | _bf16      | BFloat16类型；尾数7位，指数8位             |
| tfloat32_t   | _tf32      | Tensor Float 32类型；尾数10位，指数8位     |
| int4_4       | _s4        | 有符号4位整型                              |
| uint4_t      | _u4        | 无符号4位整型                              |
| bin1_t       | _b1        | 一位二进制位                               |
| complex\<T\> |            | 复数类型，其实部或虚部的类型由实数类型指定 |

在cutlass/array.h头文件中，提供Array\<T,N\>容器和AlignedArray\<T,N,Align\>容器的定义，如下所示。

```c++
template<typename T, int N, bool RegisterSized = sizeof_bits<T>::value >= 32> struct Array;
template<typename T, int N>
struct Array<T, N, true> {
    static constexpr size_t kElements = N;
    using Storage = T;
    Storage storage[kElements];
  	typedef T value_type;
    typedef value_type& reference;
    typedef value_type* pointer;
    pointer data() { return reinterpret_cast<pointer>(storage); }
    reference operator[](size_type pos) { return reinterpret_cast<reference>(storage[pos]); }
};
template<typename T, int N, int Alignment = (sizeof_bits<T>::value * N + 7) / 8>
class alignas(Alignment) AlignedArray: public Array<T,N> {};
```

Array\<T,N\>是一个固定大小的数组，与C++标准库std::array相似，但可存储小于1B的类型，且小类型的对象之间紧凑存储。在使用sizeof(Array\<T,N\>)运算符时，其返回结果仍然是以字节为单位，且最小是1个字节。应尽量避免对Array单个元素的操作，而应使用其成员方法，这些方法会使用效率更高的向量化指令。

AlignedArray\<T,N\>是一个固定大小的数组，可指定其内存空间按多少字节对齐。

在cutlass/aligned_buffer.h头文件中，提供AlignedBuffer\<T,N,Align\>容器的定义，如下所示。

```c++
template<typename T, int N, int Align = 16>
struct AlignedBuffer {
    static int const kBytes = (sizeof_bits<T>::value * N + 7) / 8;
    using Storage = uint8_t;
    alignas(Align) Storage storage[kBytes];
    typedef T value_type;
    typedef value_type* pointer;
    pointer data() { return reinterpret_cast<pointer>(storage); }
};
```

AlignedBuffer\<T,N,Align\>是一个固定大小的缓冲区，不会调用所持有类型的构造方法。可使用AlignedBuffer<>::data()方法获得内存空间的地址指针。常用于获取一段以给定字节对齐的连续内存空间，如设备全局内存或共享内存，以用于向量化操作，一个示例如下所示。

```c++
__global__ void array_demo_kernel() {
    const int kN = 1024;
    __shared__ AlignedBuffer<half_t, kN> smem_buffer;
    AlignedArray<half_t, 8> *ptr = reinterpret_cast<AlignedArray<half_t, 8>*>(smem_buffer.data());
    AlignedArray<half_t, 8> value = ptr[threadIdx.x];  // 128-bit shared memory load
}
```

在cutlass/numeric_conversion.h头文件中，提供NumericConverter\<T,S\>数值类型转换器的定义，如下所示。

```c++
enum class FloatRoundStyle {
    round_indeterminate,          // rounding mode unknown
    round_toward_zero,            // round toward zero
    round_to_nearest,             // round to nearest even
    round_to_nearest_satfinite,   // round to nearest even, capping value to min and max of destination type
    round_toward_infinity,        // round toward infinity
    round_toward_neg_infinity,    // round toward negative infinity
    round_half_ulp_truncate,      // add 0.5ulp to integer representation then round toward zero
    round_half_ulp_trunc_dntz     // like round_half_ulp_truncate, except denorms are rounded *toward* zero
};
template<typename T, typename S, FloatRoundStyle Round = FloatRoundStyle::round_to_nearest>
struct NumericConverter {
    using result_type = T;
    using source_type = S;
    static FloatRoundStyle const round_style = Round;
    static result_type convert(source_type const &s) { return static_cast<result_type>(s); }
    result_type operator()(source_type const &s) const { return convert(s); }
};
```

NumericConverter\<T,S\>会尽可能地在目标架构上使用硬件加速，并支持多种舍入模式。此外，NumericArrayConverter\<T,S,N\>支持转换Array数组类型，一个示例如下所示。

```c++
void convert_demo() {
    int const kN = 16;
    Array<int8_t, kN> destination;
    Array<int, kN> source;
    NumericArrayConverter<int8_t, int, kN> convert;
    destination = convert(source);
}
```

在cutlass/coord.h头文件中，提供Coord\<Rank\>容器的定义，如下所示。

```c++
template<int Rank, typename Index = int, typename LongIndex = int64_t>
struct Coord {
    static int const kRank = Rank;
    Index idx[kRank];
    Index& operator[](int dim) { return idx[dim]; }
};
```

Coord\<Rank\>是一个通用的逻辑坐标，或表示维数形状，可用于张量中的索引下标，并支持两个坐标之间的加减乘除操作。

在cutlass/matrix_coord.h头文件和cutlass/tensor_coord.h头文件中，提供MatrixCoord坐标和Tensor4DCoord坐标的定义，如下所示。

```c++
struct MatrixCoord : public Coord<2, int> {
    static int const kRow = 0;
    static int const kColumn = 1;
    Index& row() { return this->at(kRow); }
    Index& column() { return this->at(kColumn); }
};
struct Tensor4DCoord : public Coord<4> {
    static int const kN = 0;
    static int const kH = 1;
    static int const kW = 2;
    static int const kC = 3;
    Index& n() { return this->at(kN); }
    Index& h() { return this->at(kH); }
    Index& w() { return this->at(kW); }
    Index& c() { return this->at(kC); }
}
```

MatrixCoord和Tensor4DCoord分别提供专用于二维矩阵和四维张量情况下的坐标，并提供相关特定的成员方法。

在cutlass/predicate_vector.h头文件中，提供PredicateVector的定义，如下所示。

```c++
template <int kPredicates, int kPredicatesPerByte = 4, int kPredicateStart = 0>
struct PredicateVector {
    // Storage type of individual elements
    typedef uint32_t Storage;
    // Number of bytes needed
    static constexpr int kBytes = (kPredicates + kPredicatesPerByte - 1) / kPredicatesPerByte;
    // Number of storage elements needed
    static constexpr int kWordCount = (kBytes + int(sizeof(Storage)) - 1) / int(sizeof(Storage));
    Storage storageData[kWordCount];
}
```

PredicateVector是一个由谓词构成的固定长度的向量，也即掩码向量，可以在循环展开代的码段中使用寄存器加速访问。

在cutlass/functional.h头文件中，提供一些模板函数的定义，该头文件是模仿C++标准库的functional头文件，如下所示。

```c++
template<typename A, typename B = A, typename C = A>
struct multiply_add {
    C operator()(A const &a, B const &b, C const &c) const {
		return C(a) * C(b) + c;
    }
};
```

其中，CUTLASS扩展了multiply_add\<T\>的定义，支持复数complex\<T\>类型的乘加操作，并尽可能调用本地硬件指令。

## Layouts and Tensors

> 注意，本节讨论的布局仅用于CUTLASS 2.x版本，在CUTLASS 3.x版本中，使用cute::Layout<Shape,Stride>的概念描述线程和数据张量的布局。

张量是一个多维对象，由内存中多维的数值元素数组表示。例如，二维矩阵通常用于经典数值计算，多维张量通常用于深度学习任务等。本节描述CUTLASS库的设计，如何使用Layout概念将逻辑索引空间映射到内存布局，如何使用TensorRef和TensorView概念间接访问内存中的张量元素。同时，CUTLASS提供一些与C++标准库一致的概念；size指张量的元素总数；capacity指实际存储的元素总数；rank指张量逻辑维度的数目；extent指张量每个维度上的维数。

布局Layout将逻辑索引空间映射到内存空间中存储位置的实际偏移，并存储用于计算映射的状态，定义其它CUTLASS组件需要使用的部分实例化。

在cutlass/layout目录的若干头文件中，提供各种布局类型的定义。例如cutlass/layout/vector.h头文件、cutlass/layout/matrix.h头文件、cutlass/layout/tensor.h头文件等，还有cutlass/layout/permute.h头文件提供变换概念的定义。一个通用设计如下。

```c++
struct LayoutConcept {
    static int const kRank;                        // Logical rank of tensor
    static int const kStrideRank;                  // Rank of stride vector
    using Index = int32_t;                         // Index type used for coordinates
    using LongIndex = int64_t;                     // Long index type used for offsets
    using TensorCoord = Coord<kRank, Index>;       // Logical coordinate
    using Stride = Coord<kStrideRank, LongIndex>;  // Stride object
    Stride stride_;                                // Stride data member  
    ColumnMajor(LongIndex ldm = 0): stride_(ldm) {}          // Constructor with leading dimension
    ColumnMajor(Stride stride): stride_(stride) {}           // Constructor
    static LayoutConcept packed(TensorCoord const &extent);  // Return a layout to a tightly packed tensor
    LongIndex operator()(TensorCoord const &coord) const;    // Return the offset of a coordinate in linear memory
    TensorCoord inverse(LongIndex offset) const;             // mapping linear offset to logical coordinate
    Stride stride() const();                                 // Returns the stride of the layout
    LongIndex capacity(TensorCoord const &extent) const;     // The number of contiguous elements needed to store a tensor
};
```

在cuBLAS库中，存在前导维数的概念，在默认采用列主序存储的矩阵布局时，这意味着矩阵元素{rid,cid}具有值为rid+cid\*ld的偏移，等价于CUTLASS提供的ColumnMajor布局类型；同时CUTLASS也提供RowMajor等布局类型，如下所示。

```c++
void layout_demo() {
    int64_t ld = 32;
    ColumnMajor col_layout(ld);
    RowMajor    row_layout(ld);
    int64_t col_offset = col_layout({7, 23});  // rid + cid * ld
    int64_t row_offset = row_layout({7, 23});  // rid * ld + cid
    printf("%ld, %ld\n", col_offset, row_offset);  // 743, 247
}
```

在上述两种情况下，逻辑坐标{rid,cid}表示矩阵中同一个元素，这允许采用逻辑索引空间的算法实现保持通用性，并由Layout提供到实际存储位置的映射。

在cutlass/tensor_ref.h头文件中，提供TensorRef\<T,Layout\>结构体的定义，该结构体持有一个张量的数据地址指针以及布局对象，用于访问张量元素，可作为函数参数传递，如下所示。

```c++
template<typename Element, typename Layout>
class TensorRef {
    using Reference = Element&;
    Element* ptr_;   // Pointer
    Layout layout_;  // Layout object maps logical coordinates to linear offsets
    TensorRef(Element *ptr, Layout const &layout): ptr_(ptr), layout_(layout) {}  // Constructs a TensorRef
    // Returns the pointer to referenced data
    Element* data() const { return ptr_; }
    // Returns a reference to the element at a given linear index
    Reference data(LongIndex idx) const { return ptr_[idx]; }
    // Computes the offset of an index from the origin of the tensor
    LongIndex offset(TensorCoord const &coord) const { return layout_(coord); }
    // Returns a reference to the element at a given Coord
    Reference operator[](TensorCoord const &coord) const { return data(offset(coord)); }
};
```

在cutlass/tensor_view.h头文件中，提供TensorView\<T,Layout\>类的定义，用于描述线性代数计算中限维确定的张量。该类继承自TensorRef\<T,Layout\>结构体，并提供extent()方法获得某个特定维度轴上的维数，如下所示。

```c++
template<typename Element, typename Layout>
class TensorView : public TensorRef<Element, Layout> {
    using TensorCoord = typename Layout::TensorCoord;
    TensorCoord extent_;  // View extent
    // Constructs a TensorView object
    TensorView(Element *ptr, Layout const &layout, TensorCoord const &extent): Base(ptr, layout), extent_(extent) {}
    TensorCoord const& extent() const { return extent_; }  // Returns the extent of the view
};
```

使用TensorRef或TensorView访问张量元素的示例如下所示。

```c++
void tensor_view_demo() {
    int8_t *ptr = (int8_t*)malloc(sizeof(int8_t) * 16 * 9);
    for (int i = 0; i < 16 * 9; ptr[i++] = i);
    TensorView<int8_t, ColumnMajor> view(ptr, ColumnMajor(16), MatrixCoord(16, 9));
    if (view.contains({9, 5})) {
        printf("%d\n", view[{9, 5}]);  // 89
    }
    free(ptr);
}
```

注意，使用一个问题规模，以及每个操作数的TensorRef对象，可以避免在确定计算操作时的一些过度冗余指定。

# CuTe Template

```shell
cute       # CuTe Template Library, CuTe Layout, layout algebra, MMA/Copy atoms, tiled MMA/Copy
├── *          # Core library types such as Shape, Stride, Layout, Tensor, and associated operations
├── numeric    # CuTe's internal numerics implementation
├── atom       # Meta-information either link to or built from arch/ operators
├── arch       # Bare bones PTX wrapper structs for copy and math instructions
├── container  # Core container types used across CuTe, namely, cute::tuple
└── algorithm  # Definitions of core operations such as copy, gemm, and operations on cute::tuples
```

# CUTLASS GEMM API

根据层级划分，CUTLASS抽象出每个层级的矩阵乘法累加（matrix multiply-accumulate，MMA）操作，包括设备、线程块、线程束、线程、指令层级。下述伪代码展示的是使用线程束同步矩阵乘法指令（如mma.sync）的通用矩阵乘法kernel模型，整个操作称为Gemm操作，并假设结尾操作只执行矩阵更新。

```c++
// cutlass::gemm::device::Gemm
for (int bn = 0; bn < gemmN; bn += blockN)  // for each Block
for (int bm = 0; bm < gemmM; bm += blockM)  // for each Block
for (int bk = 0; bk < gemmK; bk += blockK)  // GEMM mainloop, no unrolling; one iteration is one "stage"
    // cutlass::gemm::threadblock::Mma
    for (int wn = 0; wn < blockN; wn += warpN)  // for each Warp
    for (int wm = 0; wm < blockM; wm += warpM)  // for each Warp
    for (int wk = 0; wk < blockK; wk += warpK)  // fully unroll across blockK; one iteration is one "k Group"
        // cutlass::gemm::warp::Mma
        for (int mk = 0; mk < warpK; mk += mmaK)  // outer product loop, fully unroll across warpK
        for (int mn = 0; mn < warpN; mn += mmaN)  // for each Thread
        for (int mm = 0; mm < warpM; mm += mmaM)  // for each Thread
            mma_instruction(d, a, b, c);  // cutlass::arch::mma, warp-wide matrix multiply instruction
```

最外两层循环对应着线程块层级的硬件并行性，并没有显式地写在代码中，而是使用CUDA并行编程模型中的线程网格语义并发启动。注释cutlass::gemm::threadblock::Mma指的是线程块范围的矩阵乘法累加操作，由一个线程块负责计算一部分矩阵乘积；注释cutlass::gemm::warp::Mma指的是线程束范围的矩阵乘法累加，由一个线程束负责计算一系列外积累加。最内层操作指硬件直接支持的操作，该示例中是线程束同步的TensorCore的矩阵乘法指令；此外也可以在线程层级执行单个线程的乘法累加指令。

该嵌套循环在CUTLASS中由下图所示的数据类型、布局、数学指令等进行描述，如下所示。省略公共命名空间cutlass::前缀。
