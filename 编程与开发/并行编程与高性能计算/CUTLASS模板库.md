# CUTLASS

CUTLASS是CUDA Templates for Linear Algebra Subroutines and Solvers的缩写，是基于CUDA运行时的线性代数例程与求解器的C++模板库，用于实现高性能的矩阵乘法GEMM及其相关计算。除通用矩阵乘法之外，CUTLASS通过隐式GEMM算法实现高性能的卷积操作。

> 使用模板库的优势在于，一些在计算过程中不变的配置，例如分片形状与迭代策略，可以使用模板参数在编译期间确定，从而只使用函数参数传递数据。

CUTLASS库的源码可在https://github.com/NVIDIA/cutlass网址获得，其包括CUTLASS模板库与CuTe模板库。其中CUTLASS模板库是指CUTLASS 2.X实现版本，通过各层级的模板库抽象提供GEMM实现；而CuTe模板库是自CUTLASS 3.0版本引入的新模板库，通过Layout对象和Tensor对象提供GEMM实现。需要注意的是，CUTLASS 3.0版本需要CUDA 11.4及以上版本，且GPU设备的计算能力为SM70及以上版本。

CUTLASS库包括若干组件。在顶层include目录中提供CUTLASS模板库和CuTe模板库的头文件，应用程序编程需要将顶层include目录添加到编译器的头文件搜索路径；在顶层tools目录中提供CUTLASS Instance模板实例、CUTLASS Profiler分析器、CUTLASS Utilities额外工具；在顶层examples目录中提供使用示例；在顶层media目录中提供文档；在顶层test目录中提供测试组件。

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

> 在项目结构中，通常文件目录与命名空间的组成方式是一致的，例如，命名空间cutlass::gemm::device对应到cutlass/gemm/device目录。因为CUTLASS模板库的所有代码都位于cutlass根命名空间中，故在介绍时默认省略cutlass::命名空间。

多维对象（multidimensional object）是一个统称，可以指数组（array）、矩阵（matrix）、张量（tensor）、索引空间（index space）、形状（shape）、跨步（stride）、布局（layout）等。逻辑数目（logical number）是指，在逻辑表示上，有效元素的数目。实际存储数目（physical number）是指，在内存空间中进行存储时，占用物理存储空间的实际存储的元素数目，包括有效元素和填充元素。

使用Index表示某个逻辑维度轴上的索引，使用Extent表示某个逻辑维度轴上的逻辑维数，使用Rank表示维度轴的数目，使用Size表示全部逻辑元素的数目；使用LongIndex表示在内存空间中存储位置的线性偏移，使用Capacity表示多维对象在内存中实际需要存储的元素数目，包括填充元素。

# Host Utility

在项目顶层的tools/util/include/cutlass目录中，提供CUTLASS的各种功能的工具模板类，实际使用时可查阅目录中所提供的头文件，此处只是列举一些常用的工具模板类。注意，应用程序需要将顶层tools/util/include目录添加到编译器的头文件搜索路径，完整的是tools/util/include/cutlass/util路径。

在cutlass/util/device_memory.h头文件中，提供GPU设备全局内存管理函数的C++包装接口DeviceAllocation\<T\>模板类，其使用smart_ptr智能指针对内存空间地址指针进行管理，在模板类的实例对象超出作用域时，会自动释放已分配的设备内存，避免内存泄漏问题。

```c++
__global__ void demo_device_alloc_kernel(float *device_ptr) {}

void demo_device_alloc() {
    int num_of_float = 1024;
    // using allocation = cutlass::DeviceAllocation<T>;
    cutlass::device_memory::allocation<float> device_alloc(num_of_float);
    demo_device_alloc_kernel<<<128, 128>>>(device_alloc.get());
    // Device memory is automatically freed when device_alloc goes out of scope
}
```

在cutlass/util/host_tensor.h头文件中，提供HostTensor<T,Layout>模板类，用于表示一个张量对象，并在主机端或设备端分配存储空间。

```c++
template <
    typename Element,  // Data type of element stored within tensor (concept: NumericType)
    typename Layout    // Defines a mapping from logical coordinate to linear memory (concept: Layout)
>
class HostTensor {
public:
    // Note: Below is used to handle packing of subbyte elements
    // kBitsStoredVec          : The bits of store vec that could be divisiable by the element
    // kElementsPerStoredVec   : The number of elements could be stored in per store vec
    // kNumStoragePerStoredVec : How much storage(i.e. sizeof(element storage)) the store vec needs to consume.
    //                           Usually the element storage of subbyte is uint8_t.
    // Example
    //  int2:  kBitsStoredVec = 8; kElementsPerStoredVec = 4; kNumStoragePerStoredVec = 1 uint8_t;
    //  int4:  kBitsStoredVec = 8; kElementsPerStoredVec = 2; kNumStoragePerStoredVec = 1 uint8_t;
    static constexpr int kBitsStoredVec = (sizeof_bits<Element>::value < 8)
        ? cutlass::lcm(sizeof_bits<Element>::value, 8) : sizeof_bits<Element>::value;
    static constexpr int kElementsPerStoredVec = kBitsStoredVec / sizeof_bits<Element>::value;
    static constexpr int kNumStoragePerStoredVec = kBitsStoredVec / (sizeof(Element) * 8);

private:
    TensorCoord extent_;  // Extent of tensor in logical dimensions
    Layout layout_;       // Layout object
    
    // Host-side memory allocation. Avoid the std::vector<bool> specialization
    std::vector<std::conditional_t<std::is_same_v<Element,bool>, uint8_t, Element>> host_;
    // Device-side memory. using allocation = cutlass::DeviceAllocation<T>
    device_memory::allocation<Element> device_;

public:
    // Constructs a tensor given an extent and layout
    HostTensor(TensorCoord const &extent, Layout const &layout, bool device_backed = true) {
        this->reset(extent, layout, device_backed);
    }
    
    // Updates the extent and layout of the HostTensor. Allocates memory according to the new extent and layout.
    void reset(TensorCoord const &extent, Layout const &layout, bool device_backed_ = true) {                        
        extent_ = extent;
        layout_ = layout;
        this->reserve(size_t(layout_.capacity(extent_)), device_backed_);
    }
    
    // Resizes internal memory allocations without affecting layout or extent
    // @param count          : size of tensor in elements
    // @param device_backed_ : if true, device memory is also allocated
    void reserve(size_t count, bool device_backed_ = true) {
        device_.reset();
        host_.clear();
        count = (count + kElementsPerStoredVec - 1) / kElementsPerStoredVec * kNumStoragePerStoredVec;
        host_.resize(count);
        // Allocate memory
        Element* device_memory = nullptr;
        if (device_backed_) { device_memory = device_memory::allocate<Element>(count); }
        device_.reset(device_memory, device_backed_ ? count : 0);
    }
```

一个示例如下所示，使用单精度列主序存储一个二维矩阵张量，并获得该矩阵的主机内存地址指针与设备内存地址指针，及其TensorRef和TensorView对象。

```c++
void demo_tensor() {
    int rows = 128;
    int columns = 96;
    cutlass::HostTensor<float, cutlass::layout::ColumnMajor> tensor({rows, columns});
    float *host_ptr = tensor.host_data();
    cutlass::TensorRef<float, cutlass::layout::ColumnMajor> host_ref = tensor.host_ref();
    cutlass::TensorView<float, cutlass::layout::ColumnMajor> host_view = tensor.host_view();
    float *device_ptr = tensor.device_data();
    cutlass::TensorRef<float, cutlass::layout::ColumnMajor> device_ref = tensor.device_ref();
    cutlass::TensorView<float, cutlass::layout::ColumnMajor> device_view = tensor.device_view();
}
```

在使用HostTensor<T,Layout>模板类时，应用程序需要保证主机内存中数据与设备内存中数据的同步，该模板类提供若干同步方法，如下所示。

```c++
template <typename Element, typename Layout>
class HostTensor {
private:
    std::vector<std::conditional_t<std::is_same_v<Element,bool>, uint8_t, Element>> host_;
    device_memory::allocation<Element> device_;

public:
    // Returns true if device memory is allocated
    bool device_backed() const { return (device_.get() == nullptr) ? false : true; }
    
    // Copies data from device to host
    void sync_host() {
        if (device_backed()) { device_memory::copy_to_host(host_data(), device_data(), size()); }
    }
    
    // Copies data from host to device
    void sync_device() {
        if (device_backed()) { device_memory::copy_to_device(device_data(), host_data(), size()); }
    }
};
```

在cutlass/util/tensor_view_io.h头文件中，对位于主机端上的TensorView对象重载了流输出运算符operator<<()，以方便打印元素数据，如下所示。

```c++
void demo_print() {
    int rows = 2;
    int columns = 3;
    cutlass::HostTensor<int, cutlass::layout::ColumnMajorInterleaved<2>> tensor({rows, columns});
    cutlass::TensorView<int, cutlass::layout::ColumnMajorInterleaved<2>> host_view = tensor.host_view();
    int val = 1;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            host_view[{i, j}] = val++;
        }
    }
    std::cout << tensor.host_view() << std::endl;
    int *host_ptr = tensor.host_data();
    for (int i = 0; i < tensor.capacity(); printf("%d ", host_ptr[i++]));
    printf("\n");
}
```

```shell
1, 2, 3,
4, 5, 6
1 2 4 5 3 0 6 0 
```

在cutlass/util/reference/host/tensor_fill.h头文件和cutlass/util/reference/device/tensor_fill.h头文件中，提供用于初始化TensorView对象的各种辅助方法，可对主机内存对象或设备内存对象进行指定模式的初始化，包括填充指定值、正则随机初始化、高斯随机初始化等。

```c++
void demo_fill() {
    int rows = 128;
    int columns = 96;
    cutlass::HostTensor<float, cutlass::layout::ColumnMajor> tensor({rows, columns});

    // 填充给定值
    float x = 3.14159f;
    cutlass::reference::host::TensorFill(tensor.host_view(), x);
    cutlass::reference::device::TensorFill(tensor.device_view(), x);

    uint64_t seed = 0x2024;
    int non_zero_bits = 2;

    // 正则随机初始化
    float maximum = 4;
    float minimum = -4;
    cutlass::reference::host::TensorFillRandomUniform(tensor.host_view(), seed, maximum, minimum, non_zero_bits);
    cutlass::reference::device::TensorFillRandomUniform(tensor.device_view(), seed, maximum, minimum, non_zero_bits);

    // 高斯初始化
    float mean = 0.5;
    float stddev = 2.0;
    cutlass::reference::host::TensorFillRandomGaussian(tensor.host_view(), seed, mean, stddev, non_zero_bits);
    cutlass::reference::device::TensorFillRandomGaussian(tensor.device_view(), seed, mean, stddev, non_zero_bits);
}
```

其中，随机初始化方法都可以接受一个non_zero_bits参数，用于指定二进制小数部分至少多少位数字不为零值。

在cutlass/util/reference/host/gemm.h头文件中，提供主机端GEMM通用矩阵乘法计算的实现，一个使用示例如下所示。

```c++
void demo_host_gemm() {
    int M = 64, N = 32, K = 16;
    cutlass::half_t alpha = 1.5_hf, beta = -1.25_hf;

    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> A({M, K});
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> B({K, N});
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> C({M, N});
    uint64_t seed = 0x2024;
    cutlass::half_t mean = 0.5_hf;
    cutlass::half_t stddev = 2.0_hf;
    cutlass::reference::host::TensorFillRandomGaussian(A.host_view(), seed, mean, stddev);
    cutlass::reference::host::TensorFillRandomGaussian(B.host_view(), seed, mean, stddev);
    cutlass::reference::host::TensorFillRandomGaussian(C.host_view(), seed, mean, stddev);

    cutlass::reference::host::Gemm<
        cutlass::half_t, cutlass::layout::ColumnMajor,
        cutlass::half_t, cutlass::layout::ColumnMajor,
        cutlass::half_t, cutlass::layout::ColumnMajor,
        cutlass::half_t, cutlass::half_t
    > gemm_op;

    gemm_op({M, N, K}, alpha, A.host_view(), B.host_view(), beta, C.host_view());
    std::cout << C.host_view() << std::endl;
}
```

在cutlass/util/reference/host/tensor_compare.h头文件中，提供主机端的TensorEquals()方法，用于判断两个主机端的HostTensor对象是否相等。

```c++
bool same = cutlass::reference::host::TensorEquals(tensor1.host_view(), tensor2.host_view());
```

在cutlass/util/reference/host/tensor_elementwise.h头文件中，提供主机端内存中TensorView对象的逐元素操作，例如TensorAdd()函数、TensorSub()函数、TensorMul()函数、TensorDiv()函数、TensorModulus()函数，以及自定义的TensorFuncBinaryOp结构体等。

# Common Concept

在项目顶层的cutlass目录中，提供CUTLASS在各个硬件层级对GEMM的实现代码，以及所需要的辅助类型，如下所示。

```shell
cutlass  # CUTLASS Template Library
├── *          # Fundamental types
├── layout     # Layout type for matrix, tensor and other mathematical Object in memory
├── detail     # Helper for macros and others
├── platform   # Platform features
├── arch       # Architecture features (including instruction implementation)
├── gemm       # GEneral Matrix Multiply computations
│   ├── device       # Launch kernels
│   ├── kernel       # Kernels
│   ├── threadblock  # Cta Tile
│   ├── warp         # Warp Tile
│   └── thread       # Thread Tile
├── transform  # Code specialized for layout, type, and domain transformations
├── epilogue   # Epilogue rearranges result to canonical layouts, and supports conversion and reduction operations
├── reduction  # Reduction kernels
└── conv       # Implict GEMM for Convolution
```

## Fundamental Type

CUTLASS沿用C++标准库的基本类型，可用于主机端代码与设备端代码，并且与设备的计算能力无关。此外，CUTLASS还额外定义了一些数值类型。需要注意的是，一些类型或函数在较低的架构上并不支持，例如hrsqrt函数，可在编译时使用-arch=sm_70指定目标架构。

在cutlass/numeric_types.h头文件中，提供一些特殊数值类型的定义，如下所示。

| 数值类型   | 字面量后缀 | 描述                                   |
| ---------- | ---------- | -------------------------------------- |
| half_t     | _hf        | IEEE半精度浮点数；尾数10位，指数5位    |
| bfloat16_t | _bf16      | BFloat16类型；尾数7位，指数8位         |
| tfloat32_t | _tf32      | Tensor Float 32类型；尾数10位，指数8位 |
| int4_t     | _s4        | 有符号4位整型                          |
| uint4_t    | _u4        | 无符号4位整型                          |
| bin1_t     | _b1        | 一位二进制位                           |

```c++
template <int Bits, bool Signed = true>
struct integer_subbyte {
    using Storage = uint8_t;  // Storage type
    static constexpr Storage bits_mask_ = Storage(Storage(-1) >> (8 - Bits));       // bitmask for truncation
    static constexpr Storage sign_mask_ = Storage((Signed ? 1 : 0) << (Bits - 1));  // bitmask for the sign bit
    Storage storage;
}

using int4b_t = integer_subbyte<4, true>;    // 4-bit Integer type
using uint4b_t = integer_subbyte<4, false>;  // 4-bit Unsigned integer type
using bin1_t = bool;                         // 1-bit binary type
```

在cutlass/numeric_size.h头文件中，提供辅助模板sizeof_bits\<T\>的定义，用于获取一个类型所占用的二进制位的数目。

```c++
// defines the size of an element in bits
template <typename T>
struct sizeof_bits { static constexpr int value = int(sizeof(T) * 8); };

template <int Bits, bool Signed>
struct sizeof_bits<integer_subbyte<Bits,Signed>> { static constexpr int value = Bits; };

template <>
struct sizeof_bits<bin1_t> { static constexpr int value = 1; };

template <>
struct sizeof_bits<void> { static constexpr int value = 0; };
```

## Macro and Platform

在cutlass/cutlass.h头文件中，提供一个枚举类型Status的定义，用于标识CUTLASS库的执行状态，此外还提供一些常量定义。

```c++
/// Status code returned by CUTLASS operations
enum class Status {
    kSuccess,                  ///< Operation was successful.
    kErrorMisalignedOperand,   ///< operands fail alignment requirements.
    kErrorInvalidDataType,     ///< DataType fails requirement.
    kErrorInvalidLayout,       ///< Layout fails alignment requirement.
    kErrorInvalidProblem,      ///< Specified problem size is not supported by operator.
    kErrorNotSupported,        ///< Operation is not supported on current device.
    kErrorWorkspaceNull,       ///< The given workspace is null when it is required to be non-null.
    kErrorInternal,            ///< An error within CUTLASS occurred.
    kErrorArchMismatch,        ///< CUTLASS runs on a device that it was not compiled for.
    kErrorInsufficientDriver,  ///< CUTLASS runs with a driver that is too old.
    kErrorMemoryAllocation,    ///< Kernel launch failed due to insufficient device memory.
    kInvalid                   ///< Status is unspecified.
};

static const int NumThreadsPerWarp = 32;
static const int NumThreadsPerWarpGroup = 128;
static const int NumWarpsPerWarpGroup = NumThreadsPerWarpGroup / NumThreadsPerWarp;
static const int NumThreadsPerHalfWarp = NumThreadsPerWarp / 2;
static const int NumThreadsPerQuad = 4;
static const int NumThreadsPerQuadPair = NumThreadsPerQuad * 2;
```

在cutlass/detail/helper_macros.hpp头文件中，提供一些辅助宏定义，如下所示。

```c++
#define CUTLASS_HOST_DEVICE __forceinline__ __device__ __host__
#define CUTLASS_DEVICE      __forceinline__ __device__
#define CUTLASS_HOST        __host__
#define CUTLASS_GLOBAL      __global__ static
```

在cutlass/platform/platform.h头文件中，提供一些模板元的定义，如下所示。这些模板元与C++标准库中的类似，用于在编译过程中对一些类型做出假设。

```c++
template<typename _Tp, _Tp __v>
struct integral_constant {
    static constexpr _Tp value = __v;
    typedef integral_constant<_Tp, __v> type;
    typedef _Tp value_type;
    constexpr   operator value_type() const noexcept { return value; }
    constexpr value_type operator()() const noexcept { return value; }
};

using true_type  = integral_constant<bool, true>;   // compile-time boolean with true value
using false_type = integral_constant<bool, false>;  // compile-time boolean with false value

template<typename _Tp, typename _Up>
struct is_same : public false_type {};

template<typename _Tp>               
struct is_same<_Tp, _Tp> : public true_type {};

template<bool, typename _Tp = void>
struct enable_if {};

template<typename _Tp>
struct enable_if<true, _Tp> { typedef _Tp type; };                        // Partial specialization for true

template<bool _Cond, typename _Iftrue, typename _Iffalse>
struct conditional { typedef _Iftrue type; };

template<typename _Iftrue, typename _Iffalse>
struct conditional<false, _Iftrue, _Iffalse> { typedef _Iffalse type; };  // Partial specialization for false
```

## Array

模板类Array<T,N>是一个固定长度的数组，存储N个T类型的元素，元素的数据类型可以是小于一个字节的亚类型（Sub-Type），亚类型的元素之间紧凑存储。需要注意的是，对于亚类型元素而言，在使用sizeof(Array<T,N>)运算符时，其返回结果仍然是以字节为单位的，且最小是一个字节。

在CUTLASS中实例化Array<T,N>对象时，一个线程通常会使用多个寄存器来存储数组元素，并且常被用于表示一个Fragment矩阵片段。当线程的寄存器溢出时，则会使用线程的局部内存来存储数组元素。当使用\_\_shared\_\_修饰符时，则会使用共享内存来存储数组元素。

### Array<T,N>

在cutlass/array.h头文件中，提供Array<T,N>的定义，如下所示。

```c++
/// Statically sized array for any data type
template <typename T, int N, bool RegisterSized = sizeof_bits<T>::value >= 32>
struct Array;

/// Statically sized array for any data type
template<typename T, int N>
struct Array<T, N, true> {
    using Storage = T;  /// Storage type
    using Element = T;  /// Element type
    static constexpr size_t kStorageElements = N;  /// Number of storage elements
    static constexpr size_t kElements = N;         /// Number of logical elements
    
    Storage storage[kElements];  /// Internal storage
    
    typedef T value_type;
    typedef size_t size_type;
    typedef value_type& reference;
    typedef value_type* pointer;

    reference operator[](size_type pos) { return reinterpret_cast<reference>(storage[pos]); }
    pointer data()                      { return reinterpret_cast<pointer>(storage); }
    constexpr size_type size() const    { return kElements; }
};
```

AlignedArray继承自Array<T,N>模板类，但AlignedArray<T,N,Align>可以指定其内部的用于存储元素的内存空间按照Alignment字节进行对齐。

在cutlass/array.h头文件中，提供AlignedArray<T,N,Align>的定义，如下所示。

```c++
/// Aligned array type
template<typename T, int N, int Alignment = (sizeof_bits<T>::value * N + 7) / 8>
class alignas(Alignment) AlignedArray: public Array<T,N> { };
```

### AlignedBuffer

模板类AlignedBuffer<T,N,Align>是一个固定长度的缓冲区，存储N个T类型的元素，并且内部的用于存储元素的内存空间按照Align字节进行对齐。AlignedBuffer常用于获取一段按照指定字节对齐的连续内存空间，例如设备的全局内存或共享内存，以用于向量化操作。

在cutlass/aligned_buffer.h头文件中，提供AlignedBuffer<T,N,Align>的定义，如下所示。

```c++
/// Modifies semantics of cutlass::Array<> to provide guaranteed alignment. 
template <typename T, int N, int Align = 16>
struct AlignedBuffer {
public:
    using Storage = uint8_t;          /// Internal storage type
    static int const kCount = N;      /// Number of logical elements held in buffer
    static int const kAlign = Align;  /// Alignment requirement in bytes
    static int const kBytes = (sizeof_bits<T>::value * N + 7) / 8;  /// Number of storage elements

    typedef T value_type;
    typedef size_t size_type;
    typedef value_type& reference;
    typedef value_type* pointer;

private:
    alignas(Align) Storage storage[kBytes];  /// Internal storage

public:
    pointer data()                   { return reinterpret_cast<pointer>(storage); }
    constexpr size_type size() const { return kCount; }
};
```

如下一个示例，在共享内存上获取一段连续的内存空间，元素是half_t类型。

```c++
__global__ void demo_aligned_buffer_kernel() {
    const int kN = 1024;
    __shared__ AlignedBuffer<half_t, kN> smem_buffer;
    AlignedArray<half_t, 8> *ptr = reinterpret_cast<AlignedArray<half_t, 8>*>(smem_buffer.data());
    AlignedArray<half_t, 8> value = ptr[threadIdx.x];  // 128-bit shared memory load
}
```

## Type Caster

模板类NumericConverter<T,S>是一个类型转换器，用于将一个一个对象从S类型转换成T类型，该类型转换的过程会尽可能地在目标架构上使用硬件加速。模板类NumericArrayConverter<T,S,N>是一个数组的类型转换器，用于将一个数组中的所有N个元素从S类型转换成T类型。

### NumericConverter

在cutlass/numeric_conversion.h头文件中，提供一个枚举类型FloatRoundStyle的定义，如下所示。该枚举类的值用于标识转换过程中的浮点数舍入方式。

```c++
/// Floating-point rounding style similare to Standard Library's formats but supporting additional rounding options.
enum class FloatRoundStyle {
    round_indeterminate,         ///< rounding mode unknown
    round_toward_zero,           ///< round toward zero
    round_to_nearest,            ///< round to nearest even
    round_to_nearest_satfinite,  ///< round to nearest even, capping value to min and max of destination type
    round_toward_infinity,       ///< round toward infinity
    round_toward_neg_infinity,   ///< round toward negative infinity
    round_half_ulp_truncate,     ///< add 0.5ulp to integer representation then round toward zero
    round_half_ulp_trunc_dntz    ///< like round_half_ulp_truncate, except denorms are rounded *toward* zero
};
```

在cutlass/numeric_conversion.h头文件中，提供NumericConverter<T,S>的定义，如下所示。

```c++
template <typename T, typename S, FloatRoundStyle Round = FloatRoundStyle::round_to_nearest>
struct NumericConverter {
    using result_type = T;
    using source_type = S;
    static FloatRoundStyle const round_style = Round;

    static result_type convert(source_type const & s)  { return static_cast<result_type>(s); }
    result_type operator()(source_type const &s) const { return convert(s); }
};
```

### NumericArrayConverter

在cutlass/numeric_conversion.h头文件中，提供NumericArrayConverter<T,S,N>的定义，如下所示。

```c++
/// Conversion operator for Array
template <
    typename T, typename S, int N,
    FloatRoundStyle Round = FloatRoundStyle::round_to_nearest,
    typename Transform = cutlass::transform::thread::UnaryTransform::Identity
>
struct NumericArrayConverter {
    using result_type = Array<T, N>;
    using source_type = Array<S, N>;
    static FloatRoundStyle const round_style = Round;

    static result_type convert(source_type const & s) {
        result_type result;
        NumericConverter<T, S, Round> convert_;

        for (int i = 0; i < N; ++i) {
            if (platform::is_same<Transform, cutlass::transform::thread::UnaryTransform::Identity>::value) {
                result[i] = convert_(s[i]);
            } else {
                // platform::is_same<Transform, cutlass::transform::thread::UnaryTransform::Conjugate>::value == true
                result[i] = conj(convert_(s[i]));
            }
        }
        return result;
    }

    result_type operator()(source_type const &s) const {
        return convert(s);
    }
};
```

如下一个示例，将一个int类型的数组，转换为一个int8_t类型的数组。

```c++
void demo_converter() {
    int const kN = 16;
    Array<int8_t, kN> destination;
    Array<int, kN> source;
    NumericArrayConverter<int8_t, int, kN> convert;
    destination = convert(source);
}
```

## Coordinate

模板类Coord\<Rank\>是一个通用的逻辑坐标（Logical Coordinate），它具有Rank个维度轴，每个维度轴上可以使用一个索引坐标来确定该维度轴上的元素位置。多个Coord\<Rank\>坐标对象之间支持四则运算，这种坐标之间的加减乘除运算是逐元素（element-wise）的。

在CUTLASS中，常用的坐标是二维的矩阵坐标MatrixCoord、三维的矩阵乘法坐标GemmCoord、四维的批量矩阵乘法坐标BatchedGemmCoord，以及四维的张量坐标Tensor4DCoord和五维的张量坐标Tensor5DCoord。这些坐标常用于在各个层级中表示某种类型的某个元素的逻辑位置。

<img src="CUTLASS模板库.assets/MatrixCoord和MatrixShape.png" style="zoom:15%;" />

### Coord\<Rank\>

在cutlass/coord.h头文件中，提供Coord\<Rank\>的定义，如下所示。

```c++
/// Statically-sized array specifying Coords within a tensor
template <
    int Rank_,                     ///< Logical rank of coordinate
    typename Index_ = int,         ///< Index type used for each dimension
    typename LongIndex_ = int64_t  ///< Long index type used for linear offsets
>
struct Coord {
public:
    static int const kRank = Rank_;  /// Number of elements in Coord
    using Index = Index_;            /// Index type used to store elements
    using LongIndex = LongIndex_;    /// Type used to represent linear offsets

private:
    Index idx[kRank];  /// Indices

public:
    /// Default ctor initializes uniformly
    explicit Coord(Index value = Index(0)) {
        for (int i = 0; i < kRank; ++i) {
            idx[i] = value;
        }
    }

    /// Constructs from an array of integers
    Coord(Index const (&_idx)[kRank]) {
        for (int i = 0; i < kRank; ++i) {
            idx[i] = _idx[i];
        }
    }

    /// Member access operator
    Index& operator[](int dim) { return idx[dim]; }
    
    /// Access via index; may limit unrolling potential
    Index& at(int dim) { return idx[dim]; }

    /// Element-wise operators
    Coord operator+(Coord const& b) const { Coord c; for (int i = 0; i < kRank; ++i) { c.idx[i] = idx[i] + b.idx[i]; } return c; }
    Coord operator-(Coord const& b) const { Coord c; for (int i = 0; i < kRank; ++i) { c.idx[i] = idx[i] - b.idx[i]; } return c; }
    Coord operator*(Coord const& b) const { Coord c; for (int i = 0; i < kRank; ++i) { c.idx[i] = idx[i] * b.idx[i]; } return c; }
    Coord operator/(Coord const& b) const { Coord c; for (int i = 0; i < kRank; ++i) { c.idx[i] = idx[i] / b.idx[i]; } return c; }

    /// In-place element-wise operators
    Coord& operator+=(Coord const& b) { for (int i = 0; i < kRank; ++i) { idx[i] += b.idx[i]; } return *this; }
    Coord& operator-=(Coord const& b) { for (int i = 0; i < kRank; ++i) { idx[i] -= b.idx[i]; } return *this; }
    Coord& operator*=(Coord const& b) { for (int i = 0; i < kRank; ++i) { idx[i] *= b.idx[i]; } return *this; }
    Coord& operator/=(Coord const& b) { for (int i = 0; i < kRank; ++i) { idx[i] /= b.idx[i]; } return *this; }

    /// Compare operators
    bool operator< (Coord const &b) const { for (int i = 0; i < kRank; ++i) { if (!(idx[i] <  b[i])) { return false; } } return true; }
    bool operator<=(Coord const &b) const { for (int i = 0; i < kRank; ++i) { if (!(idx[i] <= b[i])) { return false; } } return true; }
    bool operator> (Coord const &b) const { return !(*this <= b); }
    bool operator>=(Coord const &b) const { return !(*this <  b); }
};
```

在cutlass/coord.h头文件中，还提供关于Coord\<Rank\>的辅助函数，如下所示。

```c++
/// Scalar multiplication
template <int Rank, typename Index>
Coord<Rank, Index> operator*(Index s, Coord<Rank, Index> coord) {
    for (int i = 0; i < Rank; ++i) { coord[i] *= s; }
    return coord;
}
/// Scalar multiplication
template <int Rank, typename Index>
Coord<Rank, Index> operator*(Coord<Rank, Index> coord, Index s) {
    for (int i = 0; i < Rank; ++i) { coord[i] *= s; }
    return coord;
}

/// Scalar division
template <int Rank, typename Index>
Coord<Rank, Index> operator/(Index s, Coord<Rank, Index> coord) {
    for (int i = 0; i < Rank; ++i) { coord[i] = s / coord[i]; }
    return coord;
}
/// Scalar division
template <int Rank, typename Index>
Coord<Rank, Index> operator/(Coord<Rank, Index> coord, Index s) {
    for (int i = 0; i < Rank; ++i) { coord[i] = coord[i] / s; }
    return coord;
}

/// Helper to make a 2-element coordinate
template <typename T>
Coord<2, T> make_Coord(T _0, T _1) {
    T values[2] = { _0, _1 };
    return Coord<2, T>(values);
}

/// Helper to make a 3-element coordinate
template <typename T>
Coord<3, T> make_Coord(T _0, T _1, T _2) {
    T values[3] = { _0, _1, _2 };
    return Coord<3, T>(values);
}

/// Helper to make a 4-element coordinate
template <typename T>
Coord<4, T> make_Coord(T _0, T _1, T _2, T _3) {
    T values[4] = { _0, _1, _2, _3 };
    return Coord<4, T>(values);
}

/// Helper to make a 5-element coordinate
template <typename T>
Coord<5, T> make_Coord(T _0, T _1, T _2, T _3, T _4) {
    T values[5] = { _0, _1, _2, _3, _4 };
    return Coord<5, T>(values);
}
```

### MatrixCoord

在cutlass/matrix_coord.h头文件中，提供MatrixCoord的定义，如下所示。

```c++
/// MatrixCoord wraps Coord<2, int> to provide a helper for accessing named dimensions.
/// Classes expecting a coordinate in the rank=2 index space of a matrix should use MatrixCoord.
struct MatrixCoord : public Coord<2, int> {
public:
    using Index = int;                           /// Integer-valued index
    using Base = Coord<2, Index>;                /// Base type is a Coord of rank=2
    using LongIndex = typename Base::LongIndex;  /// LongIndex type

private:
    /// Syntax = (row, column)
    static int const kRow = 0;     /// Rows dimension
    static int const kColumn = 1;  /// Columns dimension

public:
    /// Default ctor
    MatrixCoord() {}

    /// Helper to construct from a row and column
    MatrixCoord(Index row, Index column) : Base(make_Coord(row, column)) {}

    /// Returns the row of the coordinate
    Index & row() { return this->at(kRow); }

    /// Returns the column of the coordinate
    Index & column() { return this->at(kColumn); }
};
```

### GemmCoord

在cutlass/gemm_coord.h头文件中，提供GemmCoord和BatchedGemmCoord的定义，如下所示。

```c++
/// GemmCoord is a structure derived from Coord<3> that specifies a location within the coordinate space of a GEMM problem.
struct GemmCoord : public Coord<3, int> {
    typedef int Index;             /// Integer-valued index
    typedef Coord<3, Index> Base;  /// Base type is a Coord of rank=3

    /// Syntax = (m, n, k)
    static int const kM = 0;  /// GEMM M dimension - rows of the output C matrix
    static int const kN = 1;  /// GEMM N dimension - columns of the output C matrix
    static int const kK = 2;  /// GEMM K dimension - inner dimension of the GEMM problem

    /// Default ctor
    GemmCoord() {}

    /// Helper to construct from a K, N, M, batch variables
    GemmCoord(Index m, Index n, Index k) : Base(make_Coord(m, n, k)) {}

    /// Returns reference to the GEMM M coordinate
    Index & m() { return this->at(kM); }

    /// Returns reference to the GEMM N coordinate
    Index & n() { return this->at(kN); }

    /// Returns reference to the GEMM K coordinate
    Index & k() { return this->at(kK); }
};

/// BatchedGemmCoord is a structure derived from Coord<4> that specifies a location within the coordinate space of a batched GEMM problem.
struct BatchedGemmCoord : public Coord<4, int> {
    typedef int Index;             /// Integer-valued index
    typedef Coord<4, Index> Base;  /// Base type is a Coord of rank=4

    /// Syntax = (m, n, k, batch)
    static int const kM = 0;      /// GEMM M dimension - rows of the output C matrix
    static int const kN = 1;      /// GEMM N dimension - columns of the output C matrix
    static int const kK = 2;      /// GEMM K dimension - inner dimension of the GEMM problem
    static int const kBatch = 3;  /// GEMM Batch dimension - inner dimension of the GEMM problem

    /// Default ctor
    BatchedGemmCoord() {}

    /// Helper to construct from a K, N, M, and batch variables
    BatchedGemmCoord(Index m, Index n, Index k, Index b) : Base(make_Coord(m, n, k, b)) {}

    /// Returns reference to the GEMM M coordinate
    Index & m() { return this->at(kM); }

    /// Returns reference to the GEMM N coordinate
    Index & n() { return this->at(kN); }

    /// Returns reference to the GEMM K coordinate
    Index & k() { return this->at(kK); }

    /// Returns reference to the GEMM batch coordinate
    Index & batch() { return this->at(kBatch); }
};
```

### Tensor[4D|5D]Coord

在cutlass/tensor_coord.h头文件中，提供Tensor4DCoord和Tensor5DCoord的定义，如下所示。

```c++
/// Defines a canonical 4D coordinate used by tensor operations.
struct Tensor4DCoord : public Coord<4> {
    using Base = Coord<4>;                       /// Base class
    using Index = typename Base::Index;          /// Index type
    using LongIndex = typename Base::LongIndex;  /// LongIndex type

    /// Syntax = (n, h, w, c)
    static int const kN = 0;  /// Batch dimension
    static int const kH = 1;  /// Height dimension
    static int const kW = 2;  /// Width dimension
    static int const kC = 3;  /// Channels dimension

    /// Default ctor
    Tensor4DCoord() {}

    /// Helper to construct from N, H, W, and C.
    Tensor4DCoord(Index n, Index h, Index w, Index c) : Base(make_Coord(n, h, w, c)) {}

    /// Returns the batch of the coordinate
    Index & n() { return this->at(kN); }

    /// Returns the row of the coordinate
    Index & h() { return this->at(kH); }

    /// Returns the column of the coordinate
    Index & w() { return this->at(kW); }

    /// Returns the channel of the coordinate
    Index & c() { return this->at(kC); }
};

/// Defines a canonical 5D coordinate used by tensor operations.
struct Tensor5DCoord : public Coord<5> {
    using Base = Coord<5>;                       /// Base class
    using Index = typename Base::Index;          /// Index type
    using LongIndex = typename Base::LongIndex;  /// LongIndex type

    /// Syntax = (n, d, h, w, c)
    static int const kN = 0;  /// Batch dimension
    static int const kD = 1;  /// Depth dimension
    static int const kH = 2;  /// Height dimension
    static int const kW = 3;  /// Width dimension
    static int const kC = 4;  /// Channels dimension

    /// Default ctor
    Tensor5DCoord() {}

    /// Helper to construct from N, D, H, W, and C.
    Tensor5DCoord(Index n, Index d, Index h, Index w, Index c) : Base(make_Coord(n, d, h, w, c)) {}

    /// Returns the batch of the coordinate
    Index & n() { return this->at(kN); }

    /// Returns the batch of the coordinate
    Index & d() { return this->at(kD); }

    /// Returns the row of the coordinate
    Index & h() { return this->at(kH); }

    /// Returns the column of the coordinate
    Index & w() { return this->at(kW); }

    /// Returns the channel of the coordinate
    Index & c() { return this->at(kC); }
};
```

## Shape

在CUTLASS中，与坐标类似的概念是形状（Shape），模板类MatrixShape<Row,Column>用于表示一个二维矩阵的维数形状，模板类GemmShape<M,N,K>用于表示一个三维矩阵乘法的维数形状。实际上，也可以直接使用一个Coord\<Rank\>来表示形状，这能起到同样的作用，但为了代码可读性，还是提供常用形状MatrixShape和GemmShape的类型定义。

### MatrixShape

在cutlass/matrix_shape.h头文件中，提供MatrixShape<Row,Column>的定义，如下所示。

```c++
/// Describes the size of a matrix tile
template <
    int Row_,    ///< rows of a matrix
    int Column_  ///< columns of a matrix
>
struct MatrixShape {
    static int const kRow = Row_;              ///< rows of a matrix
    static int const kColumn = Column_;        ///< columns of a matrix
    static int const kCount = Row_ * Column_;  ///< total number of elements in a matrix

    /// Returns a Coord object
    static Coord<2> toCoord() {
        return make_Coord(kRow, kColumn);
    }
};
```

### GemmShape

在cutlass/gemm_coord.h头文件中，提供GemmShape<M,N,K>的定义，如下所示。

```c++
/// Shape of a matrix multiply-add operation
template <
    int M = 1,  /// Rows of matrix product
    int N = 1,  /// Columns of matrix product
    int K = 1   /// Inner dimension of matrix product
>
struct GemmShape {
    static int const kM = M;
    static int const kN = N;
    static int const kK = K;
    static int const kMN = M * N;
    static int const kMK = M * K;
    static int const kKN = N * K;
    static int const kMNK = M * N * K;
    static int const kCount = kMNK;

    /// Returns a Coord object
    static Coord<3> toCoord() {
        return make_Coord(kM, kN, kK);
    }
};

/// Type alias of the transpose of a GemmShape
template <
    typename Shape  /// concept: GemmShape
>
using GemmShapeTranspose = GemmShape<Shape::kN, Shape::kM, Shape::kK>;
```

## Layout

布局（Layout）是一个用于将元素坐标转换为偏移量的映射（Mapping），它将一个逻辑坐标映射为一个偏移量（Offset）。偏移值是指，多维数组的某个元素的存储位置，与第一个元素的存储位置之间的间距。一个布局由存储顺序（Storage Order）和跨步（Stride）确定，存储顺序是指多个维度轴存储时的先后顺序，跨步是指两个相应元素的存储位置之间的间距。

需要注意的是，在CUTLASS中，偏移和跨步，都是以元素类型Element为单位的，而不是以字节为单位的。

CUTLASS常用的布局是行主序布局RowMajor、列主序布局ColumnMajor、行主序交错布局RowMajorInterleaved、列主序交错布局ColumnMajorInterleaved。这些布局常用于在各个层级中将某种类型的某个元素的逻辑坐标，转换成该元素在内存空间中的偏移量。

![](CUTLASS模板库.assets/MatrixLayout.png)

### RowMajor

在cutlass/layout/matrix.h头文件中，提供RowMajor的定义，如下所示。

```c++
/// Mapping function for row-major matrices.
class RowMajor {
public:
    static int const kRank = 2;        /// Logical rank of tensor
    static int const kStrideRank = 1;  /// Rank of stride vector

    using Index = int32_t;                         /// Index type used for coordinates
    using LongIndex = int64_t;                     /// Long index type used for offsets
    using TensorCoord = MatrixCoord;               /// Logical coordinate
    using Stride = Coord<kStrideRank, LongIndex>;  /// Stride vector

private:
    Stride stride_;  /// Stride data member

public:
    /// Constructor
    RowMajor(Stride stride) : stride_(stride) {}

    /// Constructor
    RowMajor(LongIndex ldm = 0) : stride_(ldm) {}

    /// Helper returns a layout to a tightly packed tensor
    static RowMajor packed(MatrixCoord const &extent) {
        return RowMajor(extent.column());
    }

    /// Returns the offset of a coordinate in linear memory.
    /// Assumes coordinate has convention (row, column)
    LongIndex operator()(MatrixCoord const &coord) const {
        return LongIndex(coord.row()) * LongIndex(stride_[0]) + coord.column();
    }

    /// Inverse of layout function, mapping linear offset to logical coordinate
    MatrixCoord inverse(LongIndex offset) const {
        return MatrixCoord(Index(offset / stride_[0]), Index(offset % stride_[0]));
    }

    /// Compute the number of contiguous elements needed to store a tensor with the given size
    LongIndex capacity(MatrixCoord const &extent) const {
        return LongIndex(extent.row()) * LongIndex(stride_[0]);
    }
};
```

### ColumnMajor

在cutlass/layout/matrix.h头文件中，提供ColumnMajor的定义，如下所示。

```c++
/// Mapping function for column-major matrices.
class ColumnMajor {
public:
    static int const kRank = 2;        /// Logical rank of tensor
    static int const kStrideRank = 1;  /// Rank of stride vector

    using Index = int32_t;                         /// Index type used for coordinates
    using LongIndex = int64_t;                     /// Long index type used for offsets
    using TensorCoord = MatrixCoord;               /// Logical coordinate
    using Stride = Coord<kStrideRank, LongIndex>;  /// Stride vector

private:
    Stride stride_;  /// Stride data member

public:
    /// Constructor
    ColumnMajor(Stride stride) : stride_(stride) {}

    /// Constructor
    ColumnMajor(LongIndex ldm = 0) : stride_(ldm) {}

    /// Helper returns a layout to a tightly packed tensor
    static ColumnMajor packed(MatrixCoord const &extent) {
        return ColumnMajor(extent.row());
    }

    /// Returns the offset of a coordinate in linear memory. 
    /// Assumes coordinate has convention (row, column)
    LongIndex operator()(MatrixCoord const &coord) const {
        return LongIndex(coord.column()) * LongIndex(stride_[0]) + coord.row();
    }

    /// Inverse of layout function, mapping linear offset to logical coordinate
    MatrixCoord inverse(LongIndex offset) const {
        return MatrixCoord(Index(offset % stride_[0]), Index(offset / stride_[0]));
    }

    /// Compute the number of contiguous elements needed to store a tensor with the given size
    LongIndex capacity(MatrixCoord const &extent) const {
        return LongIndex(extent.column()) * LongIndex(stride_[0]);
    }
};
```

### RowMajorInterleaved

在cutlass/layout/matrix.h头文件中，提供RowMajorInterleaved\<Interleave\>的定义，如下所示。

```c++
/// Mapping function for interleaved matrices. 
/// Matrix is structured as row-major arrangement of fixed-size columns.
template <int Interleave>
struct RowMajorInterleaved {
public:
    static int const kRank = 2;        /// Logical rank of tensor
    static int const kStrideRank = 1;  /// Rank of stride vector

    using Index = int32_t;                         /// Index type used for coordinates
    using LongIndex = int64_t;                     /// Long index type used for offsets
    using TensorCoord = MatrixCoord;               /// Logical coordinate
    using Stride = Coord<kStrideRank, LongIndex>;  /// Stride vector

    static int const kInterleave = Interleave;  /// Size of interleaved columns

private:
    Stride stride_;  /// Stride data member

public:
    /// Constructor
    RowMajorInterleaved(Stride stride) : stride_(stride) {}

    /// Constructor
    RowMajorInterleaved(LongIndex ldm = 0) : stride_(ldm) {}

    /// Helper returns a layout to a tightly packed tensor
    static RowMajorInterleaved packed(MatrixCoord const &extent) {
        return RowMajorInterleaved(extent.column() * kInterleave);
    }

    /// Returns the offset of a coordinate in linear memory. 
    /// Assumes coordinate has convention (row, column)
    LongIndex operator()(MatrixCoord const &coord) const {
        Index row_major = coord.row() / kInterleave;
        Index row_minor = coord.row() % kInterleave;
        return LongIndex(row_major) * LongIndex(stride_[0]) + LongIndex(coord.column()) * kInterleave + row_minor;
    }

    /// Inverse of layout function, mapping linear offset to logical coordinate
    MatrixCoord inverse(LongIndex offset) const {
        Index row_major = Index(offset / stride_[0]);
        Index residual  = Index(offset % stride_[0]);
        Index column    = residual / kInterleave;
        Index row_minor = residual % kInterleave;
        return MatrixCoord(row_major * kInterleave + row_minor, column);
    }

    /// Compute the number of contiguous elements needed to store a tensor with the given size
    LongIndex capacity(MatrixCoord const &extent) const {
        return (extent.row() + kInterleave - 1) / kInterleave * stride_[0];
    }
};
```

### ColumnMajorInterleaved

在cutlass/layout/matrix.h头文件中，提供ColumnMajorInterleaved\<Interleave\>的定义，如下所示。

```c++
/// Mapping function for interleaved matrices. 
/// Matrix is structured as row-major arrangement of fixed-size columns.
template <int Interleave>
struct RowMajorInterleaved {
public:
    static int const kRank = 2;        /// Logical rank of tensor
    static int const kStrideRank = 1;  /// Rank of stride vector

    using Index = int32_t;                         /// Index type used for coordinates
    using LongIndex = int64_t;                     /// Long index type used for offsets
    using TensorCoord = MatrixCoord;               /// Logical coordinate
    using Stride = Coord<kStrideRank, LongIndex>;  /// Stride vector

    static int const kInterleave = Interleave;  /// Size of interleaved columns

private:
    Stride stride_;  /// Stride data member

public:
    /// Constructor
    RowMajorInterleaved(Stride stride) : stride_(stride) {}

    /// Constructor
    RowMajorInterleaved(LongIndex ldm = 0) : stride_(ldm) {}

    /// Helper returns a layout to a tightly packed tensor
    static RowMajorInterleaved packed(MatrixCoord const &extent) {
        return RowMajorInterleaved(extent.column() * kInterleave);
    }

    /// Returns the offset of a coordinate in linear memory. 
    /// Assumes coordinate has convention (row, column)
    LongIndex operator()(MatrixCoord const &coord) const {
        Index row_major = coord.row() / kInterleave;
        Index row_minor = coord.row() % kInterleave;
        return LongIndex(row_major) * LongIndex(stride_[0]) + LongIndex(coord.column()) * kInterleave + row_minor;
    }

    /// Inverse of layout function, mapping linear offset to logical coordinate
    MatrixCoord inverse(LongIndex offset) const {
        Index row_major = Index(offset / stride_[0]);
        Index residual  = Index(offset % stride_[0]);
        Index column    = residual / kInterleave;
        Index row_minor = residual % kInterleave;
        return MatrixCoord(row_major * kInterleave + row_minor, column);
    }

    /// Compute the number of contiguous elements needed to store a tensor with the given size
    LongIndex capacity(MatrixCoord const &extent) const {
        return (extent.row() + kInterleave - 1) / kInterleave * stride_[0];
    }
};
```

## Pitch Linear Mode

连续线性模式（Pitch Linear Mode）是一种元素在内存中的组织方式，其最内层维度轴上的元素连续存储，外层维度轴上的元素以内存维度轴的维数为单位进行连续存储。在CUTLASS中，使用的是，二维的连续线性坐标PitchLinearCoord、二维的连续线性形状PitchLinearShape、二维的连续线性布局PitchLinear。

![](CUTLASS模板库.assets/PitchLinear.png)

### PitchLinearCoord

在cutlass/pitch_linear_coord.h头文件中，提供PitchLinearCoord的定义，如下所示。

```c++
/// Coordinate in pitch-linear space
struct PitchLinearCoord : public Coord<2, int> {
public:
    using Index = int;                           /// Integer-valued index
    using Base = Coord<2, Index>;                /// Base type is a Coord of rank=2
    using LongIndex = typename Base::LongIndex;  /// Long integer type

private:
    static int const kContiguous = 0;  /// Contiguous dimension
    static int const kStrided = 1;     /// Strided dimension

public:
    /// Default ctor
    PitchLinearCoord() {}

    /// Helper to construct from a row and column
    PitchLinearCoord(Index contiguous_, Index strided_) : Base(make_Coord(contiguous_, strided_)) {}

    /// Returns the contiguous dimension
    Index & contiguous() { return this->at(kContiguous); }

    /// Returns the strided dimension
    Index & strided() { return this->at(kStrided); }
};
```

### PitchLinearShape

在cutlass/pitch_linear_coord.h头文件中，提供PitchLinearShape的定义，如下所示。

```c++
/// Template defining a shape used by pitch-linear operators
template <
    int Contiguous,
    int Strided
>
struct PitchLinearShape {
    static int const kContiguous = Contiguous;
    static int const kStrided = Strided;
    static int const kCount = Contiguous * Strided;
};
```

### PitchLinear

在cutlass/layout/pitch_linear.h头文件中，提供PitchLinear的定义，如下所示。

```c++
/// Mapping function for pitch-linear memory
class PitchLinear {
public:
    static int const kRank = 2;        /// Logical rank of tensor
    static int const kStrideRank = 1;  /// Rank of stride vector

    using Index = int32_t;                         /// Index type used for coordinates
    using LongIndex = int64_t;                     /// Long index type used for offsets
    using TensorCoord = PitchLinearCoord;          /// Logical coordinate
    using Stride = Coord<kStrideRank, LongIndex>;  /// Stride vector

private:
    Stride stride_;  /// Stride data member

public:
    /// Constructor
    PitchLinear(Stride _stride) : stride_(_stride) {}

    /// Constructor
    PitchLinear(LongIndex ldm = 0) : stride_(ldm) {}

    /// Helper returns a layout to a tightly packed tensor
    static PitchLinear packed(TensorCoord const &extent) {
        return PitchLinear(extent.contiguous());
    }

    /// Returns the offset of a coordinate in linear memory. 
    /// Assumes coordinate has convention (contiguous, strided)
    LongIndex operator()(TensorCoord const &coord) const {
        return LongIndex(coord.contiguous()) + LongIndex(coord.strided()) * LongIndex(stride_[0]);
    }

    /// Returns the logical coordinate given an offset.
    TensorCoord inverse(LongIndex index) const {
        return make_Coord(TensorCoord::Index(index % stride_[0]), TensorCoord::Index(index / stride_[0]));
    }
    
    /// Returns the stride of the layout
    LongIndex & stride(int rank) {
        return stride_[rank];
    }
};
```

## TensorOp Layout

为高效利用Tensor Core硬件单元进行矩阵乘加运算，需要对矩阵A、矩阵B、矩阵C和矩阵D的元素布局方式做出规定，使其满足mma.xxx系列指令的要求。

在CUTLASS中，相关布局的基本概念是TensorOpMultiplicand布局，该布局是基于元素位数（Element Size in bits）和交叉数目（Crosswise Size in elements）来定义，适用于.b8、.b16、.b32位数的元素，并且假设所使用的内存是连续线性的PitchLinear内存。

<img src="CUTLASS模板库.assets/TensorOpMultiplicand.png" style="zoom:20%;" />

当一个PitchLinear布局的维数大于一个TensorOpMultiplicand布局的维数时，会先在Contiguous维度轴上使用多个TensorOpMultiplicand布局，并且以tile_contiguous_idx标识每个TensorOpMultiplicand布局；然后再在Strided维度轴上使用多个TensorOpMultiplicand布局。

### TensorOpMultiplicand

在cutlass/layout/tensor_op_multiplicand_sm75.h头文件中，提供TensorOpMultiplicand的定义，如下所示。

```c++
/// Template based on element size (in bits) - defined in terms of pitch-linear memory and Crosswise size (in elements).
/// This one is the base class of all Ampere/Turing fp16/bf16/int8/int4/int1 tensor core kernels. tf32 TN uses this too.
/// 通常情况下，Crosswise = platform::min(128 / sizeof(Element), ThreadblockShape::k[M|N]);
/// 于是，Crosswise的取值，对于.b8为128，对于.b16为64，对于.b32为32
template <int ElementSize, int Crosswise>
struct TensorOpMultiplicand {
    static int const kRank = 2;        /// Logical rank of tensor
    static int const kStrideRank = 1;  /// Rank of stride vector

    using Index = int32_t;                                /// Index type used for coordinates
    using LongIndex = int64_t;                            /// Long index type used for offsets
    using TensorCoord = PitchLinearCoord;                 /// Logical coordinate
    using Stride = Coord<kStrideRank, Index, LongIndex>;  /// Stride vector

    /// This layout is optimized for 128b accesses.
    /// 一次访问的粒度是128位，后文将128位的数据称为一个vector向量，通常使用形如.shared.v4.b32的向量访问
    static int const kAccessSize = 128;
    /// 一个元素的位数，例如.b8、.b16、.b32类型
    static int const kElementSize = ElementSize;
    /// 一次128位的向量访问，能访问几个元素，即，16个.b8元素，8个.b16元素，4个.b32元素
    static int const kElementsPerAccess = kAccessSize / kElementSize;
    /// 交叉数目，指定多少个元素为“一折”，如'Z'型布局中的“一折”
    static int const kCrosswise = Crosswise;

    /// Contiguous dimension of the tile shape matches one shared memory cache line - 128B.
    /// For 128bit access size, it equals to 8 accesses.
    /// 缓冲行是128字节，也是共享内存32个Bank一层的字节数，这等于8次粒度为128位的向量访问
    static int const kTileShapeContiguous = 128 / (kAccessSize / 8);
    /// Number of kblocks to store PartitionShape::kContiguous Elements.
    /// 8次粒度为128位的向量访问的元素数目，除以kCrosswise交叉数目，得到，一个128字节访问的所有元素能构成几个一折
    static int const kFactor = kTileShapeContiguous * kElementsPerAccess / kCrosswise;
    /// The strided dimension needs to be at least (WarpSize(32) / kTileShapeContiguous) = 4 for a warp to access.
    /// To ensure conflict free access, it also needs to be at least (kTileShapeContiguous / kFactor) = 8 / kFactor.
    static int const kTileShapeStride = platform::max(kTileShapeContiguous / kFactor, 32 / kTileShapeContiguous);

    /// Fundamental tile shape in units of vectors to guarantee bank conflict free shared memory load/store.
    /// For kFactor = 1, TileShape = PitchLinearShape<8, 8>;
    /// For kFactor > 1, TileShape = PitchLinearShape<8, 4>;
    using TileShape = PitchLinearShape<kTileShapeContiguous, kTileShapeStride>;

    /// Fundamental partition shape in units of vectors
    using PartitionShape = PitchLinearShape<4, 4>;
    using PartitionCount = PitchLinearShape<
        TileShape::kContiguous / PartitionShape::kContiguous, TileShape::kStrided / PartitionShape::kStrided>;
    using AccessCount = PitchLinearShape<PartitionShape::kContiguous, PartitionShape::kStrided>;

private:
    /// Stride data member. For GEMM, it equals to `kCrosswise * stage`.
    Stride stride_;

public:
    /// Constructor
    TensorOpMultiplicand(Stride stride) : stride_(stride) {}

    /// Constructor
    TensorOpMultiplicand(Index ldm = 0) : stride_(ldm) {}

    /// Helper returns a layout to a tightly packed tensor
    static TensorOpMultiplicand packed(TensorCoord const &extent) {
        return TensorOpMultiplicand(extent[0]);
    }

    /// Returns the offset of a coordinate in linear memory.
    /// Assumes coordinate has convention (contiguous, strided)
    LongIndex operator()(TensorCoord const &coord) const {
        // First, compute cIdx and sIdx of vector within source (in units of vector accesses)
        int vec_contiguous_idx = coord.contiguous() / kElementsPerAccess;
        int vec_strided_idx = coord.strided() / kFactor;

        // Compute the fundamental tile being accessed
        int tile_contiguous_idx = vec_contiguous_idx / (TileShape::kContiguous / kFactor);
        int tile_contiguous_residual = vec_contiguous_idx % (TileShape::kContiguous / kFactor)
            + ((coord.strided() % kFactor) * (TileShape::kContiguous / kFactor));
        int tile_strided_residual = vec_strided_idx % TileShape::kStrided;

        // Compute the 'partition' within the fundamental tile
        int partition_contiguous_idx = tile_contiguous_residual / PartitionShape::kContiguous;
        int partition_strided_idx = tile_strided_residual / PartitionShape::kStrided;
        int partition_contiguous_residual = tile_contiguous_residual % PartitionShape::kContiguous;
        int partition_strided_residual = tile_strided_residual % PartitionShape::kStrided;

        // Then swizzle
        int permuted_vec_contiguous_within_partition = partition_contiguous_residual ^ (partition_strided_residual % 4);
        int permuted_partition_contiguous_within_tile = partition_contiguous_idx ^ (partition_strided_idx % 2);

        // Compute final element location
        int element_contiguous = (tile_contiguous_idx * TileShape::kContiguous
            + permuted_partition_contiguous_within_tile * PartitionShape::kContiguous
            + permuted_vec_contiguous_within_partition)
            * kElementsPerAccess
            + (coord.contiguous() % kElementsPerAccess);
        int element_strided = vec_strided_idx;
        return element_contiguous + element_strided * stride_[0] * kFactor;
    }
};
```

### TensorOpMultiplicandCongruous

以TensorOpMultiplicand布局为基本（称之为基本布局），CUTLASS提出各种其它用途的相关布局。例如，TensorOpMultiplicandCongruous布局是对基本布局的包装，ColumnMajorTensorOpMultiplicandCongruous布局将列主序布局映射为基本布局，RowMajorTensorOpMultiplicandCongruous布局将行主序布局映射为基本布局。

在cutlass/layout/tensor_op_multiplicand_sm75.h头文件中，提供TensorOpMultiplicandCongruous的定义，如下所示。

```c++
/// Template based on element size (in bits) - defined in terms of pitch-linear memory and Crosswise size (in elements).
template <int ElementSize, int Crosswise>
struct TensorOpMultiplicandCongruous {
    static int const kRank = 2;        /// Logical rank of tensor
    static int const kStrideRank = 1;  /// Rank of stride vector

    using Index = int32_t;                                /// Index type used for coordinates
    using LongIndex = int64_t;                            /// Long index type used for offsets
    using TensorCoord = PitchLinearCoord;                 /// Logical coordinate
    using Stride = Coord<kStrideRank, Index, LongIndex>;  /// Stride vector
    
    /// This layout is optimized for 128b accesses
    using Base = TensorOpMultiplicand<ElementSize, Crosswise>;
    static int const kAccessSize = Base::kAccessSize;
    static int const kElementSize = Base::kElementSize;
    static int const kElementsPerAccess = Base::kElementsPerAccess;
    static int const kCrosswise = Base::kCrosswise;
    static int const kFactor = Base::kFactor;
    using TileShape = typename Base::TileShape;
    using PartitionShape = typename Base::PartitionShape;
    using PartitionCount = typename Base::PartitionCount;
    using AccessCount = typename Base::AccessCount;

private:
    Base layout_;

public:
    /// Constructor
    TensorOpMultiplicandCongruous(Stride stride) : layout_(stride) {}
    
    /// Constructor
    TensorOpMultiplicandCongruous(Index ldm = 0) : layout_(ldm) {}

    /// Helper returns a layout to a tightly packed tensor
    static TensorOpMultiplicandCongruous packed(TensorCoord const &extent) {
        return TensorOpMultiplicandCongruous(extent[0]);
    }

    /// Returns the offset of a coordinate in linear memory.
    /// Assumes coordinate has convention (contiguous, strided)
    LongIndex operator()(TensorCoord const &coord) const {
        return layout_(coord);
    }

    /// Inverse of layout function, mapping linear offset to logical coordinate
    TensorCoord inverse(LongIndex offset) const {
        PitchLinearCoord coord = layout_.inverse(offset);
        return coord;
    }
};
```

在cutlass/layout/tensor_op_multiplicand_sm75.h头文件中，提供ColumnMajorTensorOpMultiplicandCongruous的定义，如下所示。

```c++
/// Template mapping a column-major view of pitch-linear memory to TensorOpMultiplicand
template <int ElementSize, int Crosswise>
struct ColumnMajorTensorOpMultiplicandCongruous {
    static int const kRank = 2;        /// Logical rank of tensor
    static int const kStrideRank = 1;  /// Rank of stride vector

    using Index = int32_t;                                /// Index type used for coordinates
    using LongIndex = int64_t;                            /// Long index type used for offsets
    using TensorCoord = MatrixCoord;                      /// Logical coordinate
    using Stride = Coord<kStrideRank, Index, LongIndex>;  /// Stride vector

    /// This layout is optimized for 128b accesses
    using Base = TensorOpMultiplicandCongruous<ElementSize, Crosswise>;
    static int const kAccessSize = Base::kAccessSize;
    static int const kElementSize = Base::kElementSize;
    static int const kElementsPerAccess = Base::kElementsPerAccess;
    static int const kCrosswise = Base::kCrosswise;
    static int const kFactor = Base::kFactor;
    using TileShape = typename Base::TileShape;
    using PartitionShape = typename Base::PartitionShape;
    using PartitionCount = typename Base::PartitionCount;
    using AccessCount = typename Base::AccessCount;

private:
    Base layout_;

public:
    /// Constructor
    ColumnMajorTensorOpMultiplicandCongruous(Stride stride) : layout_(stride) {}

    /// Constructor
    ColumnMajorTensorOpMultiplicandCongruous(Index ldm = 0) : layout_(ldm) {}

    /// Helper returns a layout to a tightly packed tensor
    static ColumnMajorTensorOpMultiplicandCongruous packed(TensorCoord const &extent) {
        return ColumnMajorTensorOpMultiplicandCongruous(extent.row());
    }

    /// Returns the offset of a coordinate in linear memory. 
    /// Assumes coordinate has convention (contiguous, strided)
    LongIndex operator()(TensorCoord const &coord) const {
        return layout_(PitchLinearCoord(coord.row(), coord.column()));
    }

    /// Inverse of layout function, mapping linear offset to logical coordinate
    TensorCoord inverse(LongIndex offset) const {
        PitchLinearCoord coord = layout_.inverse(offset);
        return MatrixCoord(coord.contiguous(), coord.strided());
    }
};
```

在cutlass/layout/tensor_op_multiplicand_sm75.h头文件中，提供RowMajorTensorOpMultiplicandCongruous的定义，如下所示。

```c++
/// Template mapping a row-major view of pitch-linear memory to TensorOpMultiplicand
template <int ElementSize, int Crosswise>
struct RowMajorTensorOpMultiplicandCongruous {
    static int const kRank = 2;        /// Logical rank of tensor
    static int const kStrideRank = 1;  /// Rank of stride vector

    using Index = int32_t;                                /// Index type used for coordinates
    using LongIndex = int64_t;                            /// Long index type used for offsets
    using TensorCoord = MatrixCoord;                      /// Logical coordinate
    using Stride = Coord<kStrideRank, Index, LongIndex>;  /// Stride vector

    /// This layout is optimized for 128b accesses
    using Base = TensorOpMultiplicandCongruous<ElementSize, Crosswise>;
    static int const kAccessSize = Base::kAccessSize;
    static int const kElementSize = Base::kElementSize;
    static int const kElementsPerAccess = Base::kElementsPerAccess;
    static int const kCrosswise = Base::kCrosswise;
    static int const kFactor = Base::kFactor;
    using TileShape = typename Base::TileShape;
    using PartitionShape = typename Base::PartitionShape;
    using PartitionCount = typename Base::PartitionCount;
    using AccessCount = typename Base::AccessCount;

private:
    Base layout_;

public:
    /// Constructor
    RowMajorTensorOpMultiplicandCongruous(Stride stride) : layout_(stride) {}

    /// Constructor
    RowMajorTensorOpMultiplicandCongruous(Index ldm = 0) : layout_(ldm) {}

    /// Helper returns a layout to a tightly packed tensor
    static RowMajorTensorOpMultiplicandCongruous packed(TensorCoord const &extent) {
        return RowMajorTensorOpMultiplicandCongruous(extent.column());
    }

    /// Returns the offset of a coordinate in linear memory. 
    /// Assumes coordinate has convention (contiguous, strided)
    LongIndex operator()(TensorCoord const &coord) const {
        return layout_(PitchLinearCoord(coord.column(), coord.row()));
    }

    /// Inverse of layout function, mapping linear offset to logical coordinate
    TensorCoord inverse(LongIndex offset) const {
        PitchLinearCoord coord = layout_.inverse(offset);
        return MatrixCoord(coord.strided(), coord.contiguous());
    }
}
```

### TensorOpMultiplicandCrosswise

以TensorOpMultiplicand布局为基本（称之为基本布局），CUTLASS提出各种其它用途的相关布局。例如，TensorOpMultiplicandCrosswise布局是对基本布局的包装，ColumnMajorTensorOpMultiplicandCrosswise布局将列主序布局映射为基本布局，RowMajorTensorOpMultiplicandCrosswise布局将行主序布局映射为基本布局。

在cutlass/layout/tensor_op_multiplicand_sm75.h头文件中，提供TensorOpMultiplicandCrosswise的定义，如下所示。

```c++
/// Template based on element size (in bits) - defined in terms of pitch-linear memory and Crosswise size (in elements).
template <int ElementSize, int Crosswise>
struct TensorOpMultiplicandCrosswise {
    static int const kRank = 2;        /// Logical rank of tensor
    static int const kStrideRank = 1;  /// Rank of stride vector

    using Index = int32_t;                                /// Index type used for coordinates
    using LongIndex = int64_t;                            /// Long index type used for offsets
    using TensorCoord = PitchLinearCoord;                 /// Logical coordinate
    using Stride = Coord<kStrideRank, Index, LongIndex>;  /// Stride vector

    /// This layout is optimized for 128b accesses
    using Base = TensorOpMultiplicand<ElementSize, Crosswise>;
    static int const kAccessSize = Base::kAccessSize;
    static int const kElementSize = Base::kElementSize;
    static int const kElementsPerAccess = Base::kElementsPerAccess;
    static int const kCrosswise = Base::kCrosswise;
    static int const kFactor = Base::kFactor;
    using TileShape = typename Base::TileShape;
    using PartitionShape = typename Base::PartitionShape;
    using PartitionCount = typename Base::PartitionCount;
    using AccessCount = typename Base::AccessCount;

private:
    Base layout_;

public:
    /// Constructor
    TensorOpMultiplicandCrosswise(Stride stride) : layout_(stride) {}

    /// Constructor
    TensorOpMultiplicandCrosswise(Index ldm = 0) : layout_(ldm) {}

    /// Helper returns a layout to a tightly packed tensor
    static TensorOpMultiplicandCrosswise packed(TensorCoord const &extent) {
        return TensorOpMultiplicandCrosswise(extent[0]);
    }

    /// Returns the offset of a coordinate in linear memory.
    /// Assumes coordinate has convention (contiguous, strided)
    LongIndex operator()(TensorCoord const &coord) const {
        return layout_(coord);
    }

    /// Inverse of layout function, mapping linear offset to logical coordinate
    TensorCoord inverse(LongIndex offset) const {
        PitchLinearCoord coord = layout_.inverse(offset);
        return coord;
    }
}
```

在cutlass/layout/tensor_op_multiplicand_sm75.h头文件中，提供ColumnMajorTensorOpMultiplicandCrosswise的定义，如下所示。

```c++
/// Template mapping a column-major view of pitch-linear memory to TensorOpMultiplicandCrosswise
template <int ElementSize, int Crosswise>
struct ColumnMajorTensorOpMultiplicandCrosswise {
    static int const kRank = 2;        /// Logical rank of tensor
    static int const kStrideRank = 1;  /// Rank of stride vector

    using Index = int32_t;                                /// Index type used for coordinates
    using LongIndex = int64_t;                            /// Long index type used for offsets
    using TensorCoord = MatrixCoord;                      /// Logical coordinate
    using Stride = Coord<kStrideRank, Index, LongIndex>;  /// Stride vector
    
    /// This layout is optimized for 128b accesses
    using Base = TensorOpMultiplicandCrosswise<ElementSize, Crosswise>;
    static int const kAccessSize = Base::kAccessSize;
    static int const kElementSize = Base::kElementSize;
    static int const kElementsPerAccess = Base::kElementsPerAccess;
    using TileShape = typename Base::TileShape;
    using PartitionShape = typename Base::PartitionShape;
    using PartitionCount = typename Base::PartitionCount;
    using AccessCount = typename Base::AccessCount;

private:
    Base layout_;

public:
    /// Constructor
    ColumnMajorTensorOpMultiplicandCrosswise(Stride stride) : layout_(stride) {}

    /// Constructor
    ColumnMajorTensorOpMultiplicandCrosswise(Index ldm = 0) : layout_(ldm) {}

    /// Helper returns a layout to a tightly packed tensor
    static ColumnMajorTensorOpMultiplicandCrosswise packed(TensorCoord const &extent) {
        return ColumnMajorTensorOpMultiplicandCrosswise(extent.row());
    }

    /// Returns the offset of a coordinate in linear memory.
    /// Assumes coordinate has convention (contiguous, strided)
    LongIndex operator()(TensorCoord const &coord) const {
        return layout_(PitchLinearCoord(coord.row(), coord.column()));
    }

    /// Inverse of layout function, mapping linear offset to logical coordinate
    TensorCoord inverse(LongIndex offset) const {
        PitchLinearCoord coord = layout_.inverse(offset);
        return MatrixCoord(coord.contiguous(), coord.strided());
    }
}
```

在cutlass/layout/tensor_op_multiplicand_sm75.h头文件中，提供RowMajorTensorOpMultiplicandCrosswise的定义，如下所示。

```c++
/// Template mapping a row-major view of pitch-linear memory to TensorOpMultiplicandCrosswise
template <int ElementSize, int Crosswise>
struct RowMajorTensorOpMultiplicandCrosswise {
    static int const kRank = 2;        /// Logical rank of tensor
    static int const kStrideRank = 1;  /// Rank of stride vector

    using Index = int32_t;                                /// Index type used for coordinates
    using LongIndex = int64_t;                            /// Long index type used for offsets
    using TensorCoord = MatrixCoord;                      /// Logical coordinate
    using Stride = Coord<kStrideRank, Index, LongIndex>;  /// Stride vector

    /// This layout is optimized for 128b accesses
    using Base = TensorOpMultiplicandCrosswise<ElementSize, Crosswise>;
    static int const kAccessSize = Base::kAccessSize;
    static int const kElementSize = Base::kElementSize;
    static int const kElementsPerAccess = Base::kElementsPerAccess;
    using TileShape = typename Base::TileShape;
    using PartitionShape = typename Base::PartitionShape;
    using PartitionCount = typename Base::PartitionCount;
    using AccessCount = typename Base::AccessCount;

private:
    Base layout_;

public:
    /// Constructor
    RowMajorTensorOpMultiplicandCrosswise(Stride stride) : layout_(stride) {}

    /// Constructor
    RowMajorTensorOpMultiplicandCrosswise(Index ldm = 0) : layout_(ldm) {}

    /// Helper returns a layout to a tightly packed tensor
    static RowMajorTensorOpMultiplicandCrosswise packed(TensorCoord const &extent) {
        return RowMajorTensorOpMultiplicandCrosswise(extent.column());
    }

    /// Returns the offset of a coordinate in linear memory.
    /// Assumes coordinate has convention (contiguous, strided)
    LongIndex operator()(TensorCoord const &coord) const {
        return layout_(PitchLinearCoord(coord.column(), coord.row()));
    }

    /// Inverse of layout function, mapping linear offset to logical coordinate
    TensorCoord inverse(LongIndex offset) const {
        PitchLinearCoord coord = layout_.inverse(offset);
        return MatrixCoord(coord.strided(), coord.contiguous());
    }
}
```

## Tensor Accessor

张量访问器（Accessor），用于访问在内存中张量的一个元素，它接收一个元素的逻辑坐标，根据张量的起始指针和排列布局，从相应的内存位置上访问元素。

### TensorRef

模板类TensorRef<Element,Layout>持有一个指向张量起始元素位置的ptr\_指针，和一个描述张量元素排列方式的layout\_布局。在访问时，TensorRef接收一个元素的逻辑坐标，然后使用布局layout\_获得该元素在内存中的偏移量，然后根据起始元素的指针ptr\_，使用基址偏移寻址，从相应的内存位置上访问到该元素的值。

值得注意的是，在CUTLASS中，使用TensorRef<Element,Layout>访问内存位置时，其元素类型Element通常是一个基本数据类型，或者是一个Array数组。

在cutlass/tensor_ref.h头文件中，提供TensorRef<Element,Layout>的定义，如下所示。

```c++
/// TensorRef is a template for objects pointing to the start of tensors of arbitrary rank and layout within memory.
/// A TensorRef combines a pointer and a Layout concept.
template <
    typename Element_,  /// Data type of element stored within tensor (concept: NumericType)
    typename Layout_    /// Defines a mapping from logical coordinate to linear memory (concept: Layout)
>
class TensorRef {
public:
    using Element = Element_;    /// Data type of individual access
    using Layout = Layout_;      /// Mapping function from logical coordinate to linear memory
    using Reference = Element&;  /// Reference type to an element

    static int const kRank = Layout::kRank;  /// Logical rank of tensor index space

    using Index = typename Layout::Index;              /// Index type
    using LongIndex = typename Layout::LongIndex;      /// Long index used for pointer offsets
    using TensorCoord = typename Layout::TensorCoord;  /// Coordinate in logical tensor space
    using Stride = typename Layout::Stride;            /// Layout's stride vector

private:
    Element* ptr_;   /// Pointer
    Layout layout_;  /// Layout object maps logical coordinates to linear offsets

public:
    /// Constructs a TensorRef.
    TensorRef() : ptr_(nullptr) {}

    /// Constructs a TensorRef with a pointer and layout object.
    TensorRef(
        Element *ptr,         ///< pointer to start of tensor
        Layout const &layout  ///< layout object containing stride and mapping function
    ) : ptr_(ptr), layout_(layout) {}

    /// Updates the pointer and layout object
    void reset(Element* ptr, Layout const &layout) {
        ptr_ = ptr;
        layout_ = layout;
    }

    /// Returns the pointer to referenced data
    Element * data() const {
        return ptr_;
    }

    /// Returns a reference to the element at a given linear index
    Reference data(LongIndex idx) const { 
        return ptr_[idx];
    }

    /// Computes the offset of an index from the origin of the tensor
    LongIndex offset(TensorCoord const& coord) const {
        return layout_(coord);
    }

    /// Returns a reference to the element at a given Coord
    Reference at(TensorCoord const& coord) const {
        return data(offset(coord));
    }

    /// Returns a reference to the element at a given Coord
    Reference operator[](TensorCoord const& coord) const {
        return data(offset(coord));
    }

    /// Adds an offset to each pointer
    TensorRef & add_pointer_offset(LongIndex offset_) {
        ptr_ += offset_;
        return *this;
    }

    /// Adds an offset to each pointer
    TensorRef & add_coord_offset(TensorCoord const &coord) {
        add_pointer_offset(offset(coord));
        return *this;
    }

    /// Returns a TensorRef offset by a given amount
    TensorRef operator+(TensorCoord const& b) const {
        TensorRef result(*this);
        result.add_coord_offset(b);
        return result;
    }

    /// Returns a TensorRef offset by a given amount
    TensorRef & operator+=(TensorCoord const& b) {
        add_coord_offset(b);
        return *this;
    }

    /// Returns a TensorRef offset by a given amount
    TensorRef operator-(TensorCoord const& b) const {
        TensorRef result(*this);
        result.add_pointer_offset(-offset(b));
        return result;
    }

    /// Returns a TensorRef offset by a given amount
    TensorRef & operator-=(TensorCoord const& b) {
        add_pointer_offset(-offset(b));
        return *this;
    }
    
    /// Returns the layout object's stride in a given physical dimension
    typename Layout::Stride::Index & stride(int dim) {
        return layout_.stride().at(dim);
    }
};
```

### TensorView

TensorView<Element,Layout>继承自TensorRef<Element,Layout>模板类，但TensorView<Element,Layout>是一个维数都确定的访问器，它假设所访问的张量是确定维数的，即张量在每个维度轴上的维数都是提前确定的。由此性质，TensorView<Element,Layout>可以定义一些有用的函数。

在cutlass/tensor_view.h头文件中，提供TensorView<Element,Layout>的定义，如下所示。

```c++
template <
    typename Element_,  /// Data type of element stored within tensor
    typename Layout_    /// Maps a Coord<Rank_> in the logical tensor index space to the internal n-D array
>
class TensorView : public TensorRef<Element_, Layout_> {
public:
    using Base = cutlass::TensorRef<Element_, Layout_>;  /// Base tensor reference
    using Layout = Layout_;                              /// Mapping function from logical coordinate to internal n-D array
    using TensorRef = Base;                              /// Underlying TensorRef type
    using Element = Element_;                            /// Data type of individual access
    using Reference = Element&;                          /// Reference type to an element

    static int const kRank = Layout::kRank;  /// Logical rank of tensor index space

    using Index = typename Layout::Index;              /// Index type
    using LongIndex = typename Layout::LongIndex;      /// Long index used for pointer offsets
    using TensorCoord = typename Layout::TensorCoord;  /// Coordinate in logical tensor space
    using Stride = typename Layout::Stride;            /// Coordinate in storage n-D array

private:
    TensorCoord extent_;  /// View extent

public:
    /// Constructs a TensorView object
    TensorView() {}

    /// Constructs a TensorView object
    TensorView(
        Element *ptr,              ///< pointer to start of tensor
        Layout const &layout,      ///< layout object containing stride and mapping function
        TensorCoord const &extent  ///< size of the view in logical coordinates
    ) : Base(ptr, layout), extent_(extent) {}

    /// Updates the pointer and layout object
    void reset(Element* ptr, Layout const &layout, TensorCoord const &extent) {
        Base::reset(ptr, layout);
        this->resize(extent);
    }

    /// Changes the size of the view without affecting pointer or layout
    void resize(TensorCoord const &extent) {
        this->extent_ = extent;
    }

    /// Returns the number of logical elements
    LongIndex size() const {
        return extent_.product();
    }

    /// Determines whether a location is within a tensor
    bool contains(TensorCoord const& coord) const {
        for (int dim = 0; dim < kRank; ++dim) {
            if (!(coord[dim] >= 0 && coord[dim] < extent(dim))) {
                return false;
            }
        }
        return true;
    }
};
```

# Usage Examples

在cutlass/gemm/device目录中，提供设备层级的GEMM接口，用于在GPU设备上启动矩阵乘法的kernel核函数，主要包括标准GEMM计算、分组GEMM计算、批量GEMM计算、SplitK算法GEMM计算。由模板类提供实现，即cutlass::gemm::device::Gemm模板类、cutlass::gemm::device::GemmArray模板类、cutlass::gemm::device::GemmBatched模板类、cutlass::gemm::device::GemmSplitKParallel模板类。一些GEMM计算的示例如下。

```c++
void demo_gemm() {
    using Gemm = cutlass::gemm::device::Gemm<
        float, cutlass::layout::ColumnMajor,
        float, cutlass::layout::ColumnMajor,
        float, cutlass::layout::ColumnMajor, float
    >;
    Gemm gemm_op;
    cutlass::Status stat = gemm_op(
        {{M, N, K}, {d_A, M}, {d_B, K}, {d_C, M}, {d_C, M}, {alpha, beta}}
    );
}

void demo_gemm_batched() {
    using GemmBatched = cutlass::gemm::device::GemmBatched<
        float, cutlass::layout::ColumnMajor,
        float, cutlass::layout::ColumnMajor,
        float, cutlass::layout::ColumnMajor, float
    >;
    GemmBatched gemm_batched_op;
    cutlass::Status status = gemm_batched_op(
        {{M, N, K}, {d_A, M}, M * K, {d_B, K}, K * N, {d_C, M}, M * N, {d_C, M}, M * N, {alpha, beta}, Batch}
    );
}

void demo_gemm_array() {
    using GemmArray = cutlass::gemm::device::GemmArray<
        float, cutlass::layout::ColumnMajor,
        float, cutlass::layout::ColumnMajor,
        float, cutlass::layout::ColumnMajor, float
    >;
    GemmArray gemm_array_op;
    gemm_array_op(
        {{M, N, K}, d_A_array, M, d_B_array, K, d_C_array, M, d_C_array, M, {alpha, beta}, Batch}
    );
}

void demo_gemm_splitK() {
    using GemmSplitK = cutlass::gemm::device::GemmSplitKParallel<
        float, cutlass::layout::ColumnMajor,
        float, cutlass::layout::ColumnMajor,
        float, cutlass::layout::ColumnMajor, float
    >;
    GemmSplitK gemm_splitK_op;
    int split_num = 16;  // Split K dimension into 16 partitions
    GemmSplitK::Arguments args({M, N, K}, {d_A, M}, {d_B, K}, {d_C, M}, {d_C, M}, {alpha, beta}, split_num);
    size_t workspace_size = GemmSplitK::get_workspace_size(args);
    cutlass::device_memory::allocation<uint8_t> workspace_buffer(workspace_size);
    cutlass::Status status = gemm_splitK_op.initialize(args, workspace_buffer.get());
    status = gemm_splitK_op();
}
```

# GEMM API ^$

CUTLASS模板库对于矩阵乘法GEMM的核心实现代码如下目录所示。并且，在实现代码的几乎每个层级都提供了以default为前缀的默认配置default_xxx.cu，若不清楚每个层级的模板参数如何指定，可以参考这些默认配置。

```shell
cutlass
├── arch       # Architecture features (including instruction implementation)
├── gemm       # GEneral Matrix Multiply computations
│   ├── device       # Launch kernels
│   ├── kernel       # Kernels
│   ├── threadblock  # Cta Tile
│   ├── warp         # Warp Tile
│   └── thread       # Thread Tile
├── transform  # Code specialized for layout, type, and domain transformations
├── epilogue   # Epilogue rearranges result to canonical layouts, and supports conversion and reduction operations
└── reduction  # Reduction kernels
```

CUTLASS对通用矩阵乘法GEMM进行并行分片，映射到CUDA并行编程模型中的多个层级资源上，整个流程的示意图如下所示。

![](CUTLASS模板库.assets/CUTLASS的GEMM示意图.png)

## Common Tag

在cutlass/gemm/gemm_enumerated_types.h头文件中，提供一些枚举类型的定义，用于标识操作数、指定操作类型、设置共享内存清除策略，如下所示。

```c++
/// GEMM operand enumeration: D = A * B + C
enum class Operand {
    kA,  /// A multiplicand
    kB,  /// B multiplicand
    kC,  /// Source accumulator
    kD   /// Destination accumulator
};

enum class GemmUniversalMode {
    kGemm,
    kGemmSplitKParallel,
    kBatched,
    kArray,
    kGrouped,
    kInvalid
};

/// Some options for clearing shared memory
enum class SharedMemoryClearOption {
    kNone,           ///< SMEM is in don't-care state
    kZfill,          ///< Kernels fill out of bounds accesses with zeros
    kClearLastStage  ///< Last SMEM stage is explicitly cleared. Mainloop uses 'kNone'
};
```

## Architecture

在cutlass/arch目录中，提供一些基本操作的PTX汇编指令实现，主要包括内存访问指令、mma.xxx系列矩阵乘加指令、wmma.xxx系列矩阵乘加指令。这些PTX汇编指令直接与硬件架构进行交互，不同的硬件架构支持不同的PTX汇编指令，该层级的代码是对PTX汇编指令的包装。

<img src="CUTLASS模板库.assets/gemm-arch.png" style="zoom:15%;" />

在cutlass/arch/arch.h头文件中，提供LaneId()与SmId()辅助函数，以及设备架构与计算能力的标识。

```c++
/// Computes laneId within a warp
int LaneId() {
    int ret;
    asm("mov.u32 %0, %%laneid;" : "=r"(ret) : );
    return ret;
}

/// Computes SM number the thread is running on
int SmId() {
    int ret;
    asm("mov.u32 %0, %%smid;" : "=r"(ret) : );
    return ret;
}

struct Sm50 { static int const kMinComputeCapability = 50; };
struct Sm60 { static int const kMinComputeCapability = 60; };
struct Sm61 { static int const kMinComputeCapability = 61; };
struct Sm70 { static int const kMinComputeCapability = 70; };
struct Sm72 { static int const kMinComputeCapability = 72; };
struct Sm75 { static int const kMinComputeCapability = 75; };
struct Sm80 { static int const kMinComputeCapability = 80; };
struct Sm86 { static int const kMinComputeCapability = 86; };
struct Sm89 { static int const kMinComputeCapability = 89; };
struct Sm90 { static int const kMinComputeCapability = 90; };
```

在cutlass/arch/cache_operation.h头文件中，提供标识Cache缓存行为的枚举类。

```c++
/// Controls PTX cache operations
struct CacheOperation {
    enum Kind {
        Always,       /// Cache at all levels - accessed again
        Global,       /// Cache at global level
        Streaming,    /// Streaming - likely to be accessed once
        LastUse,      /// Indicates the line will not be used again
        Volatile,     /// Don't cache, and fetch again
        WriteBack,    /// Write back at all coherent levels
        WriteThrough  /// Write through to system memory
    };
};
```

### global_load, global_store

在cutlass/arch/memory.h头文件中，提供从全局内存中加载数据的操作，并支持不同的加载粒度，即支持一次性加载1、2、4、8、16、32个字节的数据。

```c++
template <
    typename AccessType,                                    /// Fragment type to store loaded data
    int LoadBytes,                                          /// The bytes of loading
    CacheOperation::Kind cache_op = CacheOperation::Always  /// Cache operation
>
struct global_load;

template <typename AccessType>
struct global_load<AccessType, 16, CacheOperation::Always> {
    global_load(AccessType &D, void const *ptr, bool pred_guard) {
        uint4 &data = reinterpret_cast<uint4 &>(D);
        // The redundant mov PTX instruction is used to enforce the compiler to keep the initializing code before ld.global
        asm volatile(
            "{\n"
            "  .reg .pred p;\n"
            "  setp.ne.b32 p, %5, 0;\n"
            "  mov.b32 %0, %6;\n"
            "  mov.b32 %1, %7;\n"
            "  mov.b32 %2, %8;\n"
            "  mov.b32 %3, %9;\n"
        #if CUTLASS_ENABLE_L2_PREFETCH
            "  @p ld.global.L2::128B.v4.u32 {%0, %1, %2, %3}, [%4];\n"
        #else
            "  @p ld.global.v4.u32 {%0, %1, %2, %3}, [%4];\n"
        #endif
            "}\n"
            : "=r"(data.x), "=r"(data.y), "=r"(data.z), "=r"(data.w)
            : "l"(ptr), "r"((int)pred_guard), "r"(data.x), "r"(data.y), "r"(data.z), "r"(data.w)
        );
    }
};
```

在cutlass/arch/memory.h头文件中，提供向全局内存中写入数据的操作，并支持不同的写入粒度，即支持一次性写入1、2、4、8、16、32个字节的数据。

```c++
template <
    typename AccessType,  /// Fragment type to store data
    int StoreBytes        /// The bytes of storing
>
struct global_store;

template <typename AccessType>
struct global_store<AccessType, 16> {
    global_store(AccessType const &D, void *ptr, bool pred_guard) {
        uint4 const &data = reinterpret_cast<uint4 const &>(D);
        asm volatile(
            "{\n"
            "  .reg .pred p;\n"
            "  setp.ne.b32 p, %5, 0;\n"
            "  @p st.global.v4.u32 [%0], {%1, %2, %3, %4};\n"
            "}\n"
            :
            : "l"(ptr), "r"(data.x), "r"(data.y), "r"(data.z), "r"(data.w), "r"((int)pred_guard)
        );
    }
};
```

### shread_load, shared_store

在cutlass/arch/memory.h头文件中，提供从共享内存中加载数据的操作，并支持不同的加载粒度，即支持一次性加载2、4、8、16个字节的数据。

```c++
/// ld.shared
template <int Bytes>
void shared_load(void *dst, uint32_t ptr);

/// ld.shared - 128b
template <>
void shared_load<16>(void *dst, uint32_t ptr) {
    uint4 *dst_u128 = reinterpret_cast<uint4 *>(dst);
    asm volatile(
        "ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(dst_u128->x), "=r"(dst_u128->y), "=r"(dst_u128->z), "=r"(dst_u128->w)
        : "r"(ptr)
    );
}
```

在cutlass/arch/memory.h头文件中，提供向共享内存中写入数据的操作，并支持不同的写入粒度，即支持一次性写入2、4、8、16个字节的数据。

```c++
/// st.shared
template <int Bytes>
void shared_store(uint32_t ptr, void const *src);

/// st.shared - 128b
template <>
void shared_store<16>(uint32_t ptr, void const *src) {
    uint4 const *dst_u128 = reinterpret_cast<uint4 const *>(src);
    asm volatile(
        "st.shared.v4.u32 [%0], {%1, %2, %3, %4};\n"
        : 
        : "r"(ptr), "r"(dst_u128->x), "r"(dst_u128->y), "r"(dst_u128->z), "r"(dst_u128->w)
    );
}
```

需要注意的是，在使用PTX汇编指令ld.shared和st.shared访问共享内存时，所使用的地址是位于.shared共享存储状态空间的，而不是一个通用地址，因此需要使用cvta指令将通用地址转换为一个位于共享存储状态空间的地址。相应的辅助函数如下所示。

```c++
// helper to cast SMEM pointer to unsigned integer
uint32_t cast_smem_ptr_to_uint(void const* const ptr) {
    uint32_t smem_ptr;
    asm volatile(
        "{\n"
        "  .reg .u64 smem_ptr;"
        "  cvta.to.shared.u64 smem_ptr, %1;"
        "  cvt.u32.u64 %0, smem_ptr;"
        "}\n"
        : "=r"(smem_ptr)
        : "l"(ptr)
    );
    return smem_ptr;
    #endif
}
```

### ldsm

在cutlass/arch/memory_sm75.h头文件中，提供从共享内存中加载数据的操作，这是对ldmatrix指令的包装，可以一次性加载1、2、3、4个矩阵。

```c++
template <
    typename Layout,  /// Layout of destination matrix (column-major implies transpose)
   int MatrixCount    /// .x1, .x2, or .x4
>
void ldsm(Array<unsigned, MatrixCount> & D, void const* ptr);

template <>
void ldsm<layout::RowMajor, 4>(Array<unsigned, 4> & D, void const* ptr) {
    unsigned addr = cutlass_get_smem_pointer(ptr);
    int x, y, z, w;
    asm volatile (
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
        : "r"(addr)
    );
    reinterpret_cast<int4 &>(D) = make_int4(x, y, z, w);
}

template <>
void ldsm<layout::ColumnMajor, 4>(Array<unsigned, 4> & D, void const* ptr) {
    unsigned addr = cutlass_get_smem_pointer(ptr);
    int x, y, z, w;
    asm volatile (
        "ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];" 
        : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
        : "r"(addr)
    );
    reinterpret_cast<int4 &>(D) = make_int4(x, y, z, w);
}
```

### SIMD

大多数指令都是在CUDA Core硬件上执行的。对于一个线程而言，有的操作是单指令多数据（SIMD）的，这对于GPU而言是依赖于循环展开的。对于一个Warp调度器而言，有的操作时单指令多线程（SIMT）的，这依赖于Warp调度器的调度机制。

在cutlass/arch/simd.h头文件中，提供一个线程依赖于循环展开的SIMD操作，此处列举几个有代表性的操作。

```c++
template <typename T, int N>
Array<T, N> operator*(Array<T, N> const &a, Array<T, N> const &b) {
    Array<T, N> d;
    for (int i = 0; i < N; ++i) { d[i] = a[i] * b[i]; }
    return d;
}

template <typename T, int N>
Array<T, N> mac(Array<T, N> const &a, Array<T, N> const &b, Array<T, N> const &c) {
    Array<T, N> d;
    for (int i = 0; i < N; ++i) { d[i] = a[i] * b[i] + c[i]; }
    return d;
}

template <typename T, typename Accumulator, int N>
Accumulator dot(Array<T, N> const &a, Array<T, N> const &b, Accumulator accum) {
    for (int i = 0; i < N; ++i) { accum += a[i] * b[i]; }
    return accum;
}
```

### MMA

矩阵乘加操作可以在Tensor Core硬件上由mma.xxx系列指令执行，这些指令在CUTLASS中被抽象为arch::Mma模板类。

在cutlass/arch/mma.h头文件中，提供对各类操作的标识符（用于提示编译器选择合适的部分实例化的模板类），如下所示。

```c++
struct OpMultiplyAdd {};        // Tag indicating the operation implied by MMA.
struct OpClassSimt {};          // Tag classifying math operators as thread-level operations.
struct OpClassTensorOp {};      // Tag classifying operators as Tensor Core operations.
struct OpClassWmmaTensorOp {};  // Tag classifying operators as WMMA Tensor Core operations.
```

在cutlass/arch/mma.h头文件中，提供arch::Mma的定义，如下所示。

```c++
/// Matrix multiply-add operation
template <
    typename Shape,     /// Size of the matrix product (concept: GemmShape)
    int kThreads,       /// Number of threads participating
    typename ElementA,  /// Data type of A elements
    typename LayoutA,   /// Layout of A matrix (concept: MatrixLayout)
    typename ElementB,  /// Data type of B elements
    typename LayoutB,   /// Layout of B matrix (concept: MatrixLayout)
    typename ElementC,  /// Element type of C matrix
    typename LayoutC,   /// Layout of C matrix (concept: MatrixLayout)
    typename Operator   /// Inner product operator
>
struct Mma;

/// Matrix multiply-add operation - specialized for 1x1x1x1 matrix multiply operation
template <
    typename ElementA,  /// Data type of A elements
    typename LayoutA,   /// Layout of A matrix (concept: MatrixLayout)
    typename ElementB,  /// Data type of B elements
    typename LayoutB,   /// Layout of B matrix (concept: MatrixLayout)
    typename ElementC,  /// Element type of C matrix
    typename LayoutC,   /// Layout of C matrix (concept: MatrixLayout)
    typename Operator   /// Inner product operator
>
struct Mma<gemm::GemmShape<1, 1, 1>, 1, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, Operator> {
    void operator()(
        Array<ElementC, 1> &d,
        Array<ElementA, 1> const &a,
        Array<ElementB, 1> const &b,
        Array<ElementC, 1> const &c
    ) {
        multiply_add<ElementA, ElementB, ElementC> op;
        d[0] = op(a[0], b[0], c[0]);
    }
};
```

在诸如cutlass/arch/mma_sm80.h等头文件中，提供arch::Mma特定于一个硬件架构的代码实现，如下所示。

```c++
/// Matrix multiply-add operation: F32 = F16 * F16 + F32
template <>
struct Mma<
    gemm::GemmShape<16, 8, 16>,
    32,
    half_t,
    layout::RowMajor,
    half_t,
    layout::ColumnMajor,
    float,
    layout::RowMajor,
    OpMultiplyAdd
> {
    using Shape = gemm::GemmShape<16, 8, 16>;
    using ElementA = half_t;
    using LayoutA = layout::RowMajor;
    using FragmentA = Array<half_t, 8>;
    using ElementB = half_t;
    using LayoutB = layout::ColumnMajor;
    using FragmentB = Array<half_t, 4>;
    using ElementC = float;
    using LayoutC = layout::RowMajor;
    using FragmentC = Array<float, 4>;
    using Operator = OpMultiplyAdd;
    using ArchTag = arch::Sm80;

    /// Computes multiply-add
    void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b, FragmentC const &c) const {
        uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
        uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
        float const *C = reinterpret_cast<float const *>(&c);
        float *D = reinterpret_cast<float *>(&d);
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
            : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
            : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3])
        );
    }
};
```

### WMMA

矩阵乘加操作可以在Tensor Core硬件上由wmma.xxx系列指令执行，这些指令在CUTLASS中被抽象为arch::Wmma模板类。

在cutlass/arch/wmma.h头文件中，提供arch::Wmma的定义，如下所示。

```c++
// WMMA template structure defines nvcuda::wmma::fragments and static assertion chaeks
// for a specific template paramterized data type (Element[A|B|C]), layout (Layout[A|B|C]), and native wmma size (Shape)
template <
    typename Shape_,                                   ///< Size of the matrix product (concept: GemmShape)
    typename ElementA_,                                ///< Data type of A elements 
    typename LayoutA_,                                 ///< Layout of A matrix (concept: MatrixLayout)  
    typename ElementB_,                                ///< Data type of B elements
    typename LayoutB_,                                 ///< Layout of B matrix (concept: MatrixLayout)  
    typename ElementC_,                                ///< Element type of C matrix  
    typename LayoutC_,                                 /// Layout of C matrix (concept: MatrixLayout)
    typename Operator_ = cutlass::arch::OpMultiplyAdd  ///< Inner product operator (multiply-add, xor.popc)
>
struct Wmma;
```

在诸如cutlass/arch/wmma_sm70.h等头文件中，提供arch::Wmma特定于一个硬件架构的代码实现，如下所示。

```c++
// WMMA template structure defines nvcuda::wmma::fragments and static assert for
// wmma native instruction sizes supported for half.
template <typename Shape_, typename LayoutA_, typename LayoutB_, typename ElementC_, typename LayoutC_>
struct Wmma<
    Shape_,                       ///< Size of the matrix product (concept: GemmShape)
    cutlass::half_t,              ///< ElementA
    LayoutA_,                     ///< LayoutA
    cutlass::half_t,              ///< ElementB
    LayoutB_,                     ///< LayoutB
    ElementC_,                    ///< ElementC
    LayoutC_,                     ///< LayoutC
    cutlass::arch::OpMultiplyAdd  ///< Operator (multiply-add, xor.popc)
> {
    using Shape = Shape_;
    using ElementA = cutlass::half_t;
    using LayoutA = LayoutA_;
    using ElementB = cutlass::half_t;
    using LayoutB = LayoutB_;
    using ElementC = ElementC_;
    using LayoutC = LayoutC_;
    using Operator = cutlass::arch::OpMultiplyAdd;
    using ArchTag = arch::Sm70;

    // Wmma Fragment
    using FragmentA = nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_a, Shape::kM, Shape::kN, Shape::kK,
        typename CutlassToWmmaDataType<ElementA>::Type, typename CutlassToWmmaLayout<LayoutA>::Layout
    >;
    using FragmentB = nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_b, Shape::kM, Shape::kN, Shape::kK,
        typename CutlassToWmmaDataType<ElementB>::Type, typename CutlassToWmmaLayout<LayoutB>::Layout
    >;
    using FragmentC = nvcuda::wmma::fragment<
        nvcuda::wmma::accumulator, Shape::kM, Shape::kN, Shape::kK,
        typename CutlassToWmmaDataType<ElementC>::Type
    >;

    /// Performs a nvcuda::wmma matrix multiply-accumulate operation
    void operator()(FragmentC &D, FragmentA const &A, FragmentB const &B, FragmentC const &C) const {
        nvcuda::wmma::mma_sync(D, A, B, C);
    }
};
```

## Thread Level

在cutlass/gemm/thread目录中，提供矩阵乘加操作在Thread线程层级的实现，主要是线程使用SIMD浮点指令在CUDA Core上的实现，这些操作被抽象为gemm::thread::Mma模板类。

![](CUTLASS模板库.assets/gemm-thread.png)

在cutlass/gemm/thread/mma.h头文件中，提供gemm::thread::Mma的定义，如下所示。

```c++
/// Structure to compute the matrix product
template <
    typename Shape,                           /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename ElementA,                        /// Data type of A elements
    typename LayoutA,                         /// Layout of A matrix (concept: MatrixLayout)
    typename ElementB,                        /// Data type of B elements
    typename LayoutB,                         /// Layout of B matrix (concept: MatrixLayout)
    typename ElementC,                        /// Element type of C matrix
    typename LayoutC,                         /// Layout of C matrix (concept: MatrixLayout)
    typename Operator = arch::OpMultiplyAdd,  /// Concept: arch::OpMultiplyAdd or arch::Mma<>
    typename Enable = bool                    /// Used for partial specialization
>
struct Mma;
```

在cutlass/gemm/thread/mma_sm50.h头文件中，提供通用的gemm::thread::MmaGeneric实现，并且该实现由gemm::thread::Mma使用，如下所示。

```c++
/// Gemplate that handles all packed matrix layouts
template <
    typename Shape_,     /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename ElementA_,  /// Data type of A elements
    typename LayoutA_,   /// Layout of A matrix (concept: layout::MapFunc)
    typename ElementB_,  /// Data type of B elements
    typename LayoutB_,   /// Layout of B matrix (concept: layout::MapFunc)
    typename ElementC_,  /// Element type of C matrix
    typename LayoutC_,   /// Layout of C matrix (concept: layout::MapFunc)
    typename Operator_   /// Operator used to compute GEMM
>
struct MmaGeneric {
    using Shape = Shape_;        /// Size of the Gemm problem - concept: gemm::GemmShape<>
    using ElementA = ElementA_;  /// Data type of operand A
    using LayoutA = LayoutA_;    /// Layout of A matrix (concept: layout::MapFunc)
    using ElementB = ElementB_;  /// Data type of operand B
    using LayoutB = LayoutB_;    /// Layout of B matrix (concept: layout::MapFunc)
    using ElementC = ElementC_;  /// Element type of operand C
    using LayoutC = LayoutC_;    /// Layout of C matrix (concept: layout::MapFunc)
    using Operator = Operator_;  /// Underlying mathematical operator

    using FragmentA = Array<ElementA, Shape::kMK>;  /// A operand storage
    using FragmentB = Array<ElementB, Shape::kKN>;  /// B operand storage
    using FragmentC = Array<ElementC, Shape::kMN>;  /// C operand storage

    /// Instruction
    using MmaOp = arch::Mma<gemm::GemmShape<1, 1, 1>, 1, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, Operator>;

    /// Computes a matrix product D = A * B + C
    void operator()(FragmentC & D, FragmentA const & A, FragmentB const & B, FragmentC const & C) {
        TensorRef<ElementA const, LayoutA> a_ref(reinterpret_cast<ElementA const *>(&A), LayoutA::packed({ Shape::kM, Shape::kK }));
        TensorRef<ElementB const, LayoutB> b_ref(reinterpret_cast<ElementB const *>(&B), LayoutB::packed({ Shape::kK, Shape::kN }));
        TensorRef<ElementC, LayoutC> d_ref(reinterpret_cast<ElementC *>(&D), LayoutC::packed(make_Coord(Shape::kM, Shape::kN)));
        MmaOp mma_op;

        // Copy accumulators
        D = C;
        // Compute matrix product
        for (int k = 0; k < Shape::kK; ++k) {
            for (int n = 0; n < Shape::kN; ++n) {
                for (int m = 0; m < Shape::kM; ++m) {
                    // ◦┌┐┌┐
                    // ↓↑↓↑↓
                    // └┘└┘◦
                    int m_serpentine = (n % 2) ? (Shape::kM - 1 - m) : m;
                    MatrixCoord mn(m_serpentine, n);
                    MatrixCoord mk(m_serpentine, k);
                    MatrixCoord kn(k, n);
                    Array<ElementC, 1> d;
                    Array<ElementA, 1> a;
                    Array<ElementB, 1> b;
                    d[0] = d_ref.at(mn);
                    a[0] = a_ref.at(mk);
                    b[0] = b_ref.at(kn);
                    mma_op(d, a, b, d);
                    d_ref.at(mn) = d[0];
                }
            }
        }
    }
};

/// Gemplate that handles conventional layouts for FFMA and DFMA GEMM
template <
    typename Shape_,     /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename ElementA_,  /// Data type of A elements
    typename LayoutA_,   /// Layout of A matrix (concept: layout::MapFunc)
    typename ElementB_,  /// Data type of B elements
    typename LayoutB_,   /// Layout of B matrix (concept: layout::MapFunc)
    typename ElementC_,  /// Element type of C matrix
    typename LayoutC_    /// Layout of C matrix (concept: layout::MapFunc)
>
struct Mma<Shape_, ElementA_, LayoutA_, ElementB_, LayoutB_, ElementC_, LayoutC_, arch::OpMultiplyAdd, bool> {
    using Shape = Shape_;                  /// Size of the Gemm problem - concept: gemm::GemmShape<>
    using ElementA = ElementA_;            /// Data type of operand A
    using LayoutA = LayoutA_;              /// Layout of A matrix (concept: layout::MapFunc)
    using ElementB = ElementB_;            /// Data type of operand B
    using LayoutB = LayoutB_;              /// Layout of B matrix (concept: layout::MapFunc)
    using ElementC = ElementC_;            /// Element type of operand C
    using LayoutC = LayoutC_;              /// Layout of C matrix (concept: layout::MapFunc)
    using Operator = arch::OpMultiplyAdd;  /// Underlying mathematical operator
    
    using FragmentA = Array<ElementA, Shape::kMK>;  /// A operand storage
    using FragmentB = Array<ElementB, Shape::kKN>;  /// B operand storage
    using FragmentC = Array<ElementC, Shape::kMN>;  /// C operand storage

    /// Underlying matrix multiply operator (concept: arch::Mma)
    using ArchMmaOperator = typename MmaGeneric<Shape, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, Operator>::MmaOp;

    /// Computes a matrix product D = A * B + C
    void operator()(FragmentC & D, FragmentA const & A, FragmentB const & B, FragmentC const & C) {
        MmaGeneric<Shape, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, Operator> mma;
        mma(D, A, B, C);
    }
};
```

## Warp Level

在cutlass/gemm/warp目录中，提供矩阵乘加操作在Warp线程束层级的实现，主要包括使用CUDA Core硬件的实现与使用Tensor Core硬件的实现，这些操作被抽象为gemm::warp::MmaSimt模板类和gemm::warp::MmaTensorOp模板类。此外，还提供一些相关的辅助类型，包括线程排列策略，数据访问策略等。

![](CUTLASS模板库.assets/gemm-warp.png)

### MmaSimtPolicy

当使用CUDA Core硬件实现矩阵乘加操作时，需要明确，一整个Warp中32个线程的排列布局、一个MMA计算的形状，以及一个线程负责几个MMA计算。这些属性被抽象为gemm::warp::MmaSimtPolicy模板类。

在cutlass/gemm/warp/mma_simt_policy.h头文件中，提供gemm::warp::MmaSimtPolicy的定义，如下所示。

```c++
/// Describes the arrangement and configuration of per-lane operations in warp-level matrix multiply 
template <
    typename WarpShape_,   /// shape of the warp in lanes (concept: MatrixShape). 一整个Warp的线程形状，如4×8或8×4
    typename LaneLayout_,  /// layout function of lanes. 一整个Warp的线程排列布局，如'Z'形布局
    /// size of each lane's thread-level matrix product (concept: GemmShape). 一个线程负责的一个MMA形状，一个线程可执行多个MMA计算
    typename LaneMmaShape_
>
struct MmaSimtPolicy {
    using WarpShape = WarpShape_;
    using LaneLayout = LaneLayout_;
    using LaneMmaShape = LaneMmaShape_;
    using MmaShape = LaneMmaShape;

    /// Returns a layout functor mapping lane position in the warp to thread ID
    static LaneLayout get_lane_layout() {
        return LaneLayout::packed({ WarpShape::kRow, WarpShape::kColumn });
    }
};
```

### MmaSimtTileIterator

当在线程束Warp层级执行矩阵乘加计算时，需要将数据从共享内存中读取到寄存器中，这种数据迭代器被抽象为gemm::warp::MmaSimtTileIterator模板类。

<img src="CUTLASS模板库.assets/gemm-warp-MmaSimtTileIterator.png" style="zoom:12%;" />

在cutlass/gemm/warp/mma_simt_tile_iterator.h头文件中，提供gemm::warp::MmaSimtTileIterator的定义，如下所示。

```c++
/// Iterates over operands to warp-level matrix multiply operations targeting SIMT instructions
/// concept: MutableRandomAccessContiguousTileIteratorConcept
template <
    typename Shape_,            /// Size of the matrix to load (concept: MatrixShape). 一整个Warp要加载的矩阵形状
    Operand Operand,            /// Operand identity. 用于标识操作数矩阵，可以是矩阵A、矩阵B、矩阵C、矩阵D
    typename Element_,          /// Data type of elements. 元素的基本类型
    typename Layout_,           /// Layout of operand. 标识矩阵元素的排列布局，可以是行主序或列主序，矩阵A和B还支持sliced-K布局
    typename Policy_,           /// Shape of the warp in units of thread (concept: MmaSimtPolicy). 线程束Warp的策略
    int PartitionsK = 1,        /// Number of partitions along K dimension - used in sliced-K
    int PartitionGroupSize = 1  /// Group Size along kPartition - used in sliced-K
>
class MmaSimtTileIterator;
```

在实现时，会根据所访问的矩阵操作数，以及矩阵操作数的排列布局，来提供最高效的访问方式，即模板类的部分实例化。例如，对于矩阵A而言，最优布局是列主序存储，对于矩阵B而言，最优布局是行主序存储，对于矩阵C和矩阵D而言，最优布局是行主序布局或列主序布局。

当矩阵A在共享内存中按照列主序存储时，gemm::warp::MmaSimtTileIterator访问器的实现代码如下所示。当矩阵B在共享内存中按照行主序存储时，gemm::warp::MmaSimtTileIterator访问器的实现代码与矩阵A的情况类似，此处不再赘述。

```c++
/// Specialization for A operands of column-major layouts
/// Concept: MutableRandomAccessContiguousTileIteratorConcept
template <
    typename Shape_,        /// Size of the matrix to load (concept: MatrixShape)
    typename Element_,      /// Data type of A elements
    typename Policy_,       /// Shape of the warp in units of thread (concept: MmaSimtPolicy)
    int PartitionsK,        /// Number of partitions along K dimension - used in sliced-K
    int PartitionGroupSize  /// Group Size along kPartition - used in sliced-K
>
class MmaSimtTileIterator<Shape_, Operand::kA, Element_, layout::ColumnMajor, Policy_, PartitionsK, PartitionGroupSize> {
public:
    using Shape = Shape_;
    static Operand const kOperand = Operand::kA;
    using Element = Element_;
    using Layout = layout::ColumnMajor;
    using Policy = Policy_;
    using TensorRef = TensorRef<Element, Layout>;         /// TensorRef type for loading element from a tensor
    using Index = typename TensorRef::Index;              /// Index type
    using LongIndex = typename TensorRef::LongIndex;      /// Long Index type
    using TensorCoord = typename TensorRef::TensorCoord;  /// Coordinate for an element in the tensor

    /// Thread-level shape of a fragment.
    using ThreadShape = MatrixShape<Shape::kRow / Policy::WarpShape::kRow, Shape::kColumn>;
    /// Number of individual loads.
    using Iterations = MatrixShape<ThreadShape::kRow / Policy::LaneMmaShape::kM, ThreadShape::kColumn>;
    /// Fragment object holding a thread's part of a tile.
    using Fragment = Array<Element, ThreadShape::kCount>;

private:
    /// Internal reference. 张量引用，类型单位是一个Array，表示乘加计算中矩阵A对应的一个MMA形状
    /// 加载和存储，都是以一个Array为单位的，即每次访问一个Array类型，偏移计算也以Array为单位，而不是以单个元素为单位的
    cutlass::TensorRef<Array<Element, Policy::LaneMmaShape::kM>, layout::ColumnMajor> ref_;

public:
    /// Default ctor constructs null iterator
    MmaSimtTileIterator() {}
    
    /// Constructor from TensorRef
    MmaSimtTileIterator(TensorRef ref, int lane_id) {
        // compute offset based on thread ID and lane layout
        typename Policy::LaneLayout lane_layout = Policy::get_lane_layout();
        MatrixCoord lane_offset = lane_layout.inverse(lane_id) * MatrixCoord(Policy::LaneMmaShape::kM, 0);
        ref.add_coord_offset(lane_offset);
        ref_.reset(reinterpret_cast<Array<Element, Policy::LaneMmaShape::kM> *>(ref.data()), ref.stride(0) / Policy::LaneMmaShape::kM);
    }

    /// Advances an iterator along logical dimensions of matrix in units of whole tiles
    MmaSimtTileIterator & add_tile_offset(TensorCoord const &coord) {
        ref_.add_coord_offset({ coord.row() * Shape::kRow / Policy::LaneMmaShape::kM, coord.column() * Shape::kColumn });
        return *this;
    }

    /// Advances the iterator along the advance dimension
    MmaSimtTileIterator & operator++() {
        ref_.add_coord_offset({ 0, Shape::kColumn });
        return *this;
    }

    /// Loads a fragment from memory at the location pointed to by the iterator. (vector loads)
    void load_with_pointer_offset(Fragment &frag, Index pointer_offset) const {
        Array<Element, Policy::LaneMmaShape::kM> *dst_ptr = reinterpret_cast<Array<Element, Policy::LaneMmaShape::kM> *>(&frag);
        for (int k = 0; k < Iterations::kColumn; ++k) {
            for (int m = 0; m < Iterations::kRow; ++m) {
                auto src_ptr = ref_.data() + ref_.offset({ m * Policy::WarpShape::kRow, k }) + pointer_offset / Policy::LaneMmaShape::kM;
                arch::shared_load(dst_ptr[m + k * Iterations::kRow], src_ptr);
            }
        }
    }

    /// Loads a fragment from memory at the location pointed to by the iterator.
    void load(Fragment &frag) const {
        load_with_pointer_offset(frag, 0);
    }
};
```

当矩阵C或矩阵D在共享内存中按照行主序存储时，gemm::warp::MmaSimtTileIterator访问器的实现代码如下所示。

```c++
/// Specialization for C operands of row-major layouts
/// Concept: MutableRandomAccessContiguousTileIteratorConcept
template <
    typename Shape_,    /// Size of the matrix to load (concept: MatrixShape)
    typename Element_,  /// Data type of A elements
    typename Policy_    /// Shape of the warp in units of thread (concept: MmaSimtPolicy)
>
class MmaSimtTileIterator<Shape_, Operand::kC, Element_, layout::RowMajor, Policy_> {
public:
    using Shape = Shape_;
    static Operand const kOperand = Operand::kC;
    using Element = Element_;
    using Layout = layout::RowMajor;
    using Policy = Policy_;
    using TensorRef = TensorRef<Element, Layout>;         /// TensorRef type for loading element from a tensor
    using Index = typename TensorRef::Index;              /// Index type
    using LongIndex = typename TensorRef::LongIndex;      /// Long Index type
    using TensorCoord = typename TensorRef::TensorCoord;  /// Coordinate for an element in the tensor

    /// Thraed-level shape of a fragment
    using ThreadShape = MatrixShape<Shape::kRow / Policy::WarpShape::kRow, Shape::kColumn / Policy::WarpShape::kColumn>;
    /// Number of individual loads
    using Iterations = MatrixShape<ThreadShape::kRow / Policy::LaneMmaShape::kM, ThreadShape::kColumn / Policy::LaneMmaShape::kN>;
    /// Delta of MmaShape in unit of element in shared memory
    using Delta = MatrixShape<Policy::WarpShape::kRow * Policy::LaneMmaShape::kM, Policy::WarpShape::kColumn * Policy::LaneMmaShape::kN>;
    /// Fragment object holding a thread's part of a tile
    using Fragment = Array<Element, ThreadShape::kCount>;

private:
    TensorRef ref_;  /// Internal reference

public:
    /// Default ctor constructs null iterator
    MmaSimtTileIterator() {}
    
    /// Constructor from TensorRef
    MmaSimtTileIterator(TensorRef const &ref, int lane_id) : ref_(ref) {
        // compute offset based on thread ID and lane layout
        typename Policy::LaneLayout lane_layout = Policy::get_lane_layout();
        MatrixCoord lane_offset = lane_layout.inverse(lane_id) * MatrixCoord(Policy::LaneMmaShape::kM, Policy::LaneMmaShape::kN);
        ref_.add_coord_offset(lane_offset);
    }

    /// Advances an iterator along logical dimensions of matrix in units of whole tiles
    MmaSimtTileIterator & add_tile_offset(TensorCoord const &coord) {
        ref_.add_coord_offset({ coord.row() * Shape::kRow, coord.column() * Shape::kColumn });
        return *this;
    }
    
    /// Advances the iterator along the advance dimension
    MmaSimtTileIterator & operator++() {
        ref_.add_coord_offset({ Shape::kRow, 0 });
        return *this;
    }

    /// Loads a fragment from memory with additional logical offset; linear offset (in units of Element) when loading
    void load_with_pointer_offset(Fragment &frag, Index pointer_offset) const {
        for (int mma_m = 0; mma_m < Iterations::kRow; ++mma_m) {
            for (int m = 0; m < Policy::LaneMmaShape::kM; ++m) {
                Array<Element, Policy::LaneMmaShape::kN> const *src_ptr = reinterpret_cast<Array<Element, Policy::LaneMmaShape::kN> const *>
                    (ref_.data() + pointer_offset + ref_.offset({ mma_m * Delta::kRow + m, 0 }));
                for (int mma_n = 0; mma_n < Iterations::kColumn; ++mma_n) {
                    Array<Element, Policy::LaneMmaShape::kN> *dst_ptr = reinterpret_cast<Array<Element, Policy::LaneMmaShape::kN> *>
                        (&frag) + mma_n + Iterations::kColumn * (m + mma_m * Policy::LaneMmaShape::kM);
                    *dst_ptr = src_ptr[mma_n * Policy::WarpShape::kColumn];
                }
            }
        }
    }

    /// Loads a fragment from memory at the location pointed to by the iterator.
    void load(Fragment &frag) const {
        load_with_pointer_offset(frag, 0);
    }

    /// Stores a fragment to memory at the location pointed to by the iterator
    void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) const {
        for (int mma_m = 0; mma_m < Iterations::kRow; ++mma_m) {
            for (int m = 0; m < Policy::LaneMmaShape::kM; ++m) {
                Array<Element, Policy::LaneMmaShape::kN> *dst_ptr = reinterpret_cast<Array<Element, Policy::LaneMmaShape::kN> *>
                    (ref_.data() + pointer_offset + ref_.offset({ mma_m * Delta::kRow + m, 0 }));
                for (int mma_n = 0; mma_n < Iterations::kColumn; ++mma_n) {
                    Array<Element, Policy::LaneMmaShape::kN> const *src_ptr = reinterpret_cast<Array<Element, Policy::LaneMmaShape::kN> const *>
                        (&frag) + mma_n + Iterations::kColumn * (m + mma_m * Policy::LaneMmaShape::kM);
                    dst_ptr[mma_n * Policy::WarpShape::kColumn] = *src_ptr;
                }
            }
        }
    }

    /// Stores a fragment to memory at the location pointed to by the iterator
    void store(Fragment const &frag) const {
        store_with_pointer_offset(frag, 0);
    }
};
```

### MmaSimt

当在CUDA Core硬件上使用SIMD指令实现矩阵乘加操作时，在Warp线程束层级的实现被抽象为gemm::warp::MmaSimt模板类。

在cutlass/gemm/warp/mma_simt.h头文件中，提供gemm::warp::MmaSimt的定义，如下所示。

```c++
/// Structure to compute the matrix product targeting CUDA cores and SIMT math instructions.
template <
    typename Shape_,      /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename ElementA_,   /// Data type of A elements
    typename LayoutA_,    /// Layout of A matrix (concept: MatrixLayout)
    typename ElementB_,   /// Data type of B elements
    typename LayoutB_,    /// Layout of B matrix (concept: MatrixLayout)
    typename ElementC_,   /// Element type of C matrix
    typename LayoutC_,    /// Layout of C matrix (concept: MatrixLayout)
    typename Policy_,     /// Shape of the warp in units of thread (concept: MmaSimtPolicy)
    int PartitionsK = 1,  /// Number of partitions along K dimension
    ComplexTransform TransformA = ComplexTransform::kNone,  /// Complex transformation on operand A
    ComplexTransform TransformB = ComplexTransform::kNone,  /// Complex transformation on operand B
    typename Enable = bool                                  /// Used for partial specialization
>
class MmaSimt {
public:
    using Shape = Shape_;                     /// Shape of warp-level matrix operation (concept: GemmShape)
    using ElementA = ElementA_;               /// Data type of multiplicand A
    using LayoutA = LayoutA_;                 /// Layout of multiplicand A
    using ElementB = ElementB_;               /// Data type of multiplicand B
    using LayoutB = LayoutB_;                 /// Layout of multiplicand B
    using ElementC = ElementC_;               /// Data type of accumulator matrix C
    using LayoutC = LayoutC_;                 /// Layout of accumulator matrix C
    using Policy = Policy_;                   /// Shape of the warp in units of thread (concept: MmaLanePolicySimt)
    using ArchTag = arch::Sm50;               /// Hard-coded for now
    using OperatorClass = arch::OpClassSimt;  /// Indicates class of matrix operator
    static ComplexTransform const kTransformA = TransformA;  /// Complex transform on A operand
    static ComplexTransform const kTransformB = TransformB;  /// Complex transform on B operand

    /// Layout of element at thread level
    using ThreadLayoutA = typename platform::conditional<
        platform::is_same<layout::ColumnMajorInterleaved<4>, LayoutA>::value, layout::ColumnMajor,
        typename platform::conditional<platform::is_same<layout::RowMajorInterleaved<4>, LayoutA>::value, layout::RowMajor, LayoutA>::type
    >::type;
    using ThreadLayoutB = typename platform::conditional<
        platform::is_same<layout::ColumnMajorInterleaved<4>, LayoutB >::value, layout::ColumnMajor,
        typename platform::conditional<platform::is_same<layout::RowMajorInterleaved<4>, LayoutB >::value, layout::RowMajor, LayoutB>::type
    >::type;

    /// Thread-level matrix multiply accumulate operator
    using ThreadMma = thread::Mma<
        GemmShape<Shape::kM / Policy::WarpShape::kRow, Shape::kN / Policy::WarpShape::kColumn, Policy::LaneMmaShape::kK>,
        ElementA, ThreadLayoutA, ElementB, ThreadLayoutB, ElementC, LayoutC, arch::OpMultiplyAdd>;
    using ArchMmaOperator = typename ThreadMma::ArchMmaOperator;  /// Underlying matrix multiply operator (concept: arch::Mma)
    using MathOperator = typename ArchMmaOperator::Operator;      /// Indicates math operator
    using InstructionShape = GemmShape<1, 1, 1>;                  /// Shape of the underlying instruction

public:
    /// Iterates over the A operand in memory
    using IteratorA = MmaSimtTileIterator<MatrixShape<Shape::kM, Policy::LaneMmaShape::kK>,
        Operand::kA, ElementA, LayoutA, Policy, PartitionsK, Shape::kK>;
    using FragmentA = typename IteratorA::Fragment;  /// Storage for A tile

    /// Iterates over the B operand in memory
    using IteratorB = MmaSimtTileIterator<MatrixShape<Policy::LaneMmaShape::kK, Shape::kN>,
        Operand::kB, ElementB, LayoutB, Policy, PartitionsK, Shape::kK>;
    using FragmentB = typename IteratorB::Fragment;  /// Storage for B tile

    /// Iterates over the C operand in memory
    using IteratorC = MmaSimtTileIterator<MatrixShape<Shape::kM, Shape::kN>, Operand::kC, ElementC, LayoutC, Policy>;
    using FragmentC = typename ThreadMma::FragmentC;  /// Storage for C tile

public:
    /// Constructor
    MmaSimt() {}

    /// Performs a warp-level matrix multiply-accumulate operation
    void operator()(FragmentC &d, FragmentA a, FragmentB b, FragmentC const &c, int group_idx = 0) const {
        ThreadMma mma;
        if (kTransformA == ComplexTransform::kConjugate) { a = conjugate<FragmentA>()(a); }
        if (kTransformB == ComplexTransform::kConjugate) { b = conjugate<FragmentB>()(b); }
        mma(d, a, b, c);
    }
};
```

### DefaultMmaTensorOp

当使用Tensor Core硬件实现矩阵乘加操作时，CUTLASS提供一个默认的模板配置，抽象为gemm::warp::DefaultMmaTensorOp模板类。

在cutlass/gemm/warp/default_mma_tensor_op.h头文件中，提供gemm::warp::DefaultMmaTensorOp的定义，如下所示。

```c++
template <
    typename WarpShape_,                       /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename InstructionShape_,                /// Shape of one matrix production operation (concept: GemmShape)
    typename ElementA_,                        /// Data type of A elements
    typename LayoutA_,                         /// Layout of A matrix (concept: MatrixLayout)
    typename ElementB_,                        /// Data type of B elements
    typename LayoutB_,                         /// Layout of B matrix (concept: MatrixLayout)
    typename ElementC_,                        /// Element type of C matrix
    typename LayoutC_,                         /// Layout of C matrix (concept: MatrixLayout)
    typename Operator_ = arch::OpMultiplyAdd,  /// Operator describing the tensor operation
    int PartitionsK = 1,                       /// Number of partitions along K dimension
    /// Store the accumulators in row major or column major. Row major is used when output layout is interleaved.
    bool AccumulatorsInRowMajor = false
>
struct DefaultMmaTensorOp;

/// Partial specialization for m-by-n-by-kgroup
template <
    typename WarpShape_,         /// Shape of one matrix production operation (concept: GemmShape)
    typename InstructionShape_,  /// Shape of one matrix production operation (concept: GemmShape)
    typename ElementA,           /// Data type of A elements
    typename LayoutA,            /// Layout of A matrix (concept: MatrixLayout)
    typename ElementB,           /// Data type of B elements
    typename LayoutB,            /// Layout of B matrix (concept: MatrixLayout)
    typename ElementC,           /// Element type of C matrix
    typename LayoutC,            /// Layout of C matrix (concept: MatrixLayout)
    typename Operator_,          /// Operator describing the tensor operation
    int PartitionsK,             /// Number of partitions along K dimension
    /// Store the accumulators in row major or column major. Row major is used when output layout is interleaved.
    bool AccumulatorsInRowMajor
>
struct DefaultMmaTensorOp {
    using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<cutlass::arch::Mma<InstructionShape_, 32, ElementA, cutlass::layout::RowMajor,
            ElementB, cutlass::layout::ColumnMajor, ElementC, cutlass::layout::RowMajor, Operator_>, cutlass::MatrixShape<1, 1>>;

    // Define the warp-level tensor op
    using Type = cutlass::gemm::warp::MmaTensorOp<WarpShape_, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
        Policy, PartitionsK, AccumulatorsInRowMajor>;
};
```

### MmaTensorOpPolicy

当使用Tensor Core硬件实现矩阵乘加操作时，需要明确，所使用的底层arch::Mma<>操作，由此指定具体使用的mma.xxx指令，以及一个Warp线程束所负责的多个MMA形状之间的间距，间距以一个MMA形状为单位。这些属性被抽象为gemm::warp::MmaTensorOpPolicy模板类。

在cutlass/gemm/warp/mma_tensor_op_policy.h头文件中，提供gemm::warp::MmaTensorOpPolicy的定义，如下所示。

```c++
/// Mma Tensor Op Policy
template <
    typename Operator_,  ///< hardware instruction(s) performing TensorOp (concept: arch::Mma)
    typename OpDelta_    ///< distance between operations (concept: MatrixShape)
>
struct MmaTensorOpPolicy {
    using Operator = Operator_;  ///< hardware instruction(s) performing TensorOp (concept: arch::Mma)
    using OpDelta = OpDelta_;    ///< distance between operations (concept: MatrixShape)
    using MmaShape = typename Operator::Shape;
};
```

### MmaTensorOpMultipicandTileIterator

当在线程束Warp层级执行矩阵乘加计算时，需要将数据从共享内存中读取到寄存器中，这被抽象为gemm::warp::MmaTensorOpMultipicandTileIterator模板类，分别适用于操作数矩阵A和矩阵B。值得注意的是，在从共享内存中加载矩阵数据时，使用的是ldmatrix指令。

![](CUTLASS模板库.assets/gemm-warp-MmaTensorOpMultiplicandTileIterator.png)

在cutlass/gemm/warp/mma_tensor_op_tile_iterator.h头文件中，提供gemm::warp::MmaTensorOpMultipicandTileIterator的定义，如下所示。

```c++
template <
    typename Shape_,             /// Size of the matrix to load (concept: MatrixShape)
    Operand Operand,             /// Operand identity
    typename Element_,           /// Data type of A elements
    typename Layout_,            /// Layout of operand
    typename InstructionShape_,  /// Shape of one matrix production operation (concept: GemmShape)
    int OpDelta_,                /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
    int Threads,                 /// Number of threads participating in one matrix operation
    int PartitionsK_ = 1         /// Number of partitions along K dimension
>
class MmaTensorOpMultiplicandTileIterator;
```

当使用layout::TensorOpMultiplicandCongruous基础布局时，gemm::warp::MmaTensorOpMultipicandTileIterator访问器的实现代码如下所示。

```c++
/// This tile iterator is specialized for 32-thread TensorOps. 
/// It uses LDSM to load from shared memory and therefore must be initialized with a TensorRef to shared memory.
/// Satisfies: ReadableRandomAccessContiguousTileIteratorConcept
template <
    typename Shape_,             /// Size of the matrix to load (concept: PitchLinearShape)
    Operand Operand_,            /// Identifies A or B multiplicand
    typename Element_,           /// Data type of elements
    typename InstructionShape_,  /// Shape of one matrix product operation (concept: PitchLinearShape)
    int OpDelta_,                /// Interval between adjacent *MMA instructions (in units of MMA instructions)
    int PartitionsK_             /// Number of partitions along K dimension
>
class MmaTensorOpMultiplicandTileIterator<Shape_, Operand_, Element_,
    cutlass::layout::TensorOpMultiplicandCongruous<sizeof_bits<Element_>::value, 64>,
    InstructionShape_, OpDelta_, 32, PartitionsK_
> {
public:
    using Shape = Shape_;                      /// Shape of tile to load (concept: PitchLinearShape)
    static Operand const kOperand = Operand_;  /// Operand tag
    using Element = Element_;                  /// Element type
    using Layout = cutlass::layout::TensorOpMultiplicandCongruous<sizeof_bits<Element_>::value, 64>;  /// Layout of source tile
    using InstructionShape = InstructionShape_;    /// Shape of one matrix product operation (concept: GemmShape)
    static int const kOpDelta = OpDelta_;          /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
    static int const kThreads = 32;                /// Number of participating threads
    static int const kPartitionsK = PartitionsK_;  /// Number of partitions along K dimension

    using TensorRef = TensorRef<Element, Layout>;                   /// TensorRef type for loading element from a tensor
    using Index = typename TensorRef::Index;                        /// Index type
    using LongIndex = typename TensorRef::LongIndex;                /// Long Index type
    using StrideIndex = typename TensorRef::Layout::Stride::Index;  /// Long Index type
    using TensorCoord = typename TensorRef::TensorCoord;            /// Coordinate for an element in the tensor

    /// Internal structure of iterator - made public to enable introspection
    struct Policy {
        /// Determine number of elements along outer and inner dimension per individual LDSM op (.m8n8)
        static int const kLdsmOpOuter = Layout::kElementsPerAccess;  /// along InstructionShape::kContiguous
        static int const kLdsmOpInner = 8;                           /// along InstructionShape::kStrided

        /// Shape of one individual LDSM instruction.
        /// 使用一条`ldmatrix.sync.aligned.m8n8.x4.shared.b16`指令，同时加载4个矩阵（8x8），这4个矩阵的排布即是LdsmShape
        static int const LdsmShapeStrided = InstructionShape::kStrided / kLdsmOpInner;
        static int const LdsmShapeContiguous = 4 / LdsmShapeStrided;
        using LdsmShape = layout::PitchLinearShape<LdsmShapeContiguous, LdsmShapeStrided>;  // <4, 1> or <2, 2>

        /// Number and arrangement of LDSM instructions
        using LdsmIterations = layout::PitchLinearShape<Shape::kContiguous / Layout::kElementsPerAccess / LdsmShapeContiguous, 1>;
        /// Number of groups for each tile
        static int const kGroupsPerTile = Shape::kStrided / InstructionShape::kStrided;
    };

    /// Fragment object holding a thread's part of a tile
    using Fragment = Array<Element, Shape::kContiguous * InstructionShape::kStrided / kThreads>;

private:
    /// Pointer type used for accesses
    using AccessType = Array<Element, Layout::kElementsPerAccess>;

    /// Number of internal pointers needed to reference shared memory
    static int const kPointerCount = Layout::TileShape::kContiguous / Policy::LdsmShape::kContiguous;

    AccessType const *pointer_[kPointerCount];  /// Shared memory base pointers - not advanced
    StrideIndex stride_;                        /// Layout object storing stride values
    Index byte_offset_;                         /// Byte offset incremented as iterator advances
    int k_group_idx_;                           /// Internal counter used to jump to next K partition

public:
    /// Default ctor constructs null iterator
    MmaTensorOpMultiplicandTileIterator() : stride_(0), byte_offset_(0) {}

    /// Constructor from TensorRef
    MmaTensorOpMultiplicandTileIterator(TensorRef const &ref, int lane_id)
        : stride_(ref.stride(0) / Layout::kElementsPerAccess), byte_offset_(0), k_group_idx_(0) {
        // 因为一个`ldmatrix.x4`指令加载4个矩阵（8×8），共32个行地址由32个线程负责，那么
        // 一个quad指的是32的1/4，即是8；一个quad_pair指的是32的1/4的1/2，即是4；一个quad_quad指的是32的1/4的1/4，即是2
        int quad_pair = (lane_id >> 3);          // 0 ~ 3
        int quad_quad = (lane_id >> 4);          // 0 ~ 1
        int lane_in_quad = (lane_id & 3);        // 0 ~ 3
        int lane_in_quad_pair = (lane_id & 7);   // 0 ~ 7
        int lane_in_quad_quad = (lane_id & 15);  // 0 ~ 15
        
        for (int i = 0; i < kPointerCount; ++i) {
            int partition_contiguous_idx = -1;
            int access_contiguous_idx = -1;
            int access_strided_idx = -1;
            // Q stands for one 8x128bit block, i.e. one 8x8 matrix with .b16 type
            if (Policy::LdsmShape::kContiguous == 4) {
                // Matrix multiply .m16n8k8 A/B
                // Q0 Q1 Q2 Q3
                // Four blocks are next to each other in the contiguous dimension.
                partition_contiguous_idx = ((lane_in_quad_pair >> 2) ^ i);
                access_contiguous_idx = (quad_pair ^ lane_in_quad);
                access_strided_idx = lane_in_quad_pair;
            } else if (Policy::LdsmShape::kContiguous == 2 && kOperand == Operand::kA) {
                // Matrix multiply .m16n8k16 A
                // Q0 Q1
                // Q2 Q3
                partition_contiguous_idx = ((lane_in_quad_pair >> 2) ^ (i >> 1));
                access_contiguous_idx = (((quad_pair & 1) + ((i & 1) << 1)) ^ lane_in_quad);
                access_strided_idx = lane_in_quad_pair + (lane_id >> 4 << 3);
            } else if (Policy::LdsmShape::kContiguous == 2 && kOperand == Operand::kB) {
                // Matrix multiply .m16n8k16 B
                // Q0 Q2
                // Q1 Q3
                partition_contiguous_idx = ((lane_in_quad_pair >> 2) ^ (i >> 1));
                access_contiguous_idx = ((quad_quad + ((i & 1) << 1)) ^ lane_in_quad);
                access_strided_idx = lane_in_quad_quad;
            } else if (Policy::LdsmShape::kContiguous == 1) {
                // Matrix multiply .m16n8k32.SP B
                // Q0
                // Q1
                // Q2
                // Q3
                partition_contiguous_idx = ((lane_in_quad_pair >> 2) ^ (i >> 2));
                access_contiguous_idx = ((i & 3) ^ lane_in_quad);
                access_strided_idx = lane_id;
            }
            int access_contiguous = partition_contiguous_idx * Layout::PartitionShape::kContiguous + access_contiguous_idx;
            int access_strided = access_strided_idx;
            pointer_[i] = reinterpret_cast<AccessType const *>(ref.data()) + access_contiguous + access_strided * stride_;
        }
    }

    /// Adds a pointer offset to internal pointer(s) to advance through memory
    MmaTensorOpMultiplicandTileIterator & add_pointer_offset(LongIndex offset) {
        byte_offset_ += offset * sizeof(Element);
        return *this;
    }

    /// Advances an iterator along logical dimensions of matrix in units of whole tiles
    MmaTensorOpMultiplicandTileIterator & add_tile_offset(TensorCoord const &tile_offset) {
        int contiguous_offset = tile_offset.contiguous();
        if (Shape::kContiguous == Layout::PartitionShape::kContiguous * Layout::kElementsPerAccess) {
            if (tile_offset.contiguous() % 2) {
                for (int i = 0; i < kPointerCount / 2; ++i) {
                    AccessType const *tmp_pointer = pointer_[i];
                    pointer_[i] = pointer_[i + kPointerCount / 2];
                    pointer_[i + kPointerCount / 2] = tmp_pointer;
                }
            }
            contiguous_offset = (tile_offset.contiguous() >> 1) << 1;
        }
        int offset = (tile_offset.strided() * InstructionShape::kStrided) * stride_ * Layout::kElementsPerAccess
            + contiguous_offset * Shape::kContiguous;
        add_pointer_offset(offset);
        return *this;
    }

    /// Advances the iterator along the advance dimension
    MmaTensorOpMultiplicandTileIterator & operator++() {
        add_tile_offset({ 0, 1 });
        if (kPartitionsK > 1) {
            ++k_group_idx_;
            // Jump to next stage
            if (k_group_idx_ == Policy::kGroupsPerTile) {
                k_group_idx_ = 0;
                add_tile_offset({ 0, ((kPartitionsK - 1) * Policy::kGroupsPerTile) });
            }
        }
        return *this;
    }

    /// Loads a fragment from memory with additional logical offset
    void load_with_byte_offset(Fragment &frag, Index byte_offset) const {
        Array<unsigned, Policy::LdsmShape::kCount> *fetch_ptr = reinterpret_cast<Array<unsigned, Policy::LdsmShape::kCount> *>(&frag);
        for (int s = 0; s < Policy::LdsmIterations::kStrided; ++s) {
            for (int c = 0; c < Policy::LdsmIterations::kContiguous; ++c) {
                int access_idx = c + s * Policy::LdsmIterations::kContiguous;
                AccessType const *source_ptr = pointer_[c % kPointerCount] + Layout::TileShape::kContiguous * (c / kPointerCount)
                    + Policy::kLdsmOpInner * Policy::LdsmShape::kStrided * s * stride_;
                char const *source_byte_ptr = reinterpret_cast<char const *>(source_ptr) + byte_offset + byte_offset_;
                cutlass::arch::ldsm<layout::ColumnMajor, Policy::LdsmShape::kCount>(fetch_ptr[access_idx], source_byte_ptr);
            }
        }
    }

    /// Loads a fragment from memory at the location pointed to by the iterator.
    void load(Fragment &frag) const {
        load_with_byte_offset(frag, 0);
    }
        
    /// Notify the iterator which k-group it is currently pointing to.
    /// This does not advance the iterator. Rather, it overrides its internal tracking with constant-valued k-group index
    /// to enable the compiler to fold constants and achieve more efficient code.
    /// This is used by some nontrivial permuted layouts.
    void set_kgroup_index(int k_group) {
        // no op
    }
};
```

在实现时，会根据所访问的矩阵操作数，以及矩阵操作数的排列布局，来提供最高效的访问方式，即模板类的部分实例化。例如，对于矩阵A而言，最优布局是layout::ColumnMajorTensorOpMultiplicandCongruous布局，对于矩阵B而言，最优布局是layout::RowMajorTensorOpMultiplicandCongruous布局。

当矩阵A使用layout::ColumnMajorTensorOpMultiplicandCongruous布局时，gemm::warp::MmaTensorOpMultipicandTileIterator访问器的实现代码如下。当矩阵B使用layout::RowMajorTensorOpMultiplicandCongruous布局时，gemm::warp::MmaTensorOpMultipicandTileIterator访问器的实现代码与矩阵A的情况类似，此处不再赘述。

```c++
/// This tile iterator is specialized for 32-thread TensorOps.
/// It uses LDSM to load from shared memory and therefore must be initialized with a TensorRef to shared memory. 
/// Satisfies: ReadableRandomAccessContiguousTileIteratorConcept
template <
    typename Shape_,             /// Size of the matrix to load (concept: MatrixShape)
    Operand Operand_,            /// Identifies A or B multiplicand
    typename Element_,           /// Data type of elements
    typename InstructionShape_,  /// Shape of one matrix product operation (concept: MatrixShape)
    int OpDelta_,                /// Interval between adjacent *MMA instructions (in units of MMA instructions)
    int Crosswise,               /// Element number when the layout crosses (in units of elements)
    int PartitionsK_             /// Number of partitions along K dimension
>
class MmaTensorOpMultiplicandTileIterator<Shape_, Operand_, Element_,
    cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<sizeof_bits<Element_>::value, Crosswise>,
    InstructionShape_, OpDelta_, 32, PartitionsK_
> {
public:
    using Shape = Shape_;                      /// Shape of tile to load (concept: PitchLinearShape)
    static Operand const kOperand = Operand_;  /// Operand tag
    using Element = Element_;                  /// Element type
    static int const kCrosswise = Crosswise;   /// MBlock or NBlock size
    using Layout = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<sizeof_bits<Element_>::value, kCrosswise>;  /// source layout
    using InstructionShape = InstructionShape_;  /// Shape of one matrix product operation (concept: MatrixShape)
    static int const kOpDelta = OpDelta_;        /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
    static int const kThreads = 32;              /// Number of participating threads

    using TensorRef = TensorRef<Element, Layout>;                   /// TensorRef type for loading element from a tensor
    using Index = typename TensorRef::Index;                        /// Index type
    using LongIndex = typename TensorRef::LongIndex;                /// Long Index type
    using StrideIndex = typename TensorRef::Layout::Stride::Index;  /// Long Index type
    using TensorCoord = typename TensorRef::TensorCoord;            /// Coordinate for an element in the tensor

    /// Underlying tile iterator implementation
    using Base = MmaTensorOpMultiplicandTileIterator<
        layout::PitchLinearShape<Shape::kRow, Shape::kColumn>, kOperand, Element,
        layout::TensorOpMultiplicandCongruous<sizeof_bits<Element_>::value, kCrosswise>,
        layout::PitchLinearShape<InstructionShape::kRow, InstructionShape::kColumn>,
        kOpDelta, kThreads, PartitionsK_>;

    /// Underlying tile iterator
    Base iterator_;
        
    /// Fragment object holding a thread's part of a tile
    using Fragment = typename Base::Fragment;

public:
    /// Default Constructor constructs null iterator
    MmaTensorOpMultiplicandTileIterator() {}

    /// Constructor from TensorRef
    MmaTensorOpMultiplicandTileIterator(TensorRef const &ref, int lane_id) : iterator_({ ref.data(), ref.stride() }, lane_id) {}

    /// Adds a pointer offset to internal pointer(s) to advance through memory
    MmaTensorOpMultiplicandTileIterator & add_pointer_offset(LongIndex offset) {
        iterator_.add_pointer_offset(offset);
        return *this;
    }

    /// Advances an iterator along logical dimensions of matrix in units of whole tiles
    MmaTensorOpMultiplicandTileIterator & add_tile_offset(TensorCoord const &tile_offset) {
        iterator_.add_tile_offset({ tile_offset.row(), tile_offset.column() });
        return *this;
    }

    /// Advances the iterator along the advance dimension
    MmaTensorOpMultiplicandTileIterator & operator++() {
        ++iterator_;
        return *this;
    }

    /// Loads a fragment from memory with additional logical offset
    void load_with_byte_offset(Fragment &frag, Index byte_offset) const {
        iterator_.load_with_byte_offset(frag, byte_offset);
    }

    /// Loads a fragment from memory at the location pointed to by the iterator.
    void load(Fragment &frag) const {
        iterator_.load(frag);
    }

    /// Notify the iterator which k-group it is currently pointing to.
    /// This does not advance the iterator. Rather, it overrides its internal tracking with constant-valued k-group index 
    /// to enable the compiler to fold constants and achieve more efficient code.
    /// This is used by some nontrivial permuted layouts.
    void set_kgroup_index(int k_group) {
        iterator_.set_kgroup_index(k_group);
    }
};
```

### MmaTensorOpAccumulatorTileIterator

当在线程束Warp层级执行矩阵乘加计算时，需要将数据从共享内存中读取到寄存器中，并将计算结果从寄存器中写入到共享内存中，这被抽象为gemm::warp::MmaTensorOpAccumulatorTileIterator模板类，适用于累加器矩阵C和矩阵D。

![](CUTLASS模板库.assets/gemm-warp-MmaTensorOpAccumulatorTileIterator.png)

在cutlass/gemm/warp/mma_tensor_op_tile_iterator.h头文件中，提供gemm::warp::MmaTensorOpAccumulatorTileIterator的定义，如下所示。

```c++
template <
    typename Shape_,             /// Size of the matrix to load (concept: MatrixShape)
    typename Element_,           /// Element type
    typename Layout_,            /// Layout of operand in memory
    typename InstructionShape_,  /// Shape of one matrix product operation (concept: MatrixShape)
    typename OpDelta_            /// Interval between adjacent *MMA instructions (in units of MMA instructions, concept: MatrixShape)
>
class MmaTensorOpAccumulatorTileIterator;
```

当矩阵C和矩阵D使用layout::RowMajor布局时，性能最优，gemm::warp::MmaTensorOpAccumulatorTileIterator访问器的实现代码如下。

```c++
/// This tile iterator is specialized for 32-thread TensorOps.
/// It is used to load or store accumulators from memory and is agnostic to layout.
/// It could be faster if it assumed row-major accumulator layout.
/// Satisfies: ReadableRandomAccessContiguousTileIteratorConcept | WriteableRandomAccessContiguousTileIteratorConcept
template <
    typename Shape_,             /// Size of the matrix to load (concept: MatrixShape)
    typename Element_,           /// Element type
    typename InstructionShape_,  /// Shape of one matrix product operation (concept: MatrixShape)
    typename OpDelta_            /// Interval between adjacent *MMA instructions (in units of MMA instructions, concept: MatrixShape)
>
class MmaTensorOpAccumulatorTileIterator<Shape_, Element_, cutlass::layout::RowMajor, InstructionShape_, OpDelta_> {
public:
    using Shape = Shape_;                         /// Shape of tile to load (concept: MatrixShape)
    static Operand const kOperand = Operand::kC;  /// Operand tag
    using Element = Element_;                     /// Element type
    using Layout = cutlass::layout::RowMajor;     /// Layout of source tile
    using InstructionShape = InstructionShape_;   /// Shape of one matrix product operation (concept: MatrixShape)
    using OpDelta = OpDelta_;                     /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
    static int const kThreads = 32;               /// Number of participating threads

    using TensorRef = TensorRef<Element, Layout>;         /// TensorRef type for loading element from a tensor
    using Index = typename TensorRef::Index;              /// Index type
    using LongIndex = typename TensorRef::LongIndex;      /// Long Index type
    using TensorCoord = typename TensorRef::TensorCoord;  /// Coordinate for an element in the tensor

    /// Internal structure of iterator - made public to enable introspection
    struct Policy {
        /// Number of mma operations performed
        using MmaIterations = MatrixShape<(Shape::kRow + InstructionShape::kM - 1) / InstructionShape::kM,
            (Shape::kColumn + InstructionShape::kN - 1) / InstructionShape::kN>;
    };

    /// Fragment object holding a thread's part of a tile
    using Fragment = Array<Element, Policy::MmaIterations::kCount * InstructionShape::kMN / kThreads>;

private:
    // Assume accumulator tile is an arrangement of 8-by-8 tiles replicated over the entire shape,
    // with each quad mapped to one row and each thread mapped to 1/4 of the elements of that row.
    // The accumulators within one row are assumed to be consecutive.
    static int const kElementsPerAccess = InstructionShape::kN / 4;
    static int const kRowsPerTile = 8;
    static int const kAccumulatorRows = InstructionShape::kM / kRowsPerTile;

    /// Reference to output tensor
    TensorRef ref_;

public:
    /// Default Constructor constructs null iterator
    MmaTensorOpAccumulatorTileIterator() {}

    /// Constructor from TensorRef
    MmaTensorOpAccumulatorTileIterator(TensorRef const &ref, int lane_id) : ref_(ref) {
        // Each thread has two elements with .b16 type
        // | T0  | T1  | T2  | T3  |
        // | T4  | T5  | T6  | T7  |
        // | T8  | T9  | T10 | T11 |
        // | T12 | T13 | T14 | T15 |
        // | T16 | T17 | T18 | T19 |
        // | T20 | T21 | T22 | T23 |
        // | T24 | T25 | T26 | T27 |
        // | T28 | T29 | T30 | T31 |
        int quad = (lane_id >> 2);
        int lane_in_quad = (lane_id & 3);
        MatrixCoord lane_offset(quad, lane_in_quad * kElementsPerAccess);
        ref_.add_coord_offset(lane_offset);
    }

    /// Advances an iterator along logical dimensions of matrix in units of whole tiles
    MmaTensorOpAccumulatorTileIterator & add_tile_offset(TensorCoord const &tile_offset) {
        ref_.add_coord_offset(tile_offset * make_Coord(Shape::kRow, Shape::kColumn));
        return *this;
    }

    /// Loads a fragment from memory with additional logical offset
    void load_with_pointer_offset(Fragment &frag, Index pointer_offset) const {
        TensorRef offset_ref(ref_);
        offset_ref.add_pointer_offset(pointer_offset);
        for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn; ++mma_n) {
            for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {
                int mma_accum_start = kAccumulatorRows * kElementsPerAccess * (mma_n * Policy::MmaIterations::kRow + mma_m);
                for (int row = 0; row < kAccumulatorRows; ++row) {
                    for (int col = 0; col < kElementsPerAccess; ++col) {
                        int accum_m = mma_m * InstructionShape::kM * OpDelta::kRow + row * kRowsPerTile;
                        int accum_n = mma_n * InstructionShape::kN * OpDelta::kColumn + col;
                        frag[mma_accum_start + row * kElementsPerAccess + col] = offset_ref.at({ accum_m, accum_n });
                    }
                }
            }
        }
    }

    /// Loads a fragment from memory at the location pointed to by the iterator.
    void load(Fragment &frag) const {
        load_with_pointer_offset(frag, 0);
    }

    /// Stores a fragment to memory with additional pointer offset
    void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) const {
        TensorRef offset_ref(ref_);
        offset_ref.add_pointer_offset(pointer_offset);
        for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn; ++mma_n) {
            for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {
                int mma_accum_start = kAccumulatorRows * kElementsPerAccess * (mma_n * Policy::MmaIterations::kRow + mma_m);
                for (int row = 0; row < kAccumulatorRows; ++row) {
                    for (int col = 0; col < kElementsPerAccess; ++col) {
                        int accum_m = mma_m * InstructionShape::kM * OpDelta::kRow + row * kRowsPerTile;
                        int accum_n = mma_n * InstructionShape::kN * OpDelta::kColumn + col;
                        offset_ref.at({ accum_m, accum_n }) = frag[mma_accum_start + row * kElementsPerAccess + col];
                    }
                }
            }
        }
    }

    /// Stores a fragment to memory
    void store(Fragment const &frag) const {
        store_with_pointer_offset(frag, 0);
    }
};
```

### MmaTensorOp

当在Tensor Core硬件上使用mma.xxx系列指令实现矩阵乘加操作时，在Warp线程束层级的实现被抽象为gemm::warp::MmaTensorOp模板类。

在cutlass/gemm/warp/mma_tensor_op.h头文件中，提供gemm::warp::MmaTensorOp的定义，如下所示。

```c++
/// Structure to compute the matrix product targeting Tensor Cores.
template <
    typename Shape_,        /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename ElementA_,     /// Data type of A elements
    typename LayoutA_,      /// Layout of A matrix (concept: MatrixLayout)
    typename ElementB_,     /// Data type of B elements
    typename LayoutB_,      /// Layout of B matrix (concept: MatrixLayout)
    typename ElementC_,     /// Element type of C matrix
    typename LayoutC_,      /// Layout of C matrix (concept: MatrixLayout)
    typename Policy_,       /// Policy describing warp-level MmaTensorOp (concept: MmaTensorOp policy)
    int PartitionsK_ = 1,   /// Number of partitions along K dimension
    /// Store the accumulators in row major or column major. Row major is used when output layout is interleaved.
    bool AccumulatorsInRowMajor = false,
    typename Enable = bool  /// Used for partial specialization
>
class MmaTensorOp {
public:
    using Shape = Shape_;        /// Shape of warp-level matrix operation (concept: GemmShape)
    using ElementA = ElementA_;  /// Data type of multiplicand A
    using LayoutA = LayoutA_;    /// Layout of multiplicand A
    using ElementB = ElementB_;  /// Data type of multiplicand B
    using LayoutB = LayoutB_;    /// Layout of multiplicand B
    using ElementC = ElementC_;  /// Data type of accumulator matrix C
    using LayoutC = LayoutC_;    /// Layout of accumulator matrix C

    using Policy = Policy_;                                    /// Shape of the warp in units of thread (concept: MmaLanePolicySimt)
    using ArchMmaOperator = typename Policy::Operator;         /// Underlying matrix multiply operator (concept: arch::Mma)
    using MathOperator = typename ArchMmaOperator::Operator;   /// Indicates math operator
    using ArchTag = typename ArchMmaOperator::ArchTag;         /// Architecture tag from underlying instruction
    using OperatorClass = arch::OpClassTensorOp;               /// Indicates class of matrix operator
    using InstructionShape = typename ArchMmaOperator::Shape;  /// Shape of underlying instruction

    static int const kThreadCount = 32;            /// Number of threads participating in warp-level matrix product
    static int const kPartitionsK = PartitionsK_;  /// Number of partitions along K dimension

public:
    /// Iterates over the A operand in memory
    using IteratorA = MmaTensorOpMultiplicandTileIterator<MatrixShape<Shape::kM, Shape::kK>, Operand::kA, ElementA, LayoutA,
        MatrixShape<ArchMmaOperator::Shape::kM, ArchMmaOperator::Shape::kK>, Policy::OpDelta::kRow, kThreadCount, kPartitionsK>;
    using FragmentA = typename IteratorA::Fragment;  /// Storage for A tile
    using TransformedFragmentA = Array<typename ArchMmaOperator::ElementA, FragmentA::kElements>;  /// Storage for transformed A tile

    /// Iterates over the B operand in memory
    using IteratorB = MmaTensorOpMultiplicandTileIterator<MatrixShape<Shape::kK, Shape::kN>, Operand::kB, ElementB, LayoutB,
        MatrixShape<ArchMmaOperator::Shape::kK, ArchMmaOperator::Shape::kN>, Policy::OpDelta::kRow, kThreadCount, kPartitionsK>;
    using FragmentB = typename IteratorB::Fragment;  /// Storage for B tile
    using TransformedFragmentB = Array<typename ArchMmaOperator::ElementB, FragmentB::kElements>;  /// Storage for transformed B tile

    /// Iterates over the C operand in memory
    using IteratorC = MmaTensorOpAccumulatorTileIterator<MatrixShape<Shape::kM, Shape::kN>, ElementC, LayoutC,
        typename ArchMmaOperator::Shape, typename Policy::OpDelta>;
    using FragmentC = typename IteratorC::Fragment;  /// Storage for C tile

    /// Number of mma operations performed
    using MmaIterations = MatrixShape<(Shape::kM + ArchMmaOperator::Shape::kM - 1) / ArchMmaOperator::Shape::kM,
        (Shape::kN + ArchMmaOperator::Shape::kN - 1) / ArchMmaOperator::Shape::kN>;

public:
    /// Underlying matrix multiply operator (concept: arch::Mma)
    ArchMmaOperator mma;

public:
    /// Constructor
    MmaTensorOp() {}

    /// Performs a warp-level matrix multiply-accumulate operation
    void operator()(FragmentC &D, TransformedFragmentA const &A, TransformedFragmentB const &B, FragmentC const &C) const {
        using MmaOperandA = typename ArchMmaOperator::FragmentA;
        using MmaOperandB = typename ArchMmaOperator::FragmentB;
        using MmaOperandC = typename ArchMmaOperator::FragmentC;
        D = C;
        MmaOperandA const *ptr_A = reinterpret_cast<MmaOperandA const *>(&A);
        MmaOperandB const *ptr_B = reinterpret_cast<MmaOperandB const *>(&B);
        MmaOperandC *ptr_D = reinterpret_cast<MmaOperandC *>(&D);

        for (int n = 0; n < MmaIterations::kColumn; ++n) {
            for (int m = 0; m < MmaIterations::kRow; ++m) {
                int m_serpentine = ((n % 2) ? (MmaIterations::kRow - 1 - m) : m);
                mma(ptr_D[m_serpentine + n * MmaIterations::kRow], ptr_A[m_serpentine], ptr_B[n],
                    ptr_D[m_serpentine + n * MmaIterations::kRow]
                );
            }
        }
    }
};
```

## Threadblock Level

在cutlass/gemm/threadblock目录中，提供矩阵乘加操作在Threadblock线程块层级的实现，主要包括双缓冲的实现与多级流水线的实现，这些操作被抽象为gemm::threadblock::MmaPipelined模板类和gemm::threadblock::MmaMultiStage模板类。此外，还提供一些相关的辅助类型，例如默认执行配置等。此外，在线程块级别，负责将数据从全局内存加载到寄存器当中，以及将寄存器中数据写入到共享内存当中，这需要cutlass/transform目录中提供的数据转移器。

![](CUTLASS模板库.assets/gemm-threadblock.png)

### MmaPolicy

当在Threadblock线程块层级实现矩阵乘加操作时，需要明确，一个Warp线程束所使用的矩阵乘加实现、共享内存偏移、维度K上的划分数目。这些属性被抽象为gemm::threadblock::MmaPolicy模板类。

在cutlass/gemm/threadblock/mma_base.h头文件中，提供gemm::threadblock::MmaPolicy的定义，如下所示。

```c++
/// Policy object describing MmaTensorOp and MmaSimt
template <
    typename Operator_,      /// Warp-level GEMM operator (concept: gemm::warp::Mma)
    typename SmemPaddingA_,  /// Padding used for A operand in shared memory (concept: MatrixShape)
    typename SmemPaddingB_,  /// Padding used for B operand in shared memory (concept: MatrixShape)
    int PartitionsK = 1      /// Number of partitions of K dimension of GEMM
>
struct MmaPolicy {
    using Operator = Operator_;             /// Warp-level GEMM operator (concept: gemm::warp::MmaTensorOp or gemm::warp::MmaSimt)
    using SmemPaddingA = SmemPaddingA_;     /// Padding used for A operand in shared memory
    using SmemPaddingB = SmemPaddingB_;     /// Padding used for B operand in shared memory
    static int kPartitionsK = PartitionsK;  /// Number of partitions of K dimension
};
```

### DefaultMma

当在Threadblock线程块层级实现矩阵乘加操作时，CUTLASS提供一个默认的模板配置，抽象为gemm::threadblock::DefaultMma模板类。在默认的模板配置中，会使用一个默认的核心模板配置，抽象为gemm::threadblock::DefaultMmaCore模板类，见下一小节所述。此外，根据底层实现使用的CUDA Core硬件或Tensor Core硬件的不同，以及底层实现使用的双缓冲或多级流水线的方式不同，默认的核心模板配置会使用不同的Warp线程束层级的矩阵乘加实现，以及使用不同的数据转移器。

在cutlass/gemm/threadblock/default_mma.h头文件中，提供gemm::threadblock::DefaultMma的定义，如下所示。

```c++
template <
    typename ElementA_,            /// Element type for A matrix operand
    typename LayoutA_,             /// Layout type for A matrix operand
    int kAlignmentA,               /// Access granularity of A matrix in units of elements
    typename ElementB_,            /// Element type for B matrix operand
    typename LayoutB_,             /// Layout type for B matrix operand
    int kAlignmentB,               /// Access granularity of B matrix in units of elements
    typename ElementAccumulator_,  /// Element type for internal accumulation
    typename LayoutC_,             /// Layout type for C and D matrix operands
    typename OperatorClass_,       /// Operator class tag
    typename ArchTag_,             /// Tag indicating architecture to tune for
    typename ThreadblockShape_,    /// Threadblock-level tile size (concept: GemmShape)
    typename WarpShape_,           /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape_,    /// Instruction-level tile size (concept: GemmShape)
    int Stages,                    /// Number of stages used in the pipelined mainloop
    typename Operator,             /// Operation perfomed by GEMM
    /// Store the accumulators in row major or column major. Row major is used when output layout is interleaved.
    bool AccumulatorsInRowMajor = false,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear = SharedMemoryClearOption::kNone,
    bool GatherA = false,                         /// Gather operand A by using an index array
    bool GatherB = false,                         /// Gather operand B by using an index array
    typename PermuteALayout = layout::NoPermute,  /// Permute operand A
    typename PermuteBLayout = layout::NoPermute   /// Permute operand B
>
struct DefaultMma;
```

当使用gemm::threadblock::MmaPipelined实现时，gemm::threadblock::DefaultMma的实现代码如下所示，这可用CUDA Core或Tensor Core实现。

```c++
/// Specialization for row-major output (OperatorClass Simt)
template <
    typename ElementA,            /// Element type for A matrix operand
    typename LayoutA,             /// Layout type for A matrix operand
    int kAlignmentA,              /// Access granularity of A matrix in units of elements
    typename ElementB,            /// Element type for B matrix operand
    typename LayoutB,             /// Layout type for B matrix operand
    int kAlignmentB,              /// Access granularity of B matrix in units of elements
    typename ElementAccumulator,  /// Element type for internal accumulation
    typename LayoutC,             /// Layout type for C and D matrix operand
    typename ArchTag,             /// Tag indicating architecture to tune for
    typename ThreadblockShape,    /// Threadblock-level tile size (concept: GemmShape)
    typename WarpShape,           /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape,    /// Instruction-level tile size (concept: GemmShape)
    typename Operator,            /// Operation performed by GEMM
    bool GatherA,                 /// Gather operand A by using an index array
    bool GatherB,                 /// Gather operand B by using an index array
    typename PermuteALayout,      /// Permute operand A
    typename PermuteBLayout       /// Permute operand B
>
struct DefaultMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB, ElementAccumulator, LayoutC,
    arch::OpClassSimt, ArchTag, ThreadblockShape, WarpShape, InstructionShape, 2, Operator, false,
    SharedMemoryClearOption::kNone, GatherA, GatherB, PermuteALayout, PermuteBLayout
> {
    // Define the MmaCore components
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<ThreadblockShape, WarpShape, InstructionShape,
        ElementA, LayoutA, ElementB, LayoutB, ElementAccumulator, LayoutC, arch::OpClassSimt, 2, Operator>;

    // Define iterators over tiles from the A operand. Iterators to read from global memory.
    using IteratorA = cutlass::transform::threadblock::PredicatedTileIterator<
        cutlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>,
        ElementA, LayoutA, 1, typename MmaCore::IteratorThreadMapA, kAlignmentA, GatherA, PermuteALayout>;

    // Define iterators over tiles from the B operand. Iterators to read from global memory.
    using IteratorB = cutlass::transform::threadblock::PredicatedTileIterator<
        cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
        ElementB, LayoutB, 0, typename MmaCore::IteratorThreadMapB, kAlignmentB, GatherB, PermuteBLayout>;

    // Define the threadblock-scoped pipelined matrix multiply
    using ThreadblockMma = cutlass::gemm::threadblock::MmaPipelined<typename MmaCore::Shape,
        IteratorA, typename MmaCore::SmemIteratorA, IteratorB, typename MmaCore::SmemIteratorB,
        ElementAccumulator, LayoutC, typename MmaCore::MmaPolicy>;
};
```

```c++
/// Specialization for row-major output (OperatorClass TensorOp)
template <
    typename ElementA,                          /// Element type for A matrix operand
    typename LayoutA,                           /// Layout type for A matrix operand
    int kAlignmentA,                            /// Access granularity of A matrix in units of elements
    typename ElementB,                          /// Element type for B matrix operand
    typename LayoutB,                           /// Layout type for B matrix operand
    int kAlignmentB,                            /// Access granularity of B matrix in units of elements
    typename ElementAccumulator,                /// Element type for internal accumulation
    typename ArchTag,                           /// Tag indicating architecture to tune for
    typename ThreadblockShape,                  /// Threadblock-level tile size (concept: GemmShape)
    typename WarpShape,                         /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape,                  /// Instruction-level tile size (concept: GemmShape)
    typename Operator,                          /// Operation performed by GEMM
    SharedMemoryClearOption SharedMemoryClear,  /// Use zfill or predicate for out-of-bound cp.async
    bool GatherA,                               /// Gather operand A by using an index array
    bool GatherB,                               /// Gather operand B by using an index array
    typename PermuteALayout,                    /// Permute operand A
    typename PermuteBLayout                     /// Permute operand B
>
struct DefaultMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB, ElementAccumulator, layout::RowMajor,
    arch::OpClassTensorOp, ArchTag, ThreadblockShape, WarpShape, InstructionShape, 2, Operator, false, 
    SharedMemoryClear, GatherA, GatherB, PermuteALayout, PermuteBLayout
> {
    // Define the MmaCore components
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<ThreadblockShape, WarpShape, InstructionShape,
        ElementA, LayoutA, ElementB, LayoutB, ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp, 2, Operator>;

    // Define iterators over tiles from the A operand. Iterators to read from global memory.
    using IteratorA = cutlass::transform::threadblock::PredicatedTileIterator<
        cutlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>,
        ElementA, LayoutA, 1, typename MmaCore::IteratorThreadMapA, kAlignmentA, GatherA, PermuteALayout>;

    // Define iterators over tiles from the B operand. Iterators to read from global memory.
    using IteratorB = cutlass::transform::threadblock::PredicatedTileIterator<
        cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
        ElementB, LayoutB, 0, typename MmaCore::IteratorThreadMapB, kAlignmentB, GatherB, PermuteBLayout>;

    // Define the threadblock-scoped pipelined matrix multiply
    using ThreadblockMma = cutlass::gemm::threadblock::MmaPipelined<typename MmaCore::Shape,
        IteratorA, typename MmaCore::SmemIteratorA, IteratorB, typename MmaCore::SmemIteratorB,
        ElementAccumulator, layout::RowMajor, typename MmaCore::MmaPolicy>;
};
```

当使用gemm::threadblock::MmaMultiStage实现时，gemm::threadblock::DefaultMma的实现代码如下所示，这可用CUDA Core或Tensor Core实现。

```c++
/// Specialization for row-major output (OperatorClass Simt)
template <
    typename ElementA,            /// Element type for A matrix operand
    typename LayoutA,             /// Layout type for A matrix operand
    int kAlignmentA,              /// Access granularity of A matrix in units of elements
    typename ElementB,            /// Element type for B matrix operand
    typename LayoutB,             /// Layout type for B matrix operand
    int kAlignmentB,              /// Access granularity of B matrix in units of elements
    typename ElementAccumulator,  /// Element type for internal accumulation
    typename LayoutC,             /// Layout type for C and D matrix operand
    typename ArchTag,             /// Tag indicating architecture to tune for
    typename ThreadblockShape,    /// Threadblock-level tile size (concept: GemmShape)
    typename WarpShape,           /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape,    /// Instruction-level tile size (concept: GemmShape)
    int Stages,                   /// Number of stages used in the multistage mainloop
    typename Operator,            /// Operation perfomed by GEMM
    bool GatherA,                 /// Gather operand A by using an index array
    bool GatherB,                 /// Gather operand B by using an index array
    typename PermuteALayout,      /// Permute operand A
    typename PermuteBLayout       /// Permute operand B
>
struct DefaultMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB, ElementAccumulator, LayoutC,
    arch::OpClassSimt, ArchTag, ThreadblockShape, WarpShape, InstructionShape, Stages, Operator, false,
    SharedMemoryClearOption::kNone, GatherA, GatherB, PermuteALayout, PermuteBLayout
> {
    // Define the MmaCore components
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<ThreadblockShape, WarpShape, InstructionShape,
        ElementA, LayoutA, ElementB, LayoutB, ElementAccumulator, LayoutC, arch::OpClassSimt, Stages, Operator>;

    // Define iterators over tiles from the A operand. Iterators to read from global memory.
    using ThreadMapA = typename MmaCore::IteratorThreadMapA;
    using AccessTypeA = cutlass::Array<ElementA, kAlignmentA>;
    using IteratorA = cutlass::transform::threadblock::PredicatedTileAccessIterator<
        cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
        ElementA, LayoutA, 1, ThreadMapA, AccessTypeA, GatherA, PermuteALayout>;

    // Define iterators over tiles from the B operand. Iterators to read from global memory.
    using ThreadMapB = typename MmaCore::IteratorThreadMapB;
    using AccessTypeB = cutlass::Array<ElementB, kAlignmentB>;
    using IteratorB = cutlass::transform::threadblock::PredicatedTileAccessIterator<
        cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
        ElementB, LayoutB, 0, ThreadMapB, AccessTypeB, GatherB, PermuteBLayout>;

    // Define the threadblock-scoped multistage matrix multiply
    using ThreadblockMma = cutlass::gemm::threadblock::MmaMultistage<typename MmaCore::Shape,
        IteratorA, typename MmaCore::SmemIteratorA, MmaCore::kCacheOpA, IteratorB, typename MmaCore::SmemIteratorB, MmaCore::kCacheOpB,
        ElementAccumulator, LayoutC, typename MmaCore::MmaPolicy, Stages>;
};
```

```c++
/// Specialization for row-major output (OperatorClass TensorOp)
template <
    typename ElementA,                          /// Element type for A matrix operand
    typename LayoutA,                           /// Layout type for A matrix operand
    int kAlignmentA,                            /// Access granularity of A matrix in units of elements
    typename ElementB,                          /// Element type for B matrix operand
    typename LayoutB,                           /// Layout type for B matrix operand
    int kAlignmentB,                            /// Access granularity of B matrix in units of elements
    typename ElementAccumulator,                /// Element type for internal accumulation
    typename LayoutC,                           /// Layout type for C and D matrix operand
    typename ArchTag,                           /// Tag indicating architecture to tune for
    typename ThreadblockShape,                  /// Threadblock-level tile size (concept: GemmShape)
    typename WarpShape,                         /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape,                  /// Instruction-level tile size (concept: GemmShape)
    int Stages,                                 /// Number of stages used in the multistage mainloop
    typename Operator,                          /// Operation perfomed by GEMM
    SharedMemoryClearOption SharedMemoryClear,  /// Use zfill or predicate for out-of-bound cp.async
    bool GatherA,                               /// Gather operand A by using an index array
    bool GatherB,                               /// Gather operand B by using an index array
    typename PermuteALayout,                    /// Permute operand A
    typename PermuteBLayout                     /// Permute operand B
>
struct DefaultMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB, ElementAccumulator, LayoutC,
    arch::OpClassTensorOp, ArchTag, ThreadblockShape, WarpShape, InstructionShape, Stages, Operator, false,
    SharedMemoryClear, GatherA, GatherB, PermuteALayout, PermuteBLayout
> {
    static cutlass::arch::CacheOperation::Kind const CacheOpA = ((sizeof_bits<ElementA>::value * kAlignmentA) == 128)
        ? cutlass::arch::CacheOperation::Global : cutlass::arch::CacheOperation::Always;

    static cutlass::arch::CacheOperation::Kind const CacheOpB = ((sizeof_bits<ElementB>::value * kAlignmentB) == 128)
        ? cutlass::arch::CacheOperation::Global : cutlass::arch::CacheOperation::Always;

    // Define the MmaCore components
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<ThreadblockShape, WarpShape, InstructionShape,
        ElementA, LayoutA, ElementB, LayoutB, ElementAccumulator, LayoutC, arch::OpClassTensorOp, Stages, Operator, false, CacheOpA, CacheOpB>;

    // Define iterators over tiles from the A operand. Iterators to read from global memory.
    using ThreadMapA = typename MmaCore::IteratorThreadMapA;
    using AccessTypeA = cutlass::Array<ElementA, kAlignmentA>;
    using IteratorA = cutlass::transform::threadblock::PredicatedTileAccessIterator<
        cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
        ElementA, LayoutA, 1, ThreadMapA, AccessTypeA, GatherA, PermuteALayout>;

    // Define iterators over tiles from the B operand. Iterators to read from global memory.
    using ThreadMapB = typename MmaCore::IteratorThreadMapB;
    using AccessTypeB = cutlass::Array<ElementB, kAlignmentB>;
    using IteratorB = cutlass::transform::threadblock::PredicatedTileAccessIterator<
        cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
        ElementB, LayoutB, 0, ThreadMapB, AccessTypeB, GatherB, PermuteBLayout>;

    // Define the threadblock-scoped multistage matrix multiply
    using ThreadblockMma = cutlass::gemm::threadblock::MmaMultistage<typename MmaCore::Shape,
        IteratorA, typename MmaCore::SmemIteratorA, MmaCore::kCacheOpA, IteratorB, typename MmaCore::SmemIteratorB, MmaCore::kCacheOpB,
        ElementAccumulator, LayoutC, typename MmaCore::MmaPolicy, Stages, SharedMemoryClear>;
};
```

### DefaultMmaCore

当在Threadblock线程块层级实现矩阵乘加操作时，CUTLASS提供一个默认的核心模板配置，抽象为gemm::threadblock::DefaultMmaCore模板类。此外，根据底层实现使用的CUDA Core硬件或Tensor Core硬件的不同，以及底层实现使用的双缓冲或多级流水线的方式不同，默认的核心模板配置会使用不同的Warp线程束层级的矩阵乘加实现，以及使用不同的数据转移器。

在cutlass/gemm/threadblock/default_mma_core.h头文件中，提供gemm::warp::DefaultMmaCore的定义，如下所示。

```c++
/// Template defininng default matrix multiply operators inferred from threadblock tile size,
/// global memory data layout, and target math instruction.
template <
    typename Shape,                                    /// Shape of threadblock-scoped matrix multiply operator
    typename WarpShape,                                /// Shape of warp-level matrix multiply operator
    typename InstructionShape,                         /// Shape of one matrix production operation (concept: GemmShape)
    typename ElementA,                                 /// Element data type of A operand
    typename LayoutA,                                  /// Layout of operand A
    typename ElementB,                                 /// Element data type of B operand
    typename LayoutB,                                  /// Layout of operand B
    typename ElementC,                                 /// Data type of accumulator
    typename LayoutC,                                  /// Layout of accumulator
    typename OperatorClass,                            /// Indicates type of math operator (arch::OpClassSimt or arch::OpClassTensorOp)
    int Stages = 2,                                    /// Number of stages
    typename Operator = cutlass::arch::OpMultiplyAdd,  /// Operation performed by MMA
    /// Store the accumulators in row major or column major. Row major is used when output layout is interleaved.
    bool AccumulatorsInRowMajor = false,
    cutlass::arch::CacheOperation::Kind CacheOpA = cutlass::arch::CacheOperation::Global,  /// Cache operation of operand A
    cutlass::arch::CacheOperation::Kind CacheOpB = cutlass::arch::CacheOperation::Global,  /// Cache operation of operand B
    ComplexTransform TransformA = ComplexTransform::kNone,  /// per-element transformation for elements of A
    ComplexTransform TransformB = ComplexTransform::kNone,  /// per-element transformation for elements of B
    bool IsComplex = false                                  /// (is_complex<ElementA>::value || is_complex<ElementB>::value)
>
struct DefaultMmaCore;
```

在cutlass/gemm/threadblock/default_mma_core_simt.h头文件中，CUTLASS在CUDA Core硬件上使用SIMD指令，gemm::threadblock::DefaultMmaCore的实现代码如下所示。

```c++
namespace detail {
// convert a WarpShape which is the whole tile of elements into warp num threads.
// The goal is for each thread's tile of elements to be as square as possible for performance (4x4 will be faster than 2x8).
template<typename WarpShape>
constexpr int simt_get_warp_threads_m() {
    return (WarpShape::kM > WarpShape::kN) ? 8 : 4;
}
// Computes padding in shared memory to perform efficient transpose without bank conflicts.
constexpr int simt_transpose_padding(int threads, int crosswise, int size_in_bits) {
    return (size_in_bits >= 32
        ? threads / crosswise / (size_in_bits / 32)
        : threads / crosswise * (32 / size_in_bits));
}
}
```

```c++
/// A: row-major    : needs transposition and smem padding
/// B: column-major : needs transposition and smem padding
/// Operator: simt class
/// This uses the default warp-level operator given tile sizes.
template <
    typename Shape_,      /// Shape of threadblock-scoped matrix multiply operator (concept: GemmShape)
    typename WarpShape_,  /// Shape of warp-level matrix multiply operator (concept: GemmShape)
    typename ElementA_,   /// Data type of A operand
    typename ElementB_,   /// Data type of B operand
    typename ElementC_,   /// Data type of accumulator
    typename LayoutC_,    /// Layout of accumulator
    typename Operator_    /// Operation performed by GEMM
>
struct DefaultMmaCore<Shape_, WarpShape_, GemmShape<1, 1, 1>,
    ElementA_, layout::RowMajor, ElementB_, layout::ColumnMajor, ElementC_, LayoutC_, arch::OpClassSimt, 2, Operator_
> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = GemmShape<1, 1, 1>;
    using ElementA = ElementA_;
    using LayoutA = layout::ColumnMajor;
    using ElementB = ElementB_;
    using LayoutB = layout::RowMajor;
    using ElementC = ElementC_;
    using LayoutC = LayoutC_;
    using OperatorClass = arch::OpClassSimt;
    using Operator = Operator_;  /// Default Operator

    static int const PartitionsK = Shape::kK / WarpShape::kK;
    /// Number of warps present
    using WarpCount = GemmShape<Shape::kM / WarpShape::kM, Shape::kN / WarpShape::kN, PartitionsK>;
    static int const kWarpSize = warp::WarpSize<arch::OpClassSimt>::value;  /// Number of threads per warp
    static int const kThreads = WarpCount::kCount * kWarpSize;              /// Number of threads total
    static int const kElementsPerAccess = 1;

    /// ThreadMap of iterator A
    using IteratorThreadMapA = transform::PitchLinearStripminedThreadMap<
        layout::PitchLinearShape<Shape::kK, Shape::kM>, kThreads, kElementsPerAccess>;
    /// Transpose the ThreadMap of iterator A
    using SmemThreadMapA = transform::TransposePitchLinearThreadMapSimt<IteratorThreadMapA>;
    /// Shared memory layout of A
    using SmemLayoutA = layout::ColumnMajor;
    /// Shared memory iterator of A operand. Iterators to write to shared memory.
    using SmemIteratorA = transform::threadblock::RegularTileIterator<
        MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 1, SmemThreadMapA>;  /// SmemThreadMapA was IteratorThreadMapA

    /// ThreadMap of iterator B
    using IteratorThreadMapB = transform::PitchLinearStripminedThreadMap<
        layout::PitchLinearShape<Shape::kK, Shape::kN>, kThreads, kElementsPerAccess>;
    /// Transpose the ThreadMap of iterator B
    using SmemThreadMapB = transform::TransposePitchLinearThreadMapSimt<IteratorThreadMapB>;
    /// Shared memory layout of B
    using SmemLayoutB = layout::RowMajor;
    /// Shared memory iterator to B operand. Iterators to write to shared memory.
    using SmemIteratorB = transform::threadblock::RegularTileIterator<
        MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 0, SmemThreadMapB>;  /// SmemThreadMapB was IteratorThreadMapB

    /// Warp-level matrix multiply operator
    static const int WarpNumThreadsM = detail::simt_get_warp_threads_m<WarpShape>();
    static const int WarpNumThreadsN = kWarpSize / WarpNumThreadsM;
    static const int ThreadTileM = WarpShape::kM / WarpNumThreadsM;
    static const int ThreadTileN = WarpShape::kN / WarpNumThreadsN;
    static const int LaneLayoutKI = ThreadTileM > 4 && ThreadTileN > 4 ? 2 : 1;
    static const int numElementsA = 128 / sizeof_bits<ElementA>::value;
    static const int numElementsB = 128 / sizeof_bits<ElementB>::value;
    static const int LaneM = cutlass::const_min(numElementsA, ThreadTileM);
    static const int LaneN = cutlass::const_min(numElementsB, ThreadTileN);
    /// these should have max of thread tile also
    using LaneMmaShape = cutlass::gemm::GemmShape<LaneM, LaneN, 1>;
    using Policy = cutlass::gemm::warp::MmaSimtPolicy<
        cutlass::MatrixShape<WarpNumThreadsM, WarpNumThreadsN>,   /// WarpShape in unit of Thread
        cutlass::layout::RowMajorInterleaved<LaneLayoutKI>,       /// LaneLayout
        LaneMmaShape
    >;

    using MmaWarpSimt = cutlass::gemm::warp::MmaSimt<
        WarpShape,    /// Size of the Gemm problem - concept: gemm::GemmShape<> 128, 128, 8
        ElementA,     /// Data type of A elements
        SmemLayoutA,  /// Layout of A matrix (concept: MatrixLayout)
        ElementB,     /// Data type of B elements
        SmemLayoutB,  /// Layout of B matrix (concept: MatrixLayout)
        ElementC,     /// Element type of C matrix
        LayoutC,      /// Layout of C matrix (concept: MatrixLayout)
        Policy        /// Policy describing warp-level MmaSimtOp (concept: MmaSimtOp policy)
    >;

    static int const kPaddingM = detail::simt_transpose_padding(kWarpSize, Shape::kK, sizeof_bits<ElementA>::value);
    static int const kPaddingN = detail::simt_transpose_padding(kWarpSize, Shape::kK, sizeof_bits<ElementB>::value);

    /// Policy used to define MmaPipelined 
    using MmaPolicy = MmaPolicy<
        MmaWarpSimt,
        MatrixShape<kPaddingM, 0>,    // skew for A matrix to avoid SMEM bank conflicts
        MatrixShape<0, kPaddingN>,    // skew for B matrix to avoid SMEM bank conflicts
        WarpCount::kK
    >;
};
```

在cutlass/gemm/threadblock/default_mma_core_sm75.h头文件中，CUTLASS在Tensor Core硬件上使用mma指令，gemm::threadblock::DefaultMmaCore的实现代码如下所示。

```c++
/// A: column-major
/// B: row-major
/// Operator: tensor op class
/// This uses the default warp-level operator given tile sizes
template <
    typename Shape_,             /// Shape of threadblock-scoped matrix multiply operator (concept: GemmShape)
    typename WarpShape_,         /// Shape of warp-level matrix multiply operator (concept: GemmShape)
    typename InstructionShape_,  /// Shape of one matrix production operation (concept: GemmShape)
    typename ElementA_,          /// Data type of A operand
    typename ElementB_,          /// Data type of B operand
    typename ElementC_,          /// Data type of accumulator
    typename LayoutC_,           /// Layout of accumulator
    typename Operator_           /// Operation performed by GEMM
>
struct DefaultMmaCore<Shape_, WarpShape_, InstructionShape_,
    ElementA_, layout::ColumnMajor, ElementB_, layout::RowMajor, ElementC_, LayoutC_, arch::OpClassTensorOp, 2, Operator_
> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = InstructionShape_;
    using ElementA = ElementA_;
    using LayoutA = layout::ColumnMajor;
    using ElementB = ElementB_;
    using LayoutB = layout::RowMajor;
    using ElementC = ElementC_;
    using LayoutC = LayoutC_;
    using OperatorClass = arch::OpClassTensorOp;
    using Operator = Operator_;  /// Default Operator

    /// Number of warps present
    using WarpCount = GemmShape<Shape::kM / WarpShape::kM, Shape::kN / WarpShape::kN, Shape::kK / WarpShape::kK>;
    static int const kWarpSize = warp::WarpSize<arch::OpClassTensorOp>::value;  /// Number of threads per warp
    static int const kThreads = WarpCount::kCount * kWarpSize;                  /// Number of threads total
    static int const kAccessSizeInBits = 128;                                   /// Size of a threadblock-scoped access

    /// Warp thread arrangement
    static int const kWarpThreadArrangementContiguousA = platform::min(Shape::kM / (kAccessSizeInBits / sizeof_bits<ElementA>::value), 8);
    static int const kWarpThreadArrangementStridedA = kWarpSize / kWarpThreadArrangementContiguousA;
    /// ThreadMap of iterator A
    using IteratorThreadMapA = transform::PitchLinearWarpRakedThreadMap<layout::PitchLinearShape<Shape::kM, Shape::kK>,
        kThreads, layout::PitchLinearShape<kWarpThreadArrangementContiguousA, kWarpThreadArrangementStridedA>,
        kAccessSizeInBits / sizeof_bits<ElementA>::value>;
    /// Shared memory layout of A
    static int const Crosswise_A = platform::min(int(128 / sizeof(ElementA)), Shape::kM);
    using SmemLayoutA = layout::ColumnMajorTensorOpMultiplicandCongruous<sizeof_bits<ElementA>::value, Crosswise_A>;
    /// Shared memory iterator to A operand. Iterators to write to shared memory.
    using SmemIteratorA = transform::threadblock::RegularTileIterator<
        MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 1, IteratorThreadMapA>;

    /// Warp thread arrangement
    static int const kWarpThreadArrangementContiguousB = platform::min(Shape::kN / (kAccessSizeInBits / sizeof_bits<ElementB>::value), 8);
    static int const kWarpThreadArrangementStridedB = kWarpSize / kWarpThreadArrangementContiguousB;
    /// ThreadMap of iterator B
    using IteratorThreadMapB = transform::PitchLinearWarpRakedThreadMap<layout::PitchLinearShape<Shape::kN, Shape::kK>,
        kThreads, layout::PitchLinearShape<kWarpThreadArrangementContiguousB, kWarpThreadArrangementStridedB>,
        kAccessSizeInBits / sizeof_bits<ElementB>::value>;
    /// Shared memory layout of B
    static int const Crosswise_B = platform::min(int(128 / sizeof(ElementB)), Shape::kN);
    using SmemLayoutB = layout::RowMajorTensorOpMultiplicandCongruous<sizeof_bits<ElementB>::value, Crosswise_B>;
    /// Shared memory iterator to B operand. Iterators to write to shared memory.
    using SmemIteratorB = transform::threadblock::RegularTileIterator<
        MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 0, IteratorThreadMapB>;

    /// Warp-level matrix multiply operator
    using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<WarpShape, InstructionShape,
        ElementA, SmemLayoutA, ElementB, SmemLayoutB, ElementC, LayoutC, Operator, WarpCount::kK>::Type;

    /// Policy used to define MmaPipelined
    using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>, MatrixShape<0, 0>, WarpCount::kK>;
};
```

### MmaBase

矩阵乘加操作在Threadblock线程块层级实现时，包括双缓冲的实现与多级流水线的实现，这些实现都派生自一个gemm::threadblock::MmaBase模板类。

在cutlass/gemm/threadblock/mma_base.h头文件中，提供gemm::threadblock::MmaBase的定义，如下所示。

```c++
/// Policy object describing MmaTensorOp and MmaSimt
template <
    typename Operator_,      /// Warp-level GEMM operator (concept: gemm::warp::Mma)
    typename SmemPaddingA_,  /// Padding used for A operand in shared memory (concept: MatrixShape)
    typename SmemPaddingB_,  /// Padding used for B operand in shared memory (concept: MatrixShape)
    int PartitionsK = 1      /// Number of partitions of K dimension of GEMM
>
struct MmaPolicy {
    using Operator = Operator_;          /// Warp-level GEMM operator (concept: gemm::warp::MmaTensorOp or gemm::warp::MmaSimt)
    using SmemPaddingA = SmemPaddingA_;  /// Padding used for A operand in shared memory
    using SmemPaddingB = SmemPaddingB_;  /// Padding used for B operand in shared memory
    static int const kPartitionsK = PartitionsK;  /// Number of partitions of K dimension
};

////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product targeting CUDA cores and SIMT math instructions.
template <
    typename Shape_,        /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename Policy_,       /// Policy describing tuning details (concept: MmaPolicy)
    int Stages,             /// Number of stages,
    typename Enable = bool  /// Used for partial specialization
>
class MmaBase {
public:
    using Shape = Shape_;               /// Size of the Gemm problem - concept: gemm::GemmShape<>
    using Policy = Policy_;             /// Policy describing tuning details
    static int const kStages = Stages;  /// Number of stages

    /// Warp-level Mma
    using Operator = typename Policy::Operator;
    /// Shape describing the overall GEMM computed from shared memory by each warp.
    using WarpGemm = typename Policy::Operator::Shape;
    /// Shape describing the number of warps filling the CTA
    using WarpCount = GemmShape<Shape::kM / WarpGemm::kM, Shape::kN / WarpGemm::kN, Shape::kK / WarpGemm::kK>;
    /// Number of warp-level GEMM oeprations
    static int const kWarpGemmIterations = (WarpGemm::kK / Operator::Policy::MmaShape::kK);

    using TensorRefA = TensorRef<typename Operator::ElementA, typename Operator::LayoutA>;  /// Tensor reference to the A operand
    using TensorRefB = TensorRef<typename Operator::ElementB, typename Operator::LayoutB>;  /// Tensor reference to the B operand

    /// Shared storage object needed by threadblock-scoped GEMM
    class SharedStorage {
    public:
        /// Shape of the A matrix operand in shared memory
        using ShapeA = MatrixShape<Shape::kM + Policy::SmemPaddingA::kRow, Shape::kK * kStages + Policy::SmemPaddingA::kColumn>;
        /// Shape of the B matrix operand in shared memory
        using ShapeB = MatrixShape<Shape::kK * kStages + Policy::SmemPaddingB::kRow, Shape::kN + Policy::SmemPaddingB::kColumn>;

        AlignedBuffer<typename Operator::ElementA, ShapeA::kCount> operand_A;  /// Buffer for A operand
        AlignedBuffer<typename Operator::ElementB, ShapeB::kCount> operand_B;  /// Buffer for B operand

    public:
        /// Returns a layout object for the A matrix
        static typename Operator::LayoutA LayoutA() {
            return Operator::LayoutA::packed({ ShapeA::kRow, ShapeA::kColumn });
        }

        /// Returns a layout object for the B matrix
        static typename Operator::LayoutB LayoutB() {
            return Operator::LayoutB::packed({ ShapeB::kRow, ShapeB::kColumn });
        }

        /// Returns a TensorRef to the A operand
        TensorRefA operand_A_ref() {
            return TensorRefA{ operand_A.data(), LayoutA() };
        }

        /// Returns a TensorRef to the B operand
        TensorRefB operand_B_ref() {
            return TensorRefB{ operand_B.data(), LayoutB() };
        }
    };

protected:
    typename Operator::IteratorA warp_tile_iterator_A_;  /// Iterator to load a warp-scoped tile of A operand from shared memory
    typename Operator::IteratorB warp_tile_iterator_B_;  /// Iterator to load a warp-scoped tile of B operand from shared memory

public:
    /// Construct from tensor references
    /// @param shared_storage : Shared storage needed for internal use by threadblock-scoped GEMM
    MmaBase(SharedStorage &shared_storage, int thread_idx, int warp_idx, int lane_idx) :
        warp_tile_iterator_A_(shared_storage.operand_A_ref(), lane_idx),
        warp_tile_iterator_B_(shared_storage.operand_B_ref(), lane_idx) {}
};
```

### MmaPipelined

当使用双缓冲实现在Threadblock线程块层级的矩阵乘加操作时，相关实现被抽象为gemm::threadblock::MmaPipelined模板类。

在cutlass/gemm/threadblock/mma_pipelined.h头文件中，提供gemm::threadblock::MmaPipelined的定义，如下所示。

```c++
/// Structure to compute the matrix product targeting CUDA cores and SIMT math instructions.
template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename Shape_,
    /// Iterates over tiles of A operand in global memory. (concept: ReadableTileIterator | ForwardTileIterator | MaskedTileIterator)
    typename IteratorA_,
    /// Iterates over tiles of A operand in shared memory. (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorA_,
    /// Iterates over tiles of B operand in global memory. (concept: ReadableTileIterator | ForwardTileIterator | MaskedTileIterator)
    typename IteratorB_,
    /// Iterates over tiles of B operand in shared memory. (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorB_,
    typename ElementC_,  /// Data type of accumulator matrix
    typename LayoutC_,   /// Data type of accumulator matrix
    typename Policy_,    /// Policy describing tuning details (concept: MmaPolicy)
    /// Transformation applied to A operand
    typename TransformA_ = NumericArrayConverter<
        typename SmemIteratorA_::Element, typename IteratorA_::Element, IteratorA_::Fragment::kElements>,
    /// Transformation applied to B operand
    typename TransformB_ = NumericArrayConverter<
        typename SmemIteratorB_::Element, typename IteratorB_::Element, IteratorB_::Fragment::kElements>,
    /// Used for partial specialization
    typename Enable = bool
>
class MmaPipelined : public MmaBase<Shape_, Policy_, 2> {
public:
    using Base = MmaBase<Shape_, Policy_, 2>;  /// Base class
    using Shape = Shape_;                      /// Size of the Gemm problem - concept: gemm::GemmShape<>
    using IteratorA = IteratorA_;              /// Iterates over tiles of A operand in global memory
    using IteratorB = IteratorB_;              /// Iterates over tiles of B operand in global memory
    using SmemIteratorA = SmemIteratorA_;      /// Iterates over tiles of A operand in shared memory
    using SmemIteratorB = SmemIteratorB_;      /// Iterates over tiles of B operand in shared memory
    using ElementC = ElementC_;                /// Data type of accumulator matrix
    using LayoutC = LayoutC_;                  /// Layout of accumulator matrix
    using Policy = Policy_;                    /// Policy describing tuning details
    using TransformA = TransformA_;            /// Transformation applied to A operand
    using TransformB = TransformB_;            /// Transformation applied to B operand

    using FragmentA = typename IteratorA::Fragment;          /// Fragment of operand A loaded from global memory
    using FragmentB = typename IteratorB::Fragment;          /// Fragment of operand B loaded from global memory
    using FragmentC = typename Policy::Operator::FragmentC;  /// Fragment of accumulator tile
    using Operator = typename Policy::Operator;              /// Warp-level Mma
    using ArchTag = typename Policy::Operator::ArchTag;      /// Obtain the arch tag from the warp-level operator
    
    static ComplexTransform const kTransformA = Operator::kTransformA;  /// Complex transform on A operand
    static ComplexTransform const kTransformB = Operator::kTransformB;  /// Complex transform on B operand

protected:
    Operator warp_mma;               /// Warp-level MMA operator
    SmemIteratorA smem_iterator_A_;  /// Iterator to write threadblock-scoped tile of A operand to shared memory
    SmemIteratorB smem_iterator_B_;  /// Iterator to write threadblock-scoped tile of B operand to shared memory
    TransformA transform_A_;         /// transformation applied to A fragment
    TransformB transform_B_;         /// transformation applied to B fragment
    int smem_write_stage_idx;        /// Shared memory write stage index

public:
    /// Construct from tensor references
    /// @param shared_storage : Shared storage needed for internal use by threadblock-scoped GEMM
    /// @param thread_idx     : ID of each thread within the threadblock
    /// @param warp_idx       : ID of each warp within the threadblock
    /// @param lane_idx       : ID of each thread within a warp
    /// @param transform_A    : transformation applied to A fragment
    /// @param transform_B    : transformation applied to B fragment
    MmaPipelined(typename Base::SharedStorage &shared_storage, int thread_idx, int warp_idx, int lane_idx,
        TransformA transform_A = TransformA(), TransformB transform_B = TransformB()) :
        Base(shared_storage, thread_idx, warp_idx, lane_idx),
        smem_iterator_A_(shared_storage.operand_A_ref(), thread_idx),
        smem_iterator_B_(shared_storage.operand_B_ref(), thread_idx),
        transform_A_(transform_A), transform_B_(transform_B), smem_write_stage_idx(0) {
        // Compute warp location within threadblock tile by mapping the warp_id to three coordinates:
        // idx_m: the warp's position within the threadblock along the M dimension
        // idx_n: the warp's position within the threadblock along the N dimension
        // idx_k: the warp's position within the threadblock along the K dimension
        int warp_idx_mn = warp_idx % (Base::WarpCount::kM * Base::WarpCount::kN);
        int warp_idx_m = warp_idx_mn % Base::WarpCount::kM;
        int warp_idx_n = warp_idx_mn / Base::WarpCount::kM;
        int warp_idx_k = warp_idx / (Base::WarpCount::kM * Base::WarpCount::kN);
        // Add per-warp offsets in units of warp-level tiles
        this->warp_tile_iterator_A_.add_tile_offset({ warp_idx_m, Base::kWarpGemmIterations * warp_idx_k });
        this->warp_tile_iterator_B_.add_tile_offset({ Base::kWarpGemmIterations * warp_idx_k, warp_idx_n });
    }

    /// Advance shared memory write-iterators to the next stage
    void advance_smem_write_stage() {
        ++this->smem_iterator_A_;
        ++this->smem_iterator_B_;
        // Add negative offsets to return iterators to the 'start' of the circular buffer in shared memory
        if (smem_write_stage_idx == 1) {
            this->smem_iterator_A_.add_tile_offset({ 0, -Base::kStages });
            this->smem_iterator_B_.add_tile_offset({ -Base::kStages, 0 });
        }
        smem_write_stage_idx ^= 1;
    }

    /// Advance shared memory read-iterators and write-iterators to the next stage
    void advance_smem_stages() {
        ++this->smem_iterator_A_;
        ++this->smem_iterator_B_;
        // Add negative offsets to return iterators to the 'start' of the circular buffer in shared memory
        if (smem_write_stage_idx == 1) {
            // wrap write stage
            this->smem_iterator_A_.add_tile_offset({ 0, -Base::kStages });
            this->smem_iterator_B_.add_tile_offset({ -Base::kStages, 0 });
        } else {
            // wrap read stage
            this->warp_tile_iterator_A_.add_tile_offset({ 0, -Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations });
            this->warp_tile_iterator_B_.add_tile_offset({ -Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations, 0 });
        }
        smem_write_stage_idx ^= 1;
    }

    /// GEMM prologue. `gmem --> smem`.
    /// The last kblock is loaded in the prologue.
    /// Bootstrap the global->shared memory pipeline by fetching the global fragments
    /// needed by the first `kStages - 1` threadblock mainloop iterations.
    /// @param iterator_A        : [in|out] iterator over A operand in global memory
    /// @param iterator_B        : [in|out] iterator over B operand in global memory
    /// @param gemm_k_iterations : [in|out] number of threadblock mainloop iterations remaining
    void prologue(IteratorA &iterator_A, IteratorB &iterator_B, int &gemm_k_iterations) {
        // Load A fragment from global A
        FragmentA tb_frag_A;
        tb_frag_A.clear();
        iterator_A.load(tb_frag_A);
        ++iterator_A;
        // Load B fragment from global B
        FragmentB tb_frag_B;
        tb_frag_B.clear();
        iterator_B.load(tb_frag_B);
        ++iterator_B;
        // Store A and B fragments to shared
        this->smem_iterator_A_.store(transform_A_(tb_frag_A));
        this->smem_iterator_B_.store(transform_B_(tb_frag_B));
        // Advance write stage
        advance_smem_write_stage();
    }

    /// Wait until we have at least one completed global fetch stage
    void gmem_wait() {
        __syncthreads();
    }

    /// Perform the specified number of threadblock mainloop iterations of matrix multiply-accumulate.
    /// Assumes prologue has been initiated.
    /// @param gemm_k_iterations :     [in] number of threadblock mainloop iterations
    /// @param accum             : [in|out] accumulator tile
    /// @param iterator_A        : [in|out] iterator over A operand in global memory
    /// @param iterator_B        : [in|out] iterator over B operand in global memory
    void gemm_iters(int gemm_k_iterations, FragmentC &accum, IteratorA &iterator_A, IteratorB &iterator_B) {
        // Avoid reading out of bounds
        iterator_A.clear_mask(gemm_k_iterations <= 1);
        iterator_B.clear_mask(gemm_k_iterations <= 1);

        using WarpFragmentA = typename Operator::FragmentA;
        using WarpFragmentB = typename Operator::FragmentB;
        // Pair of fragments used to overlap shared memory loads and math instructions
        WarpFragmentA warp_frag_A[2];
        WarpFragmentB warp_frag_B[2];
        // Load A fragment from shared A
        this->warp_tile_iterator_A_.set_kgroup_index(0);
        this->warp_tile_iterator_A_.load(warp_frag_A[0]);
        ++this->warp_tile_iterator_A_;
        // Load B fragment from shared B
        this->warp_tile_iterator_B_.set_kgroup_index(0);
        this->warp_tile_iterator_B_.load(warp_frag_B[0]);
        ++this->warp_tile_iterator_B_;

        // Pair of fragments used to overlap global memory loads and math instructions
        FragmentA tb_frag_A;
        FragmentB tb_frag_B;

        // Mainloop.
        // Note: The main loop does not support Base::kWarpGemmIterations == 2.
        for (; gemm_k_iterations > 0; --gemm_k_iterations) {
            // Loop over GEMM K dimension
            for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations; ++warp_mma_k) {
                // Load warp-level tiles from shared memory, wrapping to k offset if this is the last group as the case may be.
                if (warp_mma_k == Base::kWarpGemmIterations - 1) {
                    // Write fragments to shared memory
                    this->smem_iterator_A_.store(transform_A_(tb_frag_A));
                    this->smem_iterator_B_.store(transform_B_(tb_frag_B));
                    // Wait until we have at least one completed global fetch stage
                    gmem_wait();
                    // Advance smem read and write stages
                    advance_smem_stages();
                }
                this->warp_tile_iterator_A_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
                this->warp_tile_iterator_B_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
                this->warp_tile_iterator_A_.load(warp_frag_A[(warp_mma_k + 1) % 2]);
                this->warp_tile_iterator_B_.load(warp_frag_B[(warp_mma_k + 1) % 2]);
                ++this->warp_tile_iterator_A_;
                ++this->warp_tile_iterator_B_;
                if (warp_mma_k == 0) {
                    // Load fragment from global A
                    tb_frag_A.clear();
                    iterator_A.load(tb_frag_A);
                    ++iterator_A;
                    // Load fragment from global B
                    tb_frag_B.clear();
                    iterator_B.load(tb_frag_B);
                    ++iterator_B;
                    // Avoid reading out of bounds if this was the last loop iteration
                    iterator_A.clear_mask(gemm_k_iterations <= 2);
                    iterator_B.clear_mask(gemm_k_iterations <= 2);
                }
                warp_mma(accum, warp_frag_A[warp_mma_k % 2], warp_frag_B[warp_mma_k % 2], accum);
            }
        }
    }

    /// Prepares the class for another prologue.
    void wind_down() {
        // First, increment remaining warp tiles to catch it up with the write stage.
        for (int warp_mma_k = 1; warp_mma_k < Base::kWarpGemmIterations; ++warp_mma_k) {
            this->warp_tile_iterator_A_.set_kgroup_index(warp_mma_k);
            this->warp_tile_iterator_B_.set_kgroup_index(warp_mma_k);
            ++this->warp_tile_iterator_A_;
            ++this->warp_tile_iterator_B_;
        }
        // If we bumped the read iterators to the end of the circular buffer, wrap them around to align them with the write iterators.
        if (smem_write_stage_idx == 0) {
            this->warp_tile_iterator_A_.add_tile_offset({ 0, -Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations });
            this->warp_tile_iterator_B_.add_tile_offset({ -Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations, 0 });
        }
    }

    /// Perform a threadblock-scoped matrix multiply-accumulate
    /// @param gemm_k_iterations : number of iterations of the mainloop
    /// @param accum             : destination accumulator tile
    /// @param iterator_A        : iterator over A operand in global memory
    /// @param iterator_B        : iterator over B operand in global memory
    /// @param src_accum         : source accumulator tile
    void operator()(int gemm_k_iterations, FragmentC &accum, IteratorA iterator_A, IteratorB iterator_B, FragmentC const &src_accum) {
        // Prologue
        prologue(iterator_A, iterator_B, gemm_k_iterations);
        // Wait until we have at least one completed global fetch stage
        gmem_wait();
        // Perform accumulation in the 'd' output operand
        accum = src_accum;
        // Perform the MAC-iterations
        gemm_iters(gemm_k_iterations, accum, iterator_A, iterator_B);
    }
};
```

### MmaMultiStage

当使用多级流水线实现在Threadblock线程块层级的矩阵乘加操作时，相关实现被抽象为gemm::threadblock::MmaMultiStage模板类。

在cutlass/gemm/threadblock/mma_multistage.h头文件中，提供gemm::threadblock::MmaMultiStage的定义，如下所示。

```c++

```

## Transform API ^$

在cutlass/transform目录中，提供在不同域之间进行数据转移的代码实现，主要用于解决数据布局变换带来的问题。该功能主要用在Threadblock线程块层级，用于将数据从设备全局内存加载到寄存器，以及将寄存器中的数据写入到共享内存。其中，从全局内存加载数据的操作被抽象为XXX模板类，向共享内存写入数据的操作被抽象为XXX模板类。

