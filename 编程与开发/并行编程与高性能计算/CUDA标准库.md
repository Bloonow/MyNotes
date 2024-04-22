# CUDA标准库

CUDA开发工具套装中有很多有用的库，涵盖线性代数、图像处理、机器学习等众多应用领域。对科学与工程计算领域来说比较重要的库如下所示。

| 库名     | 简介                                                   |
| -------- | ------------------------------------------------------ |
| Thrust   | 类似于C++的标准模板库（standard template library）     |
| cuBLAS   | 基本线性代数子程序（basic linear algebra subroutines） |
| cuFFT    | 快速傅里叶变换（fast Fourier transforms）              |
| cuSPARSE | 稀疏矩阵（sparse matrix）                              |
| cuRAND   | 随机数生成器（random number generator）                |
| cuSolver | 稠密（dense）矩阵和稀疏（sparse）矩阵计算库            |
| cuDNN    | 深度神经网络（deep neural networks）                   |

使用CUDA程序库，有许多优点，如下所示。

(1)可以节约程序开发时间，有些库的功能自己实现需要花费很多人力物力与时间。

(2)可以获得更值得信赖的程序，这些常用库都是业界专家精英们的智慧结晶，一般来说比自己实现更加可靠。

(3)可以简化代码，有些功能自己实现可能需要成百上千行代码，但适当使用库函数也许用几十行代码就能完成。

(4)可以加速程序，对于常见的计算来说，库函数能够获得的性能往往是比较高的。

但是，对于某些特定问题，使用库函数得到的性能不一定能胜过自己的实现。例如，Thrust和cuBLAS库中的很多功能是很容易实现的，有时一个计算任务通过编写一个核函数就能完成，而使用这些库却可能需要调用几个函数，从而增加全局内存的访问量。此时，用这些库就有可能得到比较差的性能。

# Thrust库

Thrust是一个实现了众多基本并行算法的C++模板库，类似于C++的标准模板库（standard template library，STL），该库自动包含在CUDA工具箱中。这是一个模板库，仅仅由一些头文件组成，在使用该库的某个功能时，包含需要的头文件即可。该库中的所有类型与函数都在命名空间thrust中定义，都以thrust::开头。

Thrust中的数据结构主要是矢量容器（vector container），类似于STL中的std::vector，在Thrust中，有两种矢量。一种是存储于主机的矢量thrust::host\_vector\<typename\>模板，另一种是存储于设备的矢量thrust::device\_vector\<typename\>模板，这里的typename可以是任何数据类型。这两种矢量分别位于thrust/host\_vector.h头文件和thrust/device_vector.h头文件中。

Thrust提供了五类常用算法，包括：(1)变换（transformation），例如数组求和计算就是一种变换操作；(2)归约（reduction），例如求和归约计算等；(3)前缀和（prefix sum）；(4)排序（sorting）与搜索（searching）；(5)选择性复制、替换、移除、分区等重排（reordering）操作。

除了thrust::copy()函数外，Thrust算法的参数必须都来自于主机矢量host\_vector或都来自于设备矢量device\_vector，否则编译器会报错。

此处以求前缀和为例，也常称为**扫描（scan）**，该操作是将一个序列
$$
x_0, x_1, x_2, x_3, \cdots
$$
变成另一个序列
$$
y_0=x_0, y_1 = x_0 \circ x_1, y_2 = x_0 \circ x_1 \circ x_2, \cdots
$$
其中，符号$\circ$表示一种数学运算，例如，加法$+$求累加和、乘法$\times$求累乘积，其默认是求累加和。这样定义的扫描称为包含扫描（inclusive scan），即$y_i$的表达式中包含$x_i$，相对地，非包含扫描（exclusive scan）的操作如下
$$
y_0=x_0, y_1 = x_0, y_2 = x_0 \circ x_1, y_3 = x_0 \circ x_1 \circ x_2, \cdots
$$
此处，使用device_vector实现的一个示例代码如下所示。

```c++
#include <thrust/device_vector.h>
#include <thrust/scan.h>

int main(int argc, char *argv[]) {
    int N = 10;
    thrust::device_vector<int> x(N, 0);
    thrust::device_vector<int> y(N, 0);
    for (int i = 0; i < x.size(); i++) x[i] = i + 1;
    thrust::inclusive_scan(x.begin(), x.end(), y.begin());
    for (int i = 0; i < y.size(); i++) {
        printf("%d ", (int)y[i]);
    }
    return 0;
}
```

除使用device_vector外，还可以直接使用设备内存的指针，代码如下所示。

```c++
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

int main(int argc, char *argv[]) {
    int N = 10;
    int *h_x = (int*)malloc(sizeof(int) * N);
    int *h_y = (int*)malloc(sizeof(int) * N);
    for (int i = 0; i < N; i++) h_x[i] = i + 1;
    int *x, *y;
    cudaMalloc(&x, sizeof(int) * N);
    cudaMalloc(&y, sizeof(int) * N);
    cudaMemcpy(x, h_x, sizeof(int) * N, cudaMemcpyHostToDevice);

    thrust::inclusive_scan(thrust::device, x, x + N, y);

    cudaMemcpy(h_y, y, sizeof(int) * N, cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) {
        printf("%d ", (int)h_y[i]);
    }
    return 0;
}
```

相比使用设备矢量的版本，该版本的inclusive_scan()使用了一个thrust::device参数，它表示执行策略（execution policy），位于thrust/execution_policy.h头文件中。

如果程序中大量使用了Thrust库提供的功能，那么直接使用设备矢量存储数据是比较好的方法；而如果程序中大部分代码都是手写的核函数，只是偶尔使用Thrust库提供的功能，那么使用设备内存指针是比较好的方法。

# cuBLAS库

cuBLAS是BLAS在CUDA运行时的实现，其全称是basic linear algebra subroutines，即基本线性代数子程序。这一套子程序最早是在CPU中通过Fortran语言实现的，所以后来的各种实现都带有Fortran风格，其与C风格最大的区别就是，Fortran中的多维数组是列主序存储的。

cuBLAS库包含四个API，具体为：(1)cuBLAS API，标准BLAS的CUDA运行时实现；(2)cuBLASXt API，将BLAS计算扩展到多GPU环境；(3)cuBLASLt API，一个专门用于处理矩阵乘法的API接口，在CUDA 10.1中引入；(4)cuBLASDx API，在CUDA核函数中执行BLAS计算，用于算子融合以降低全局访存开销，属于MathDx的一部分，需要单独下载。

其中，cuBLAS API实现了三个层级的函数，具体为：(1)第一层级，处理矢量之间的运算，如矢量的内积；(2)第二层级，处理矩阵矢量之间的运算，例如矩阵与矢量的乘法；(3)第三层级，处理矩阵之间的运算，如矩阵与矩阵相乘。

## 注意事项

对于使用cuBLAS库的CUDA程序来说，需要包含cublas_v2.h头文件，并在编译时指定其动态链接库cublas，即如下所示。

```shell
nvcc demo.cu -o demo.exe -l cublas
```

使用cuBLAS库的基本代码结构如下所示。

```c++
int main(int argc, char* argv[]) {
    /* prepare data allocated on GPU */
    cublasStatus_t cublas_stat;
	cublasHandle_t handle;
	blas_stat = cublasCreate(&handle);
    if (cublas_stat != CUBLAS_STATUS_SUCCESS) exit(0);
    /* call cuBLAS API over GPU data */
    cublasDestroy(handle);
    /* operation else */
}
```

需要注意的是，在cuBLAS中，将矢量、矩阵、张量等多维数组结构视作**列主序**存储（也即从索引为0号的第一个维度轴开始存储），而非C风格的行主序存储（从索引为-1的最后一个维度轴开始存储）。

例如，一个2行3列的矩阵$A$，其各个元素的值如下所示。
$$
A = \begin{bmatrix} 0.0 & 1.0 & 2.0 \\ 3.0 & 4.0 & 5.0 \end{bmatrix}
$$
若采用行主序存储，则其在连续内存空间中，存储的一维数组如下所示。

```c++
double A[] = { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 };  // row-major
```

若采用列主序存储，则其在连续内存空间中，存储的一维数组如下所示。

```c++
double A[] = { 0.0, 3.0, 1.0, 4.0, 2.0, 5.0 };  // col-major
```

本文中，为符合人的阅读习惯，使用(M,N)表示M行N列矩阵，使用(rId,cId)表示行索引为rId列索引为cId的相应元素。推广到多维张量，相应地对应到各个维度轴上。

假设，对于行主序存储的M行N列的矩阵$A$，对其进行转置，如果仅从逻辑上变换元素的相对位置，底层数据的一维存储顺序保持不变，则得到列主序存储的N行M列的矩阵$A^T$；而如果同时对底层数据的一维存储顺序进行调整，使其真正按转置后的矩阵进行行主序存储，则得到行主序存储的N行M列的矩阵$A^T$。注意，第一种是从逻辑上切换了程序的多维数组存储方式（行主序变为列主序），而第二种在实现上没有改变多维数组的存储方式（行主序仍是行主序）。

此外，还需了解前导维数（leading dimension）的概念，也称为主维数。它表示，当沿着主维度轴的索引增加1时，其底层数据的一维存储跨过了几个元素。例如，对于行主序存储的M行N列的矩阵，其前导维数为N（一行元素个数），表示其一维存储每跨过连续N个元素，行索引增加1；对于列主序存储的M行N列的矩阵，其前导维数为M（一列元素个数），表示其一维存储每跨过连续M个元素，列索引增加1。

## GEMM

下面用一个通用矩阵乘法GEMM的示例，展示如何使用cuBLAS库。首先需要用到如下函数。

```c++
cublasStatus_t cublasSetMatrix(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb);
```

将主机内存上的矩阵$A$，复制到设备内存上的矩阵$B$，其中$A$指向主机内存空间，$B$指向设备内存空间。参数rows和cols表示要复制的矩阵的行列数，它可以是完整矩阵的一个子块，参数elemSize表示要复制的字节数，参数lda和ldb分别表示列主序存储的矩阵$A$和矩阵$B$的前导维数。

```c++
cublasStatus_t cublasGetMatrix(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb);
```

将设备内存上的矩阵$A$，复制到主机内存上的矩阵$B$，其中$A$指向设备内存空间，$B$指向主机内存空间。参数rows和cols表示要复制的矩阵的行列数，它可以是完整矩阵的一个子块，参数elemSize表示要复制的字节数，参数lda和ldb分别表示列主序存储的矩阵$A$和矩阵$B$的前导维数。

当然，也可以使用cudaMemcpy()函数进程主机与设备的数据复制。

```c++
cublasStatus_t cublasSgemm(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float *alpha,
    const float *A, int lda,
    const float *B, int ldb,
    const float *beta,
          float *C, int ldc
);
cublasStatus_t cublasDgemm(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const double *alpha,
    const double *A, int lda,
    const double *B, int ldb,
    const double *beta,
          double *C, int ldc
);
```

单精度和双精度版本的通用矩阵乘法GEMM，此外，还有半精度和复数类型的函数，该函数实现如下公式所表示的操作
$$
C = \alpha \cdot \text{op}(A)\text{op}(B) + \beta \cdot C
$$
其中，$\alpha,\beta$为标量，当$\alpha=1,\beta=0$时为标准矩阵乘法，$\text{op}(\cdot)$表示在执行矩阵乘法之前对矩阵先进行的操作，其可取值如下所示
$$
\text{op}(A) = \begin{cases}
A   & \text{if } \texttt{transa} == \text{CUBLAS\_OP\_N} \\
A^T & \text{if } \texttt{transa} == \text{CUBLAS\_OP\_T} \\
A^H & \text{if } \texttt{transa} == \text{CUBLAS\_OP\_C}
\end{cases}
$$
其中，$A,B,C$为列主序存储的矩阵，矩阵维度分别为$\text{op}(A)$是$m\times k$阶矩阵，$\text{op}(B)$是$k\times n$阶矩阵，$C$是$m\times n$阶矩阵。参数lda,ldb,ldc分别表示矩阵列主序存储的矩阵$A,B,C$的前导维数。

上述函数，若返回CUBLAS_STATUS_SUCCESS则表示只需成功，若返回CUBLAS_STATUS_INVALID_VALUE则表示传值错误，若返回CUBLAS_STATUS_EXECUTION_FAILED则表示无法在GPU启动核函数。

一个示例如下所示，假设矩阵以C语言的行主序方式存储。

```c++
__global__ void dis_matrix(double *d_M, int eles) {
    for (int i = 0; i < eles; i++) printf("%.1f\t", d_M[i]); printf("\n");
}

int main(int argc, char* argv[]) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    const int M = 2;
    const int K = 3;
    const int N = 4;
    
    double *A = (double*)malloc(sizeof(double) * M * K);
    double *B = (double*)malloc(sizeof(double) * K * N);
    double *C = (double*)malloc(sizeof(double) * M * N);
    for (int i = 0; i < M * K; i++) A[i] = i;  // 行主序(M,K)矩阵，仅逻辑上转置，可看成列主序(K,M)矩阵
    for (int i = 0; i < K * N; i++) B[i] = i;  // 行主序(K,N)矩阵，仅逻辑上转置，可看成列主序(N,K)矩阵
    for (int i = 0; i < M * N; i++) C[i] = 0;  // 行主序(M,N)矩阵，仅逻辑上转置，可看成列主序(N,M)矩阵
    printf("A\t=\t"); for (int i = 0; i < M * K; i++) printf("%.1f\t", A[i]); printf("\n");
    printf("B\t=\t"); for (int i = 0; i < K * N; i++) printf("%.1f\t", B[i]); printf("\n");

    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeof(double) * M * K);
    cudaMalloc(&d_B, sizeof(double) * K * N);
    cudaMalloc(&d_C, sizeof(double) * M * N);
    // cublasSetMatrix(M, K, sizeof(double), A, M, d_A, M);  // 列主序(M,K)的情况
    // cublasSetMatrix(K, N, sizeof(double), B, K, d_B, K);  // 列主序(M,K)的情况
    cublasSetMatrix(K, M, sizeof(double), A, K, d_A, K);  // 看成列主序(K,M)矩阵时，应该使用的参数
    cublasSetMatrix(N, K, sizeof(double), B, N, d_B, N);  // 看成列主序(N,K)矩阵时，应该使用的参数
    printf("d_A\t=\t"); dis_matrix<<<1,1>>>(d_A, M * K); cudaDeviceSynchronize();
    printf("d_B\t=\t"); dis_matrix<<<1,1>>>(d_B, K * N); cudaDeviceSynchronize();

    double alpha = 1.0;
    double beta = 0.0;
    // 列主序情况，m,n,k参数分别表示(M,K)的op(A)矩阵，(K,N)的op(B)矩阵，lda,ldb为A,B的前导维数M,K，ldc为C的前导维数
    // cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M);
    // 行主序情况，要将A,B看成列主序，(K,M)矩阵与(N,K)矩阵无法相乘，故需要转置op(A)和op(B)，转置后分别为(M,K)矩阵和(K,N)矩阵，
    // lda,ldb为A,B的前导维数，因将A,B看成列主序的(K,M)矩阵和(N,K)矩阵，故分别为K,N，
    // ldc为C的前导维数，因为无法控制矩阵C进行转置，故输出结果C是列主序存储的(M,N)矩阵，其前导维数为M
    cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &alpha, d_A, K, d_B, N, &beta, d_C, M);
    
    printf("d_C\t=\t"); dis_matrix<<<1,1>>>(d_C, M * N); cudaDeviceSynchronize();
    // 因为无法控制矩阵C进行转置，故输出结果C是列主序存储的(M,N)矩阵，其前导维数为M
    cublasGetMatrix(M, N, sizeof(double), d_C, M, C, M);
    printf("C\t=\t"); for (int i = 0; i < M * N; i++) printf("%.1f\t", C[i]); printf("\n");
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);
    cublasDestroy(handle);
    return 0;
}
```

数学上，计算结果如下。
$$
\begin{align}
C &= AB \\
&= \begin{bmatrix} 0.0 & 1.0 & 2.0 \\ 3.0 & 4.0 & 5.0 \end{bmatrix}
\begin{bmatrix} 0.0 & 1.0 & 2.0 & 3.0 \\ 4.0 & 5.0 & 6.0 & 7.0 \\ 8.0 & 9.0 & 10.0 & 11.0 \end{bmatrix} \\
&= \begin{bmatrix} 20.0 & 23.0 & 26.0 & 29.0 \\ 56.0 & 68.0 & 80.0 & 92.0 \end{bmatrix}
\end{align}
$$
上述代码编译执行结果如下，需要注意的是，结果矩阵$C$的存储是列主序的。

```shell
nvcc gemm.cu -o run -l cublas
./run
```

```
A       =       0.0     1.0     2.0     3.0     4.0     5.0
B       =       0.0     1.0     2.0     3.0     4.0     5.0     6.0     7.0     8.0     9.0     10.0    11.0
d_A     =       0.0     1.0     2.0     3.0     4.0     5.0
d_B     =       0.0     1.0     2.0     3.0     4.0     5.0     6.0     7.0     8.0     9.0     10.0    11.0
d_C     =       20.0    56.0    23.0    68.0    26.0    80.0    29.0    92.0
C       =       20.0    56.0    23.0    68.0    26.0    80.0    29.0    92.0
```

由上述代码可以看到，因为无法控制输出矩阵$C$的存储方式，其只能是作为列主序存储的矩阵输出，故当语言环境是行主序存储时，使用行主序存储的矩阵$A$和矩阵$B$相乘，也只能得到$C$的列主序存储。故而实际上，可以通过在行主序环境下，求输出矩阵$C$的转置，利用cublas\<t\>gemm()函数只能输出列主序矩阵的特性，实际输出的即是最初要求的矩阵$C$，公式如下所示。
$$
\begin{align}
C &= AB \\
C^T &= (AB)^T = B^T A^T
\end{align}
$$
故，求矩阵$C=AB$，实际调用cublas\<t\>gemm()函数时，按照求$C^T=B^TA^T$传入参数，从而其输出的列主序的$C^T$，就是原来要求的行主序的矩阵$C$。将计算封装如下所示的函数。

```c++
void gemm_row_major(double *d_A, double *d_B, double *d_C, int M, int N, int K, cublasHandle_t handle) {
    double alpha = 1.0;
    double beta = 0.0;
    // C = A B, but C is col-major
    // cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &alpha, d_A, K, d_B, N, &beta, d_C, M);
    // C^T = B^T A^T, and C is row-major
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
}
```

使用gemm_row_major()函数，对于上述示例，结果如下所示。

```
A       =       0.0     1.0     2.0     3.0     4.0     5.0
B       =       0.0     1.0     2.0     3.0     4.0     5.0     6.0     7.0     8.0     9.0     10.0    11.0
d_A     =       0.0     1.0     2.0     3.0     4.0     5.0
d_B     =       0.0     1.0     2.0     3.0     4.0     5.0     6.0     7.0     8.0     9.0     10.0    11.0
d_C     =       20.0    23.0    26.0    29.0    56.0    68.0    80.0    92.0
C       =       20.0    23.0    26.0    29.0    56.0    68.0    80.0    92.0
```

## 批量GEMM

对于深度学习的应用场景，使用更多的是批量的矩阵相乘，或者是跨步的批量矩阵乘法。批量矩阵相乘的API函数接口，如下所示。

```c++
cublasStatus_t cublasSgemmBatched(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float *alpha,
    const float *Aarray[], int lda,
    const float *Barray[], int ldb,
    const float *beta,
          float *Carray[], int ldc,
    int batchCount
);
cublasStatus_t cublasDgemmBatched(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const double *alpha,
    const double *Aarray[], int lda,
    const double *Barray[], int ldb,
    const double *beta,
          double *Carray[], int ldc,
    int batchCount
);
```

单精度和双精度版本的通用批量矩阵乘法GEMM，此外，还有半精度和复数类型的函数，该函数实现如下公式所表示的操作
$$
C[i] = \alpha \cdot \text{op}(A[i])\text{op}(B[i]) + \beta \cdot C[i]
$$
其中，$i\in[0,\text{batchCount}-1]$，$\alpha,\beta$为标量，当$\alpha=1,\beta=0$时为标准矩阵乘法，$\text{op}(\cdot)$表示在执行矩阵乘法之前对矩阵先进行的操作，其可取值如下所示
$$
\text{op}(A) = \begin{cases}
A   & \text{if } \texttt{transa} == \text{CUBLAS\_OP\_N} \\
A^T & \text{if } \texttt{transa} == \text{CUBLAS\_OP\_T} \\
A^H & \text{if } \texttt{transa} == \text{CUBLAS\_OP\_C}
\end{cases}
$$
其中，$A,B,C$为列主序存储的矩阵数组指针，其0号维度轴上的元素表示整个矩阵的索引，即表示第几个矩阵。矩阵维度分别为$\text{op}(A[i])$是$m\times k$阶矩阵，$\text{op}(B[i])$是$k\times n$阶矩阵，$C[i]$是$m\times n$阶矩阵。参数lda,ldb,ldc分别表示矩阵列主序存储的矩阵$A[i],B[i],C[i]$的前导维数。

上述函数，若返回CUBLAS_STATUS_SUCCESS则表示只需成功，若返回CUBLAS_STATUS_INVALID_VALUE则表示传值错误，若返回CUBLAS_STATUS_EXECUTION_FAILED则表示无法在GPU启动核函数。

一个示例如下所示，假设矩阵以C语言的行主序方式存储。

```c++
__global__ void dis_matrix(double *d_M, int eles) {
    for (int i = 0; i < eles; i++) printf("%.1f\t", d_M[i]); printf("\n");
}

int main(int argc, char* argv[]) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    const int M = 2;
    const int K = 3;
    const int N = 4;
    const int batchCount = 2;
    
    double **A = (double**)malloc(sizeof(double*) * batchCount);
    double **B = (double**)malloc(sizeof(double*) * batchCount);
    double **C = (double**)malloc(sizeof(double*) * batchCount);
    for (int bidx = 0; bidx < batchCount; bidx++) {
        A[bidx] = (double*)malloc(sizeof(double) * M * K);
        B[bidx] = (double*)malloc(sizeof(double) * K * N);
        C[bidx] = (double*)malloc(sizeof(double) * M * N);
    }
    for (int bidx = 0; bidx < batchCount; bidx++) {
        for (int i = 0; i < M * K; i++) A[bidx][i] = i;    // 行主序(M,K)矩阵，仅逻辑上转置，可看成列主序(K,M)矩阵
        for (int i = 0; i < K * N; i++) B[bidx][i] = i;    // 行主序(K,N)矩阵，仅逻辑上转置，可看成列主序(N,K)矩阵
        for (int i = 0; i < M * N; i++) C[bidx][i] = 0;    // 行主序(M,N)矩阵，仅逻辑上转置，可看成列主序(N,M)矩阵
    }
    for (int bidx = 0; bidx < batchCount; bidx++) {
        printf("A[%d]\t=\t", bidx); for (int i = 0; i < M * K; i++) printf("%.1f\t", A[bidx][i]); printf("\n");
        printf("B[%d]\t=\t", bidx); for (int i = 0; i < K * N; i++) printf("%.1f\t", B[bidx][i]); printf("\n");
    }

    double **d_A_ptr = (double**)malloc(sizeof(double*) * batchCount);
    double **d_B_ptr = (double**)malloc(sizeof(double*) * batchCount);
    double **d_C_ptr = (double**)malloc(sizeof(double*) * batchCount);
    for (int bidx = 0; bidx < batchCount; bidx++) {
        cudaMalloc(&d_A_ptr[bidx], sizeof(double) * M * K);
        cudaMalloc(&d_B_ptr[bidx], sizeof(double) * K * N);
        cudaMalloc(&d_C_ptr[bidx], sizeof(double) * M * N);
    }
    for (int bidx = 0; bidx < batchCount; bidx++) {
        cublasSetMatrix(K, M, sizeof(double), A[bidx], K, d_A_ptr[bidx], K);  // 看成列主序(K,M)矩阵时，应该使用的参数
        cublasSetMatrix(N, K, sizeof(double), B[bidx], N, d_B_ptr[bidx], N);  // 看成列主序(N,K)矩阵时，应该使用的参数
    }
    for (int bidx = 0; bidx < batchCount; bidx++) {
        printf("d_A[%d]\t=\t", bidx); dis_matrix<<<1,1>>>(d_A_ptr[bidx], M * K); cudaDeviceSynchronize();
        printf("d_B[%d]\t=\t", bidx); dis_matrix<<<1,1>>>(d_B_ptr[bidx], K * N); cudaDeviceSynchronize();
    }

    double alpha = 1.0;
    double beta = 0.0;
    // 因为d_A_ptr[bidx],d_B_ptr[bidx],d_C_ptr[bidx]变量虽然指向设备地址，但其本身是位于主机内存上的，
    // 而cublas<t>gemmBatched()函数需要根据地址访问d_A_ptr[bidx]等变量，故需要再将d_A_ptr[bidx]等本身复制到设备内存上
    double **d_A;
    double **d_B;
    double **d_C;
    for (int bidx = 0; bidx < batchCount; bidx++) {
        cudaMalloc((double**)&d_A, sizeof(double*) * batchCount);
        cudaMalloc((double**)&d_B, sizeof(double*) * batchCount);
        cudaMalloc((double**)&d_C, sizeof(double*) * batchCount); 
    }
    cudaMemcpy(d_A, d_A_ptr, sizeof(double*) * batchCount, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, d_B_ptr, sizeof(double*) * batchCount, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, d_C_ptr, sizeof(double*) * batchCount, cudaMemcpyHostToDevice);
    // 求C，列主序
    cublasDgemmBatched(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &alpha, d_A, K, d_B, N, &beta, d_C, M, batchCount);

    for (int bidx = 0; bidx < batchCount; bidx++) {
        printf("d_C[%d]\t=\t", bidx); dis_matrix<<<1,1>>>(d_C_ptr[bidx], M * N); cudaDeviceSynchronize();
    }
    for (int bidx = 0; bidx < batchCount; bidx++) {
        // 因为无法控制矩阵C进行转置，故输出结果C是列主序存储的(M,N)矩阵，其前导维数为M
        cublasGetMatrix(M, N, sizeof(double), d_C_ptr[bidx], M, C[bidx], M);
    }
    for (int bidx = 0; bidx < batchCount; bidx++) {
        printf("C[%d]\t=\t", bidx); for (int i = 0; i < M * N; i++) printf("%.1f\t", C[bidx][i]); printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    for (int bidx = 0; bidx < batchCount; bidx++) {
        cudaFree(d_A_ptr[bidx]);
        cudaFree(d_B_ptr[bidx]);
        cudaFree(d_C_ptr[bidx]);
        free(A[bidx]);
        free(B[bidx]);
        free(C[bidx]);
    }
    free(d_A_ptr);
    free(d_B_ptr);
    free(d_C_ptr);
    free(A);
    free(B);
    free(C);
    cublasDestroy(handle);
    return 0;
}
```

上述代码编译执行结果如下，需要注意的是，结果矩阵$C$的存储是列主序的。

```shell
nvcc gemm.cu -o run -l cublas
./run
```

```
A[0]    =       0.0     1.0     2.0     3.0     4.0     5.0
B[0]    =       0.0     1.0     2.0     3.0     4.0     5.0     6.0     7.0     8.0     9.0     10.0    11.0
A[1]    =       0.0     1.0     2.0     3.0     4.0     5.0
B[1]    =       0.0     1.0     2.0     3.0     4.0     5.0     6.0     7.0     8.0     9.0     10.0    11.0
d_A[0]  =       0.0     1.0     2.0     3.0     4.0     5.0
d_B[0]  =       0.0     1.0     2.0     3.0     4.0     5.0     6.0     7.0     8.0     9.0     10.0    11.0
d_A[1]  =       0.0     1.0     2.0     3.0     4.0     5.0
d_B[1]  =       0.0     1.0     2.0     3.0     4.0     5.0     6.0     7.0     8.0     9.0     10.0    11.0
d_C[0]  =       20.0    56.0    23.0    68.0    26.0    80.0    29.0    92.0
d_C[1]  =       20.0    56.0    23.0    68.0    26.0    80.0    29.0    92.0
C[0]    =       20.0    56.0    23.0    68.0    26.0    80.0    29.0    92.0
C[1]    =       20.0    56.0    23.0    68.0    26.0    80.0    29.0    92.0
```

可以看到，因为一个矩阵用一级指针表示，则要表示批量矩阵，需要使用二级指针表示，例如double \*\*dev_ptr形式，其dev_ptr指向设备地址，持有batchCount个元素，分别为各个矩阵的设备地址，例如dev_ptr[0]指向0号矩阵的设备地址。只有这样，才能保证cublas\<t\>gemmBatched()函数内部在使用诸如dev_ptr[bidx]的形式时，才能正确的指向每个矩阵数据的设备地址。

而不能直接使用一级指针连续存储多个矩阵，例如double \*dev_ptr形式，虽然可以用其连续存储多个矩阵的数据，但对于cublas\<t\>gemmBatched()函数来说，它并不知道每个矩阵的元素个数，故当其试图使用dev_ptr[bidx]的形式访问第二个矩阵时，它实际访问到的是0号矩阵的bidx号元素，这是错误的。

可以看出，这种组织方式较为复杂繁琐，故对于批量矩阵乘法，可以使用cublas\<t\>gemmStridedBatched()函数，它通过手动指定stride参数表示每个矩阵元素个数，从而可以使用一级指针连续存储多个矩阵。

## 跨步批量GEMM

```c++
cublasStatus_t cublasSetMatrix(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb);
cublasStatus_t cublasGetMatrix(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb);
```

对于cublasSetMatrix()和cublasGetMatrix()辅助函数来说，其可以很自然的扩展到批量矩阵的形式，只需将非前导维度的次级维度的参数，变成原来的batchCount倍即可，其他参数无需变动。例如，对于列主序存储的批量个rows行cols列矩阵块的传输，因为rows参数与其前导维度一致，故cols与其次级维度一致，此时只需将cols参数乘以batchCount倍即可。

跨步的批量矩阵乘法的API函数接口，如下所示。

```c++
cublasStatus_t cublasSgemmStridedBatched(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float *alpha,
    const float *A, int lda,
          long long int strideA,
    const float *B, int ldb,
          long long int strideB,
    const float *beta,
          float *C, int ldc,
          long long int strideC,
    int batchCount
);
cublasStatus_t cublasDgemmStridedBatched(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const double *alpha,
    const double *A, int lda,
           long long int strideA,
    const double *B, int ldb,
           long long int strideB,
    const double *beta,
          double *C, int ldc,
           long long int strideC,
    int batchCount
);
```

单精度和双精度版本的通用批量矩阵乘法GEMM，此外，还有半精度和复数类型的函数，该函数实现如下公式所表示的操作
$$
C+i*\text{strideC} = \alpha \cdot \text{op}(A+i*\text{strideA})\text{op}(B+i*\text{strideB}) + \beta \cdot (C+i*\text{strideC})
$$
其中，$i\in[0,\text{batchCount}-1]$，$\alpha,\beta$为标量，当$\alpha=1,\beta=0$时为标准矩阵乘法，$\text{op}(\cdot)$表示在执行矩阵乘法之前对矩阵先进行的操作，其可取值如下所示
$$
\text{op}(A) = \begin{cases}
A   & \text{if } \texttt{transa} == \text{CUBLAS\_OP\_N} \\
A^T & \text{if } \texttt{transa} == \text{CUBLAS\_OP\_T} \\
A^H & \text{if } \texttt{transa} == \text{CUBLAS\_OP\_C}
\end{cases}
$$
其中，$A,B,C$为列主序存储的矩阵序列，相邻矩阵的数据在内存中连续存储，每两个矩阵之间间隔stride个元素。矩阵维度分别为$\text{op}(A[i])$是$m\times k$阶矩阵，$\text{op}(B[i])$是$k\times n$阶矩阵，$C[i]$是$m\times n$阶矩阵。参数lda,ldb,ldc分别表示矩阵列主序存储的矩阵$A[i],B[i],C[i]$的前导维数。

上述函数，若返回CUBLAS_STATUS_SUCCESS则表示只需成功，若返回CUBLAS_STATUS_INVALID_VALUE则表示传值错误，若返回CUBLAS_STATUS_EXECUTION_FAILED则表示无法在GPU启动核函数。

一个示例如下所示，假设矩阵以C语言的行主序方式存储。

```c++
__global__ void dis_matrix(double *d_M, int eles) {
    for (int i = 0; i < eles; i++) printf("%.1f\t", d_M[i]); printf("\n");
}

int main(int argc, char* argv[]) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    const int M = 2;
    const int K = 3;
    const int N = 4;
    const int batchCount = 2;
    
    double *A = (double*)malloc(sizeof(double) * M * K * batchCount);
    double *B = (double*)malloc(sizeof(double) * K * N * batchCount);
    double *C = (double*)malloc(sizeof(double) * M * N * batchCount);
    for (int i = 0; i < M * K * batchCount; i++) A[i] = i;  // 行主序(M,K)矩阵，仅逻辑上转置，可看成列主序(K,M)矩阵
    for (int i = 0; i < K * N * batchCount; i++) B[i] = i;  // 行主序(K,N)矩阵，仅逻辑上转置，可看成列主序(N,K)矩阵
    for (int i = 0; i < M * N * batchCount; i++) C[i] = 0;  // 行主序(M,N)矩阵，仅逻辑上转置，可看成列主序(N,M)矩阵
    for (int bidx = 0; bidx < batchCount; bidx++) {
        printf("A[%d]\t=\t", bidx); for (int i = 0; i < M * K; i++) printf("%.1f\t", A[bidx*M*K+i]); printf("\n");
        printf("B[%d]\t=\t", bidx); for (int i = 0; i < K * N; i++) printf("%.1f\t", B[bidx*K*N+i]); printf("\n");
    }

    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeof(double) * M * K * batchCount);
    cudaMalloc(&d_B, sizeof(double) * K * N * batchCount);
    cudaMalloc(&d_C, sizeof(double) * M * N * batchCount);
    cublasSetMatrix(K, M * batchCount, sizeof(double), A, K, d_A, K);  // 看成列主序(K,M)矩阵时，应该使用的参数
    cublasSetMatrix(N, K * batchCount, sizeof(double), B, N, d_B, N);  // 看成列主序(N,K)矩阵时，应该使用的参数
    for (int bidx = 0; bidx < batchCount; bidx++) {
        printf("d_A[%d]\t=\t", bidx); dis_matrix<<<1,1>>>(&d_A[bidx*M*K], M * K); cudaDeviceSynchronize();
        printf("d_B[%d]\t=\t", bidx); dis_matrix<<<1,1>>>(&d_B[bidx*K*N], K * N); cudaDeviceSynchronize();
    }

    double alpha = 1.0;
    double beta = 0.0;
    // 求C，列主序
    cublasDgemmStridedBatched(
        handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, 
        &alpha, d_A, K, M*K, d_B, N, K*N, &beta, d_C, M, M*N, 
        batchCount);

    for (int bidx = 0; bidx < batchCount; bidx++) {
        printf("d_C[%d]\t=\t", bidx); dis_matrix<<<1,1>>>(&d_C[bidx*M*N], M * N); cudaDeviceSynchronize();
    }

    for (int bidx = 0; bidx < batchCount; bidx++) {
        // 因为无法控制矩阵C进行转置，故输出结果C是列主序存储的(M,N)矩阵，其前导维数为M
        cublasGetMatrix(M, N * batchCount, sizeof(double), d_C, M, C, M);
    }
    for (int bidx = 0; bidx < batchCount; bidx++) {
        printf("C[%d]\t=\t", bidx); for (int i = 0; i < M * N; i++) printf("%.1f\t", C[bidx*M*N+i]); printf("\n");
    }
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);
    cublasDestroy(handle);
    return 0;
}
```

上述代码编译执行结果如下，需要注意的是，结果矩阵$C$的存储是列主序的。

```shell
nvcc gemm.cu -o run -l cublas
./run
```

```
A[0]    =       0.0     1.0     2.0     3.0     4.0     5.0
B[0]    =       0.0     1.0     2.0     3.0     4.0     5.0     6.0     7.0     8.0     9.0     10.0    11.0
A[1]    =       6.0     7.0     8.0     9.0     10.0    11.0
B[1]    =       12.0    13.0    14.0    15.0    16.0    17.0    18.0    19.0    20.0    21.0    22.0    23.0
d_A[0]  =       0.0     1.0     2.0     3.0     4.0     5.0
d_B[0]  =       0.0     1.0     2.0     3.0     4.0     5.0     6.0     7.0     8.0     9.0     10.0    11.0
d_A[1]  =       6.0     7.0     8.0     9.0     10.0    11.0
d_B[1]  =       12.0    13.0    14.0    15.0    16.0    17.0    18.0    19.0    20.0    21.0    22.0    23.0
d_C[0]  =       20.0    56.0    23.0    68.0    26.0    80.0    29.0    92.0
d_C[1]  =       344.0   488.0   365.0   518.0   386.0   548.0   407.0   578.0
C[0]    =       20.0    56.0    23.0    68.0    26.0    80.0    29.0    92.0
C[1]    =       344.0   488.0   365.0   518.0   386.0   548.0   407.0   578.0
```

# cuBLASLt库

cuBLASLt库是一个新的专用于执行GeMM（General Matrix-to-Matrix multiply）操作的轻量级库，它提供了灵活的API接口。它支持更灵活的矩阵数据布局，输入类型，计算类型，还可以通过编程参数灵活选择算法的实现和启发式方法。用户只需指定一次GeMM操作的选项集合，就可以将其重复应用于不同的输入。

在使用cuBLASt库时，需要在代码中包含cublasLt.h头文件，并在链接时指定动态库cublasLt，如下所示。

```shell
nvcc demo.cu -o demo.exe -l cublasLt
```

## 执行配置

句柄cublasLtHandle_t是一个结构体指针，该结构体对象持有cuBLASLt库的上下文。可以使用cublasLtCreate()函数初始化句柄，使用cublasLtDestroy()函数释放资源。

```c++
cublasStatus_t cublasLtCreate(cublasLtHandle_t* lightHandle);
cublasStatus_t cublasLtDestroy(cublasLtHandle_t lightHandle);
```

值得注意的是，cublasHandle_t句柄中包含着一个cublasLtHandle_t句柄，一个合法的cublasHandle_t句柄可以简单的替换cublasLtHandle_t的位置，然而，所不同的是，cublasLtHandle_t句柄不绑定任何具体的CUDA上下文。

cublasLtMatmulDesc_t是一个结构体指针，其描述了矩阵乘法的操作，用于cublasLtMatmul()函数。可以使用cublasLtMatmulDescCreate()函数创建也给描述，使用cublasLtMatmulDescDestroy()函数销毁。它用于表示计算操作的各种细节，由cublasLtMatmulDescAttributes_t各种属性指定，包括计算过程中的数据类型，矩阵存储顺序，矩阵布局等，可通过cublasLtMatmulDescSetAttribute()函数对其进行设置，当然，也可以使用后续介绍的描述符，分别指定相应的属性。

```c++
cublasStatus_t cublasLtMatmulDescCreate(
    cublasLtMatmulDesc_t* matmulDesc, cublasComputeType_t computeType, cudaDataType_t scaleType
);
cublasStatus_t cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t matmulDesc);
cublasStatus_t cublasLtMatmulDescSetAttribute(
    cublasLtMatmulDesc_t matmulDesc,
    cublasLtMatmulDescAttributes_t attr,
    const void* buf,
    size_t sizeInBytes
);
```

其中，cublasComputeType_t是一个枚举类型，用于表示计算过程中所以使用的数据类型，而cudaDataType_t是一个枚举类型，用于表示参与计算的数据类型。这两个枚举类型的定义如下所示。

```c++
typedef enum {
  CUBLAS_COMPUTE_32F           = 68,  /* float - default */
  CUBLAS_COMPUTE_32F_PEDANTIC  = 69,  /* float - pedantic */
  CUBLAS_COMPUTE_64F           = 70,  /* double - default */
  CUBLAS_COMPUTE_64F_PEDANTIC  = 71,  /* double - pedantic */
  CUBLAS_COMPUTE_32I           = 72,  /* signed 32-bit int - default */
  CUBLAS_COMPUTE_32I_PEDANTIC  = 73,  /* signed 32-bit int - pedantic */
} cublasComputeType_t;
```

```c++
typedef enum cudaDataType_t {
    CUDA_R_32F  =  0,  /* real as a float */
    CUDA_C_32F  =  4,  /* complex as a pair of float numbers */
    CUDA_R_64F  =  1,  /* real as a double */
    CUDA_C_64F  =  5,  /* complex as a pair of double numbers */
    CUDA_R_32I  = 10,  /* real as a signed 32-bit int */
    CUDA_C_32I  = 11,  /* complex as a pair of signed 32-bit int numbers */
    CUDA_R_64I  = 24,  /* real as a signed 64-bit int */
    CUDA_C_64I  = 25,  /* complex as a pair of signed 64-bit int numbers */
} cudaDataType;
```

cublasLtMatrixLayout_t是一个结构体指针，其描述了矩阵布局，默认存储是列主序（column-major）的。可以使用cublasLtMatrixLayoutCreate()函数创建一个描述，使用cublasLtMatrixLayoutDestroy()函数销毁。

```c++
cublasStatus_t cublasLtMatrixLayoutCreate(
    cublasLtMatrixLayout_t* matLayout, cudaDataType type, uint64_t rows, uint64_t cols, int64_t ld
);
cublasStatus_t cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t matLayout);
```

其中，rows参数和cols参数指定矩阵所期望的行数与列数，这里的“所期望”表示的意思是，即使矩阵实际数据就1行N列，对其广播到M行N列，则这里的rows仍是M而不是实际数据存储的1行。ld参数指定矩阵的前导维数（leading dimension），在列主序的布局当中，它表示每经过ld个连续元素后就会到达下一列，因此ld会大于等于rows行的数目。将ld参数置为0，表示对前导维度进行广播。

cublasLtMatmulAlgo_t是一个结构体指针，用于表示矩阵相乘的算法。该结构可以被序列化并重新加载，它会使用与原来相同的cuBLAS库，能够节省再次进行算法配置的时间，可使用cublasLtMatmulAlgoConfigSetAttribute()方法设置算法的细节。若使用NULL指针，则表示使用默认的启发式方法选择矩阵相乘的算法。

cuBLASLt库所提供的矩阵乘法API接口为cublasLtMatmul()函数，该函数的原型如下所示。

```c++
cublasStatus_t cublasLtMatmul(
    cublasLtHandle_t lightHandle,
    cublasLtMatmulDesc_t computeDesc,
    const void* alpha, /* host or device pointer */
    const void* A,
    cublasLtMatrixLayout_t Adesc,
    const void* B,
    cublasLtMatrixLayout_t Bdesc,
    const void* beta, /* host or device pointer */
    const void* C,
    cublasLtMatrixLayout_t Cdesc,
    void* D,
    cublasLtMatrixLayout_t Ddesc,
    const cublasLtMatmulAlgo_t* algo,
    void* workspace,
    size_t workspaceSizeInBytes,
    cudaStream_t stream
);
```

其所执行的计算如下面的公式所示，其中$A,B,C$是输入矩阵，$\alpha,\beta$是标量，$D$是输出矩阵。
$$
D = \alpha * AB + \beta * C
$$
注意，这个函数支持就地矩阵乘法（$C,\text{Cdesc}$与$D,\text{Ddesc}$相等，即指向同一内存地址）；以及非就地的矩阵乘法（$C$与$D$不相等，即指向不同内存地址），它们应该具有相同的数据类型，rows行数与cols列数，相同的batch size大小，以及相同的内存存储顺序。在非就地的矩阵乘法中，$C$的前导维数可以与$D$的前导维数不同，特别地，将矩阵$C$的前导维数置为0，表示对其按行或列进行广播，如果省略Cdesc参数，则矩阵$C$自动采用矩阵$D$的布局描述Ddesc。

其中，stream参数表示当前矩阵操作要在哪个CUDA流上执行，若为NULL，则使用默认的空流。

其中，workspace参数是指向GPU内存的指针（地址必须16字节对齐），用作cuBlas库及cuBlasLt库的工作缓冲区，workspaceSizeInBytes表示缓冲区的字节数。可以使用cublasSetWorkspace()函数设置工作缓冲区的大小，函数原型如下所示。

```c++
cublasStatus_t cublasSetWorkspace(
    cublasHandle_t handle, void* workspace, size_t workspaceSizeInBytes
);
```

该函数用于设置cuBLAS库所使用的用户持有的设备缓冲区（必须256字节对齐），以用于在当前CUDA流上，执行所有cuBLAS库的函数调用。如果未设置工作区，也即将workspace参数置为NULL且将workspaceSizeInBytes置为0，表示采用默认的工作缓冲区，所有kernel将使用默认的在cuBLAS上下文创建时所分配的工作区池。注意，cublasSetStream()函数会无条件地将cuBLAS库的工作区重设为默认工作区池。特别地，该函数能够用于在不同kernel函数启动之间切换工作区。对于Hopper架构来说，推荐使用32 MiB的工作区大小，而其他架构推荐使用4 MiB的工作区大小。

需要注意的是，对于cuBLAS库来说，将workspace设为0表示使用默认的缓冲区；而对于cuBLASLt来说，将workspace设为0表示不使用缓冲区，也即在使用cuBLASLt库时，需要手动的显式指定workspace缓冲区大小。当然缓冲区不是必要的，而且设置缓冲区并不一定能够提升性能。

## 使用示例

对于cublasLtMatmul()函数的一个使用示例如下所示。

```c++
int main(int argc, char *argv[]) {
    const int M = 2;
    const int K = 3;
    const int N = 4;

    float *h_A = (float*)malloc(sizeof(float) * M * K);
    float *h_B = (float*)malloc(sizeof(float) * K * N);
    float *h_C = (float*)malloc(sizeof(float) * M * N);
    float *h_D = (float*)malloc(sizeof(float) * M * N);
    for (int i = 0; i < M * K; i++) h_A[i] = i * 1.0;
    for (int i = 0; i < K * N; i++) h_B[i] = i * 1.0;
    for (int i = 0; i < M * N; i++) h_C[i] = i * 1.0;
    for (int i = 0; i < M * N; i++) h_D[i] = 0.0;
    printf("A\t=\t"); for (int i = 0; i < M * K; i++) printf("%.1f\t", h_A[i]); printf("\n");
    printf("B\t=\t"); for (int i = 0; i < K * N; i++) printf("%.1f\t", h_B[i]); printf("\n");
    printf("C\t=\t"); for (int i = 0; i < M * N; i++) printf("%.1f\t", h_C[i]); printf("\n");
    
    float *A, *B, *C, *D;
    cudaMalloc(&A, sizeof(float) * M * K);
    cudaMalloc(&B, sizeof(float) * K * N);
    cudaMalloc(&C, sizeof(float) * M * N);
    cudaMalloc(&D, sizeof(float) * M * N);
    cudaMemcpy(A, h_A, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(B, h_B, sizeof(float) * K * N, cudaMemcpyHostToDevice);
    cudaMemcpy(C, h_C, sizeof(float) * M * N, cudaMemcpyHostToDevice);

    cublasLtHandle_t handle;
    cublasLtCreate(&handle);
    cublasLtMatmulDesc_t computeDesc;
    cublasLtMatmulDescCreate(&computeDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc, Ddesc;  // D = AB + C
    cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, M, K, M);
    cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, K, N, K);
    cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, M, N, M);
    cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_32F, M, N, M);

    float alpha = 1.0, beta = 1.0;
    // A, B, C, D are stored as column-major  // Not use workspace
    cublasLtMatmul(handle, computeDesc, &alpha, A, Adesc, B, Bdesc, &beta, C, Cdesc, D, Ddesc, NULL, NULL, 0, NULL);
    
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatrixLayoutDestroy(Ddesc);
    cublasLtMatmulDescDestroy(computeDesc);
    cublasLtDestroy(handle);

    cudaMemcpy(h_D, D, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    printf("D\t=\t"); for (int i = 0; i < M * N; i++) printf("%.1f\t", h_D[i]); printf("\n");
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaFree(D);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_D);
    return 0;
}
```

```shell
nvcc gemm.cu -o run -l cublasLt
./run
```

```
A       =       0.0     1.0     2.0     3.0     4.0     5.0
B       =       0.0     1.0     2.0     3.0     4.0     5.0     6.0     7.0     8.0     9.0     10.0    11.0
C       =       0.0     1.0     2.0     3.0     4.0     5.0     6.0     7.0
D       =       10.0    14.0    30.0    43.0    50.0    72.0    70.0    101.0
```

在实际应用中，所需要进行的矩阵乘法往往更加复杂，这时可以使用cublasLtMatrixLayoutSetAttribute()函数，修改cublasLtMatrixLayout_t描述符的各种属性，精细控制矩阵布局，该函数原型如下所示。

```c++
cublasStatus_t cublasLtMatrixLayoutSetAttribute(
    cublasLtMatrixLayout_t matLayout,
    cublasLtMatrixLayoutAttribute_t attr,
    const void* buf, size_t sizeInBytes
);
```

其中，attr参数为要设置的矩阵布局属性，buf参数是新的属性值的地址，sizeInBytes参数是新的属性值所占用的字节数。

矩阵布局属性cublasLtMatrixLayoutAttribute_t是一个枚举类，其定义如下所示，注意各个属性值所使用的类型必须匹配。

```c++
/** Attributes of memory layout */
typedef enum {
    /** Data type. See cudaDataType. 
     *  value type: uint32_t
     */
    CUBLASLT_MATRIX_LAYOUT_TYPE = 0,
    
    /** Memory order of the data. See cublasLtOrder_t.
     *  value type: int32_t, default: CUBLASLT_ORDER_COL
     */
    CUBLASLT_MATRIX_LAYOUT_ORDER = 1,
    
    /** Number of rows. 
     *  value type: uint64_t
     *  Usually only values that can be expressed as int32_t are supported.
     */
    CUBLASLT_MATRIX_LAYOUT_ROWS = 2,
    
    /** Number of columns.
     *  value type: uint64_t
     *  Usually only values that can be expressed as int32_t are supported.
     */
    CUBLASLT_MATRIX_LAYOUT_COLS = 3,
    
    /** Matrix leading dimension. Currently only non-negative values are supported.
     *  value type: uint64_t
     */
    CUBLASLT_MATRIX_LAYOUT_LD = 4,
    
    /** Number of matmul operations to perform in the batch. See also CUBLASLT_ALGO_CAP_STRIDED_BATCH_SUPPORT. 
     *  value type: int32_t, default: 1
     */
    CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT = 5,
    
    /** Stride (in elements) to the next matrix for strided batch operation.
     *  value type: int64_t, default: 0
     */
    CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET = 6,
    
    /** Stride (in bytes) to the imaginary plane for planar complex layout. 
     *  value type: int64_t, default: 0
     */
    CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET = 7,
} cublasLtMatrixLayoutAttribute_t;
```

此处，通过矩阵布局属性控制，实现批量的矩阵乘加操作的代码如下所示。

```c++
int main(int argc, char *argv[]) {
    const int M = 2;
    const int K = 3;
    const int N = 2;
    const int32_t batch = 2;
    float *h_A = (float*)malloc(sizeof(float) * M * K * batch);  // X, input
    float *h_B = (float*)malloc(sizeof(float) * K * N);          // W, weight
    float *h_C = (float*)malloc(sizeof(float) * 1 * N);          // B, bias
    float *h_D = (float*)malloc(sizeof(float) * M * N * batch);  // Y = X W + B
    for (int i = 0; i < M * K * batch; i++) h_A[i] = (i % (M * K)) * 1.0;
    for (int i = 0; i < K * N; i++)         h_B[i] = i * 1.0;
    for (int i = 0; i < 1 * N; i++)         h_C[i] = i * 0.1;
    for (int i = 0; i < M * N * batch; i++) h_D[i] = 0.0;
    printf("A\t=\t"); for (int i = 0; i < M * K * batch; i++) printf("%.1f\t", h_A[i]); printf("\n");
    printf("B\t=\t"); for (int i = 0; i < K * N; i++)         printf("%.1f\t", h_B[i]); printf("\n");
    printf("C\t=\t"); for (int i = 0; i < 1 * N; i++)         printf("%.1f\t", h_C[i]); printf("\n");
    
    float *A, *B, *C, *D;
    cudaMalloc(&A, sizeof(float) * M * K * batch);
    cudaMalloc(&B, sizeof(float) * K * N);
    cudaMalloc(&C, sizeof(float) * 1 * N);
    cudaMalloc(&D, sizeof(float) * M * N * batch);
    cudaMemcpy(A, h_A, sizeof(float) * M * K * batch, cudaMemcpyHostToDevice);
    cudaMemcpy(B, h_B, sizeof(float) * K * N, cudaMemcpyHostToDevice);
    cudaMemcpy(C, h_C, sizeof(float) * 1 * N, cudaMemcpyHostToDevice);

    cublasLtHandle_t handle;
    cublasLtCreate(&handle);
    cublasLtMatmulDesc_t computeDesc;
    cublasLtMatmulDescCreate(&computeDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc, Ddesc;  // row-major
    cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, M, K, K);
    cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, K, N, N);
    cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, M, N, 0);  // broadcast
    cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_32F, M, N, N);
    cublasLtOrder_t row_major = CUBLASLT_ORDER_ROW;
    cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major, sizeof(cublasLtOrder_t));
    cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major, sizeof(cublasLtOrder_t));
    cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major, sizeof(cublasLtOrder_t));
    cublasLtMatrixLayoutSetAttribute(Ddesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major, sizeof(cublasLtOrder_t));
    cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(int32_t));
    cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(int32_t));
    cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(int32_t));
    cublasLtMatrixLayoutSetAttribute(Ddesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(int32_t));
    const int64_t Astride = M * K;
    const int64_t Bstride = 0;
    const int64_t Cstride = 0;
    const int64_t Dstride = M * N;
    cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &Astride, sizeof(int64_t));
    cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &Bstride, sizeof(int64_t));
    cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &Cstride, sizeof(int64_t));
    cublasLtMatrixLayoutSetAttribute(Ddesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &Dstride, sizeof(int64_t));

    float alpha = 1.0, beta = 1.0;
    // Not use workspace
    cublasLtMatmul(handle, computeDesc, &alpha, A, Adesc, B, Bdesc, &beta, C, Cdesc, D, Ddesc, NULL, NULL, 0, NULL);

    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatrixLayoutDestroy(Ddesc);
    cublasLtMatmulDescDestroy(computeDesc);
    cublasLtDestroy(handle);

    cudaMemcpy(h_D, D, sizeof(float) * M * N * batch, cudaMemcpyDeviceToHost);
    printf("D\t=\t"); for (int i = 0; i < M * N * batch; i++) printf("%.1f\t", h_D[i]); printf("\n");
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaFree(D);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_D);
    return 0;
}
```

```
A       =       0.0     1.0     2.0     3.0     4.0     5.0     0.0     1.0     2.0     3.0     4.0     5.0
B       =       0.0     1.0     2.0     3.0     4.0     5.0
C       =       0.0     0.1
D       =       10.0    13.1    28.0    40.1    10.0    13.1    28.0    40.1
```



