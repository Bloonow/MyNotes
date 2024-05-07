# CUDA运行时

CUDA Runtime运行时环境在cudart库中实现，使用CUDA的应用程序都需要链接到该运行时库，要么是cudart.lib或libcudart.a静态库，要么是cudart.dll或libcuda.so动态库。因此需要动态链接到cudart.dll或libcudart.so库的应用程序，通常会将CUDA动态库包含在安装目录。只有链接到同一个CUDA运行时实例的多个组件，才能安全地传递地址信息。

## 初始化

自CUDA 12.0以来，cudaInitDevice()调用或cudaSetDevice()调用都会初始化特定设备的CUDA运行时以及主上下文环境（primary context），若没有手动执行该初始化调用，CUDA运行时会使用编号为0的GPU设备，并在执行其它运行时API时执行自初始化。在统计API时间或分析第一次调用错误时，需要注意这一点。在CUDA 12.0之前，函数cudaSetDevice()并不会初始化CUDA运行时，应用程序通常会使用cudaFree(0)函数以初始化CUDA运行时环境。

CUDA运行时会为系统中的每个GPU设备创建一个CUDA上下文环境（context），称为该设备的主上下文环境，并在运行时API首次需要上下文环境时被初始化。在创建主上下文环境时，如果需要，设备代码会即时编译（just-in-time compile）并加载到设备内存中。主上下文环境会在应用的所有主机线程之间共享，可使用驱动API访问主上下文环境。

当主机线程调用cudaDeviceReset()函数时，当前主机线程正在使用设备的主上下文环境会被销毁，并在下次调用运行时API函数时，在相应的GPU设备上创建一个新的主上下文环境。
