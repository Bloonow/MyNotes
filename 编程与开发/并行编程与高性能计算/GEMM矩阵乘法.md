# 通用矩阵乘法GEMM

通用矩阵乘法GEMM一般是指计算数学公式
$$
\text{C} = \text{AB}
$$
其中，A,B,C分别是形状为[M,K]，[K,N]，[M,N]的矩阵，则计算矩阵C的伪代码如下所示。

```c++
for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
        C[i][j] = 0;
        for (int p = 0; p < K; k++) {
            C[i][j] += A[i][p] * B[p][j];
        }
    }
}
```

本节在NVIDIA Tesla V100 SXM2-32GB GPU上进行优化实现，其配备的是Volta GV100架构的芯片。使用FP32单精度浮点数的数据类型。

> l2cache_bandwidth: 2624 GiB/s
>
> l2cache_latency: 194 cyclel
>
> l1cache_latency: 28 cycle
>
> smem_bandwidth: 115.35 Byte/cycle
>
> smem_latency: 19 cycle
>
> dram_bandwidth: 791 GiB/s(read), 836 GiB/s(write), 764 GiB/s(copy)
>
> dram_latency: 400~440 cycle
>
> 14FLOPS(single precision)

## GEMM实现方案与理论性能分析

### 基于GEMM定义的朴素实现

对于矩阵C的每一个元素，都要读取矩阵A的一行和矩阵B的一列来计算，那么计算完整的矩阵C，矩阵A和B都要重复读取多次，所以直接按定义计算的朴素实现效率很低。若由一个线程计算结果矩阵C中的一个元素，则朴素的GEMM实现可以如下所示。

<img src="GEMM矩阵乘法.assets/Simple GEMM.png" style="zoom:12%;" />

如上图所示，一个Warp一次执行32个FFMA，对应64次浮点运算FLOP，需要读取矩阵A的1个元素和矩阵B的32个元素，共计132Byte数据。通过寄存器累加，忽略矩阵C的写回开销，则计算访存比为64FLOP÷132Byte＝0.48。虽然设备内存（device random access memory，dram）最小访问单位为一个内存事务（memory transaction），但考虑到L1缓存的存在也不会影响实际的计算访存比。

使用l2cache_bandwidth.cu可以测得V100的L2缓存的带宽约为2624GiB/s，那么最乐观估计即使L2缓存100%命中，此方案的理论上限也只有2624×0.48＝1.23TFLOPS，远低于V100理论上14TFLOPS的硬件算力。

### Threadblock Tile：利用Shared Memory减少重复访存

利用高速存储器来减少低速存储器的访问是常用的优化手段，所以可以使用共享内存（shared memory，smem）来减少重复的设备内存dram的读取。

首先把矩阵C矩阵划分为M_tile×N_tile大小的分块（称之为Threadblock Tile），每个分块由一个Threadblock计算。之后FFMA计算所需的数据全部从smem中读取，就消除了对矩阵A和B的一部分重复内存读取。考虑到smem容量有限，可以在维度K上每次读取K_tile大小的分块，直到完整遍历维度K即可得到Threadblock Tile的结果。

<img src="GEMM矩阵乘法.assets/Threadblock Tile.png" style="zoom:12%;" />

在使用smem优化之后，对于一个M_tile×N_tile分块，可以得到如下指标，即计算量（Compute）、访存量（Memory）、计算访存比（GEMM_ratio）。
$$
\begin{align}
\text{Compute} &= \text{M\_tile}\times\text{N\_tile}\times\text{K}\times2\;\text{FLOP} \\
\text{Memory} &= (\text{M\_tile}+\text{N\_tile})\times\text{K}\times4\;\text{Byte} \\
\text{GEMM\_ratio} &= \frac{\text{Compute}}{\text{Memory}}
= \frac{1}{2}\frac{1}{\frac{1}{\text{M\_tile}}+\frac{1}{\text{N\_tile}}}
\end{align}
$$
假设M_tile和N_tile都取值为64，则按上式可算得计算访存比为16，代入V100的算力可得14TFLOPS÷16＝896GiB/s，即对于64×64大小的Threadblock Tile，平均访存带宽超过896GiB/s之后，性能瓶颈就不在访存上了。使用dram_bandwidth.cu可以测得V100的设备内存的只读带宽约为791GiB/s，考虑之前测得的L2缓存的带宽约为2624GiB/s，当L2缓存的命中率超过10%时，加权访存带宽为791×0.9＋2624×0.1＝974GiB/s，已经超过896GiB/s的拐点。

位于同一行的Threadblock Tile，读取相同的矩阵A分块，位于同一列的Threadblock Tile，读取相同的矩阵B分块，这些重复读取能大大提高L2缓存命中率（通常有60%以上的命中率），所以通过矩阵C分块并使用smem优化，可以比较容易地超过访存性能拐点，消除朴素实现方法中的内存读取瓶颈。

另外，GPU设备当前的FP32算力与存储系统带宽的比值不像tensor core那么悬殊，达到访存性能拐点比较容易。而对于tensor core提供的FP16/INT8算力，就需要一些额外的技巧提高L2缓存命中率（80%以上的命中率）才能最大化算力。

### Warp Tile与Thread Tile：利用寄存器消除Shared Memory瓶颈

#### Shared Memory访存瓶颈

对于一个Threadblock Tile的计算过程而言，最直接的计算方法是把M_tile×N_tile个元素平均分配到Threadblock的每个线程上去，每个线程分得M_frag×N_frag个元素（称之为Thread Tile）。之后每个线程再对各自Thread Tile内的元素按照GEMM的定义进行计算，如下伪代码所示。

```c++
for (int i = 0; i < M_frag; i++) {
    for (int j = 0; j < N_frag; j++) {
        for (int p = 0; p < K_tile; p++) {
            A_frag[i][p] = A_tile[row_offset + i][p];
            B_frag[p][j] = B_tile[p][col_offset + j];
            C_frag[i][j] += A_frag[i][p] * B_frag[p][j];
        }
    }
}
```

上述过程包括两个步骤，即，(1)把数据A_tile和B_tile从smem中读取到寄存器A_frag和B_frag中，(2)对寄存器数据做FFMA计算。

对于最内层的在维度K_tile上进行的循环，一个Warp的执行过程如下图所示。

<img src="GEMM矩阵乘法.assets/Simple Thread Tile.png" style="zoom:12%;" />

其中，绿色表示一个Warp所要计算的C_tile的元素，浅蓝色和浅黄色在计算C_tile的一部分时需要读取的smem数据，深蓝色和深黄色表示一个Warp在每次循环迭代中读取的数据。矩阵A_tile中的深蓝色数据可以利用广播发送给Warp的32个线程，矩阵B_tile中的深黄色数据32×4Byte＝128Byte，没有bank冲突。那么一次迭代中共有32个FFMA和256Byte的smem到寄存器数据传输。

V100上每个SM的smem出口带宽为128Byte/cycle，那么32个FFMA对应需要2个cycle的smem数据传输。V100每个SM每个cycle可执行64个FFMA，则对应的smem数据读取需要4个cycle，所以此方案中smem到寄存器的数据传输会成为瓶颈，只能发挥出每个SM四分之一的算力，在V100上的理论上限为14TFLOPS÷4＝3.5TFLOPS。

> 此外，也只有在GV100、GA100、GV102、GA102等核心上smem的出口带宽才有128Byte/cycle，而在TU102等核心上smem的出口带宽只有64Byte/cycle，而且诸如TU102等核心上每个SM每个cycle可执行64个FFMA（具有64个FP32单元），所以只能发挥八分之一的硬件算力。

需要说明的是，上述分析是基于满流水并行带宽计算的，即所说的2cycle或4cycle的数据传输并不表示发射smem读取请求后经过2cycle或4cycle就能拿到数据，实际上，smem单次访问延迟有20~30个cycle。另外单条FFMA指令在Volta架构上有4个cycle的延迟，每个SM的64个FFMA单元也是分为4个处理块子分区，每个处理块子分区调度时也是分为两次Half Warp调度的。在以带宽角度分析GEMM性能时，以满流水状态下的计算或访存带宽为参考依据，就有了诸如“每个cycle执行64个FFMA”和“每个cycle传输128Byte数据”之类的指标。不要将带宽和延迟混淆。

#### Thread Tile: 利用寄存器减少Shared Memory读取

如何解决Shared Memory访存瓶颈呢？再次回顾上述的Thread Tile计算过程的伪代码如下。

```c++
for (int i = 0; i < M_frag; i++) {
    for (int j = 0; j < N_frag; j++) {
        for (int p = 0; p < K_tile; p++) {
            A_frag[i][p] = A_tile[row_offset + i][p];
            B_frag[p][j] = B_tile[p][col_offset + j];
            C_frag[i][j] += A_frag[i][p] * B_frag[p][j];
        }
    }
}
```

可以看到，上述按照M-N-K的循环嵌套顺序，实际上是矩阵乘法的向量内积表示形式，A_tile的读取位置与i,p有关，B_tile的读取位置与p,j有关，循环嵌套之下产生了重复的smem读取，这也是相对smem的计算访存比低的原因。而如果改为K-M-N的循环嵌套顺序，就变成了矩阵乘法的向量外积表示形式，如下。

```c++
for (int p = 0; p < K_tile; p++) {
    A_frag[...][p] = A_tile[row_offset + ...][p];
    B_frag[p][...] = B_tile[p][col_offset + ...];
    for (int i = 0; i < M_frag; i++) {
        for (int j = 0; j < N_frag; j++) {
            C_frag[i][j] += A_frag[i][p] * B_frag[p][j];
        }
    }
}
```

相应的Thread Tile的处理过程变为下图所示。

<img src="GEMM矩阵乘法.assets/Thread Tile.png" style="zoom:12%;" />

可以看到，在计算一个Thread Tile时，参与计算的A_tile和B_tile中的元素只被读取了一次，在单个线程内消除了向量内积实现中的smem重复读取。

但向量外积实现方案中的A_frag、B_frag、C_frag需要占用大量的寄存器，假设M_frag和N_frag都为8，那么A_frag和B_frag各需8个寄存器（如果双缓冲就是16个），C_frag需要8×8＝64个寄存器，所以此优化的本质还是用寄存器换smem访存，即高速存储器换低速存储器。

由于一个SM执行访存指令的LSU访存单元较少（通常是16个或32个），访存指令的IPC较低，而FFMA计算单元很多（通常是64个或128个），所以FFMA与其他指令流水并行状态下，FFMA要掩盖所有其他指令的延迟。因此，除计算访存比之外，还需要考虑指令调度的开销。针对smem而言，即最大化在K_tile维度上循环过程中的FFMA指令与smem访存指令的比值。

上述向量外积实现方案中，LDS指令数量与M_frag、N_frag之和成正比，与LDS指令的访存宽度成反比（例如，LDS.32、LDS.64、LDS.128等指令），设比例系数为α，FFMA指令数目为M_frag×N_frag，则有指令数目之比（I-Ratio）如下所示。
$$
\begin{align}
\text{I-Ratio} = \frac{\text{FFMA}}{\text{LDS}} 
&= \frac{1}{\alpha}\frac{\text{M\_frag}\times\text{N\_frag}}{\text{M\_frag}+\text{N\_frag}} \\
&= \frac{1}{4\alpha}\frac
{(\text{M\_frag}+\text{N\_frag})^2 - (\text{M\_frag}-\text{N\_frag})^2}{\text{M\_frag}+\text{N\_frag}}
\end{align}
$$
由此可以得到结论，(1)划分M_frag、N_frag越大，FFMA与LDS指令数目之比越高；(2)划分M_frag、N_frag之和为常数时，M_frag、N_frag之差越小，FFMA与LDS指令数目之比越高；(3)若FFMA与LDS指令数目之比为常数，M_frag、N_frag之差越小，M_frag、N_frag之和越小。

也即是说，(1)当GPU设备的FFMA指令与LDS指令的IPC越悬殊时，需要更大的M_frag、N_frag来让FFMA掩盖LDS指令延迟，随之也会消耗更多的寄存器；(2)当A_frag和B_frag占用寄存器总数固定时，M_frag、N_frag之差越小，LDS指令占比越低，越容易被FFMA指令掩盖延迟；(3)若LDS指令占比固定，则M_frag、N_frag之差越小，A_frag和B_frag占用寄存器总数越少。

计算能力3.5及之后的GPU设备，单线程最多使用255个通用寄存器，当Thread Tile处理16×16个元素时，寄存器不够用；当Thread Tile处理8×4个元素时，FFMA总延迟为32cycle＋FFMA_latency，而smem本身就具有20~30个cycle的延迟，同时global memory到smem的读取以及各种地址计算也需要指令，所以Thread Tile取8×4不足以用FFMA指令掩盖其它指令的延迟。所以，SGEMM中的Thread Tile通常取8×8或8×16等数值。注意，这里分析的是单线程内的指令延迟覆盖问题，所以使用指令延迟作为计算指标。

#### Warp Tile：最大化相对Shared Memory的计算访存比

GPU设备的实际调度的单位为一个Warp，许多开销都与整个Warp的行为密切相关，所以向量化外积方案中，除了单线程内的延迟覆盖问题，还要考虑整个Warp上的计算访存比。

一个Warp由Warp_y×Warp_x个线程组成，可以是1×32、2×16、4×8等，将这些线程对应的Thread Tile拼在一起的区域称为一个Warp Tile，尺寸为M_warp×N_warp，如下图所示。可以看到FFMA次数为M_warp×N_warp，对smem的访存量与M_warp、N_warp之和成正比，显然一个Warp的Thread摆放成4×8时计算访存比最高，摆放成1×32时最低。

<img src="GEMM矩阵乘法.assets/Warp Tile.png" style="zoom:12%;" />

假设各个硬件流水线满载运行，此处从带宽角度进行分析，只关注每条指令发射所占用的时钟周期，而忽略指令执行所需的实际周期（存储器响应并提供数据的实际时钟周期）。

以最差的一个Warp的Thread摆放成1×32为例进行分析，当Thread Tile为8×8时，矩阵A需要读取1×8×4＝32Byte，在V100设备上需要至少4个cycle（与共享内存的广播机制有关），矩阵B需要读取32×8×4＝1024Byte，需要8个cycle。所以，最差的情况下，向量外积实现方案中，计算量为8×8×32＝2048个FFMA，对应着12个cycle的smem读取，平均1个cycle的smem读取对应2048÷12＝170个FFMA计算。带宽分析上是完全满足V100最高算力的“每个SM每个cycle执行64个FFMA”的需求。

如果是TU102小核心，smem具有64Byte/cycle的出口带宽，仍然使用1×32的Thread摆放时，矩阵A读取需要4个cycle，矩阵B读取需要16个cycle，平均1个cycle的smem读取对应2048÷20＝102个FFMA计算，低于设备的“每个SM每个cycle执行128个FFMA”理论峰值。如果改为4×8的Thread摆放，则矩阵A和矩阵B的smem读取均变成4个cycle，平均1个cycle的smem读取对应2048÷8＝256个FFMA计算。可以满足设备理论FFMA上限需求，所以一般Warp都会配置为4×8或8×4的线程摆放。

但需要注意的是，正如之前所说，**以带宽作为参考而忽略高延迟的前提，是延迟能被其他过程覆盖**，如无法覆盖则跑不满理论带宽，那么带宽分析的结果也就没有了参考价值。例如，此处一个Warp内的线程，肯定要等待对应的数据从smem中读到寄存器之后才能做执行FFMA计算，如果数据读取总延迟大于FFMA总延迟，则会导致FFMA等待数据读取，那么实际的FFMA带宽也就达不到“每个cycle执行170个FFMA”了，除非有较高的occupancy占用率，有足够多的Warp填充延迟。

从延迟角度分析，再回到1×32的线程摆放情况，矩阵A的读取延迟为4cycle＋smem_latency，矩阵B的读取延迟为8cycle＋smem_latency，矩阵A和矩阵B的读取总延迟为12cycle＋smem_latency，FFMA总延迟为8×8cycle＋FFMA_latency。考虑到除smem读取之外还有global memory读取、地址计算、循环体的比较/跳转等指令，另外在很多smem吞吐为64Byte/cycle的设备上总延迟更高，1×32的线程摆放在occupancy较低时很难做到FFMA覆盖其他延迟，所以从延迟的角度，也是选取4×8或8×4的线程摆放会更好。

### Double Buffer：让GEMM流水并行起来

在NVIDIA GPU上掩盖延迟的方式主要有两种，一是Warp之间并行，二是Warp之内的指令级并行（Instruction Level Parallelism，ILP）。Warp之间的并行依赖于occupancy，当有足够多的Warp可调度时，一个Warp如果因为某些原因无法继续发射指令（例如barrier栅障、执行依赖等），则可以发射其它Warp的指令来填满硬件资源。Warp之内的ILP主要靠消除指令发射阻塞，使单个Warp内的指令序列足够填满硬件资源。

从之前上一节的分析中可以看出，要通过向量外积实现来消除smem访存瓶颈，则Thread Tile至少需要8×8或者更大，那么A_frag、B_frag、C_frag至少需要消耗8＋8＋64＝80个寄存器，此外，还有用于从global memory读取的中转寄存器、global/shared memory读写指针、Threadblock Tile循环变量等等，使得8×8的Thread Tile通常每个线程需要用到120~128个寄存器。在V100设备上，一个SM拥有4×16384个寄存器，可以调度(4×16384)/(128×32)＝16个Warp，而一个SM管理一个最多可包含64个Warp的线程池，所以占用率只有25%。V100设备拥有4个处理块子分区，每个子分区上的Warp调度器只有4个可调度的Warp，则当指令的平均发射间隔超过4个cycle之后，就无法依靠Warp之间的并行调度掩盖延迟了。考虑到GEMM中涉及smem读写的过程需要同步Threadblock，进一步限制了Warp之间的并行调度空间，所以很难依靠Warp之间并行来掩盖延迟。

于是，只能想办法提高单个Warp之内的指令级并行度了。依照之前的描述，完整的GEMM流程图如下所示。

<img src="GEMM矩阵乘法.assets/Multi Tile.png" style="zoom:12%;" />

图中，黑色字体表示方框的含义和所处的存储器，蓝色字体表示执行的指令（例如global memory读取指令LDG，smem写入指令STS等），绿色字体表示执行指令的硬件单元，橘黄色字体表示指令所涉及的存储器。

可以看出，GEMM实际上由下图所示的相互依赖的四步串联而成，每个步骤使用不同的存储器和指令执行部件。

<img src="GEMM矩阵乘法.assets/Threadblock Tile Loop.png" style="zoom:12%;" />

那么，很容易想到可以通过双缓冲（double buffer）和预取（pre-fetch）的方式实现多个步骤的流水并行。

<img src="GEMM矩阵乘法.assets/Double Buffer.png" style="zoom:12%;" />

将用于存储Threadblock Tile的smem分配两份（smem[0]、smem[1]），存储A_frag、B_frag的寄存器也分配两份（reg[0]、reg[1]），就可以消除几个步骤的前后依赖，实现Threadblock Tile读取，Fragment读取，FFMA计算之间的流水并行，也减少了一次Threadblock同步。实际上，由于global memory和shared memory之间的巨大的带宽和延迟差距，store smem相对于load gmem而言占比非常小，实现这两个步骤的流水线并行会大大增加代码复杂度导致负优化，所以直接串联即可。

### 小结

GEMM实现方案与GPU硬件特性密切相关，本节结合GPU上不同层次的并行计算部件，描述了多层分块并行的实现策略，并对各种实现方法做了理论性能定量分析。

1. Threadblock Tile配合smem解决了内存带宽瓶颈；
2. 向量外积实现方法解决了smem访问带宽瓶颈；
3. Warp Tile和Thread tile实现了用FFMA指令掩盖其他指令的延迟，并最大化相对smem的计算访存比；
4. Tile读取-Fragment读取-FFMA计算，三级软件流水设计，在向量外积实现使寄存器消耗量巨大导致occupancy较低的情况下，通过Warp之内的指令并行实现了硬件资源的充分利用。

上述方案伪代码可表示为下面形式。

```c++
__global__ void sgemm_128x128(
    const float* A_ptr, const float* B_ptr, float* C_ptr,
    const uint32_t M, const uint32_t N, const uint32_t K, const uint32_t K_tile = 8) {
    __shared__ float A_tile[2][K_tile][128];
    __shared__ float B_tile[2][K_tile][128];
    float A_frag[2][8], B_frag[2][8], C_frag[8][8] = { 0 };
    float A_ldg_buffer[4], B_ldg_buffer[4];

    // 读取第一个tile并写入共享内存，处理不足K_tile的边界情况
    load_tile(A_ptr, A_ldg_buffer);
    load_tile(B_ptr, B_ldg_buffer);
    store_tile(A_ldg_buffer, A_tile[0]);
    store_tile(B_ldg_buffer, B_tile[0]);
    // 读取第一个frag并写入寄存器
    load_frag(A_tile[0][0], A_frag[0]);
    load_frag(B_tile[0][0], B_frag[0]);
    // 共享内存双缓冲的索引
    uint32_t smem_ld_idx = 0, smem_st_idx = 1;

    // K-Loop
    for (int num_k_tiles = (K + K_tile - 1) / K_tile - 1; num_k_tiles > 0; num_k_tiles--) {
        for (int k_frag = 0; k_frag < K_tile; k_frag++) {
            // K_tile次计算即将执行完毕，将下一个tile写入共享内存
            if (k_frag == K_tile - 1) {
                store_tile(A_ldg_buffer, A_tile[smem_st_idx]);
                store_tile(B_ldg_buffer, B_tile[smem_st_idx]);
                smem_ld_idx ^= 1;
                smem_st_idx ^= 1;
            }
            // 读取下一次计算所需的frag并写入寄存器
            load_frag(A_tile[smem_ld_idx][(k_frag + 1) % K_tile], A_frag[(k_frag + 1) % 2]);
            load_frag(B_tile[smem_ld_idx][(k_frag + 1) % K_tile], B_frag[(k_frag + 1) % 2]);
            // K_tile的第一次计算之前，读取下一个tile数据
            if (k_frag == 0) {
                load_tile(A_ptr, A_ldg_buffer);
                load_tile(B_ptr, B_ldg_buffer);
            }
            // 执行FFMA计算
            ffma(A_frag[k_frag % 2], B_frag[k_frag % 2], C_frag);
        }
    }
    // 最后一个tile的迭代
    for (int k_frag = 0; k_frag < K_tile; k_frag++) {
        // 读取下一次计算所需的frag并写入寄存器
        if (k_frag < K_tile) {
            load_frag(A_tile[smem_ld_idx][(k_frag + 1) % K_tile], A_frag[(k_frag + 1) % 2]);
            load_frag(B_tile[smem_ld_idx][(k_frag + 1) % K_tile], B_frag[(k_frag + 1) % 2]);
        }
        // 执行FFMA计算
        ffma(A_frag[k_frag % 2], B_frag[k_frag % 2], C_frag);
    }

    // 重用共享内存空间，写回矩阵C的计算结果
    store_result(C_ptr, C_frag);
}
```

其中，K_tile为常数，对K_tile的内部循环体可以被编译器展开，所以代码实际只产生num_k_tiles这一循环，称之为K-Loop，或称之为Main-Loop主循环，K-Loop是GEMM的性能热点，也是优化的主要对象。

## Threadblock Tile尺寸选择

前面基本实现方案的介绍，对各种条件下的理论性能做了简要分析。可以看出，在条件允许的情况下，无论是Threadblock Tile分块、Warp Tile分块，还是Thread Tile分块，通常尺寸越大越容易优化到较高性能。但考虑到矩阵规模是多变的，不可能只用一种分块方案在各种输入规模下都有较好的性能。对于给定的矩阵规模M,N,K，此处将分析最佳Threadblock Tile尺寸的选择。

为什么不讨论Warp Tile与Thread Tile的尺寸选择呢？从之前的分析中可以看出，Warp Tile与Thread Tile的选择与一个SM中的资源关系密切，例如FFMA指令和LDS指令的IPC、smem带宽、FFMA指令和LDS指令的延迟等。另外对于SGEMM而言，Warp Tile（诸如32×64，或64×64）已经是一个很小的分割单位，所以Warp Tile和Thread Tile在一种架构上通常会取固定值，并通过改变Threadblock的大小和Warp的摆放来调整Threadblock Tile以适应输入矩阵的规模。除非遇到M或N小于64甚至32的这种极度扁长的矩阵（更像是矩阵向量乘），会单独考虑诸如SliceK和SplitK算法。

> 注意，所谓的“一种架构”是指单个SM结构完全一致的核心，有些架构代号相同但SM结构有差异的GPU设备，例如GK110和GK210、GP100和GP102、GV100和GV102、GA100和GA102等，是需要视为不同架构进行优化的。

### 等效内存带宽和L2缓存命中率估算

在之前的分析中，Threadblock Tile主要用于消除内存读取瓶颈，对于大矩阵，需要分块增大到计算访存比超过硬件性能才有机会跑满硬件算力，即满足：
$$
\text{GEMM\_ratio} = \frac{\text{FFMA}}{\text{LDG}} = \frac{1}{2}\frac{1}{\frac{1}{\text{M\_tile}}+\frac{1}{\text{N\_tile}}}
\ge \frac{\text{P\_ffma}}{\text{P\_ldg}}
$$
其中，P_ffma表示硬件FFMA算力，P_ldg表示内存与L2缓存加权之后的平均访存带宽，所以只有知道L2缓存命中率，才能算出跑满硬件算力的最小分块。L2缓存供一个GPU设备上的所有SM使用，其命中率主要依赖同时运行的Threadblock对输入矩阵的重复读取。通常来说，GPU设备的FP32算力与内存带宽比值普遍不高，无需特殊的Tile映射方法提高L2缓存命中率，所以可以基于朴素的Tile映射方法分析L2缓存的命中率，即tile_x＝blockIdx.x，tile_y＝blockIdx.y。

这里引入Wave（波）的概念，一个Wave表示一个GPU上同时执行的Threadblock。例如，一个kernel核函数的Threadblock为256个线程，每个线程使用128个寄存器，则在V100上每个SM可以同时执行2个Threadblock，并且V100拥有80个SM，于是一个Wave就是160个Threadblock。

在GEMM中，一个Wave对应的矩阵A、矩阵B、矩阵C的数据区域如下图所示。

<img src="GEMM矩阵乘法.assets/Wave Tile.png" style="zoom:12%;" />

一个Wave的大小，即在GPU上同时运行的Threadblock的数目，表示为Wave_gpu，于是得到：
$$
\begin{align}
\text{Wave\_y} &= \frac{\text{M}}{\text{M\_tile}} \\
\text{Wave\_x} &= \frac{\text{N}}{\text{N\_tile}} \\
\text{Wave\_rem} &= \text{Wave\_gpu} \;\%\; \text{Wave\_x}
\end{align}
$$
可以算出一个Wave对应的访存请求量：
$$
\text{A\_ldg\_request} = \text{Wave\_gpu} \times \text{M\_tile} \times \text{K} \\
\text{B\_ldg\_request} = \text{Wave\_gpu} \times \text{N\_tile} \times \text{K}
$$
对设备内存dram产生的实际访存量：
$$
\begin{align}
\text{A\_ldg\_dram} &= (\text{Wave\_y} + 1) \times \text{M\_tile} \times \text{K} \\
\text{B\_ldg\_dram} &= \text{N} \times \text{K}
\end{align}
$$
可以得到L2缓存理论上的命中率，大约是：
$$
\text{L2\_hit\_rate} = 1 - \frac{\text{A\_ldg\_dram}+\text{B\_ldg\_dram}}{\text{A\_ldg\_request}+\text{B\_ldg\_request}}
$$
考虑L2缓存加权后的平均访存带宽约为：
$$
\text{P\_ldg} = \text{P\_l2\_bw}\times\text{L2\_hit\_rate} + \text{P\_dram\_bw}\times(1-\text{L2\_hit\_rate})
$$
于是，可以得到GEMM的预估性能：

$$
\text{P\_gemm} = \begin{cases}
\text{P\_ldg}\times\text{GEMM\_ratio} & \text{GEMM\_ratio} < \frac{\text{P\_ffma}}{\text{P\_ldg}} \\
\text{P\_ffma} & \text{GEMM\_ratio} \ge \frac{\text{P\_ffma}}{\text{P\_ldg}}
\end{cases}
$$
上述P_gemm表达式的主要作用有两个，对于给定的M,N,K规模，一是可以估算出各种分块大小的性能，选择最快的kernel；二是可以针对特定的GPU架构与型号，算出跑满硬件算力所需的最小分块。但需要注意的是，上述分析成立需要两个条件，一是矩阵规模足够大，即M,N,K较大，矩阵A和矩阵B无法完全放进L2缓存当中；二是Threadblock分块的总数超过一个Wave的大小。

由于L2缓存的替换策略对命中率有一定影响，上面表达式算出的命中率与实际命中率会有一定误差，当矩阵A和矩阵B的数据量较大时（例如矩阵的存储空间是L2缓存容量的5倍以上），误差较小（通常10%以内），而更小的矩阵误差会更大。但这种误差通常不会导致不同分块尺寸之间的性能结果比较出现错误，也就是说按照上面的方法做性能分析是足够准确的。

### L2缓存命中率与矩阵规模的关系

上一小节分析了矩阵大小、分块大小、L2缓存命中率之间的关系。如果固定Threadblock Tile的大小，只关注M,N,K对L2命中率的影响，且忽略上一节中Wave_rem的影响，这个关系就会变得非常简单。本节使用这种简化的计算方法估算L2命中率，以直观地感受L2的实际命中率。

<img src="GEMM矩阵乘法.assets/Simple Wave Tile.png" style="zoom:12%;" />

当Threadblock Tile为常数时，占用率occupancy和一个Wave覆盖的矩阵C的范围也是常数，令一个Wave覆盖的区域为Wave_m×Wave_n常数。于是，可以得到一个Wave的访存请求量，以及对设备内存dram产生的实际访存量：
$$
\begin{align}
\text{A\_ldg\_request} &= \text{Wave\_gpu} \times \text{M\_tile} \times \text{K} \\
\text{B\_ldg\_request} &= \text{Wave\_gpu} \times \text{N\_tile} \times \text{K} \\
\text{A\_ldg\_dram} &= \text{Wave\_m} \times \text{K} \\
\text{B\_ldg\_dram} &= \text{Wave\_n} \times \text{K}
\end{align}
$$
于是，可以得到L2缓存理论上的命中率：
$$
\begin{align}
\text{L2\_hit\_rate} &= 1 - \frac{\text{A\_ldg\_dram}+\text{B\_ldg\_dram}}{\text{A\_ldg\_request}+\text{B\_ldg\_request}} \\
&= 1 - \frac{\text{Wave\_m}+\text{Wave\_n}}{(\text{M\_tile}+\text{N\_tile})\times\text{Wave\_gpu}} \\
&= 1 - \frac{1}{\text{M\_tile}+\text{N\_tile}} \frac{\text{Wave\_m}+\text{Wave\_n}}{\text{Wave\_m}\times\text{Wave\_n}} \\
&= 1 - \frac{4}{\text{M\_tile}+\text{N\_tile}}
\frac{\text{Wave\_m}+\text{Wave\_n}}{(\text{Wave\_m}+\text{Wave\_n})^2-(\text{Wave\_m}-\text{Wave\_n})^2}
\end{align}
$$
可以看出，当Wave_m、Wave_n之差约大（一个Wave覆盖区域越扁平或细长），L2命中率越低。如果Threadblock在矩阵C上以行主序排布，则当N足够大时，会使一个Wave的Threadblock摆成一行，这就变成极度扁平的情况。这也解释了用同一种分块大小，当矩阵的N变大时，或N极小而M变大时，L2命中率下降的原因。

假设一个Threadblock Tile取值128×128，一个Threadblock为256个线程，每个线程使用128个寄存器，在V100上一个Wave就是160个Threadblock。当N超过128×160＝20480时变为极度扁平的情况，此时L2命中率套用上述表达式可以求得为50%，也就是说，对于这个分块方案，无论M,N,K怎么变化，L2命中率的下限为50%。这个理论结果与上述条件下的实测结果相对误差在4%左右，实测约为52%的L2命中率。

从经验上来讲，采用朴素的Tile映射方法，大矩阵的L2命中率通常在50%~70%之间，极度扁平/细长的情况在多数应用场景的矩阵尺寸中很少出现。

### 分块大小与IPC的关系

GEMM过程主要包括计算指令（FFMA）和访存指令（LDG、STS、LDS），其中，FFMA指令由FP32单元执行，访存类指令由LSU单元执行。硬件上，两种单元的性能比例决定了计算访存指令比例与整体性能的关系。所以为了跑满FFMA算力，还应确保FFMA指令与访存指令的比例高于硬件性能的比例。

对于一个Threadblock Tile而言，可以算得Threadblock中的线程数目：
$$
\text{n\_thread} = \frac{\text{M\_tile}\times\text{N\_tile}}{\text{M\_frag}\times\text{N\_frag}}
$$
加载矩阵A和矩阵B的一个Threadblock Tile的数据量，一个线程要执行的32位的LDG指令的数目：
$$
\text{n\_ldg} = \frac{(\text{M\_tile}+\text{N\_tile})\times\text{K\_tile}}{\text{n\_thread}}
$$
将加载的Threadblock Tile存储到共享内存当中，虽然使用STS.128可以一次性写入4个float数据，但也是拆成4次memory transaction去做的，对MIO流水的占用等同于4条STS.32指令，所以STS次数按照STS.32指令数计算，有：
$$
\text{n\_sts} = \text{n\_ldg}
$$
在读取smem时，使用LDS.128指令，每个线程每条指令读取4个float元素，且一个Warp内的线程摆放为“Z”字形来最大化共享内存的broadcast广播性能（每条LDS.128只需要2次memory transaction），等效LDS指令数目：
$$
\text{n\_lds} = 2\times(\frac{\text{M\_frag}}{4}+\frac{\text{N\_frag}}{4})\times\text{K\_tile}
$$
一个线程所执行的FFMA指令数目：
$$
\text{n\_ffma} = \text{M\_frag}\times\text{N\_frag}\times\text{K\_tile}
$$
考虑到一个Threadblock Tile的执行由dram-L2读取和smem写入串联而成，且dram延迟较高，所以在double buffer流水并行设计中，为防止FFMA指令等待Tile从全局内存中读取，应该尽量让LDG指令在K-Loop开头发射，让STS指令在K-Loop结尾发射，如下图所示。

<img src="GEMM矩阵乘法.assets/Instruction Latency.png" style="zoom:12%;" />

图中两条红色虚线之间的部分为访存指令密集区域。假设编译器对访存指令完美排序，即LDG指令读到数据后刚好发射对应的STS指令，那么两条红色虚线之间的耗时占比为：

$$
\beta = 1 - \frac{\text{LDG\_latency}+\text{STS\_latency}}{\text{n\_ffma}+\text{FFMA\_latency}}
\approx 1 - \frac{\text{LDG\_latency}+\text{STS\_latency}}{\text{n\_ffma}}
$$
于是，两条红色虚线之间的全局访存指令和共享内存访问指令的数目为：
$$
\text{n\_ld\_part} = \text{n\_ldg} + \beta\times\text{n\_lds}
$$
两条红色虚线之间的FFMA指令数目为：
$$
\text{n\_ffma\_part} = \beta\times\text{n\_ffma}
$$
访存指令与计算指令之比：
$$
\begin{align}
\text{Ratio\_ld\_ffma} &= \frac{\text{n\_ld\_part}}{\text{n\_ffma\_part}} 
= \frac{\text{n\_ldg}}{\beta\times\text{n\_ffma}} + \frac{\text{n\_lds}}{\text{n\_ffma}} \\
&= \frac{\text{M\_frag}\times\text{N\_frag}\times(\frac{1}{\text{M\_tile}}+\frac{1}{\text{N\_tile}})}
{\text{M\_frag}\times\text{N\_frag}-\frac{\text{LDG\_latency}+\text{STS\_latency}}{\text{K\_tile}}}
+ \frac{\text{M\_frag}+\text{N\_frag}}{2\times\text{M\_frag}\times\text{N\_frag}}
\end{align}
$$
要跑满FFMA算力，应该满足：
$$
\text{Ratio\_ld\_ffma} \le \frac{\text{P\_lsu\_ipc}}{\text{P\_ffma\_ipc}}
$$
其中，P_lsu_ipc和P_ffma_ipc分别访存指令的IPC和FFMA指令的IPC，在V100架构上，LSU单元能够为访存指令提供的IPC为0.25，FP32单元能够为FFMA提供的IPC为0.5。考虑到指令排布与访存延迟很难完美匹配，Ratio_ld_ffma应该在上述表达式阈值的基础上保留一些裕量，也就是尽量降低访存计算指令的比值。

从Ratio_ld_ffma的表达式可以看出，若M_frag、N_frag取值固定，则M_tile、N_tile取值越大，比值越低。同时，LDG、STS的延迟是硬件相关的常数，则K_tile越大，则FFMA的总数n_ffma越大，FFMA的总时间就越长，可以用来摆放LDG指令的时间窗口也就越大，访存指令的占比也就越低。但是，由于过大的K_tile会导致smem用量过多影响occupancy，所以一般通过增大M_tile、N_tile来降低访存指令的占比。

需要注意的是，上述表达式成立需要满足一个条件，也即分母部分应该为正数：
$$
\begin{align}
& \text{M\_frag}\times\text{N\_frag}-\frac{\text{LDG\_latency}+\text{STS\_latency}}{\text{K\_tile}} > 0 \\
\Longrightarrow \;& \text{M\_frag}\times\text{N\_frag}\times\text{K\_tile} > \text{LDG\_latency}+\text{STS\_latency} \\
\Longrightarrow \;& \text{n\_ffma} > \text{LDG\_latency}+\text{STS\_latency}
\end{align}
$$
如果不满足这个条件，直观理解也就意味着一次LDG和STS的延迟都无法被FFMA的总时间覆盖，这种情况下就需要借助多个Warp之间的并行调度切换来填满FFMA时间。这也解释了在DRAM延迟较高的设备（例如GDDR内存的GPU）上，若occupancy过低，即使按照之前分析的Threadblock Tile大小能够满足FFMA峰值性能，但实测性能也有较大差距的原因。

一般来讲，Threadblock Tile减小时Threadblock变小，会更容易达到更高的occupancy，可以降低访存指令数占比对性能的影响。所以对于小的Tile，之前分析的计算访存比对性能的影响更大，而此处分析的主要目的是针对大的矩阵乘法，帮助选择合适的Tile尺寸以跑出硬件算力上限。SGEMM对于大矩阵一般选择128×64或128×128甚至128×256的Threadblock Tile数据分块。

## 读写gmem和smem时的线程摆放

假设矩阵A和矩阵B都是行主序存储，由于一个线程采用向量外积实现来计算矩阵乘法，所以一个线程每次访问A的一列与B的一行，那么对于在设备全局内存gmem中的矩阵A的数据，需要对其进行转置，以列主序的方式写入到smem中。

假设Threadblock Tile是128×128×8，线程数目为256个，则当将矩阵A和矩阵B从设备全局内存加载到共享内存中时，线程摆放如下所示。

<img src="GEMM矩阵乘法.assets/Thread Layout for LDG and STS at 128x128.png" style="zoom:12%;" />

假设Threadblock Tile是128×128×8，线程数目为256个，Thread Tile是8×8，则当将矩阵A和矩阵B从共享内存加载到寄存器当中时，线程摆放如下所示。使用向量外积的计算方式，每个线程读取连续的4个元素，采用float4向量化读取，一次性读取16字节（128bit）。

<img src="GEMM矩阵乘法.assets/Thread Layout for LDS at 128x128.png" style="zoom:12%;" />

假设Threadblock Tile是128×128×8，线程数目为256个，Thread Tile是8×8，则当将矩阵A和矩阵B从寄存器写回到设备全局内存当中时，并借助共享内存重排数据布局，则线程摆放如下所示。

<img src="GEMM矩阵乘法.assets/Thread Layout for STG at 128x128.png" style="zoom:12%;" />

从Ampere架构（计算能力8.6）开始，设备支持一个新的异步复制指令load-global-store-shared，在CUDA 11.0中提供支持，能够直接从全局内存（通常是从DRAM和L2缓存当中）加载数据到SM上的共享内存，绕过中间的L1缓存，同时避免为传输数据分配中间临时寄存器，避免寄存器文件的往返读写以节省SM内部带宽。

假设Threadblock Tile是128×256×8，线程数目为256个，则当将矩阵A和矩阵B从设备全局内存加载到共享内存中时，线程摆放如下所示。

<img src="GEMM矩阵乘法.assets/Thread Layout for LDG and STS at 128x256.png" style="zoom:12%;" />

假设Threadblock Tile是128×256×8，线程数目为256个，Thread Tile是16×8，则当将矩阵A和矩阵B从共享内存加载到寄存器当中时，线程摆放如下所示。

<img src="GEMM矩阵乘法.assets/Thread Layout for LDS at 128x256.png" style="zoom:12%;" />

## 矩阵乘法的Tensor Core实现

因为一个WMMA指令需要一个Warp执行，而一个WMMA指令所用到的数据通常是诸如M×N×K＝16×16×16之类的形状，故一条WMMA指令就会用掉在维度轴K上长达16的数据，因此相比于传统使用CUDA Core的矩阵乘法而言（其K_tile通常为8或者16），使用Tensor Core实现的矩阵乘法往往需要更大的K_tile数据。

若使用K_wmma表示一个WMMA指令所用到的数据在维度轴K上的长度（通常是8或者16），那么在划分A_tile和B_tile时，通常会将K_tile划分成K_wmma的整数倍，可以使用Chunks_K表示具有多少个K_wmma长度的数据，则K_tile＝Chunks_K×K_wmma，其中Chunks_K通常取4或者8，具体取决于设备可以提供多少共享内存的容量。

例如，对于数据类型为half-float且形如m16n16k16的WMMA指令，假设Chunks_K＝8，线程块所划分的数据形状为M_tile,N_tile,K_tile＝128,128,128，那么一个A_tile所需要的共享内存容量是128×128×sizeof(half)＝32KiB，同样B_tile所需要的共享内存容量同样是32KiB，共需要64KiB的共享内存。实际上，为避免共享内存的Bank冲突，A_tile和B_tile需要一定的偏斜（skew）量，因为wmma::load_matrix_sync指令要求内存地址256位（32字节）对齐，故维度轴K上的每一维数都需要Skews_half＝16个half类型元素的偏斜。因此，当Chunks_K取值为8时，所需的实际共享内存容量为144×128×sizeof(half)×2＝72KiB的共享内存容量。若再考虑双缓冲，则需要更多的共享内存容量。

# Efficient GEMM in CUDA

朴素地，矩阵乘法可以使用多层嵌套循环实现，在并行编程环境下，可以使用不同的并行资源对多层嵌套的矩阵乘法计算进行Tiling平铺分片，以利用并行硬件的并发性、数据的存储局部性等。CUTLASS将通用矩阵乘法GEMM映射到GPU设备上，并使用CUDA并行编程模型中的并行资源，包括Device设备、Kernel核函数、Threadblock线程块、Warp线程束、Thread线程、Instruction指令等多个层级，对矩阵乘法进行并行分片，伪代码如下所示。

```c++
for (int cta_n = 0; cta_n < GemmN; cta_n += CtaTileN)  // for each threadblock_y
for (int cta_m = 0; cta_m < GemmM; cta_m += CtaTileM)  // for each threadblock_x
for (int cta_k = 0; cta_k < GemmK; cta_k += CtaTileK)  // GEMM mainloop, no unrolling; one iteration is one "stage"
    for (int warp_n = 0; warp_n < CtaTileN; warp_n += WarpTileN)  // for each warp_y
    for (int warp_m = 0; warp_m < CtaTileM; warp_m += WarpTileM)  // for each warp_x
    for (int warp_k = 0; warp_k < CtaTileK; warp_k += WarpTileK)  // fully unroll across CtaTileK; one iteration is one "k Group"
        for (int mma_k = 0; mma_k < WarpTileK; mma_k += MmaK)  // outer product loop, fully unroll across WarpTileK
        for (int mma_n = 0; mma_n < WarpTileN; mma_n += MmaN)  // for each mma instruction
        for (int mma_m = 0; mma_m < WarpTileM; mma_m += MmaM)  // for each mma instruction
            mma_instruction(d, a, b, c);  // one single mma instruction by Tensor Core or CUDA Core
```

MMA（Matrix Multiply Accumulate）是指矩阵乘法累加操作，是矩阵乘法的实现代码中的基本操作，因为实现代码必须对K维度进行迭代循环，每次迭代都需要执行矩阵乘法操作与累加操作，也即MMA矩阵乘法累加操作。

CUTLASS对矩阵乘法的划分如下图所示，从左至右，每个层级对应着CUDA编程模型中不同的并行资源。

![](GEMM矩阵乘法.assets/CUTLASS的GEMM示意图.png)

## Tiling and Epilogue

线程块分片（Threadblock Tile），一个线程块负责计算结果矩阵的一部分，会迭代地从全局内存中加载输入矩阵分片到共享内存，并执行矩阵乘法累加操作。在线程块层级，线程块尺寸与矩阵分片策略是算法性能的关键。一个更大的线程块往往持有更大的矩阵分片，意味着更少的全局内存读取，从而能够保证DRAM带宽不是性能瓶颈；然而线程块分片与问题规模并不能总是相匹配。如果M维度或N维度较小，则线程块中的一些线程可能因为已经超出问题边界而在做无效计算。如果M维度和N维度较小而K维度较大，这种朴素的线程块分片模式只会启动很少的工作线程，而每个工作线程又会负责较长的K维度迭代计算负载，这无法充分利用GPU设备的流多处理器。在K维度上进行线程块或线程束的划分，然后对每个线程块计算得到的部分矩阵乘法结果执行求和归约，可以对这种问题规模的计算进行优化。在CUTLASS中，可以使用ThreadblockShape::{kM,kN,kK}指定线程块分片的尺寸，以匹配不同的硬件架构和问题规模。

线程束分片（Warp Tile），一个线程束负责计算线程块分片的一部分，会迭代地从共享内存中加载输入矩阵分片到寄存器，并执行矩阵乘法累加操作。在实现上，线程束的矩阵乘法累加操作，可以通过mma.sync指令或wmma指令由Tensor Core完成计算，或通过线程分片由CUDA Core完成计算。为取得最高性能，对共享内存的访问应该避免bank冲突。为重用数据，应该尽可能划分更大尺寸的线程束分片。

线程分片（Thread Tile），一个线程负责计算线程束分片的一部分，会迭代地获取寄存器中的数据，并执行矩阵乘法累加操作。因为一个线程无法访问其它线程的寄存器，故应该合理安排线程布局，使得一个线程的多条计算指令能够重用寄存器中的数据。即一个线程计算一个二维矩阵分片，从而使线程能够将一组独立的计算指令发射给CUDA Core计算，以执行矩阵乘法累加操作。SGEMM、DGEMM、HGEMM、IGEMM等通过单指令多线程SIMT指令完成计算。

在完成上述划分的矩阵乘法累加操作之后，计算所得的结果矩阵的一部分存在于一个线程的寄存器中，这种划分策略能够取得最高的矩阵乘法计算效率，但在将结果矩阵写回到全局内存中时不能实现高效的合并访存模式。

尾处理分片（Epilogue Tile）操作是一个单独阶段，一个线程负责处理结果矩阵分片的一部分，用以对计算所得的结果矩阵执行后置操作。通常情况下，一个线程节计算所得的结果矩阵分片以特定布局写回到共享内存中，这种结果矩阵分片在共享内存中的布局方式有利于线程以高效合并访存的模式写回结果。同时，一个线程可以对所负责结果矩阵分片的一部分执行其它可选的逐元素操作。CUTLASS定义一些典型的结尾操作，例如线程缩放与收缩等。

## Pipeline

层级划分结构使得每个CUDA线程需要占用大量的寄存器，且每个线程所持有的累加值至少要占用寄存器预算的一半以上；因此GPU设备的占用率较低，线程块、线程束、线程数目通常低于其它任务的工作负载；这会导致GPU难以隐藏内存延迟和切换线程上下文时所带来停顿间隔（stall）。为减轻内存延迟，CUTLASS使用软件流水线，也即使用双缓冲技术，以重叠线程的访存和计算，如下所示。

- 线程块层级，持有两个共享内存空间，一个用于为当前次的矩阵计算提供数据，另一个用于从设备全局内存中加载下一次主循环迭代所需的数据。
- 线程束层级，持有两个存储于寄存器的矩阵片段fragment，一个用于传递给CUDA Core或Tensor Core执行当前次的矩阵计算，另一个用于从共享内存中加载下一次Warp循环迭代所需的数据。

下图展示CUTLASS所使用的GEMM主循环流水线。

<img src="GEMM矩阵乘法.assets/software-pipeline.png" style="zoom:20%;" />

## SplitK and SliceK

矩阵乘法中线程块的划分具有在O(MN)上的并行性，并独立地执行内积计算。当问题规模M,N足够大时，CUTLASS的矩阵乘法kernel能够达到最大理论计算吞吐量；而当问题规模M,N较小时，则启动的线程块数目太少难以充分利用整个GPU设备。

SplitK（reduction across Block）通过将内积计算过程中的归约操作并行化，可以启动更多的线程块并发执行，从而在线程块层级充分利用计算吞吐量。CUTLASS在问题规模的K维度上进行划分，并在每个划分上启动一组线程块执行计算，然后执行并行的归约操作。用户需要管理工作缓冲区以保存中间结果。

划分维度K的GEMM允许指定问题规模以及划分数目，并且允许维度K无法被整除的情况。例如M,N,K=128,128,4096的问题规模和SplitNum=20的划分数目，会产生20个矩阵乘法kernel，前19个计算所划分到的SplitK=4096/20=204，最后一个计算所划分到的SplitK=220，这能完整处理K维度上的计算。然后再在维度K上执行归约操作，以获得最终结果。

因为每个线程块负责blockM,blockN的输出矩阵，那么线程束的划分具有在O(blockM,blockN)上的并行性。更大的线程束分片warpM,warpN允许更好的指令并行和重用，但当问题规模M,N更小时，这会限制每个线程块所持有的线程束数目，从而导致效率降低。

SliceK（reduction across Warp）通过在blockK维度上划分线程束，能够允许一个线程块产生更多线程束并发执行。SliceK策略不仅会将blockM,blockN划分给warpM,warpN，还会将线程块的计算在blockK维度进一步划分给warpK。然后在线程块的所有线程束计算完成后，再在相关的线程束之间执行归约操作。

## Warp Specialization

从Hopper架构开始，CUTLASS 3.0引入线程束专业化的概念，即一个线程块中的线程束被分为两组，分别是生产者线程束与消费者线程束。生产者使用新架构的张量内存加速器（Tensor Memory Accelerator，TMA）将数据从设备全局内存中加载到共享内存缓冲区中，并更新该阶段所关联的栅障以通知相关消费者数据已填充；消费者等待生产者的填充信号，然后启动Tensor Core的MMA操作，然后释放共享内存缓冲区，并使用新引入的Async Pipeline Class类通知生产者共享内存缓冲区已为空，以执行下一组TMA工作负载。
