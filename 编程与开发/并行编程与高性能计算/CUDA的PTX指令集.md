# 一、PTX基本语法

一般情况下，使用高级语言编写的程序，可以通过编译器直接转换成机器指令以在硬件上运行。但NVIDIA历代GPU产品的底层架构不尽相同，运行的机器指令也存在一定的差异，为保证软件层面的兼容性，CUDA定义了一套稳定的指令集架构和编程模型，称为PTX（Parallel Thread Execution），这是一种并行线程执行的底层虚拟机（low-level virtual machine）及其指令集（instruction set architecture）。PTX指令集是一种中间表示，它展示了GPU支持的功能，但是不同架构的GPU对这些功能的实现方式存在差异。这种虚拟让PTX代码与GPU底层硬件架构基本无关，能够跨越多种硬件，从而更具有通用性。

在需要时，PTX代码可以即时编译（Just-In-Time）得到更底层的二进制机器指令代码，这是由编译器根据GPU硬件的实际计算能力与具体架构所生成的二进制cubin目标代码，可以由机器直接执行的二进制目标代码对应的汇编称为SASS（Streaming Assembly）流汇编代码，又称为低级汇编指令（Low-Level Assembly Instruction），是特定于GPU架构的。机器指令SASS相比于PTX指令更接近GPU的底层架构，SASS指令集可以体现出特定架构的GPU对于PTX功能的实现方式。

PTX代码和SASS代码中的标识符都是区分大小写的，但预留关键字并不区分大小写，但通常PTX代码使用小写，SASS代码使用大写。

PTX程序的源文件通常是以.ptx作为后缀的文本文件，其语法具有汇编语言样式的风格。PTX语句可以是指令（instruction），也可以是指示（directive），语句以可选标签（label）标签开头，以分号`;`结尾，以`\n`表示换行。PTX源文件使用C风格的注释，例如`//`和`/**/`注释；使用C风格的以`#`开头的预处理指令，例如#include、#define、#if、#endif等预处理指令；源文件中所有空白字符都是等效的，并且空格将被忽略，除非它在语言中用于分隔标记。

每个PTX文件必须在开头使用.version指示PTX版本，后跟.target指示所假设的目标架构，一个示例代码如下。

```
.version 8.5
.target sm_70, sm_75, sm_86, sm_89

.reg.b32 r1, r2;
.global.f32 array[N];

start:
mov.b32 r1, %tid.x;           // move threadIdx.x to r1
shl.b32 r1, r1, 2;            // shift left tidx by 2 bits
ld.global.b32 r2, array[r1];  // thread[tidx] gets array[tidx << 2]
add.f32 r2, r2, 0.5;          // add 0.5
```

指令语句（Instruction Statement）用于表示某个具体的功能，与机器指令一一对应，所有指令语句的集合即构成一个指令集。指令由指令操作码（instruction opcode）组成，后跟以逗号分隔的操作数（operand）列表，并以分号结尾；SASS的习惯是第一个作为目的操作数，然后是源操作数；操作数可以是寄存器变量、常量表达式、地址表达式、标签（label）名称。指令具有可选的表示预测谓词（predication）的哨兵（guard），用于控制指令是否按照条件谓词的结果执行；谓词位于可选的label之后，指令的opcode之前，通常写作`@p`或者`@!p`，其中p是声明的谓词寄存器，叹号`!`表示对谓词进行取反。

指示语句（Directive Statement）用于指示一个特定属性或特定行为，对指令起到修饰和限定的作用，指示语句以点号`.`开头，如下所示。

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

许多高级语言的标识符名称遵循类似的规则，只是不允许使用百分号，而PTX允许百分号作为标识符的第一个字符，这可以用于避免名称冲突，例如，用户定义的变量名称和编译器生成的名称之间的冲突。不过，PTX预定义了一个WARP_SZ常量和少量以百分号开头的特殊寄存器，如下所示。

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

# 三、PTX指令系统

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

有符号整数具有6个比较运算符，分别是：eq、ne、lt、le、gt、ge。无符号整数具有6个比较运算符，分别是：eq、ne、lo（lower）、ls（lower-or-same）、hi（higher）、hs（higher-or-same）。位数据具有2个比较运算符，分别是：eq、ne。浮点数具有6个比较运算符，分别是：eq、ne、lt、le、gt、ge，但如果任一操作数为NaN，则比较结果为False；此外浮点数还具有6个比较运算符，分别是：equ、neu、ltu、leu、gtu、geu，如果两个操作数都是数值，则正常进行比较，若任一操作数为NaN，则比较结果为True。为测试NaN，提供两个运算符num和nan，其中，num当前仅当两个操作数都是数值时才返回True，nan当任一操作数为NaN时就返回True。

谓词可以使用and、or、xor、not、mov指令操作。谓词和整数值之间没有直接转换，也没有直接的方法来加载或存储谓词寄存器值。但是，setp可用于从整数生成谓词，而基于谓词的selp指令可用于根据谓词的值生成整数值。

# 二、状态空间和数据类型

不同架构的GPU中可用的硬件资源数目不同，但资源类型在各个架构上是通用的，这些资源在PTX中通过状态空间和数据类型进行抽象化，以一种概念表示。

## 状态空间

状态空间（State Space）指的是具有某种特征的存储区域，所有变量都驻留在某个状态空间中。状态空间的特征包括容量大小、可寻址性、访问速度、访问权限、线程之间的共享级别。在PTX中定义的状态空间如下所示。

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

寄存器.reg是快速存储空间，且数量是有限的，当超过限制时，寄存器变量将溢出到内存中，从而导致性能下降。寄存器可以是类型化的（有符号整数、无符号整数、浮点数、谓词），也可以是无类型的。寄存器变量的大小具有限制，谓词占用1位，标量可以占用8位、16位、32位、64位、128位，向量可以占用16位、32位、64位、128位。其中，8位寄存器常用于ld、st和cvt指令，或作为向量元组的元素。在加载或存储多字长（multi-word）数据时，寄存器可能需要边界对齐。

特殊寄存器.sreg是预定义的、特定于平台的专用寄存器空间，例如网格、簇组、线程块、线程等专用标识寄存器，时钟计数、性能监控寄存器等。寄存器空间与其它状态空间的不同之处在于，寄存器是不可寻址的，即不可以引用寄存器的地址。

常量.const是由主机初始化的只读的存储空间，大小限制为64KB，用于保存固定内存大小的常量数据，可通过ld.const指令访问。固定内存大小的常量变量可以指定初始值，若不指定则默认初始化为零。此外，还有一个640KB的常量内存空间，组织为10个独立的64KB区域，驱动程序可以在这些区域中分配和初始化常量缓冲区，并将指向缓冲区的指针作为内核函数的参数传递。由于这10个区域不是连续的，因此驱动程序必须确保缓冲区不会越界。

全局.global存储空间是上下文中所有线程都可以访问的内存，不同网格、簇组、线程块之中的线程可以使用全局内存进行通信，可通过ld.global指令、st.global指令和atom.global指令访问全局内存变量。全局内存中的变量可以指定初始值，若不指定则默认初始化为零。

局部.local存储空间是每个线程的私有内存，用于保留自己的数据，它通常是带缓存的标准内存。局部私有内存的大小是有限的，因为它必须基于每个线程进行分配。可通过ld.local指令和st.local指令访问局部内存变量。

参数.param存储空间用于：(1)将输入参数从主机传递给内核函数；(2)声明设备函数的输入参数和返回值参数；(3)声明局部范围的字节数组变量，通常用于按值类型（而非引用）将大型结构体参数传递给函数。需注意，内核函数的参数是在一个网格之内共享的，设备函数的参数是一个线程私有的。对于指令而言，为区分不同类型的参数，使用xxx.param::entry形式访问内核函数参数，使用xxx.param::func形式访问设备函数参数，若省略::entry或::func后缀，则会根据指令推断。

> 参数空间的位置是特定于实现的。例如，某些实现中，内核函数参数是驻留在全局内存中的，在这种情况下，参数空间和全局内存空间不提供访问保护；尽管内核参数空间的确切位置是特定于实现的，但内核参数空间窗口（kernel parameter space window）是始终包含在全局空间窗口中的。同样地，函数参数会根据ABI的函数调用约定将所传递的参数映射到寄存器或堆栈位置。因此，PTX代码不应该对.param空间变量的相对位置或顺序做出任何假设。

每个内核函数定义都包含一个可选的参数列表，这些参数是在.param存储空间中声明的可寻址的只读变量。可通过mov指令将内核参数的地址移动到寄存器中，生成的地址属于.param状态空间，可使用ld.param{::entry}指令访问这些参数变量。一个示例如下所示。
