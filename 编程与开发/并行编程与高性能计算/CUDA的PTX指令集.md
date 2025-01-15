# CUDA指令集

一般情况下，使用高级语言编写的程序，可以通过编译器直接转换成机器指令以在硬件上运行。但NVIDIA历代GPU产品的底层架构不尽相同，运行的机器指令也存在一定的差异，为保证软件层面的兼容性，CUDA定义了一套稳定的指令集架构和编程模型，称为PTX（Parallel Thread Execution），这是一种并行线程执行的底层虚拟机（low-level virtual machine）及其指令集（instruction set architecture）。PTX指令集是一种中间表示，它展示了GPU支持的功能，但是不同架构的GPU对这些功能的实现方式存在差异。这种虚拟让PTX代码与GPU底层硬件架构基本无关，能够跨越多种硬件，从而更具有通用性。

在需要时，PTX代码可以即时编译（Just-In-Time）得到更底层的二进制机器指令代码，这是由编译器根据GPU硬件的实际计算能力与具体架构所生成的二进制cubin目标代码，可以由机器直接执行的二进制目标代码对应的汇编称为SASS（Streaming Assembly）流汇编代码，又称为低级汇编指令（Low-Level Assembly Instruction），是特定于GPU架构的。机器指令SASS相比于PTX指令更接近GPU的底层架构，SASS指令集可以体现出特定架构的GPU对于PTX功能的实现方式。

PTX代码和SASS代码中的标识符都是区分大小写的，但预留关键字并不区分大小写，但通常PTX代码使用小写，SASS代码使用大写。

# PTX基本语法

PTX程序的源文件通常是以.ptx作为后缀的文本文件，其语法具有汇编语言样式的风格。PTX语句可以是指示（directive），也可以是指令（instruction），语句以可选标签（label）标签开头，以分号`;`结尾，以`\n`表示换行。PTX源文件使用C风格的注释，例如`//`和`/**/`注释；使用C风格的以`#`开头的预处理指令，例如#include、#define、#if、#endif等预处理指令；源文件中所有空白字符都是等效的，并且空格将被忽略，除非它在语言中用于分隔标记。

每个PTX文件必须在开头使用.version指示PTX版本，后跟.target指示所假设的目标架构，一个示例代码如下。

```assembly
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

指令语句（Instruction Statement）用于表示某个具体的功能，与机器指令一一对应，所有指令语句的集合即构成一个指令集。指令由指令操作码（instruction opcode）组成，后跟以逗号分隔的操作数（operand）列表，并以分号结尾；SASS的习惯是第一个作为目的操作数，然后是源操作数；操作数可以是寄存器变量、常量表达式、地址表达式、标签（label）名称。指令具有可选的表示预测断言（predication）的哨兵（guard），用于控制指令是否按照条件断言的结果执行；断言位于可选的label之后，指令的opcode之前，通常写作`@p`或者`@!p`，其中p是声明的断言寄存器，叹号`!`表示对断言进行取反。

指示语句（Directive Statement）用于指示一个特定属性或特定行为，对指令起到修饰和限定的作用，指示语句以点号`.`开头，例如.global等。

| .gmal |      |      |      |      |      |
| ----- | ---- | ---- | ---- | ---- | ---- |
|       |      |      |      |      |      |
|       |      |      |      |      |      |
|       |      |      |      |      |      |
|       |      |      |      |      |      |





用户定义的标识符（Identifier）遵循扩展的C++规则，它们以字母开头，后跟零个或多个字母、数字、下划线或美元字符，或者，它们以下划线、美元字符或百分号字符开头，后跟一个或多个字母、数字、下划线或美元字符。

```
followsym:   [a-zA-Z0-9_$]
identifier:  [a-zA-Z]{followsym}* | [_$%]{followsym}+
```

许多高级语言的标识符名称遵循类似的规则，只是不允许使用百分号，而PTX允许百分号作为标识符的第一个字符，这可以用于避免名称冲突，例如，用户定义的变量名称和编译器生成的名称之间的冲突。不过，PTX预定义了一个WARP_SZ常量和少量以百分号开头的特殊寄存器，如下表所示。

|      |      |      |
| ---- | ---- | ---- |
|      |      |      |
|      |      |      |
|      |      |      |

