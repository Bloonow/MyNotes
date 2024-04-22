# Intel和AT&T风格汇编

不同处理器架构有着不同的机器指令，构成不同的指令系统。常见的有Intel处理器的x86/x64指令架构，ARM架构，MIPS架构等。

汇编语言为这些机器指令提供了面向编程人员的助记符。在不同的平台上，汇编语言有着不同的风格，如在Intel的Window/DOS平台上的Intel风格，与Unix/Linux系列平台上的AT&T风格。本文将介绍这两种风格的主要不同之处。

寄存器命名方式不同。在Intel汇编格式中，寄存器名不需要加前缀；而AT&T汇编格式中，寄存器名需要加上`%`作为前缀。

立即数表示方式不同。在Intel汇编格式中，立即数不需要加前缀；而AT&T汇编格式中，立即数需要加上`$`作为前缀。

源操作数和目标操作数的位置正好相反。在Intel汇编格式中，目标操作数在源操作数的左边；而AT&T汇编格式中，目标操作数在源操作数的右边。

操作数的字长表示方式不同。在Intel汇编格式中，通过在操作数前添加`byte ptr`（字节，8bit）、`word ptr`（字，16bit）、`dword ptr`（双字，32bit）前缀来表示操作数字长；而AT&T汇编格式中，通过在指令语句后添加`b`（字节，8bit）、`w`（字，16bit）、`l`（长字，32bit）后缀表示操作数字长。

远程转移指令和远程子程序调用指令不同，对应的返回指令也不同。在Intel汇编格式中，为`jump far`指令、`call far`指令、`ret far`指令；而AT&T汇编格式中，为`ljump`指令、`lcall`指令、`lret`指令。

对于上述不同，下面列举一些例子。

| Intel格式               | AT&T格式                  | 不同之处                 |
| ----------------------- | ------------------------- | ------------------------ |
| push eax                | pushl %eax                | 寄存器命名               |
| push 1                  | pushl \$1                 | 立即数                   |
| add eax, 1              | addl \$1, %eax            | 源操作数与目标操作数位置 |
| mov al, byte ptr val    | movb val, %al             | 操作数字长               |
| jump far section:offset | ljump \$section, \$offset | 远程转移指令             |
| call far section:offset | lcall \$section, \$offset | 远程子程序调用指令       |
| ret far stack_adjust    | lret \$stack_adjust       | 远程返回指令             |

存储器的寻址方式不同。在Intel汇编格式中，寻址方式为`section:[base + index*scale + disp]`；而AT&T汇编格式中，寻址方式为`section:disp(base, index, scale)`，需要注意的是，在这种复合的地址表达式中，一个数字不作为立即数，它只是用来构成表达式，故不用加`$`前缀。

下面是一些内存操作的例子。

| Intel格式                     | AT&T格式                    |
| ----------------------------- | --------------------------- |
| mov eax, [ebp - 4]            | movl -4(%ebp), %eax         |
| mov eax, [eax*4 + array]      | movl array(, %eax, 4), %eax |
| mov cx, [ebx + eax*4 + array] | movw array(%ebx, %eax, 4)   |
| mov fs:[eax], 4               | movb $4, %fs:(%eax)         |

