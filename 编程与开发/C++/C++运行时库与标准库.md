# 运行时环境与运行时库

在计算机科学中，运行时环境（runtime environment）或运行时系统（runtime system）是存在于创建程序的计算机（编译程序的机器）以及要运行程序的计算机（执行程序的机器）中的子系统，包含程序创建（编译时）和在目标机器上执行（运行时）所涉及的计算机过程。

大多数编程语言都有某种形式的运行时系统，为程序提供运行环境，该环境可能解决许多问题，包括应用程序内存的管理、程序如何访问变量、在例程之间传递参数的机制、与操作系统的接口等。通常，运行时系统将负责设置和管理堆栈和堆，并可能包括垃圾收集、线程或语言内置的其他动态功能等。并且，编译器根据特定的运行时系统做出假设以生成正确的代码。

举一个极端的例子，物理CPU本身可以看作是特定汇编语言的运行时系统的实现，从这个角度看，执行模型由物理CPU和内存系统实现。类似地，高级语言的运行时系统本身也是使用其他语言实现的。这就创建了一个运行时系统层次结构，其中CPU本身（或者实际上是其在微代码层或以下的逻辑）充当最低级别的运行时系统。

> 在处理器设计中，微代码（microcode）充当中央处理单元CPU硬件与计算机程序员可见的指令集体系结构之间的中间层。它由一组硬件级指令组成，这些指令实现更高级别的机器指令。虽然微代码在现代通用CPU中使用，但它仅作为更快的硬连线控制单元无法管理的场景的后备路径。
>
> 微代码位于特殊的高速存储器中，可将机器指令、状态机数据或其他输入转换为详细的电路级操作序列。它将机器指令与底层电子设备分开，从而可以更灵活地设计和修改指令。此外，它还有助于构建复杂的多步骤指令，同时降低计算机电路的复杂性。编写微代码的行为通常称为微编程，特定处理器实现中的微代码有时称为微程序。

在计算机科学中，运行时库（runtime library）是一组低级的例程（low-level routine），运行时库通常包含用于内存管理或异常处理的内置函数；因此，运行时库始终特定于平台和编译器。

编译器在编译时，将对运行时库例程的调用，插入到可执行二进制文件，来调用运行时环境的某些行为。在程序执行期间（运行时），对运行时库的这些调用的执行会导致可执行二进制文件和运行时环境之间的通信；而运行时环境实现了编程语言的执行模型、内置函数和其他基本行为。运行时库可能实现运行时环境的一部分行为，但如果阅读可用调用的代码，它们通常只是简单的包装信息并将其发送到运行时环境或操作系统的包装器。但有时，术语“运行时库”意味着包括运行时环境本身的代码，即使其中大部分代码无法通过库调用直接访问。

例如，一些只能在运行时执行（或更高效或更准确）的语言功能在运行时环境中实现，并且可以通过运行时库API调用，例如一些逻辑错误、数组边界检查、动态类型检查、异常处理以及可能的调试功能。因此，尽管在开发过程中进行了复杂的编译时检查和测试，但直到在“实时”环境中使用真实数据测试程序时，一些编程错误才会被发现。

# GNU的C运行时库

在Unix/Linux系列的操作系统上，C运行时库由GNU项目实现，名称为GNU C Library，通常称为glibc，是GNU项目对C标准库的实现。它为Linux内核和其他内核的系统调用提供了一个包装器，供应用程序使用。尽管它的名称是glibc，但它现在也直接支持C++以及间接支持其他编程语言。

glibc项目是在20世纪80年代由自由软件基金会（Free Software Foundation，FSF）为GNU操作系统发起的，用于运行许多不同内核和不同硬件架构的系统，它最常见的用途是在x86硬件上使用Linux内核的系统，不过，官方支持的硬件包括ARM、ARC、IA-64、MIPS、PowerPC、RISC-V、x86等。

1992年2月正式发布glibc 1.0版本；1997年1月正式发布glibc 2.0版本；2012年6月正式发布glibc 2.16版本，支持x32 ABI，符合ISO C11标准；2012年12月正式发布glibc 2.17版本，支持64位ARM指令集；2013年8月正式发布glibc 2.18版本，改进对C++ 11的支持；2018年2月正式发布2.27版本，提供RISC-V支持，优化性能；2021年8月正式发布glibc 2.34版本，将libpthread、libdl、libutil、libanl库集成到libc中。详细的发布说明可查阅相关文档。

glibc库为GNU系统以及许多使用Linux作为内核的系统提供核心库，包括ISO C11、POSIX、BSD、特定于操作系统的关键API等。这些API包括open、read、write、malloc、printf、getaddrinfo、dlopen、pthread_create、crypt、login、exit等基本功能。

# Microsoft的库文件

Microsoft Windows操作系统支持一种称为动态链接库（dynamic-link library）的共享库（shared library）形式，这些代码库可供多个进程使用，但只会有一个副本被加载到内存当中。各种现代Windows操作系统都会共同提供一些核心库，大多数Windows应用程序都是在这些库上构建的。例如，HAL.DLL是内核模式的库文件，它不能被任何用户模式的程序使用；NTDLL.DLL仅被某些程序使用，但它是大多数Win32库的依赖项。

Windows硬件抽象层（Hardware Abstraction Layer，HAL）在hal.dll中实现，其中实现了许多函数，这些函数在不同的硬件平台（此处主要指的是芯片组）上以不同的方式实现。操作系统中的其他组件可以在所有平台上以相同的方式调用这些函数，而不管实际的实现方式如何。例如，在配备高级可编程中断控制器（Advanced Programmable Interrupt Controller，APIC）的机器上响应中断与没有配备该控制器的机器上响应中断的方式截然不同，HAL为此提供了一个函数，该函数可处理各种芯片组的所有类型的中断，因此其他组件无需担心这些差异。

> 在计算机系统中，芯片组（chipset）是位于一个或多个集成电路上的一组电子元件，用于管理处理器、内存和外围设备之间的数据流。芯片组通常位于计算机主板上，被设计用于特定系列的微处理器。由于它控制处理器和外部设备之间的通信，因此芯片组在确定系统性能方面起着至关重要的作用。有时，芯片组一词用于描述手机中使用的片上系统（SoC）。

HAL加载到内核地址空间并以内核模式运行，因此应用程序无法直接调用HAL中的例程，并且没有用户模式的API直接对应于HAL例程；实际上，HAL主要为Windows执行程序和内核以及内核模式的设备驱动程序提供服务。驱动程序使用HAL例程是因为不同的平台可能需要这些操作的不同实现；HAL针对每个平台适当地实现这些操作，因此可以在使用相同CPU架构的所有平台上使用相同的驱动程序可执行文件，并且驱动程序源文件可以在所有架构之间移植。虽然大多数硬件的驱动程序包含在其他文件（通常为.sys文件类型）中，但一些核心驱动程序会被编译到hal.dll当中。PCI和PCI-E等总线上的设备的内核模式的设备驱动程序直接调用HAL中的例程来访问设备的I/O端口和寄存器。

从Windows 10 2004版开始，HAL被合并（或静态链接）到ntoskrnl.exe当中，而dll仅作为向后兼容的存根。

NTDLL.DLL实现导出Windows本地接口（Windows Native API）。Native API是操作系统用户模式的组件使用的接口，这些组件必须在没有Win32或其他API子系统支持的情况下运行，大多数此类API都在NTDLL.DLL和ntoskrnl.exe（及其变体）的之上实现，这些库中导出的符号大多数都带有Nt前缀，例如NtDisplayString。Native API还用于实现由KERNEL32.DLL导出的许多内核API或基本API（kernel APIs or base APIs），大多数Windows应用程序不直接调用NTDLL.DLL。

直接链接到NTDLL.DLL库的应用程序被称为使用本地子系统（native subsystem），它们存在的主要原因是，为了执行必须在Win32子系统可用之前执行的，在系统启动序列早期运行的任务。一个明显的重要例子是创建Win32子系统的csrss.exe进程（Client/Server Runtime Subsystem），在csrss.exe进程存在之前，不能创建任何Win32进程，因此创建它的进程（smss.exe，session manager）必须使用本地子系统。

与Win32应用程序不同，本地应用程序在内核运行时代码（ntoskrnl.exe）中实例化，因此它们必须具有不同的入口点NtProcessStartup（而不是Win32应用程序中的(w)(Win)MainCRTStartup入口点），并且本地应用程序通过指向内存结构的指针获取命令行参数，使用Rtl堆API管理自己的内存（Win32的堆API只是对其进行了包装，没有本质区别），并通过调用RtlExitUserProcess（而不是ExitProcess）返回执行。与本地应用程序链接的通用库是nt.lib，它包含本地应用程序的启动代码，类似于C运行时为Win32应用程序提供启动代码的方式。

Win32应用程序接口由诸多库实现，每个库实现一部分子集。

KERNEL32.DLL向应用程序公开了大多数Win32基本API，例如内存管理、输入/输出（I/O）操作、进程和线程创建以及同步功能。

GDI32.DLL导出图形设备接口（Graphics Device Interface，GDI）函数，这些函数执行基本绘图功能，以输出到视频显示器和打印机。应用程序直接调用GDI函数来执行低级绘图（直线、矩形、椭圆）、文本输出、字体管理等功能。

USER32.DLL实现Windows USER组件，该组件创建和操作Windows用户界面的标准元素，例如桌面、窗口和菜单。因此，它使程序能够实现与Windows外观相匹配的图形用户界面（GUI）。程序调用Windows USER中的函数来执行操作，例如创建和管理窗口、接收窗口消息（主要是用户输入，例如鼠标和键盘事件，但也包括来自操作系统的通知）、在窗口中显示文本以及显示消息框。USER32.DLL中的许多函数都调用GDI32.DLL导出的GDI函数来实际渲染用户界面的各种元素，某些类型的程序还会直接调用GDI函数，在先前通过USER32函数创建的窗口中执行较低级别的绘制操作。

COMCTL32.DLL实现了各种标准Windows控件，例如文件打开、保存和另存为对话框、进度条和列表视图。它调用USER32.DLL和GDI32.DLL中的函数来创建和管理这些UI元素的窗口、在其中放置各种图形元素并收集用户输入。

COMDLG32.DLL即通用对话框库，它实现了各种Windows对话框，用于执行Microsoft所认为的通用应用程序任务（common application task）。从Windows Vista发布开始，Microsoft认为此库提供的“打开”和“另存为”对话框已弃用，并由通用项目对话框（common item dialog）API取代。

WS2_32.DLL实现了Windows Sockets API，简称为Winsock API，它提供了TCP/IP网络功能，定义了Windows TCP/IP客户端应用程序（例如FTP客户端或Web浏览器）与底层TCP/IP协议栈之间的标准接口。并提供与其他网络API的不完整的部分兼容性。wsock.dll和wsock32.dll是用于Win3.11和Win95兼容性的旧版本。

ADVAPI32.DLL，即高级Windows 32基础接口（Advanced Windows 32 Base API）的DLL，提供用于操作Windows注册表的安全调用和函数。

NETAPI32.DLL提供查询和管理网络接口的功能。

OLE32.DLL提供了组件对象模型以及对象链接和嵌入。

# Microsoft的C运行时库

MSVCRT.DLL是Microsoft Visual C++（MSVC）编译器4.2至6.0版的C标准库，它为由这些MSVC版本编译的程序提供了大多数标准C库函数，这些函数包括字符串操作、内存分配、C样式的输入/输出调用等。而MSVCP\*.DLL是相应的C++标准库。运行时库的源代码包含在Visual C++中，以供参考和调试。

MSVC 4.0之前的版本以及7.0到12.0的版本使用不同的DLL名称（例如MSVCR20.DLL、MSVCR70.DLL、MSVCR71.DLL、MSVCP110.DLL等），应用程序需要安装适当的版本，微软为此提供了Visual C++可再发行（Visual C++ Redistributable）组件包，尽管Windows通常已安装一个版本。此运行时库由使用Visual C++和一些其他编译器（例如MinGW）编写的程序使用。一些编译器有自己的运行时库。

从Visual Studio 2015提供的MSVC 14.0版开始，大多数C/C++运行时被移到新的名称为UCRTBASE.DLL的实现中，该DLL与C99标准非常接近。从Windows 10开始，通用C运行时（Universal C Run Time，UCRT）成为Windows的组成部分，因此每个非MSVC编译器（例如GCC、Clang、LLVM等）都可以链接到UCRT库。此外，使用UCRTBASE.DLL的C/C++程序需要链接到另一个新的DLL实现，即Visual C++ Runtime库。在14.0版中，这个DLL的名称是VCRUNTIME140.DLL库。这个名称在未来的版本中可能会发生变化，但截至17.0版还没有改变。

还有许多其他运行时库，例如，ATL\*.DLL提供的活动模板库，MFC\*.DLL提供的Microsoft基础类，VCOMP\*.DLL提供的Microsoft OpenMP运行时，VCRUNTIME\*.DLL提供的Microsoft Visual C++运行时，MSVCIRT.DLL提供的Microsoft C++库，以及C#、Visual Basic.NET、C++/CLI和其他.NET语言编写的程序需要.NET Framework框架，它由许多库（例如mscorlib.dll提供的多语言标准通用对象运行时库）和程序集（例如System.Windows.Forms.dll库）构成。

# C标准库

C标准库（C standard library），有时也称为libc，是C编程语言的标准库，由ISO C标准进行描述。C标准库从最初的ANSI C标准开始，并且与C Library POSIX规范同时开发，而后者是C标准库的超集。由于ANSI C被国际标准化组织采用，所以C标准库也称为ISO C标准库。C标准库的应用程序编程接口API在多个头文件中声明，每个头文件包含一个或多个函数声明、数据类型定义和宏，用于支持字符串操作、数学计算、内存管理、输入/输出处理等任务。

经过一段长时间的稳定后，1995年发布的C标准补充规范附录（Normative Addendum 1，NA1）中增加了三个新的头文件；1999年发布的C标准修订版C99中又添加了六个头文件；2011年发布的C11标准中又添加了五个文件。现在总共有29个头文件，分为若干功能主题，如下所示。

|       Header      | From |                         Description                          |                    Topic                        |
| :---------------: | :--: | :----------------------------------------------------------: | ------------------------------------------------------------ |
|  inttypes.h   | C99  | 定义精确宽度（exact-width）的整数类型。 | [Data types](https://en.wikipedia.org/wiki/C_data_types) |
|   stdint.h    | C99  | 定义精确宽度的整数类型。 | Data types |
|   limits.h    |      | 定义整数类型的特定实现属性的宏常量。 | Data types |
|    float.h    |      | 定义指定浮点库的实现特定属性的宏常量。 | Data types |
|   stdbool.h   | C99  | 定义布尔数据类型。 | Data types |
|   stddef.h    |      | 定义了几种有用的类型和宏。 | Data types |
|    ctype.h    |      | 定义一组函数，用于根据字符类型对其进行分类，或在大小写之间进行转换。 | [Character classification](https://en.wikipedia.org/wiki/C_character_classification) |
|   wctype.h    | NA1  | 定义一组函数，用于根据宽字符类型对其进行分类，或在大小写之间进行转换。 | Character classification |
|   string.h    |      | 定义字符串处理函数。 | [Strings](https://en.wikipedia.org/wiki/C_string_handling) |
|    uchar.h    | C11  | 操作Unicode字符的类型和函数。 | Strings |
|    wchar.h    | NA1  | 定义宽字符串处理函数。 | Strings |
|    math.h     |      | 定义常见的数学函数。 | [Mathematics](https://en.wikipedia.org/wiki/C_mathematical_functions) |
|   tgmath.h    | C99  | 定义类型通用（type generic）的数学函数。 | Mathematics |
|    fenv.h     | C99  | 定义一组用于控制浮点环境的函数。 | Mathematics |
|   complex.h   | C99  | 定义一组用于操作复数的函数。 | Mathematics |
|    stdio.h    |      | 定义核心的输入和输出函数。 | [File input/output](https://en.wikipedia.org/wiki/C_file_input/output) |
|    time.h     |      | 定义日期和时间处理函数。 | [Date/time](https://en.wikipedia.org/wiki/C_date_and_time_functions) |
|   locale.h    |      | 定义本地化功能。 | [Localization](https://en.wikipedia.org/wiki/C_localization_functions) |
|   stdlib.h    |      | 定义数值转换函数、伪随机数生成函数、内存分配、过程控制函数。 | [Memory allocation](https://en.wikipedia.org/wiki/C_dynamic_memory_allocation);<br/>[Process control](https://en.wikipedia.org/wiki/C_process_control) |
|  stdalign.h   | C11  | 用于查询和指定对象的对齐方式。 | Memory allocation |
|   threads.h   | C11  | 定义用于管理多个线程、互斥锁和条件变量的函数。 | Threads |
|  stdatomic.h  | C11  | 用于线程间共享数据的原子操作。 | Threads |
|   signal.h    |      | 定义信号处理函数。 | [Signals](https://en.wikipedia.org/wiki/C_signal_handling) |
|   iso646.h    | NA1  | 定义几个宏，用于实现表达几个标准标记（standard token）的替代方法，用于在ISO 646变体字符集中进行编程。 | [Alternative tokens](https://en.wikipedia.org/wiki/C_alternative_tokens) |
|     assert.h      |      | 声明断言宏，用于在调试程序时协助检测逻辑错误和其他类型的错误。 | Miscellaneous headers |
|    errno.h    |      |    用于测试库函数报告的错误代码。    |    Miscellaneous headers    |
|   setjmp.h    |      | 声明用于非本地退出（non-local exit）的setjmp和longjmp宏。 | Miscellaneous headers |
|   stdarg.h    |      | 用于访问传递给函数的可变数量的参数。 | Miscellaneous headers |
| stdnoreturn.h | C11  |            用于指定非返回函数。            |            Miscellaneous headers            |

POSIX标准库指定了许多例程，这些例程超出基本C标准库中的例程。POSIX规范包括用于多线程、网络和正则表达式等用途的头文件，这些通常与C标准库功能一起实现，紧密程度各不相同。POSIX标准为Unix特定功能添加了几个非标准C文件，许多标头已在其他体系结构中使用，例如，fcntl.h头文件和unistd.h头文件等。BSD libc是POSIX标准库的超集，由FreeBSD、NetBSD、OpenBSD和macOS等BSD操作系统附带的C库支持，其中有一些原始标准中未定义的扩展。

类Unix系统通常具有共享库形式的C库，但安装时可能缺少头文件（和编译器工具链），因此可能无法进行C开发。在类Unix系统上，C库被视为操作系统的一部分，除了C标准指定的函数外，它还包括操作系统API的其他函数，例如POSIX标准中指定的函数。C库函数（包括ISO C标准函数）被程序广泛使用，并且被视为不仅是C语言中某些内容的实现，而且事实上也是操作系统接口的一部分。如果C库被删除，类Unix操作系统通常无法运行，对于动态链接而非静态链接的应用程序来说，情况确实如此。此外，内核本身（至少在Linux的情况下）独立于任何库运行。

在Microsoft Windows上，核心系统动态库（DLL）为Microsoft Visual C++ v6.0编译器提供C标准库的实现，每个Microsoft Visual C++编译器较新版本的C标准库以及可再发行软件包由编译器单独提供。用C编写的程序要么与C库静态链接，要么链接到随这些应用程序附带的动态版本的库，而不是依赖于目标系统上存在。编译器的C库中的函数不被视为Microsoft Windows的接口。

目前存在许多C库的主流实现，随各种操作系统和C编译器提供。

GNU C库（glibc），用于GNU Hurd、GNU/kFreeBSD和大多数Linux发行版。而klibc，是标准C库的一个极简子集，主要被用于开发Linux启动过程中，并且是早期用户空间的一部分，即在内核启动期间使用的组件，但不在内核模式下运行。这些组件无法访问普通用户空间程序使用的标准库（通常是glibc）。

Microsoft C运行时库，是Microsoft Visual C++编译器的一部分，该库有两个版本，MSVCRT是可再发行版本，但于C99兼容性较低，另一个新版本UCRT（Universal C Run Time）是Windows 10操作系统的一部分，因此始终存在以进行链接，并且也符合C99标准。

# C++标准库

在C++编程语言中，C++标准库是用核心语言编写的类和函数的集合，也是ISO C++标准本身的一部分。C++程序可以从符合标准的C++标准库实现中调用大量函数，这些函数执行基础服务，并提供常用操作的高效实现。C++标准库的功能在std命名空间内声明。C++标准库提供了几个通用容器、使用和操作这些容器的函数、函数对象、通用字符串和流（包括交互式和文件I/O）、对某些语言功能的支持，以及用于常见任务的函数。

C++标准库以标准模板库（Standard Template Library，STL）引入的约定为基础，并受到泛型编程研究和STL开发人员的影响。尽管C++标准库和STL有许多共同的特征，但它们都不是对方的严格超集。C++标准库的一个值得注意的特性是，它不仅指定了泛型算法的语法和语义，而且对其性能提出了要求。这些性能要求通常对应于一个众所周知的算法，该算法是期望但不要求被使用。在大多数情况下，这需要线性时间O(n)或线性对数时间O(nlongn)。

> 尽管ISO 14882标准中定义的库的正式名称是C++标准库，但诸如Microsoft等对C++标准库的实现通常称为标准模板库（Standard Template Library，STL）。根据历史记录，STL最初是指由Alexander Stepanov编写的标准模板库，它与ISO C运行时库和部分Boost库一起，进行标准化后得到C++标准库。有时，STL是指根据Stepanov的STL改编的C++标准库的容器和算法部分。而在本文档中，标准模板库STL是指整个C++标准库。
>
> Boost库是为C++标准库提供扩展的一些C++程序库的总称，可以与C++标准库完美共同工作，由Boost社区开发和维护。

C++标准库于20世纪90年代作为ISO C++标准化工作的一部分接受了ISO标准化，自2011年起，随着C++标准的每次修订，该库每三年都会进行扩充和更新。

所有C++库实体都在在一个或多个标准标头文件中声明或定义，并且由多个组织提供主流实现。例如，由GNU项目提供的libstdc++库实现，由LLVM项目提供的libc++库实现，由Microsoft提供的MSVC STL库实现，由Nvidia提供的libcudacxx实现等。

| Header                                                       | Topic            |
| ------------------------------------------------------------ | ---------------- |
| any、chrono、concepts、expected、functional、generator、optional、scoped_allocator、tuple、type_traits、utility、variant | General          |
| compare、coroutine、initializer_list、limits、source_location、stdfloat、typeinfo、typeindex、version | Language support |
| array、deque、forward_list、list、vector、map、set、unordered_map、unordered_set、queue、stack、bitset | Container        |
| iterator、algorithm、numeric、ranges、execution              | Iterator Ranger  |
| bit、complex、ratio、random、valarray、numbers               | Math             |
| charconv、string、string_view、format、regex                 | String           |
| filesystem、fstream、iomanip、ios、iosfwd、iostream、istream、ostream、print、sstream、streambuf、strstream、syncstream | IO Stream        |
| allocators、memory、memory_resource、new                     | Memory           |
| codecvt、locale、text_encoding、cvt/wstring、cvt/wbuffer     | Localization     |
| cassert、exception、stdexcept、system_error、contract        | Execption        |
| atomic、thread、mutex、shared_mutex、condition_variable、future、barrier、latch、semaphore、stop_token | Thread           |

其中，一些以c作为前缀的头文件，是C++提供的对相应的C标准标头的包装实现。其中，hash_map，hash_set并不是ISO C++标准的一部分，位于stdext命名空间或\_\_gnu\_cxx命名空间，目前已废弃并由unordered_map，unordered_set替代。

可按照任何顺序包括标准标头，可多次包括一个标准标头，或包括定义了相同宏或相同类型的多个标准标头。声明内不能包括标准标头。

如果需要某种类型的定义，则C++库标头会包含任何其他C++库标头。但是，用户应该始终显式包含编译单元中所需的任何C++库标头，以免弄错其实际依赖项。而C标准标头从不包含其他标准标头。库中的每个函数都在标准标头中声明，与C标准库不同，C++标准库不会提供屏蔽宏。一个屏蔽宏是指与某个声明函数的名称和作用都相同的宏定义，用于屏蔽对某个函数的调用。

除C++库标头中的operator new()和operator delete()以外的所有名称都在std命名空间中定义，或者在std命名空间内的嵌套命名空间中定义。在某些转换环境（包括C++库标头）中，使用using关键字，可以将在std命名空间中声明的外部名称提升到全局命名空间当中；否则，标头不会将任何库名称引入当前命名空间中。C++标准要求C标准标头对命名空间std中的所有外部名称进行声明，然后将它们提升至全局命名空间中。但在某些转换环境中，C标准标头不包含命名空间声明，所有名称都直接在全局命名空间中进行声明。

需要注意的是，对于std::vector容器的erase(iter)操作而言，其底层操作是，删除当前迭代器iter位置的元素，并将之后的元素前移相应的位置，注意使用方式。

# C风格文件读写

程序通过读取和写入文件来与目标环境进行通信，文件可以是，(1)可重复读取和写入的数据集；(2)程序生成的字节流，例如管线；(3)从外围设备接收或发送到外围设备的字节流。最后两项是交互式文件，文件通常是程序与目标环境进行交互的主要手段。操作所有这些类型的文件的方式大致相同，就是通过调用库函数。在C风格编程中，可以使用\<stdio.h\>标准头文件中提供的函数。

概念EOF（End of File）是操作系统中用于指示无法从数据源读取更多数据的情形，‌数据源通常为文件或流。但应该注意，EOF不是文件或流中实际存在的内容，它只是用于表示“文件或流的读取已经达到结尾”这一状态，可使用feof()函数检测；此外，还可以用于表示“文件或流的读取遇到错误”这一状态，可使用ferror()函数检测。在stdio.h标准头文件中，提供EOF的定义，以及相关函数如下所示。

```c++
/* The value returned by fgetc and similar functions to indicate the end of the file. */
#define EOF (-1)

/* Return the EOF indicator for STREAM. */
extern int feof (FILE *__stream);
/* Return the error indicator for STREAM. */
extern int ferror (FILE *__stream);
/* Clear the error and EOF indicators for STREAM. */
extern void clearerr (FILE *__stream);
```

在终端输入中，‌虽然终端输入本身不会结束，‌但在将输入数据区分成多个文件时，‌需要有一种方式来指明输入的结束。‌在UNIX中，‌使用Ctrl+D组合键发送一个传输结束EOF标识，以表示文件结束，在Windows中，‌则使用Ctrl+Z组合键。

对于文件操作而言，必须先打开文件，才能对该文件执行许多操作，打开文件会将其与流对象，并与C标准库中的`FILE`数据结构关联（该数据结构屏蔽了各类文件之间的差异），标准库将维护FILE类型对象中的每个流的状态。下面代码展示GNU对FILE数据结构的实现。

```c++
/* The tag name of this struct is _IO_FILE to preserve historic C++ mangled names for functions taking FILE* arguments.
   That name should not be used in new code. */
struct _IO_FILE {
    int _flags;  /* High-order word is _IO_MAGIC; rest is flags. */

    /* The following pointers correspond to the C++ streambuf protocol. */
    char *_IO_read_ptr;    /* Current read pointer */
    char *_IO_read_end;    /* End of get area. */
    char *_IO_read_base;   /* Start of putback + get area. */
    char *_IO_write_base;  /* Start of put area. */
    char *_IO_write_ptr;   /* Current put pointer. */
    char *_IO_write_end;   /* End of put area. */
    char *_IO_buf_base;    /* Start of reserve area. */
    char *_IO_buf_end;     /* End of reserve area. */

    /* The following fields are used to support backing up and undo. */
    char *_IO_save_base;    /* Pointer to start of non-current get area. */
    char *_IO_backup_base;  /* Pointer to first valid character of backup area */
    char *_IO_save_end;     /* Pointer to end of non-current get area. */

    struct _IO_marker *_markers;
    struct _IO_FILE *_chain;

    int _fileno;
    int _flags2;
    __off_t _old_offset;  /* This used to be _offset but it's too small. */

    /* 1 + column number of pbase(); 0 is unknown. */
    unsigned short _cur_column;
    signed char _vtable_offset;
    char _shortbuf[1];

    _IO_lock_t *_lock;

    __off64_t _offset;
    
    /* Wide character stream stuff.  */
    struct _IO_codecvt *_codecvt;
    struct _IO_wide_data *_wide_data;
    struct _IO_FILE *_freeres_list;
    void *_freeres_buf;
    size_t __pad5;
    int _mode;

    /* Make sure we don't get into trouble again. */
    char _unused2[15 * sizeof(int) - 4 * sizeof(void*) - sizeof(size_t)];
};

/* The opaque type of streams. This is the definition used elsewhere. */
typedef struct _IO_FILE FILE;
```

在程序启动前，目标环境将默认打开三个文件，即标准输入流、标准输出流、标准错误输出流，它们在stdio.h标准库中定义。

```c++
/* Standard streams. */
extern FILE *stdin;   /* Standard input stream. */
extern FILE *stdout;  /* Standard output stream. */
extern FILE *stderr;  /* Standard error output stream. */
```

使用fopen()函数打开文件，返回指向打开文件的指针，若错误则返回NULL空指针；使用fclose()函数关闭文件。

```c++
/* Open a file and create a new stream for it.
   This function is a possible cancellation point and therefore not marked with __THROW. */
extern FILE *fopen(const char *__restrict __filename, const char *__restrict __modes);

/* Close STREAM.
   This function is a possible cancellation point and therefore not marked with __THROW. */
extern int fclose(FILE *__stream);
```

其中，filename参数指定文件系统上的有效路径，可使用斜杠`/`或反斜杠`\`作为路径中的目录分隔符；modes参数指定文件的打开模式，如下表所示。

| modes模式 | 访问类型                                                     |
| --------- | ------------------------------------------------------------ |
| r         | 打开文件，读取；如果文件不存在，则fopen()调用失败            |
| w         | 打开文件，写入；如果文件已存在，则清空原文件的内容           |
| a         | 打开文件，末尾写入（追加）；如果文件不存在，则创建新文件     |
| r+        | 打开文件，支持读取和写入；如果文件不存在，则fopen()调用失败  |
| w+        | 打开文件，支持读取和写入；如果文件已存在，则清空原文件的内容 |
| a+        | 打开文件，支持读取和末尾写入（追加）；如果文件不存在，则创建新文件 |

指定a访问类型、或a+访问类型打开文件时，所有写入操作均将在文件末尾进行，虽然可使用fseek()或rewind()重新定位文件指针，但在执行任何写入操作前，文件指针将始终被移回文件末尾，因此无法覆盖现有数据。

指定r+访问类型、w+访问类型、或a+访问类型时，允许读取和写入。但是，当从读取切换到写入时，写入操作必须以EOF标记结束，否则必须使用fsetpos()函数、fseek()函数、或rewind()函数对文件进行定位；从写入切换到读取时，必须使用fflush()函数刷新缓冲区。

除前述的modes模式，还支持将以下字符追加到modes参数以指定换行符（newline character）的转换模式。文件IO操作将在文本或二进制这两种转换模式之一中进行，具体取决于文件是在哪种模式下打开的。

| modes修饰符 | 访问类型                                                     |
| ----------- | ------------------------------------------------------------ |
| t           | 文本模式（默认）；输入时，CR-LF字符组合被转换为LF字符，Ctrl+Z或Ctrl+D被转换为文件尾EOF字符；输出时，LF字符被转换为CR-LF字符组合 |
| b           | 二进制模式（无转换）；禁止涉及回车CR字符和换行LF字符的转换   |

在文本模式下，Ctrl+Z或Ctrl+D被解释为输入上的EOF字符。在使用a+打开的文件中，fopen()将检查文件末尾的Ctrl+Z或Ctrl+D并删除之，将其删除是因为使用fseek()和ftell()在以Ctrl+Z或Ctrl+D结尾的文件中移动时，可能导致fseek()在文件末尾附近错误运行。

# C/C++常用功能

## main()的参数与程序退出状态

在标准的C和C++程序中，程序的入口函数有两个参数，这些参数是程序开始运行时由命令行传入的。

```c
int main(int argc, char* argv[]);
```

其中，argc是整型变量，表示命令行参数的个数（含第一个参数）；argv是字符指针的数组，每个元素为一个字符指针，指向一个字符串，这些字符串就是命令行中的每一个参数（字符串）。

需要注意的是，argv数组中0索引（即第一个）字符串是开始运行程序时使用的路径名字（绝对路径还是相对路径取决于运行程序时使用的路径）。因此，如果没有其他参数，则argv中就一个程序路径名，且argc值为1。

如下面一个例子。

```shell
demo.exe -n 4
```

则argc的值为3，argv中有3个字符串，分别为"demo.exe"、"-n"、"4"。

通常情况下，程序成功执行完一个操作正常退出的时候会带有值`EXIT_SUCCESS`，它是定义为0的宏；如果程序中存在一种错误情况，当退出程序时，通常会带有状态值`EXIT_FAILURE`，它是定义为-1的宏。如：

```c
exit(EXIT_SUCCESS);		// or
exit(EXIT_FAILURE);
```

## C++输出控制浮点数位数

```c++
#include <iostream>
#include <iomanip>
using namespace std;
int main(int argc, char* argv[]) {
    const double value = 12.3456789;	// 四舍五入
    cout << value << endl;	// 默认6精度，输出12.3457
    cout << setprecision(4) << value << endl;	// 改成4精度，输出12.35
    cout << setprecision(8) << value << endl;	// 改成8精度，输出12.345679
    // 加了fixed意味着固定点显示方式，固此处精度是指小数的位数，输出12.3457
    cout << fixed << setprecision(4) << value << endl;
    cout << value << endl;	// fixed和setprecision作用仍在，仍然输出12.3457
    cout.unsetf(ios::fixed);	// 去掉了fixed，恢复成只有setprecision(4)的精度
    cout << value << endl;	// 输出12.35
    cout.precision(6);
    cout << value << endl;	// 输出12.3457
    return 0;
}
```

## C错误处理

C语言不提供对错误处理的直接支持，但是作为一种系统编程语言，它以返回值的形式允许访问底层数据。在发生错误时，大多数的C或UNIX函数调用返回1或NULL，同时会设置一个错误代码**errno**，该错误代码是全局变量，表示在函数调用期间发生了错误。可以在`<errno.h>`头文件中找到各种各样的错误代码，在程序初始化时，把errno设置为0，表示程序中没有错误。

```c
void perror(char const* errorMessage);		// stdio.h or stdlib.h
```

- 向控制台打印错误信息，格式为`ErrorMessage: errno对应的错误描述`。

```c
char* strerror(int errno);		// string.h
```

- 返回代表errno值的文本表示形式。


## C时间处理time.h

比较有意思的一个点是，不同的操作系统的时间戳使用不同的赛博元年，Unix的时间基准为1970年1月1日12时，Linux的时间基准为1970年1月1日0时，DOS时间基准是1980年1月1日0时，Windos的时间基准是1601年1月1日0时。

`time.h`头文件定义了四个变量类型、两个宏和各种日期和时间的函数，这里仅列出与日期和时间操作有关的内容。

```c
typedef long clock_t;	// clock_t 是一个适合存储处理器时间的类型
#define CLOCKS_PER_SEC	// 该宏表示每1秒的处理器（CPU）时钟个数，通常和clock()函数搭配使用
typedef long time_t;	// time_t 是一个适合存日历时间的类型（时间戳）
// struct tm 是一个用来保存时间和日期的结构
struct tm {
    int tm_sec;		// 秒，范围从 0 到 59
    int tm_min;		// 分，范围从 0 到 59
    int tm_hour;	// 小时，范围从 0 到 23
    int tm_mday;	// 一月中的第几天，范围从 1 到 31
    int tm_mon;		// 月，范围从 0 到 11
    int tm_year;	// 自 1900 年起的年数
    int tm_wday;	// 一周中的第几天，范围从 0 到 6
    int tm_yday;	// 一年中的第几天，范围从 0 到 365
    int tm_isdst;	// 夏令时
};
```

上面是常用的类型和宏，下面列举一些常用的函数。

### 1. 时间戳time_t

```c
clock_t clock();
```

- 返回自程序开始执行起（一般为程序的开头），处理器（CPU）所用的时钟数。将函数的返回值除以CLOCKS_PER_SEC宏，即可得实际经过的秒数。
- 在32位系统中，CLOCKS_PER_SEC等于1000000，该函数大约每72分钟会返回相同的值。

```c
time_t time(time_t* timer);
```

- 返回自纪元Epoch（1970-01-01 00:00:00 UTC）起经过的**秒**数。如果参数timer不为空，则结果也存储在变量timer中，可以传参NULL，只接受返回值。
- UTC是英文CUT（Coordinated Universal Time）和法文TUC缩写不同的妥协，它表示协调世界时，又称为世界统一时间、世界标准时间、国际协调时间。协调世界时是以原子时的秒长为基础。

```c
double difftime(time_t time1, time_t time2);
```

- 返回time1和time2之间相差的秒数，即time1-time2。它们都是自纪元Epoch起经过的秒数。

```c
char* ctime(const time_t* timer);
```

- 返回一个时间戳timer所表示时间的字符串，其格式为`Www Mmm dd hh:mm:ss yyyy`，"以字母表示的星期 以字母表示的月份 第几天 时:分:秒 年"。

### 2. 时间结构体struct tm

```c
struct tm* localtime(const time_t* timer);
```

- 使用timer的值来填充struct tm结构，并用本地时区表示。

```c
time_t mktime(struct tm* timeptr);
```

- 把timeptr所指向的结构转换为一个依据本地时区的time_t值，如果错误则返回-1。

```c
struct tm* gmtime(const time_t* timer);
```

- 使用timer的值来填充struct tm结构，并用协调世界时（UTC）来表示，也称为格林尼治标准时间（GMT）。

```c
time_t mktime(struct tm* timeptr);
```

- 把timeptr所指向的结构转换为一个依据本地时区的time_t值。

```c
size_t strftime(char* str, size_t maxsize, const char* format, const struct tm* timeptr);
```

- 根据format中定义的格式化规则，格式化结构timeptr表示的时间，并把它存储在str中。
- str，这是指向目标数组的指针，用来复制产生的C字符串。
- maxsize，这是被复制到str的最大字符数。
- format，C字符串，包含了普通字符和特殊格式说明符的任何组合，格式说明符由函数替换为表示tm中所指定时间的相对应值。格式说明符如下。

| 说明符 |                          替换为                           |           实例           |
| :----: | :-------------------------------------------------------: | :----------------------: |
|   %a   |                     缩写的星期几名称                      |           Sun            |
|   %A   |                     完整的星期几名称                      |          Sunday          |
|   %b   |                      缩写的月份名称                       |           Mar            |
|   %B   |                      完整的月份名称                       |          March           |
|   %c   |                     日期和时间表示法                      | Sun Aug 19 02:56:02 2012 |
|   %d   |                  一月中的第几天（01-31）                  |            19            |
|   %H   |                 24小时格式的小时（00-23）                 |            14            |
|   %I   |                 12小时格式的小时（01-12）                 |            05            |
|   %j   |                 一年中的第几天（001-366）                 |           231            |
|   %m   |                十进制数表示的月份（01-12）                |            08            |
|   %M   |                        分（00-59）                        |            55            |
|   %p   |                        AM或PM名称                         |            PM            |
|   %S   |                        秒（00-61）                        |            02            |
|   %U   | 一年中的第几周，以第一个星期日作为第一周的第一天（00-53） |            33            |
|   %w   |        十进制数表示的星期几，星期日表示为0（0-6）         |            4             |
|   %W   | 一年中的第几周，以第一个星期一作为第一周的第一天（00-53） |            34            |
|   %x   |                        日期表示法                         |         08/19/12         |
|   %X   |                        时间表示法                         |         02:50:06         |
|   %y   |                年份，最后两个数字（00-99）                |            01            |
|   %Y   |                           年份                            |           2012           |
|   %Z   |                     时区的名称或缩写                      |           CDT            |
|   %%   |                         一个%符号                         |            %             |

```c
// 一个自定义格式化日期时间的例子
strftime(buffer, 80, "%Y-%m-%d %H:%M:%S", timeptr);
printf("%s", buffer);	// 输出如：2020-06-30 16:54:03
```

## 系统区域设置

在读取中文文件时，经常会出现乱码情况，实际上任何语言的文件都有可能乱码，这是因为编码不同造成的（程序的编码，文件的编码，操作系统的本地编码等等）。在读写文件时将编码设为一致或可以正确转换，就能够解决乱码问题。

### C区域设置

当C语言程序初始化（刚进入到main函数）时，locale被初始化为默认的C locale，其采用的字符编码是所有本地ANSI字符集编码的公共部分，是用来书写C语言源程序的最小字符集。

```c
char* setlocale(int category, const char* locale);		// #include <locale.h>
```

- category，参数指定类型范围，即区域设置影响的范围。预定义的有但不止：`LC_ALL`、`LC_COLATE`、`CTYPE`、`LC_MONETRAY`、`LC_NUMERIC`、`LC_TIME`、`LC_MESSAGES`，分别表示影响：所有、字符比较、字符分类转换处理、货币信息、数字格式、日期时间格式、POSIX规范要求的错误信息。
- locale，参数指定用来设置的区域名称，对于不同的平台和不同的编译器，区域设置的名称可能会不同，C语言标准没有干预太多，只是规定必须最少支持"C"、""、NULL三个。
- `"C"`，默认的区域设置，它是一种非常中立的区域设置，不偏向任何一个地区，把会尽量少的包含地域信息，这些信息只是让C语言程序能够正常运行。
- `""`，使用当前操作系统本地默认的区域设置。通常使用`setlocale(LC_ALL, "")`初始化，中文环境的查找系统可以使用`"chs"`字符串，效果是一样的。
- `NULL`，不指定任何区域设置，不会更改，返回当前locale名字符串，可用来获取程序当前的locale设置。
- 再次注意，locale跟不同的操作系统、编译器等因素都有关，并不一定通用。大体上一般格式如`language_area.codeset`，codeset一般可省略。常见的有：`en_US`、`en_GB`、`zh_CN`、`zh_CN.GBK`、`zh_TW`、`ja_JP.jis`、`POSIX`、`.936`（中文）、`.UTF-8`（用于指定UTF-8编码）。
- 返回值char*类型，如果执行成功，返回一个字符串，它是当前区域设置的名称；失败（例如指定的locale）不存在）则返回NULL空值。

### C++区域设置

在C++程序中，locale被泛化，设计得更有弹性，功能也更强大，它是作为一个类库出现的，头文件是`<locale>`。

```c++
class locale {
public:
    explicit locale(const char* locale, int category = all);
}
```

- 其中一个显示构造函数，参数locale、category和C语言中的分析类似。除此构造含外，类locale还有一系列其他复制构造、赋值构造函数等。
- 其中的`all`是一个`locale::category`常量，此外还有和C的可以对应：`collate`、`ctype`、`messages`、`monetary`、`numeric`、`time`、`none`（所有category的空集）。它们被作为掩码类型，可以使用`|`组合，如monetary | time。
- locale类还有一系列其他成员函数，提供更多的功能。如`name()`返回当前名字。
- locale类其实引用着一个facet对象，实际的影响是由facet实现的，此处不深究。

```c++
locale locale::global(const locale& loc);		// 类 locale 静态方法
```

- 用来设置全局locale，返回以前的全局locale。相当于C程序的setlocale()函数。
- 使用`std::locale::global(locale(""))`，可以初始化程序的全局locale为操作系统默认的本地locale。
- 对于标准IO流对象，可以使用`imbue`函数为其指定locale，如：`cin.imbue(locle(""))`。

使用C++流对象读取中文文档时，使用流操作符`>>`读取到的是空白，可以使用`getline(in, str)`。

