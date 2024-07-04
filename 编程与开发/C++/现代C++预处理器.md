# C/C++预处理器

预处理器（preprocessor）是一个文本处理器（text processor），在正式开始编译之前，由预处理器接受C/C++源文件文本，并对这些文件执行预操作（preliminary operation）。尽管一般情况下，编译器会首先调用预处理器，但仍可以在不进行编译的情况下单独使用预处理器处理源文件文本。

预处理器可以有条件地编译代码、插入文件、指定编译时错误消息，以及将特定机器规则（machine-specific rule）应用于代码段（section of code）。预处理器不会分析源文件文本，但会将源文本细分为标记（token）以定位（locate）宏调用（macro call）。

预处理器指令（preprocessor directive），例如#define和#ifdef等，通常用于简化源程序在不同的执行环境中的更改和编译。源文件中的预处理指令告知预处理器要采取的操作。例如，预处理器可以替换文本中的标记，将其他文件的内容插入源文件，或通过移除几个部分的文本来取消一部分文件的编译。值得注意的是，识别并执行预处理器指令的行为，发生在宏展开（macro expansion）之前，因此，如果宏展开到类似于预处理器指令的内容，则预处理器无法识别并执行该内容。

预处理器语句（preprocessor statement）使用的字符集（character set）与源文件语句的相同，与执行字符集（execution character set）相同，只不过转义序列（escape sequence）不受支持。

```c++
#preprocessor_directive [arguments]
```

其中，`#`数字符号（number sign）必须是指令行上的第一个非空格字符（nonwhite-space），空格（white-space）可以出现在数字符号和指令第一个字母之间，某些预处理器指还可以令包含参数。预处理器指令的行可以通过在行尾换行标记（end-of-line，如\r\n或\n）之前放置反斜杠`\`继续。

## #if与#endif指令

一些条件编译指令可以控制源文件的某部分是否参与编译，如果#if表达式后有一个非零值，则在翻译单元中保留紧跟#if指令的源文本。

```c++
#if (constant_expression)
	// something
#elif (constant_expression)
	// something
#else
	// something
#endif
```

其中，constant_expression为常量表达式，表达式必须具有整型，并且只能包含整数常量、字符常量和defined运算符，不能使用sizeof或类型转换type_cast运算符。该表达式不得查询环境，并且必须不受目标机器具体实现详情的影响。

预处理器运算符`defined`及`!defined`可用于#if和#elif的常量表达式中，如以下语法所示。

```c++
#if defined(identifier)
	// something
#endif
```

```c++
#if !defined(identifier)
	// something
#endif
```

当前如果定义了identifier，则常量表达式defined(identifier)被视为true（非零），否则常量表达式defined(identifier)被视为false（零），定义为空文本的标识符也被视为已定义。这种用法等价于使用#ifdef和#ifndef预处理器指令。

```c++
#ifdef identifier
	// something
#endif
```

```c++
#ifndef identifier
	// something
#endif
```

## #line指令

预处理器指令`#line`指示预处理器将编译器的行号和文件名的报告值（reported value）设置为给定行号和文件名。

```c++
#line digit_number "/path/to/filename"
```

```c++
int main() {
    std::cout << __FILE__ << ", " << __LINE__ << std::endl;  // D:\CodeC++\Learn\main.cpp, 6
    #line 1024 "C:/MyCode/test.cpp"
    std::cout << __FILE__ << ", " << __LINE__ << std::endl;  // C:\MyCode\test.cpp, 1024
    std::cout << __FILE__ << ", " << __LINE__ << std::endl;  // C:\MyCode\test.cpp, 1025
    return 0;
}
```

## #error指令

预处理器指令`#error`在编译时发出用户指定的错误消息，然后终止编译。

```c++
#ifndef MyMacro
#error MyMacro required.
#endif
```

## #include指令

预处理器指令`#include`告知预处理器在指令出现的位置，包含指定文件的内容。

```c++
#include <filename>
#include "filename"
#include "/path/to/filename"
```

若采用\<filename\>形式，则按以下顺序搜索包含文件：(1)编译器选项`-I`指定的包含路径，(2)环境变量INCLUDE指定的包含路径。

若采用"filename"形式，则按以下顺序搜索包含文件：(1)源文件所在的目录，(2)编译器选项`-I`指定的包含路径，(3)环境变量INCLUDE指定的包含路径。注意，若所包含的#include "header1.h"文件中，同样存在#include "header2.h"形式的头文件包含，则在搜索header2.h时，先从header1.h所在的目录开始。

若采用"/path/to/filename"形式，则编译器会遵循该路径进行搜索。

# C/C++宏

预处理器会在预处理阶段行展开宏（expand macro），在源文件文本中，除宏定义的行之外，识别到宏名称时，将被视为对该宏的调用。宏名称会被替换为宏主体，如果宏接受参数，则宏调用中的实际参数将替换掉宏主体内的形式参数。将宏调用替换为已处理的宏主体的过程称为宏调用的展开（expansion of macro call）。

C++提供了一些新功能来取代C预处理器提供的宏定义，这些新功能可增强类型安全性和可预见性，例如使用const常量取代无参数的宏定义，使用inline内联函数取代带参数的宏定义。宏定义无法确保类型安全，而它们都是类型安全的。

## #define指令

预处理器指令`#define`创建一个宏（macro），宏名是一个可接收参数（arguments）的标识符（identifier），宏主体是一个标记字符串（token string）。标记字符串token_string可由关键字（keyword）、常量（constant）、完整语句（complete statement）等一系列标记构成。若token_string标记字符串占用多行，使用`\`反斜杠进行串连。一个或多个空白字符必须将identifier与token_string隔开，此空白不会被视为替换文本的一部分。

```c++
#define identifier[(arguments)] token_string
```

可定义一个带有参数的宏，此形式用括号（parenthese）接收一个可选参数列表，用逗号分隔列表中的多个形式参数，identifier与左括号之间不能有空格。形式参数名称将出现在token_string标记字符串中，以标记实际参数的替换位置，每个参数名称可在token_string中出现多次，并且名称可按任意顺序出现，可使用括号将每个参数括起来，以便在表达式中保持适当的优先级，确保正确解释复杂的实际参数。

在定义宏之后，预处理器使用无参数的标记字符串token_string，或带参数的标记字符串token_string_with_arguments，替换掉（substitute）源文件中每个匹配宏名标识符identifier或identifier(arguments)。若是带参数的形式，宏调用中的参数数目必须与宏定义中的参数数目一致。若实际参数仍是一个宏定义，即存在宏嵌套调用，则先进行内层的实际参数的宏替换与宏展开，再进行外层的形式参数的宏替换与宏展开。

```c++
#define ADD(x,y) ((x)+(y))
#define MUL(x,y) ((x)*(y))

int main() {
    // ((1 + 2) * 3) == 9
    std::cout << MUL(ADD(1, 2), 3) << std::endl;
    return 0;
}
```

一旦定义了宏，就无法在未移除原始定义的情况下将其重定义为不同的值。但是，可以使用完全相同的定义来重定义宏。因此，相同的定义可能会在一个程序中出现多次。预处理器指令`#undef`移除一个标识符（identifier）的当前定义，后续匹配项将被预处理器忽略。指令#undef通常与指令#define成对使用，在源程序中创建一个区域，使其中的标识符具有特殊含义。指令#undef还可与指令#if一起使用来控制源程序的条件编译。此外，还可使用编译器的`-D`选项动态定义宏。

```c++
#define ADD(X,Y) ((X)+(Y))
// something
#undef ADD
```

下述一个例子，使用了do-while(0)将宏包括在其中。

```c++
#ifndef CUDA_CALL
#define CUDA_CALL(call) do {                                                                      \
    const cudaError_t stat = (call);                                                              \
    if (stat != cudaSuccess) {                                                                    \
        fprintf(stderr, "CUDA Error at File '%s', at Line '%d'\n", __FILE__, __LINE__);           \
        fprintf(stderr, "     Error Code: %d, Error Text: %s\n", stat, cudaGetErrorString(stat)); \
        cudaDeviceReset();                                                                        \
    }                                                                                             \
} while(0);
#endif
```

代码中使用了do-while(0)，这是C语言中常用的编程技巧。它创建了一个作用域，允许在其中使用局部变量，而无需考虑变量名重复。在宏定义中使用，可以让宏定义中的多条语句在逻辑上看起来像是一个语句，这样不管在调用代码中怎么使用分号和大括号，该宏总能确保其行为是一致的。

## 字符化运算符`#@`

字符化（charizing）运算符`#@`在token_string标记字符串中与宏的参数一起使用，出现在形式参数前面，使得实际参数用单引号`''`括起来，并在宏展开时被视为一个字符字面量（character literal），而不对实际参数的定义（若存在）进行展开（without expanding the parameter definition）。该运算符只能用于宏的实际参数的名称只有一个字符的情况，多个字符则行为未定义。

```c++
#define MAKECHAR(x) #@x

int main() {
    int S = 15;  // No Effect
    auto ch1 = MAKECHAR(S);   // 'S'
    auto ch2 = MAKECHAR(SS);  // 'SS'
    std::cout << typeid(ch1).name() << ": " << ch1 << std::endl;    // char: S
    std::cout << typeid(ch2).name() << ": " << ch2 << std::endl;    // int: 21331
    std::cout << typeid('SS').name() << ": " << 'SS' << std::endl;  // int: 21331
    return 0;
}
```

需要注意的是，`#@`是专用于MSVC环境的。

## 字符串化运算符`#`

字符串化（stringizing）运算符`#`在token_string标记字符串中与宏的参数一起使用，出现在形式参数前面，使得实际参数用双引号`""`括起来，并在宏展开时被视为一个字符串字面量（string literal），而不对实际参数的定义（若存在）进行展开（without expanding the parameter definition）。

注意，实际参数两侧的一个或多个连续空格会被忽略，实际参数内部的一个或多个连续空格将被减少（reduce）到至多保留一个空格，生成的字符串字面量自动与仅由空格分隔的任何相邻的字符串字面量拼接（concatenate）。此外，对于某些在字符串字面量中使用时需要转义的特殊字符（如引号`"`字符和反斜杠`\`字符），在与字符串化运算符一起使用时，必要的转义反斜杠会自动插入到该字符的前面。

```c++
#define MAKESTRING(x) #x

int main() {
    auto str1 = MAKESTRING(Hello    My Friend );  // "Hello My Friend"
    auto str2 = MAKESTRING("Good in quotes");     // "\"Good in quotes\""
    auto str3 = MAKESTRING(God\tness);            // "God\tness"
    std::cout << str1 << std::endl;  // Hello My Friend
    std::cout << str2 << std::endl;  // "Good in quotes"
    std::cout << str3 << std::endl;  // God     ness
    return 0;
}
```

## 合并运算符`##`

合并（merging or combining）运算符`##`在token_string标记字符串中使用，它将两个单独的标记（separate tokens）合并为一个完整的标记（single token），运算符前后的空白是可选的。程序应该保证合并后的标记存在声明与定义，即生成的标记必须是有效的标记。运算符`##`又称为标记粘贴运算符（token-pasting operator）。

合并运算符在宏体token_string标记字符串，可以与宏的参数一起使用。如果宏定义中的形式参数的前面或后面带有合并运算符`##`，则会立即将形式参数替换为未扩展的实际参数（unexpanded actual arguments），在替换前将不会对实际参数执行宏展开。

之后，将移除标记字符串中出现的每个合并运算符，并将其前后的标记连接在一起，生成的标记必须是有效的标记。如果标记有效，则在它表示宏名称时，扫描其中可能存在的宏替换。

```c++
#define EPS 1e-8
#define STR(x) #x
#define COMBINE(x,y) x##y

int main() {
    int HelloWorld = 14;
    auto a = COMBINE(Hello, World);            // HelloWorld
    auto b = COMBINE(E, PS);                   // EPS
    auto c = COMBINE(STR(Hello), STR(World));  // STR(Hello)STR(World)
    auto d = STR(Hello)STR(World);             // "Hello""World"
    std::cout << a << std::endl;  // 14
    std::cout << b << std::endl;  // 1e-08
    std::cout << c << std::endl;  // HelloWorld
    std::cout << d << std::endl;  // HelloWorld
    return 0;
}
```

## 可变参数的宏

可变参数的宏（variadic macro）指的是其拥有可变数量的参数（variable number of argument）。若要使用可变参数的宏，可在宏定义中使用省略号`...`作为最后一个形式参数，并在宏体token_string标记字符串使用`__VA_ARGS__`表示可变的形式参数。在宏调用中，\_\_VA_ARGS\_\_会替换为所有匹配的实际参数（包括它们之间的逗号）。C/C++标准规定，必须为由省略号指定的可变参数，传递至少一个实际参数。

```c++
#define CALL(func, ...) func(__VA_ARGS__)

int main() {
    CALL(printf, "%s,%s\n", "Hello", "World");  // Hello,World
}
```

## 标准预定义宏

ANSI/ISO C99、C11、C17标准以及ISO C++14、C++17、C++20标准要求预处理器提供一些预定义的宏（predefined macro），除非特殊说明，这些宏的定义范围适用于整个编译单元，部分宏仅针对特定生成环境或编译器选项生效。一些特定的编译环境，例如MSVC或GCC等，通常也会提供特定于编译环境的宏。

此处列举常用的标准预定义宏，若无特殊说明，下面的一些定义为字符串的宏，都是由const char\*类型表示。

`__func__`，函数未限定、未修饰的名称，即源代码中声明的函数名称。

`__STDC__`，当编译单元为标准C程序时，则定义为1；在启用C11或C17编译器选项时，它也定义为1；其他情况下则不定义。

`__TIME__`，当前源文件的编译时间，是"hh:mm:ss"格式的字符串文本。

`__DATE__`，当前源文件的编译日期，是"Mmm dd yyyy"格式的恒定长度的字符串文本。

`__LINE__`，定义为当前源文件中的整数行号，由long类型表示。可使用#line指令来更改\_\_LINE\_\_宏和\_\_FILE\_\_宏的值。

`__FILE__`，当前源文件的名称。

`__cplusplus`，当编译单元为标准C++程序时，定义为整数文本值；其他情况下则不定义。

`__STDCPP_THREADS__`，当且仅当程序可以有多个执行线程并编译为C++时，定义为1；其他情况下则不定义。

`__STDCPP_DEFAULT_NEW_ALIGNMENT__`，当指定C++17标准或更高版本标准时，此宏会扩展为size_t字面量，表示标准operator new运算符在分配内存所采用的对齐值，或手动调用operator new(std::size_t, std::align_val_t)版本指定对齐值。

`__STDC_VERSION__`，当编译单元为标准C程序，且启用C11或C17编译器选项时，存在定义，对于C11扩展到201112L，对于C17扩展到201710L。

`__STDC_HOSTED__`，如果标准C实现是托管实现（hosted implementation）并且支持整个必需的标准库，则定义为1；其他情况下则定义为0。

`__STDC_NO_ATOMICS__`，如果标准C实现不支持可选的标准原子语义（optional standard atomics），则定义为1；在启用C11或C17编译器选项时，也为1。

`__STDC_NO_COMPLEX__`，如果标准C实现不支持可选的标准复数语义（optional standard complex numbers），则定义为1；在启用C11或C17编译器选项时，它也定义为1。

`__STDC_NO_THREADS__`，如果标准C实现不支持可选的标准线程语义（optional standard threads），则定义为1；在启用C11或C17编译器选项时，也为1。

`__STDC_NO_VLA__`，如果标准C实现不支持可选的标准变长数组语义（optional standard variable length array），则定义为1；在启用C11或C17编译器选项时，它也定义为1。

# #pragma制导指令

预处理器指令`#pragma`指示特定机器（machine-specific）或特定操作系统（operating system-specific）的编译器功能。某些#pragma指令提供与编译器选项相同的功能，在源代码中遇到#pragma指令时，将重写编译器选项所指定的行为。编译器发现无法识别的#pragma时，会发出警告，并继续编译。C99标准与C++11标准提供`_Pragma`预处理器运算符，其作用与#pragma指令类似。MSVC提供`__pragma`关键字，作用与\_Pragma类似。

```c++
#pragma token_string
_Pragma(string_literal)
```

token_string是一系列字符，这些字符表示特定的编译器指令和参数（如果有）；string_literal是_Pragma的输入，它是由双引号`""`括起来的字符串字面量，在展开时会自动删除两侧的双引号，若字面量中出现特殊字符，需要使用反斜杠`\`进行转义。

```c++
#define MY_ASSERT(bool_expression) do { \
    _Pragma("warning(suppress: 4127)"); \
    if ((bool_expression) != true) { \
        printf("MY_ASSERT Failed: \"" #bool_expression "\" on %s(%d)", __FILE__, __LINE__); \
        exit(-1); \
    } \
} while (0);

int main() {
    MY_ASSERT(3.0 >= 3.14);  // MY_ASSERT Failed: "3.0 >= 3.14" on D:\CodeC++\Learn\main.cpp(14)
    return 0;
}
```

下面将介绍C++编译器所能够识别的#pragma指令。

## 程序段

目标文件（.obj）中的段（section or segment）指的是一个命名的数据块（named block of data），在程序的进程结构中，它作为一个单元（unit）加载到内存中。在MSVC环境下，可以使用dumpbin.exe查看.obj目标文件的结构；在Unix环境下，可使用objdump工具查看目标文件的结构。

### alloc_text

指令`alloc_text`指定给定函数进行定义和放置的代码段名称（name of the code section），该指令需出现在函数声明与函数定义之间，且应在同一模块中定义。由于函数地址不支持使用\_\_base进行基址寻址，因此指定函数的段位置时需要使用alloc_text指令。注意，alloc_text只适用于以C链接（C linkage）声明的函数，不能处理C++成员函数和重载函数，若要处理C++函数，需要使用extern "C"语法。

```c++
#pragma alloc_text("text_section", func_1 [, func_2, ...])
```

```c++
extern "C" void display();
#pragma alloc_text(".my_func_seg", display)
void display() {
    printf("Hello, World!");
}
```

```
Dump of file main.obj
  Summary
          42 .my_func_seg
```

### code_seg

指令`code_seg`指定目标文件中存储函数的文本段（text section），又称代码段。在程序的进程结构中，代码段是包含可执行代码（executable code）的段。默认情况下，目标文件中可执行代码的代码段名称为.text，使用不带参数的code_seg指令将后续的可执行代码的代码段名称重置为.text。

```c++
#pragma code_seg(["section_name"])
#pragma code_seg({push|pop} [, identifier] [, "section_name"])
```

选项push可将一个用identifier标识的section_name段，置于编译器内部堆栈（internal compiler stack）的栈顶，并将栈顶的section_name段作为当前代码段。

选项pop可将编译器内部堆栈上的记录弹出，直到弹出用identifier标识的section_name段为止，并将弹出后的栈顶的section_name段作为当前代码段，若未发现identifier则不会弹出任何内容。

```c++
void func1() {}  // stored in .text
#pragma code_seg(".my_code_seg1")
void func2() {}  // stored in .my_code_seg1
#pragma code_seg(push, named_bss, ".my_code_seg2")
void func3() {}  // stored in .my_code_seg2
#pragma code_seg(pop, named_bss)
void func4() {}  // stored in .my_code_seg1
```

```
Dump of file main.obj
  Summary
         134 .my_code_seg1
          35 .my_code_seg2
```

注意，code_seg不控制函数模板实例化后生成代码的放置位置（placement），也不控制编译器隐式生成的代码，例如特殊成员函数。若要控制这部分代码，可使用MSVC中的\_\_declspec(code_seg())扩展，它可以控制所有目标代码的放置位置，包括编译器生成的代码。

### bss_seg

指令`bss_seg`指定目标文件中存储未初始化变量（uninitialized variable）的段。在程序的进程结构中，BSS段是包含未初始化数据的段。默认情况下，目标文件中未初始化数据的BSS段名称为.bss，使用不带参数的bss_seg指令将后续的未初始化数据项的BSS段名称重置为.bss。

```c++
#pragma bss_seg(["section_name"])
#pragma bss_seg({push|pop} [, identifier] [, "section_name"])
```

```c++
int var1;  // stored in .bss
#pragma bss_seg(".my_bss_seg1")
int var2;  // stored in .my_bss_seg1
#pragma bss_seg(push, named_seg, ".my_bss_seg2")
int var3;  // stored in .my_bss_seg2
#pragma bss_seg(pop, named_seg)
int var4;  // stored in .my_bss_seg1
```

```
Dump of file main.obj
  Summary
           C .bss
           8 .my_bss_seg1
           4 .my_bss_seg2
```

### data_seg

指令`data_seg`指定目标文件中初始化变量（initialized variable）的段。在程序的进程结构中，数据段是包含初始化数据的段。默认情况下，目标文件中初始化数据的数据段名称为.data，使用不带参数的data_seg指令将后续的初始化数据的数据段名称重置为.data。

```c++
#pragma data_seg(["section_name"])
#pragma data_seg({push|pop} [, identifier] [, "section_name"])
```

```c++
int var1 = 1;  // stored in .data
#pragma data_seg(".my_data_seg1")
int var2 = 2;  // stored in .my_data_seg1
#pragma data_seg(push, named_seg, ".my_data_seg2")
int var3 = 3;  // stored in .my_data_seg2
#pragma data_seg(pop, named_seg)
int var4 = 4;  // stored in .my_data_seg1
```

```
Dump of file main.obj
  Summary
           4 .data
           8 .my_data_seg1
           4 .my_data_seg2
```

### const_seg

指令`const_seg`指定目标文件中常量变量（const variable）的段。在程序的进程结构中，常量段是包含常量数据的段。默认情况下，目标文件中常量数据的常量段名称为.rdata，使用不带参数的const_seg指令将后续的常量数据的常量段名称重置为.rdata。

```c++
#pragma const_seg(["section_name"])
#pragma const_seg({push|pop} [, identifier] [, "section_name"])
```

需要注意的是，源代码必须存在对常量数据的访问，才能使得常量数据存储到目标文件中；某些const变量（如标量）将自动内联到代码流（code stream）中，内联在代码段中的常量不会出现在.rdata中。如果在const_seg中定义需要动态初始化的对象，则会发生未定义行为。

```c++
const int var = 1;  // inlined in .text, not stored in .rdata
const char str1[] = "str1";  // stored in .rdata
#pragma const_seg(".my_const_seg1")
const char str2[] = "str2";  // stored in .my_const_seg1
#pragma const_seg(push, named_seg, ".my_const_seg2")
const char str3[] = "str3";  // stored in .my_const_seg2
#pragma const_seg(pop, named_seg)
const char str4[] = "str4";  // stored in .my_const_seg1

int main() {
    std::cout << str1 << str2 << str3 << str4 << std::endl;
    return 0;
}
```

```
Dump of file main.obj
  Summary
           D .my_const_seg1
           5 .my_const_seg2
           5 .rdata
```

### section

指令`section`用于在目标文件中创建一个段（section or segment）。一旦段被定义，它将对编译的其余部分保持有效，但必须使用诸如MSVC中的\_\_declspec(allocate())分配段空间，否则无法在段中放置任何内容。

```c++
#pragma section("section_name" [, attributes])
```

其中，section_name是一个表示段名称的参数，该名称不得与任何标准段名称发生冲突。

其中，attributes是一个用于指定该段属性的列表，各项之间使用逗号分隔，可使用的属性如下所述。read允许对该段上的数据进行读取、write允许对该段上的数据进行写入，execute允许在该段上执行代码，shared允许该段在所有执行该程序的进程间共享，nopage表示该段不允许分页（pageable），nocache表示该段不允许缓存，discard表示该段是可丢弃的，remove表示该段不会常驻内存（memory-resident）。如果未指定任何属性，则一个段默认具有read和write属性。

```c++
#pragma section(".my_seg", read, write)
__declspec(allocate(".my_seg"))
int myvalue = 0;
```

## 程序行为

### auto_inline

指令`auto_inline`指定之后定义的函数是否考虑自动内联展开（automatic inline expansion）。

```c++
#pragma auto_inline([{on|off}])
```

```c++
#pragma auto_inline(on)
void display() {
    printf("Hello, World!");
}
#pragma auto_inline(off)
```

### function

指令`function`指示编译器生成给定函数的调用，而不是内联它们。

```c++
#pragma function(function1 [, function2, ...])
```

内部函数（intrinsic function）通常会生成为内联代码，而不是函数调用，可以使用function指令显式强制执行函数调用，它在指令出现处生效且持续到文件末尾，或持续到intrinsic指令。

```c++
#pragma function(strlen)

int main() {
    std::cout << strlen("Hello, World!") << std::endl;  // 13
    return 0;
}
```

### intrinsic

指令`intrinsic`指示编译器对给定函数的调用是内部（intrinsic）的。

```c++
#pragma intrinsic(function1 [, function2, ...])
```

该指令告知编译器某个函数的行为是已知的，如果性能更好，编译器不用将函数调用替换为内联指令，当然也可替换为内联指令。它在指令出现处生效且持续到文件末尾，或持续到function指令。使用内部函数的程序会更快，因为没有函数调用开销。但是，由于生成了额外的代码，这些程序可能会比较大。

```c++
#pragma intrinsic(strlen)

int main() {
    std::cout << strlen("Hello, World!") << std::endl;  // 13
    return 0;
}
```

一些常用的库函数具有内部形式，如abs、fabs、memcmp、memcpy、memset、strcat、strcmp、strcpy、strlen等。

### loop

指令`loop`指定循环代码如何自动并行（auto-parallelize），或指定是否进行自动向量化（auto-vectorize）。该指令位于某个循环之前，对其之后的一个循环生效，一个循环可同时使用多个loop指令。

```c++
#pragma loop(hint_parallel(n))
#pragma loop(no_vector)
#pragma loop(ivdep)
```

当指定hint_parallel(n)时，提示编译器此循环由n个线程并行执行，如果n为零，则在运行时使用最大数量的线程。这是对编译器的一个提示，而不是命令，不能保证循环一定并行执行，如果循环有数据依赖或结构问题，则不会并行化。

默认情况下，自动向量化会尝试对所有可能受益的循环进行向量化，使用no_vector指令禁止某个循环采用自动向量化策略。使用ivdep指令提示编译器忽略循环的向量化依赖。

```c++
int main() {
    int *arr = new int[10000];
    for (int i = 0; i < 10000; arr[i++] = i);
    #pragma loop(hint_parallel(4))
    for (int i = 0; i < 10000; arr[i++] *= 2);
    return 0;
}
```

### omp

指令`omp`用于OpenMP并行编程扩展，后跟OpenMP从句，详见并行程序设计导论。

### pack

指令`pack`指定结构体、联合体、类成员的封装/打包对齐方式（packing alignment），按字节（byte）对齐，参数n的有效值从1、2、4、8、16中取值，默认取8为值。

```c++
#pragma pack([n])
#pragma pack(show)
#pragma pack({push|pop} [, identifier] [, n])
```

当使用show指令时，显示封装对齐的当前字节值，该值由警告消息显示。

当使用push指令时，将当前对齐值写入栈顶，并将当前对齐值设置为n，若未指定n，则仅将当前值写入栈顶。

当使用pop指令时，将从编译器内部栈中弹出记录，直到弹出用identifier标识的封装对齐值为止，若未指定n，则弹出记录后的栈顶的值是当前封装对齐值，若指定n则使用n作为当前的封装对齐值。

```c++
#include <stddef.h>

struct S1 { int a; short b; double c; };

#pragma pack(push, my_align, 2)
struct S2 { int a; short b; double c; };
#pragma pack(pop, my_align)

int main() {
    printf("%d %d %d\n", offsetof(S1, a), offsetof(S1, b), offsetof(S1, c));  // 0 4 8
    printf("%d %d %d\n", offsetof(S2, a), offsetof(S2, b), offsetof(S2, c));  // 0 4 6
    return 0;
}
```

## 编译链接行为

### check_stack

指令`check_stack`指定编译器是否进行栈探测（stack probe）。使用不带参数的check_stack指令将重置为默认行为，此时采用编译选项指定的行为。

```c++
#pragma check_stack([{on|off}])
#pragma check_stack{+|-}
```

当指定为on或+表示启用栈探测，指定为off或-表示关闭栈探测。

### deprecated

指令`deprecated`指示函数、类型、其他任何标识符不再受将来版本支持或者不应该再使用。可以修饰一个宏名称，需要将宏名称包含在双引号`""`内，否则宏将展开。

```c++
#pragma deprecated(identifier1 [, identifier2, ...])
```

当编译器遇到由deprecated指定的标识符时，会发出编译器警告或错误。

```c++
#pragma deprecated(foo, MyClass, "ADD")
void foo() {}
class MyClass {};
#define ADD(x,y) ((x)+(y))
```

### detect_mismatch

指令`detect_mismatch`将一条记录放在目标文件中，链接器将检查这些记录中的潜在不匹配项（potential mismatche）。链接项目时，如果项目包含两个名称相同但值不同的对象，则会引发链接器错误，使该指令可防止链接中存在不一致的目标文件。

```c++
#pragma detect_mismatch("name", "value")
```

其中，名称name和值value都是字符串字面量，遵循字符串字面量的转移规则和连接规则，区分大小写，且不能包含逗号、等号、引号，以及null字符。

此示例将创建版本标签相同但版本号不同的两个文件，链接器无法将它们编译成可执行文件。

```c++
/* mian.cpp */
#pragma detect_mismatch("my_lib_versoin", "3.14")
int main(int artc, char *argv[]) {
    foo();
    return 0;
}
```

```c++
/* mydef.cpp */
#pragma detect_mismatch("my_lib_versoin", "3.10")
void foo() {}
```

### include_alias

指令`include_alias`指定用于#include指令中的别名，当在#include指令中找到别名（alias_filename）时，在其原位置替换为实际名称（actual_filename）。该指令允许用具有不同名称或路径的文件替换源文件中所包含的头文件名。要搜索的别名必须完全一致，大小写、拼写和双引号或尖括号的使用必须全部匹配。

```c++
#pragma include_alias(<alias_filename>, <actual_filename>)
#pragma include_alias("alias_filename", "actual_filename")
```

```c++
#pragma include_alias(<myio.h>, <stdio.h>)
#include <myio.h>  // actually that is stdio.h
```

### once

指令`once`用于某个.h头文件的开始位置，指示编译器在编译源代码文件时只包含该头文件一次，可以减少构建次数。这称为多次包含优化（multiple-include optimization），其功能与使用宏定义的包含防范（include guard）语法类似，但once指令不会污染宏的全局命名空间。once指令不是C++标准，但多数常用编译器都支持该语法。

```c++
/* myheader.h */
#pragma once
// something
```

### warning

指令`warning`用于对编译器警告信息的行为进行选择性修改。

```c++
#pragma warning(warning_specifier: warning_number_list)
```

说明符warning_specifier用于指定warning指令的行为，warning_number_list用于指定编号。

当warning_specifier是default时，重置警告行为默认值，也会启用默认情况下处于关闭状态的指定警告。

当warning_specifier是disable时，表示不发出（即忽略）所产生的给定警告。

当warning_specifier是error时，表示将给定警告报告为错误。

当warning_specifier是once时，表示给定警告只报告一次。

当warning_specifier是suppress时，首先将当前警告行为的状态压入栈顶，并禁用下一行的警告行为，然后再从栈顶弹出原先的警告行为。

```c++
#pragma warning(disable: 4507)
#pragma warning(once: 4385)
#pragma warning(error: 164)
```

此外，warning还支持以下语法，其中参数n可选1、2、3、4整数值。

```c++
#pragma warning(push [, n])
#pragma warning(pop)
```

使用warning(push)指令会存储每个警告的当前警告状态。使用warning(push,n)指令会存储每个警告的当前警告状态，并将全局警告级别设置为n。

使用warning(pop)指令会弹出栈上的最后一个警告状态，在push和pop之间对警告状态所做的任何更改都将被撤消。

```c++
#pragma warning(push )
#pragma warning(disable: 4705)
#pragma warning(disable: 4706)
#pragma warning(disable: 4707)
// something
#pragma warning(pop)
```

编写头文件时，在开始处使用push指令，在结束处使用pop指令，可以确保用户所做的警告状态更改不会影响其它文件。

```c++
/* myheader.h */
#pragma warning(push, 3)
// eeclarations and definitions
#pragma warning(pop)
```

## 注入额外信息

### comment

指令`comment`将一条注解记录（comment record）放置于目标文件或可执行文件中。

```c++
#pragma comment(comment_type [, "comment_string"])
```

其中，comment_type是一个预定义的标识符（可选compiler、lib、linker、user），指定注解记录的类型，comment_string是一个字符串字面量，它提供附加信息，其中出现的特殊字符需要使用反斜杠`\`进行转义。

当comment_type是compiler时，该指令将编译器的名称和版本号（name and version number）放置于目标文件中，不支持comment_string参数。

当comment_type是lib时，该指令将库搜索路径（library-search）放置于目标文件中，使用comment_string参数指定链接器进行库搜索的名称（和可能的路径），链接器搜索此库的方式等价于在命令行中指定该库进行搜索的方式，先在当前目录中进行搜索，再在LIB环境变量指定路径下搜索。可以使用多条该指令指定多个库搜索路径，这些库将按照其指定的顺序出现在生成的目标模块中。

```c++
#pragma comment(lib, "msmpi")
```

当comment_type是linker时，该指令将链接器选项（linker option）放置于目标文件中，使用comment_string参数指定链接器选项，它等价于在命令行参数或IDE开发环境中指定。在MSVC环境中，支持如下链接器选项，包括/MERGE、/SECTION、/EXPORT、/DEFAULTLIB、/INCLUDE，在源代码中使用小写说明符。

```c++
#pragma comment(linker, "/include:_my_symbol")
```

当comment_type是user时，该指令将一般的程序注释放置于目标文件中，使用comment_string参数指定注释文本，允许使用字面量宏，编译器与链接器将忽略该条注释。

```c++
#pragma comment(user, "Compiled on " __DATE__ " at " __TIME__)
```

### component

指令`component`控制对源文件中的浏览信息（browse information）或依赖信息（dependency information）的收集行为。

```c++
#pragma component(browser, {on|off} [, references [, name]])
#pragma component(minrebuild, {on|off})
#pragma component(mintypeinfo, {on|off})
```

当指定browser时，指定启用或关闭浏览信息收集。当不指定name而仅使用references时，指定启用或关闭对引用的收集。也可以使用references与name一起指定对name的引用是否出现在浏览信息窗口中，使用此语法可忽略不感兴趣的名称和类型，并减小浏览信息文件的大小。

```c++
#pragma component(browser, on, references, DWORD)
```

为节省磁盘空间，可以在不需要收集依赖关系信息时使用#pragma component(minrebuild,off)指令，例如在不变的头文件中，在未更改类后插入#pragma component(minrebuild,on)以重新启用依赖信息收集。

使用mintypeinfo指令可以减少指定区域的调试信息，此信息的量相当大，会影响.pdb和.obj文件。注意，不能在mintypeinfo区域中调试类和结构体。

### message

指令`message`用于在编译期间将字符串发送到标准输出进行打印，其参数message_string是字符串字面量，遵循字符串的转移规则和连接规则，也可以是扩展到字符串字面量的宏定义。

```c++
#pragma message(message_string)
```

```c++
#pragma message("Compiling " __FILE__)
#pragma message("Last modified on " __TIMESTAMP__)
```