# C++标准应用

术语pimpl是一种新式C++技术，用于隐藏实现、分离接口、最小化耦合、最小化编译依赖、提高可移植性，术语pimpl是短语pointer to implementation的缩写，意为指向实现的指针。这个概念在其它技术描述中，可能存在其它名称，例如Cheshire Cat或Compiler Firewall等术语。

在接口头文件中，可以定义API如下。

```c++
// my_calss.h
#include <unique.h>

class my_class {
    //  all public and protected stuff goes here
private:
    // opaque type here
    class impl;
    std::unique_ptr<impl> pimpl;
};
```

在实现文件中，定义具体实现细节如下，并将实现编译为动态库或共享库提供。

```c++
// my_class.cpp
#include "my_class.h"

// defined privately here
class my_class::impl {
    // all private data and functions, and
    // can now change without recompiling callers
};

// constructor of my_class
my_class::my_class(): pimpl(new impl) {
    // set impl values
}
```

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

# C++库的应用

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

## C风格可变参数列表

C风格可变函数列表需要`<stdarg.h>`头文件（C++是`<cstdarg>`头文件）中的宏。

所有具有变长参数列表的函数都至少应该有一个已命名参数，可变参数使用`...`占位符。在函数中先使用`va_list 列表名;`声明一个列表，再初始化声明列表的开始位置`va_start(列表名, x);`，其中x是可变列表前一个已命名参数，最后必须用`va_end(列表名);`来释放内存，以确定函数结束后，推栈处于稳定状态。如果要访问实际参数，可使用`va_arg(列表名, 实参解释类型)`，通常用for循环遍历。

但应注意，变长参数列表无结束标志，因而要显式规定，例如可以让第一个参数计算参数的数目，或当参数是一组指针时，可以要求最后一个指针是NULL等。注：不推荐使用C风格的变长参数列表，因为其十分不安全：不知道参数的数目，不知道参数的类型。

一个例子如下：

```c++
#include <stdarg.h>
void foo(int argc, ...) {
	va_list lis;
	va_start(lis, argc);
	for (int i = 0; i < argc; ++i) {
		cout << va_arg(lis, int) << " ";
	}
	va_end(lis);
}

int main(int argc, char* argv[]) {
	foo(5, 1, 2, 3, 4, 5);	// 输出：1 2 3 4 5
	return 0;
}
```

## C时间处理time.h

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
|   %H   |                24 小时格式的小时（00-23）                 |            14            |
|   %I   |                12 小时格式的小时（01-12）                 |            05            |
|   %j   |                 一年中的第几天（001-366）                 |           231            |
|   %m   |                十进制数表示的月份（01-12）                |            08            |
|   %M   |                        分（00-59）                        |            55            |
|   %p   |                       AM 或 PM 名称                       |            PM            |
|   %S   |                        秒（00-61）                        |            02            |
|   %U   | 一年中的第几周，以第一个星期日作为第一周的第一天（00-53） |            33            |
|   %w   |        十进制数表示的星期几，星期日表示为 0（0-6）        |            4             |
|   %W   | 一年中的第几周，以第一个星期一作为第一周的第一天（00-53） |            34            |
|   %x   |                        日期表示法                         |         08/19/12         |
|   %X   |                        时间表示法                         |         02:50:06         |
|   %y   |                年份，最后两个数字（00-99）                |            01            |
|   %Y   |                           年份                            |           2012           |
|   %Z   |                     时区的名称或缩写                      |           CDT            |
|   %%   |                        一个 % 符号                        |            %             |

```c
// 一个自定义格式化日期时间的例子
strftime(buffer, 80, "%Y-%m-%d %H:%M:%S", timeptr);
printf("%s", buffer);	// 输出如：2020-06-30 16:54:03
```

## 系统区域设置

在读取中文文件时，经常会出现乱码情况，实际上任何语言的文件都有可能乱码，这是因为编码不同造成的（程序的编码，文件的编码，操作系统的本地编码等等）。在读写文件时将编码设为一致或可以正确转换，就能够解决乱码问题。

### C区域设置

当C语言程序初始化（刚进入到main函数）时，locale 被初始化为默认的 C locale，其采用的字符编码是所有本地ANSI字符集编码的公共部分，是用来书写C语言源程序的最小字符集。

```c
char* setlocale(int category, const char* locale);		// #include <locale.h>
```

- category，参数指定类型范围，即区域设置影响的范围。预定义的有但不止：`LC_ALL`、`LC_COLATE`、`CTYPE`、`LC_MONETRAY`、`LC_NUMERIC`、`LC_TIME`、`LC_MESSAGES`，分别表示影响：所有、字符比较、字符分类转换处理、货币信息、数字格式、日期时间格式、POSIX规范要求的错误信息。
- locale，参数指定用来设置的区域名称，对于不同的平台和不同的编译器，区域设置的名称可能会不同，C语言标准没有干预太多，只是规定必须最少支持 "C"、""、NULL 三个。
- `"C"`，默认的区域设置，它是一种非常中立的区域设置，不偏向任何一个地区，把会尽量少的包含地域信息，这些信息只是让C语言程序能够正常运行。
- `""`，使用当前操作系统本地默认的区域设置。通常使用`setlocale(LC_ALL, "")`初始化，中文环境的查找系统可以使用`"chs"`字符串，效果是一样的。
- `NULL`，不指定任何区域设置，不会更改，返回当前locale名字符串，可用来获取程序当前的locale设置。
- 再次注意，locale跟不同的操作系统、编译器等因素都有关，并不一定通用。大体上一般格式如`language_area.codeset`，codeset一般可省略。常见的有：`en_US`、`en_GB`、`zh_CN`、`zh_CN.GBK`、`zh_TW`、`ja_JP.jis`、`POSIX`、`.936`（中文）、`.UTF-8`（用于指定UTF-8编码）。
- 返回值 char* 类型，如果执行成功，返回一个字符串，它是当前区域设置的名称；失败（例如指定的locale）不存在）则返回NULL空值。

### C++区域设置

在C++程序中，locale被泛化，设计得更有弹性，功能也更强大，它是作为一个类库出现的，头文件是`<locale>`。

```c++
class locale {
public:
    explicit locale(const char* locale, int category = all);
}
```

- 其中一个显示构造函数，参数locale、category和C语言中的分析类似。除此构造含外，类 locale 还有一系列其他复制构造、赋值构造函数等。
- 其中的`all`是一个`locale::category`常量，此外还有和C的可以对应：`collate`、`ctype`、`messages`、`monetary`、`numeric`、`time`、`none`（所有category的空集）。它们被作为掩码类型，可以使用`|`组合，如monetary | time。
- locale 类还有一系列其他成员函数，提供更多的功能。如`name()`返回当前名字。
- locale 类其实引用着一个facet对象，实际的影响是由facet实现的，此处不深究。

```c++
locale locale::global(const locale& loc);		// 类 locale 静态方法
```

- 用来设置全局 locale，返回以前的全局 locale。相当于C程序的setlocale()函数。
- 使用`std::locale::global(locale(""))`，可以初始化程序的全局 locale 为操作系统默认的本地 locale。
- 对于标准IO流对象，可以使用`imbue`函数为其指定 locale，如：`cin.imbue(locle(""))`。

使用C++流对象读取中文文档时，使用流操作符`>>`读取到的是空白，可以使用`getline(in, str)`。

# C++标准库头文件

C++程序可以从符合标准的C++标准库实现中调用大量函数，这些函数执行基础服务（例如输入和输出），并提供常用操作的高效实现。在Windows平台上，这些基础实现由libucrt.lib、ucrt.lib等各种静态库和动态库提供。

> 尽管ISO 14882标准中定义的库的正式名称是C++标准库，但诸如Microsoft等对C++标准库的实现通常称为标准模板库（Standard Template Library，STL）。根据历史记录，STL最初是指由Alexander Stepanov编写的标准模板库，它与ISO C运行时库和部分Boost库一起，进行标准化后得到C++标准库。有时，STL是指根据Stepanov的STL改编的C++标准库的容器和算法部分。而在本文档中，标准模板库STL是指整个C++标准库。
>
> Boost库是为C++标准库提供扩展的一些C++程序库的总称，可以与C++标准库完美共同工作，由Boost社区开发和维护。

所有C++库实体都在在一个或多个标准标头文件中声明或定义。此处列举MSVC编译器对C++标准库的实现与支持情况，一些头文件在特定C++版本才被引入，而一些头文件则在特定C++版本弃用删除，使用时请注意。

数学和数字，\<bit\>，\<cfenv\>，\<cmath\>，\<complex\>，\<cstdlib\>，\<limits\>，\<numeric\>，\<random\>，\<ratio\>，\<valarray\>；字符串和字符数据，\<charconv\>，\<cctype\>，\<cstdlib\>，\<cstring\>，\<cuchar\>，\<cwchar\>，\<cwctype\>，\<regex\>，\<string\>，\<string_view\>；正则表达式，\<regex\>；

语言支持，\<cfloat\>，\<climits\>，\<codecvt\>，\<compare\>，\<contract\>，\<coroutine\>，\<csetjmp\>，\<csignal\>，\<cstdarg\>，\<cstddef\>，\<cstdint\>，\<cstdlib\>，\<exception\>，\<initializer_list\>，\<limits\>，\<new\>，\<typeinfo\>，\<version\>；

IO和格式设置，\<cinttypes\>，\<cstdio\>，\<filesystem\>，\<fstream\>，\<iomanip\>，\<ios\>，\<iosfwd\>，\<iostream\>，\<istream\>，\<ostream\>，\<sstream\>，\<streambuf\>，\<strstream\>，\<syncstream\>；本地化，\<clocale\>，\<codecvt\>，\<cvt/wbuffer\>，\<cvt/wstring\>，\<locale\>；时间，\<chrono\>，\<ctime\>；

内存管理，\<allocators\>，\<memory\>，\<memory_resource\>，\<new\>，\<scoped_allocator\>；原子操作，\<atomic\>；多线程处理，\<atomic\>，\<condition_variable\>，\<future\>，\<mutex\>，\<shared_mutex\>，\<thread\>；

序列容器，\<array\>，\<deque\>，\<forward_list\>，\<list\>，\<vector\>；有序的关联容器，\<map\>，\<set\>；无序的关联容器，\<unordered_map\>，\<unordered_set\>；哈希容器，\<hash_map\>，\<hash_set\>；容器适配器，\<queue\>，\<stack\>；容器视图，\<span\>；迭代器，\<iterator\>；算法，\<algorithm\>，\<numeric\>，\<cstdlib\>；

错误和异常处理，\<cassert\>，\<exception\>，\<stdexcept\>，\<system_error\>；

常规实用工具，\<any\>，\<bit\>，\<bitset\>，\<cstdlib\>，\<execution\>，\<functional\>，\<memory\>，\<memory_resource\>，\<optional\>，\<ratio\>，\<scoped_allocator\>，\<tuple\>，\<type_traits\>，\<typeindex\>，\<utility\>，\<variant\>；

其中，一些以c作为前缀的头文件，是C++提供的对相应的C标准标头的包装实现。其中，<hash_map>，<hash_set>并不是ISO C++标准的一部分，位于stdext命名空间或\_\_gnu\_cxx命名空间，目前已废弃并由<unordered_map>，<unordered_set>替代。

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
