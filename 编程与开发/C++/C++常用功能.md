# C++标准应用

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

所有具有变长参数列表的函数都至少应该有一个已命名参数，可变参数使用`...`占位符。在函数中先使用`va_list 列表名;`声明一个列表，再初始化声明列表的开始位置`va_start(列表名, x);`其中x是可变列表前一个已命名参数，最后必须用`va_end(列表名);`来释放内存，以确定函数结束后，推栈处于稳定状态。如果要访问突示参数，可使用`va_arg(列表名, 实参解释类型)`，通常用for循环遍历。

但应注意，变长参数列表无结束标志，因而要显式规定，如可以让第一个参数计算参数的数目，或当参数是一组指针时，可以要求最后一个指针是NULL等。注：不推荐使用C风格的变长参数列表，因为其十分不安全：不知道参数的数目，不知道参数的类型。

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

