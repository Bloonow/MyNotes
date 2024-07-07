# 概述

一个工程中的源文件不计其数，其按类型、功能、模块分别放在若干个目录中，`Makefile`定义了一系列的规则来指定哪些文件需要先编译，哪些文件需要后编译，哪些文件需要重新编译，甚至于进行更复杂的功能操作。Makefile就像一个Shell脚本一样，也可以执行操作系统的命令。

```makefile
make
```

make是一个命令工具，它解释Makefile中的指令。在Makefile文件中描述了整个工程所有文件的编译顺序、编译规则。Makefile有自己的书写格式、关键字、函数。像C语言有自己的格式、关键字和函数一样。而且在Makefile中可以使用系统shell所提供的任何命令来完成想要的工作。

make命令执行时，需要一个Makefile文件，以告诉make命令需要怎么样的去编译和链接程序，规则是：

1. 如果这个工程没有编译过，那么所有C文件都要编译并被链接。
2. 如果这个工程的某几个C文件被修改，那么只需要编译被修改的C文件，并链接目标程序。
3. 如果这个工程的头文件被改变了，那么需要编译引用了这几个头文件的C文件，并链接目标程序。

Makefile在绝大多数的IDE开发环境中都在使用，已经成为一种工程的编译方法。一般来说，大多数的IDE都有这个命令，比如：Delphi的make，Visual C++的nmake，Linux下GNU的make。很多编程语言有自己专属的、能高效配置依赖关系的方法如Ant，Maven，Gradle。

GNU make是一种make工具的版本，在当前目录中按如下顺序搜索make文件：GNUmakefile、makefile、Makefile（推荐使用的名称）。也可以用参数指定特定的make文件，如`make -f MyMakefile`。

# 用于源码编译

当下载完一个源码包并且解压后文件夹下会有一个重要的文件configure，它是一个可执行的脚本文件，用于检查目标系统的配置和可用功能，比如检查依赖或者启用禁用一些模块。configure有很多选项，在待安装的源码目录下使用命令./configure --help可以输出详细的选项列表。

通常configure的任务就是去构建Makefile，这个文件包含了有效构建项目所需的指令。而make指令就是去读取Makefile文件，并编译和安装源码包。

默认的安装路径是在/usr/local目录中，./configure --help里的path可以看到默认的安装位置，如果想指定安装位置，可以使用

```shell
./configure --prefix=/home/eb/mypath
```

使用--prefix选项的好处是可以方便地卸载软件或移植软件。当某个安装的软件不再需要时，只须简单地删除该安装目录，就可以把软件卸载干净。

之后在Makefile文件所在目录中，使用make命令即可编译源码，并进行安装，如下所示。

```shell
make -j8
make install
```

其中，-j8选项表示启用8核对源码进行编译，当软件包特别大时，编译往往需要花费许多时间，启用多核编译是必要的。

如果想要在全局中启用所安装的软件，可以在/usr/local/bin里创建软连接，指向bin文件的启动文件，如下所示。

```shell
ln -s sourcefile targetfile
```

# Makefile的规则

1. **依赖关系**

```makefile
target : prerequisites		# Annotation. 这两行被认为是一个语句。	用 \ 可以换行
	command					# 一个命令必须以一个 Tab 缩进开头
```

- `:`，分号声明一个依赖关系，左侧是目标，右侧是该目标的依赖列表。
- `target`，目标，被认为是这条语句所要处理的对象。可以是object即.o文件，也可以是可执行文件，还可以是一个标签（label）。target这一个或多个的目标文件依赖于prerequisites中的文件，其生成规则定义在command中。
- `prerequisites`，先决条件，依赖列表，即要生成target所依赖的文件或目录。这些文件只要有一个发生了变化，就会触发该语句的第三command部分，即prerequisites中如果有一个以上的文件比target文件要新的话，command所定义的命令就会被执行。
- `command`，需要执行的命令（任意的Shell命令）。语法规定Makefile中的任何命令之前都必须要有一个tab缩进，否则make就会报错。
- make会比较targets文件和prerequisites文件的修改日期，如果prerequisites文件的日期要比targets文件的日期要新，或者target不存在的话，那么，make就会执行后续定义的命令。

### 1. 执行规则

```makefile
make [target]
```

当使用make命令执行Makefile文件时，如果没有指定开始的入口target，则会从最开始一个target处执行，并把这个target作为最终要生成的目标；如指定target，则会从指定的target处执行。在遇到target时，会解析它之后的prerequisites，来执行command。

在解析prerequisites时，如果依赖列表的某个依赖项在磁盘目录中不存在时，就会去找其他语句匹配的target，看看能不能找到并构造，如果能则先保存当前执行位置，然后转而去执行这个target所在的语句；不能就会出错；执行完后再跳回原来的位置，继续执行解析原来未完的语句。该过程类似函数调用，可迭代调用。

总结一下执行步骤为：

1. 读入所有的Makefile；
2. 读入被include的其它Makefile；
3. 初始化文件中的变量；
4. 推导隐晦规则，并分析所有规则；
5. 为所有的目标文件创建依赖关系链；
6. 根据依赖关系，决定哪些目标要重新生成；
7. 执行生成命令。

GNU make可以根据Makefile自动推导（所谓隐晦规则），即自动推导依赖文件及文件依赖关系后面的命令。只要make看到一个.o文件，它就会自动地把同名的实现.c文件加在依赖关系中，并且cc -c同名.c也会被推导出来。所以一个语句，可以只写成`main.o : deps_else.h`等之类的形式，它的.c和command命令都会被推导出来。

### 2. 清空目标文件

每个Makefile中都应该写一个清空目标文件（.o和执行文件）的规则，这不仅便于重编译，也很利于保持文件的清洁。不成文的规定是clean从来都是放在文件的最后。

一般提供一个`clean`的target，方便清除生成的.o等中间文件；可以通过在编译的语句中依赖这个clean的目标来实现自动删除.o文件。如下这种形式，即只定义一个类似标号label的东西，而不在其后指出prerequisites文件，那么make就不会把它当作要生成的文件。这种形式被称为带伪目标的规则。

```makefile
[.PHONY : clean]	# 可选，.PHONY意思表示clean是一个伪目标，更为稳健
clean :
	rm *.o			# Windows下可以使用 del *.obj 删除当前目录所以 .obj 文件
	rm -rf $(OBJ)	# 也可使是宏来指定要删除的文件
```

- 可以使用`-rm`，指示make不理会使用rm命令时会出现的错误，即使出现错误也继续执行。

### 3. 预定义变量

通常而言，会在Makefile文件中定义一些预定义变量，用来指定使用的编译器、要编译的目标以及源文件等。这样一来，今后如果需要修改它们，只需要修改变量的定义即可，而不必变更之后的语句。相当于C中的宏，其中的变量都会被扩展到相应的引用位置上。

```makefile
CC = gcc
PROGRAM = main
IMPL = main.c a_ini.c b_impl.c

$(PROGRAM) : $(IMPL)
	$(CC) -o $(PROGRAM) $(IMPL)
```

- `=`，等号定义一个变量，左侧是变量，右侧是变量的值。
- `$(NAME)`，使用变量，它代表一个名为NAME的变量的值。

### 4. 模式规则

上面的例子中仅仅是解决了“当.c实现文件变动时自动重编译”，但时却无法检测.h头文件。当然可以先定义一个头文件的变量，然后为每个.o文件指定它所依赖的的.c文件和.h文件，如`main.o : main.c $(HEADERS)`；但当文件较多时，这会使Makefile文件异常啰嗦。于是如下：

```makefile
CC = gcc
PROGRAM = main
HEADER = a.h b.h
OBJ = main.o a_ini.o b_impl.o

$(PROGRAM) : $(OBJ)
	$(CC) -o $(PROGRAM) $(OBJ)

%.o : %.c $(HEADER)		# 自动将所有 .c 文件编译成 .o 文件，且依赖于 $(HEADER) 指定的 .h 文件
	$(CC) -c $< -o $@
```

- `%.o : %.c`，这是一个模式规则，它是一个特殊的宏。表示所有的.o目标都依赖于同名的.c文件；再使用变量$(HEADER)列出所需的.h头文件，就可以让一个.o目标依赖于它的.c文件和所列的.h文件。
- `$<`，它展开后是当前语句中的，依赖列表中的第一项。如果要引用整个依赖列表，可以使用`$^`。
- `$@`，它展开后是当前语句中的目标，即target。
- 值得注意的是，`%`其实是需要代入的，如前所述，在解析时如果遇到不存在的依赖项，就会寻找所匹配的其他的target，如果匹配就执行这个语句。该例中，$(OBJ)所定义的.o文件都不存在，于是就找其他的target，这时就会找到%.o匹配成功，成功执行。而使用`*.o`则表示该目录下所有的.o文件。
- 如果文件中仅存在%.o:%.c，那么由之前的分析可知，该模式规则是没有带入项的，所以就会出错。

### 5. 函数

函数的调用，很像变量的使用，也是以$来标识的，语法如下：

```makefile
$(<function> <arguments>)	# or
${<function> <arguments>}
```

- `function`，函数名，makefile支持的函数不多。
- `arguments`，参数列表，参数之间用逗号`,`分隔；参数和函数名之间用空格` `分隔。
- 为了风格的同一，变量和函数的使用最好使用统一的圆括号或花括号。

这在里列出常用的一些函数。

```makefile
$(subst <from>,<to>,<text>)
```

- 把字符串`<text>`中的`<from>`指定字符串替换为`<to>`指定的字符串，返回被替换后的字符串。

```makefile
$(dir <paths...>)
```

- 取出路径中的目录部分，如果文件不是路径中没有目录则返回当前目录，如src/a.c返回src/，b.c返回./。paths为一个参数，多个path用空格隔开。

```makefile
$(call <expression>,<param1>,<param2>,...)
```

- call函数用来创建新的参数化变量，它向表达式`<expression>`传递一些列参数`<param>`；在表达式中通常使用`$(1)`、`$(2)`等的占位符。该函数返回表达式中占位符替换后的字符串。例如：

```makefile
reverse = $(2) $(1)
foo = $(call reverse,a,b)		# foo = b a
```

```makefile
$(shell <command>)
```

- shell函数，它使用命令行窗口执行`<command>`命令，将执行结果作为返回值。例如：

```makefile
HEADER = $(shell find ./ -name "*.h")
IMPL = $(shell find ./ -name "*.c")
OBJ = $(IMPL : %.c = %.o)	# 将IMPL中匹配 %.c 的所以项替换为 %.o 项，返回给OBJ
```

### 6. 引用其他的Makefile

在Makefile中使用`include`关键字可以把别的Makefile文件包含进来，被包含的文件会原模原样的放在当前文件的包含位置。

```makefile
include <filepath...>
```

- filepath可以是当前操作系统Shell的文件模式，可以包含路径和通配符。多个filepath用空格分开。
- 在include前面可以有一些空字符，但是绝不能是Tab键开始。

在make命令开始时，会先寻找include所指出的其他Makefile，并把其内容安置在当前位置。如果文件没有指定绝对路径或者相对路径的话，make会在当前目录下寻找，如果未找到，那么make还会在下面的几个目录寻找：

1. 如果执行make时有`-I`或`--include-dir`参数，那么就会在这个参数指定的目录下去寻找。
2. 如果目录`<prefix>/include`存在（一般是/usr/local/bin或/usr/include），make也会寻找。

如果文件没有找到，make会生成一条警告信息但不会马上出现致命错误，它会继续载入其他的文件，一旦完成Makefile的读取，make会重试这些没有找到或是不能读取的文件。如果还是不行，make才会出现一条致命信息。如果不想让make理会那些无法读取的文件而继续执行，可以在命令前加一个`-`，即使用`-include`，表示无论include过程中出现什么错误，都不要报错而继续执行；和其他版本兼容的相关命令是`sinclude`，作用一样。

此外还有一个环境变量`MAKEFILES`。如果当前环境中定义了环境变量MAKEFILES，那么，make会把这个变量中的值做一个类似于include的动作。这个变量中的值是其它的Makefile，用空格分隔。只是，它和include不同的是，从这个环境变中引入的Makefile的“目标”不会起作用，如果环境变量中定义的文件发现错误，make也会不理。不建议使用。

# 编译当前目录下所有.cpp文件并生成同名可执行文件

```c++
CC = g++
CFLAGS = -O3

# 获取当前目录下所有匹配 *.cpp 的文件名称列表
SOURCES := $(wildcard *.cpp)
# 将 $(SOURCES) 列表中所有 %.cpp 项替换为 % 项
TARGETS := $(patsubst %.cpp, %, $(SOURCES))

all: $(TARGETS)

%: %.cpp
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f $(TARGETS)
```

