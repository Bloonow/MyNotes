# 自定义安装路径

若不指定./configure --prefix选项，则默认情况下，可执行文件存放至/usr/local/bin，库文件存放至/usr/local/lib，配置文件存放至/usr/local/etc，其它资源文件存放至/usr/local/share路径。


# 源码编译安装GCC

可参考https://gcc.gnu.org/install/index.html官网的安装手册，本教程是对手册内容的摘录整理和补充。

## 1. Prerequisites

在构建GCC的过程中需要依赖各种工具和包，下面逐一列出。注意，GNU GCC还支持Ada compiler和D compiler等编译器，但通常使用不到这些语言的编译器，在编译GCC时可忽略这些语言，通过--enable-languages选项指定要支持的语言。

**ISO C++11 compiler**，引导（bootstrap）构建GCC源码所必需的编译器。

可以是GCC 4.8.3或更高版本，需要支持C++11特性，在这之前更早的GCC版本，可能会遇到实现上的BUG。在GCC C++11之前的版本也允许使用ISO C++98 compiler进行引导，在GCC 4.8之前的版本也允许使用ISO C89 compiler进行引导。

要在交叉编译器（cross-compiler）上构建所有语言（all languages），或在其他不执行三阶段引导（3-stage bootstrap）的配置（configuration）上构建所有语言，需要从已有的GCC二进制文件（版本4.8.3或更高）开始，因为C以外的语言源代码可能会使用GCC扩展。

**C standard library and headers**，为构建GCC compiler编译器，所提供的C标准库和头文件必须针对（满足）所有目标平台，而不仅是本地C++ compiler编译器的平台。可从https://www.gnu.org/software/libc链接下载。

这会影响流行的x86_64-pc-linux-gnu平台（以及其他multilib目标平台），对于这种平台，64位（x86_64）和32位（i386）的libc头文件通常是单独打包的。如果在x86_64-pc-linux-gnu上构建本地编译器，需确保正确安装了32位的libc developer包（软件包的确切名称取决于Linux发行版）；或者必须通过配置--disable-multilib选项将GCC构建为只支持64位的编译器。否则，可能会遇到类似fatal error: gnu/stubs-32.h: No such file的错误。

```shell
wget https://ftp.gnu.org/gnu/libc/glibc-2.38.tar.gz
tar -xvf glibc-2.38.tar.gz
cd glibc-2.38
mkdir build
cd build
../configure --prefix=$HOME/.B/glibc
make && make install
```

需要注意的是，在GCC安装完成后，glibc等标准库不能添加到诸如LD_LIBRARY_PATH的环境变量中，否则即使是ls等最常用命令（用到libc），也会发生Segmentation fault段错误。

**GNAT**，为构建Ada compiler编译器，需要GNAT compiler编译器（可以是GCC 5.1或更高版本），其包括GNAT工具，如gnatmake和gnatlink，因为Ada前端（frontend）是用Ada编写的，并使用一些GNAT特有的扩展。

为构建交叉编译器（cross-compiler），建议先本机安装新的编译器，然后用它来构建交叉编译器；同样建议使用旧版本的GNAT来构建新版本的GNAT，使用新版本GNAT构建旧版本GNAT可能会在构建期间出现编译错误。

注意，configure没有测试GNAT安装是否有效，以及版本是否足够新；如果安装的GNAT版本太旧，并且使用了--enable-languages=ada选项，则构建将会失败。

在构建Ada编译器、Ada工具或Ada运行时库时，不能设置ADA_INCLUDE_PATH和ADA_OBJECT_PATH环境变量，可通过验证gnatls -v显式路径来检查构建环境是否干净。

**GDC**，为构建D compiler编译器，需要GDC compiler编译器（可以是GCC 9.4或更高版本），以及D运行时库libphobos，因为D前端（frontend）是用D编写的。

GDC 12之前的版本可以用ISO C++11编译器构建，然后可以用于安装并引导新版本的D frontend前端。建议使用旧版本的GDC来构建新版本的GDC，使用新版本GDC构建旧版本GDC可能会在构建期间出现编译错误（由于与弃用或删除特性）。

注意，configure没有测试GDC安装是否有效，以及版本是否足够新；虽然D frontend前端的实现没有使用任何GDC特有的扩展，或者D语言的新特性，如果安装的GDC版本太旧，并且使用了--enable-languages=D选项，则构建将会失败。

在某些目标平台上，libphobos默认是不启用的，但可以使用--enable-libphobos启用之。

**GM2**，如果想构建完整的“包含目标系统模块定义”（target SYSTEM definition module）的Modula-2文档，必须使用Python3；如果无法使用Python3，则Modula-2文档将包含与目标系统无关的系统模块定义。

**POSIX compatible shell or GNU bash**，在运行configure时是必要的，因为一些/bin/sh存在BUG，在configure目标库时可能会崩溃。

**A POSIX or SVR4 awk**，用于为GCC创建一些生成的源文件，可以是GNU awk version 3.1.5版本。

**GNU binutils**，在某些情况下是必要/可选的，参阅目标平台的具体说明，以确定具体要求。注意，binutils 2.35或高新版本，需要LTO才能正确地与GNU libtool一起工作，包括启用LTO的引导。

**gzip and bzip2 and GNU tar**，用于从源代码的压缩包中提取文件。

**Perl version 5.6.1 (or later)**，当目标平台是Darwin系统，构建libstdc++，且不使用--disable-symvers选项时，是必要的；当目标平台是Solaris系统，使用Solaris ld链接器，且不使用--disable-symvers选项时，是必要的。

如果可以，在flock不可用的情况下启用并行测试libgomp选项。

> Darwin is the core Unix operating system of macOS (previously OS X and Mac OS X), iOS, watchOS, tvOS, iPadOS, visionOS, and bridgeOS.

编译GCC需要一些库的支持，有必需的，有可选的。虽然所需工具的任何足够新版本通常都可以工作，但库的要求通常更严格，推荐使用文档中指定的确切版本。

**GNU Multiple Precision Library (GMP) version 4.3.2 (or later)**，任意精度的数学库，构建GCC所必需的，可从https://gmplib.org/下载。

如果在GCC的源码目录中，存在GMP源码（目录为gmp），那么它会和GCC一起构建。如果已经安装GMP，但不在库的搜索路径中，在对GCC进行configure时，可指定--with-gmp=/path/to/gmp选项，或指定--with-gmp-include=/path/to/gmp/include和--with-gmp-lib=/path/to/gmp/lib选项，或将GMP的lib路径添加到LD_LIBRARY_PATH环境变量中。

```shell
wget https://gmplib.org/download/gmp/gmp-6.3.0.tar.xz
tar -xvf gmp-6.3.0.tar.xz
cd gmp-6.3.0
./configure --prefix=$HOME/.B/gmp
make && make install
export LD_LIBRARY_PATH=$HOME/.B/gmp/lib:$LD_LIBRARY_PATH
```

**MPFR Library version 3.1.0 (or later)**，多精度浮点计算库，构建GCC所必需的，可从https://www.mpfr.org下载。

如果在GCC的源码目录中，存在MPFR源码（目录为mpfr），那么它会和GCC一起构建。如果已经安装MPFR，但不在库的搜索路径中，在对GCC进行configure时，可指定--with-mpfr=/path/to/mpfr选项，或指定--with-mpfr-include=/path/to/mpfr/include和--with-mpfr-lib=/path/to/mpfr/lib选项，或将MPFR的lib路径添加到LD_LIBRARY_PATH环境变量中。

```shell
wget https://www.mpfr.org/mpfr-current/mpfr-4.2.0.tar.gz
tar -xvf mpfr-4.2.0.tar.gz
cd mpfr-4.2.0
./configure --prefix=$HOME/.B/mpfr --with-gmp=$HOME/.B/gmp
make && make install
export LD_LIBRARY_PATH=$HOME/.B/mpfr/lib:$LD_LIBRARY_PATH
```

**MPC Library version 1.0.1 (or later)**，高精度复数计算库，构建GCC所必需的，可从https://www.multiprecision.org/mpc/download.html下载。

如果在GCC的源码目录中，存在MPC源码（目录为mpc），那么它会和GCC一起构建。如果已经安装MPC，但不在库的搜索路径中，在对GCC进行configure时，可指定--with-mpc=/path/to/mpc选项，或指定--with-mpc-include=/path/to/mpc/include和--with-mpc-lib=/path/to/mpc/lib选项，或将MPC的lib路径添加到LD_LIBRARY_PATH环境变量中。

```shell
wget https://ftp.gnu.org/gnu/mpc/mpc-1.3.1.tar.gz
tar -xvf mpc-1.3.1.tar.gz
cd mpc-1.3.1
./configure --prefix=$HOME/.B/mpc --with-gmp=$HOME/.B/gmp --with-mpfr=$HOME/.B/mpfr
make && make install
export LD_LIBRARY_PATH=$HOME/.B/mpc/lib:$LD_LIBRARY_PATH
```

**ISL Library version 0.15 (or later)**，集合和线性约束库，使用Graphite循环优化构建GCC所必需的，可从https://gcc.gnu.org/pub/gcc/infrastructure下载。

如果在GCC的源码目录中，存在ISL源码（目录为isl），那么它会和GCC一起构建。如果已经安装ISL，但不在库的搜索路径中，在对GCC进行configure时，可指定--with-isl=/path/to/isl选项，或指定--with-isl-include=/path/to/isl/include和--with-isl-lib=/path/to/isl/lib选项，或将ISL的lib路径添加到LD_LIBRARY_PATH环境变量中。

```shell
wget https://gcc.gnu.org/pub/gcc/infrastructure/isl-0.24.tar.bz2
tar -xvf isl-0.24.tar.bz2
cd isl-0.24
./configure --prefix=$HOME/.B/isl --with-gmp-prefix=$HOME/.B/gmp
make && make install
export LD_LIBRARY_PATH=$HOME/.B/isl/lib:$LD_LIBRARY_PATH
```

**zstd Library**，用zstd compression压缩（用于LTO bytecode字节码）来构建GCC所必需的，会在默认的库补丁（library patch）路径中搜索该库，或者，在对GCC进行configure时，使用--with-zstd=/path/to/zstd选项。

实际上，对于GMP、MPFR、MPC、ISL库来说，不必手动源码编译安装，见下一节。

## 2. Downloading the source

至此，编译GCC所需的库和包基本配置完毕，下面开始源码编译安装GCC，GCC源码可从https://gcc.gnu.org/releases.html链接下载。

源代码发行版包括C、C++、Objective-C、Fortran、Ada的编译器，以及C++、Objective-C、Fortran的运行时库。

如果打算构建binutils（以对现有版本升级，或替换操作系统提供的相应工具），需要将binutils distribution发行包解压至同一目录下，或不同目录下；对于后一情况，需要在GCC源代码目录中添加符号链接，指向打算与编译器一起构建的任何binutils组件（如bfd、binutils、gas、gprof、ld、opcodes等）。

同样，GMP、MPFR、MPC、ISL库也可以与GCC一起自动构建。在GCC源码目录中，存在contrib/download_prerequisites文件，其中声明了对这四个库的引用，如下所示。

```shell
gmp='gmp-6.1.0.tar.bz2'
mpfr='mpfr-3.1.6.tar.bz2'
mpc='mpc-1.0.3.tar.gz'
isl='isl-0.18.tar.bz2'

base_url='http://gcc.gnu.org/pub/gcc/infrastructure/'
```

可以通过直接执行contrib/download_prerequisites文件，使GCC自动配置对GMP,MPFR,MPC,ISL库的依赖，如下所示。

```shell
wget https://ftp.gnu.org/gnu/gcc/gcc-11.4.0/gcc-11.4.0.tar.gz
tar -xvf gcc-11.4.0.tar.gz
cd gcc-11.4.0
./contrib/download_prerequisites
```

## 3. configuration, make, install

与多数GNU软件一样，在构建GCC之前必须对其进行configure配置。这里，使用srcdir指向GCC源代码的顶层目录，使用objdir指向构建过程使用的临时目录。强烈建议将GCC编译构建到一个独立于源代码的目录当中，这个目录不在源代码的目录树中，即objdir不应该是srcdir的子目录。

其次，在本地配置GCC时，旧版本的cc或gcc必须存在于环境变量中，否则会导致配置脚本失败。

```shell
wget https://ftp.gnu.org/gnu/gcc/gcc-11.4.0/gcc-11.4.0.tar.gz
tar -xvf gcc-11.4.0.tar.gz
mkdir objdir
cd objdir
../gcc-11.4.0/configure --prefix=$HOME/.B/gcc114 --enable-languages=c,c++ --enable-checking=release --disable-multilib
```

configure配置完成后，推荐使用make -jN并行编译构建，以加快构建时间，编译构建完成后，使用make install进行安装，如下所示。注意，要先通过unset将一些环境变量清空，以防编译错误。

```shell
unset PKG_CONFIG_PATH CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH INCLUDE LIBRARY_PATH LD_LIBRARY_PATH LD_RUN_PATH
cd objdir
make -j16
make install
```

若在编译过程中，遇到如下错误。

```
/gcc-x.x.x/libsanitizer/sanitizer_common/sanitizer_internal_defs.h:261:72: error: size of array ‘assertion_failed__1150’ is negative
```

可以在configure时添加--disable-libsanitizer选项解决。

至此，GCC的安装已经完成，下面配置其环境变量，设prefix是指定的自定义安装路径。用户级别的二进制文件在prefix/bin目录，C++库的头文件在prefix/include目录，库文件在prefix/lib目录，编译器的内部库在prefix/lib/gcc目录。

在用户的.bashrc文件中添加以下代码，以在每次启动shell时执行，配置bash的环境变量。

```shell
GCC_BIN=$HOME/.B/gcc114/bin
GCC_INCLUDE=$HOME/.B/gcc114/include
GCC_LIB_32=$HOME/.B/gcc114/lib
GCC_LIB_64=$HOME/.B/gcc114/lib64
GCC_LIB=$GCC_LIB_32:$GCC_LIB_64
export PATH=$GCC_BIN:$PATH
export C_INCLUDE_PATH=$GCC_INCLUDE:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$GCC_INCLUDE:$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=$GCC_LIB:$LD_LIBRARY_PATH
export LD_RUN_PATH=$GCC_LIB:$LD_RUN_PATH
```

## 4. Checking

检测GCC版本，如下所示。

```shell
gcc --version
```

```
gcc (GCC) 11.4.0
Copyright (C) 2021 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```

检查GCC的搜索路径，如下所示，注意下述符号是反分号（在ESC键下方），而不是正分号。

```shell
`gcc -print-prog-name=cc1plus` -v
```

```
#include "..." search starts here:
#include <...> search starts here:
 /storage/bln/.B/gcc114/include
 .
 /storage/bln/.B/gcc114/lib/gcc/x86_64-pc-linux-gnu/11.4.0/../../../../include/c++/11.4.0
 /storage/bln/.B/gcc114/lib/gcc/x86_64-pc-linux-gnu/11.4.0/../../../../include/c++/11.4.0/x86_64-pc-linux-gnu
 /storage/bln/.B/gcc114/lib/gcc/x86_64-pc-linux-gnu/11.4.0/../../../../include/c++/11.4.0/backward
 /storage/bln/.B/gcc114/lib/gcc/x86_64-pc-linux-gnu/11.4.0/include
 /usr/local/include
 /storage/bln/.B/gcc114/lib/gcc/x86_64-pc-linux-gnu/11.4.0/include-fixed
 /usr/include
End of search list.
```

```shell
`g++ -print-prog-name=cc1plus` -v
```

```
#include "..." search starts here:
#include <...> search starts here:
 /storage/bln/.B/gcc114/include
 .
 /storage/bln/.B/gcc114/lib/gcc/x86_64-pc-linux-gnu/11.4.0/../../../../include/c++/11.4.0
 /storage/bln/.B/gcc114/lib/gcc/x86_64-pc-linux-gnu/11.4.0/../../../../include/c++/11.4.0/x86_64-pc-linux-gnu
 /storage/bln/.B/gcc114/lib/gcc/x86_64-pc-linux-gnu/11.4.0/../../../../include/c++/11.4.0/backward
 /storage/bln/.B/gcc114/lib/gcc/x86_64-pc-linux-gnu/11.4.0/include
 /usr/local/include
 /storage/bln/.B/gcc114/lib/gcc/x86_64-pc-linux-gnu/11.4.0/include-fixed
 /usr/include
End of search list.
```

# 降级安装GCC

较新内核版本的Linux系统，所自带的GCC版本一般较新，例如gcc-10.x或gcc-11.x之类，如果需要较旧版本的GCC编译器，例如gcc-7.5，无法使用新版本GCC从源码编译旧版本GCC，故选择使用编译好的旧版本的GCC软件包。

若机器可联网，可在/etc/apt/sources.list文件中，添加如下软件源，否则在使用apt安装旧版本GCC时，会因为软件仓库源较新而找不到旧版本GCC软件包。

```
deb [arch=amd64] http://archive.ubuntu.com/ubuntu focal main universe
```

使用它如下命令更新软件包的源。

```shell
sudo apt update
```

然后再通过apt安装旧版本GCC，如下所示。

```shell
sudo apt install gcc-7
sudo apt install g++-7
```

```shell
sudo ln -s /usr/bin/gcc-7 /usr/bin/gcc
sudo ln -s /usr/bin/g++-7 /usr/bin/g++
sudo ln -s /usr/bin/gcc-7 /usr/bin/cc
sudo ln -s /usr/bin/g++-7 /usr/bin/c++
```

# 源码编译安装Python

Python官方文档https://docs.python.org/3/using/index.html给出在不同平台上配置安装Python的教程，包括构建源码所需的特性。

本教程源码安装的是Python-3.9.17，下载Python的链接为https://www.python.org/ftp/python/3.9.17/Python-3.9.17.tgz。对于Python项目，在其源码目录下存在一个README.rst文件，会给出安装的配置选项和注意事项。

通常情况下，源码编译安装Python时，GCC需要用到如下库。

| module                    | lib                | install                             |
| ------------------------- | ------------------ | ----------------------------------- |
| \_dbm                     | libgdbm-compat-dev | sudo apt install libgdbm-compat-dev |
| \_bz2                     | libbz2-dev         | sudo apt install libbz2-dev         |
| \_uuid                    | uuid-dev           | sudo apt install uuid-dev           |
| \_curses                  | libncurses5-dev    | sudo apt install libncurses5-dev    |
| _curses_panel             |                    |                                     |
| _gdbm                     | libgdbm-dev        | sudo apt install libgdbm-dev        |
| _lzma                     | liblzma-dev        | sudo apt install liblzma-dev        |
| _sqlite3                  | sqlite3            | sudo apt install sqlite3            |
|                           | libsqlite3-dev     | sudo apt install libsqlite3-dev     |
| _ssl                      | openssl            | sudo apt install openssl            |
|                           | libssl-dev         | sudo apt install libssl-dev         |
| _tkinter                  | tcl8.6-dev         | sudo apt install tcl8.6-dev         |
|                           | tk8.6-dev          | sudo apt install tk8.6-dev          |
| _readline                 | libreadline-dev    | sudo apt install libreadline-dev    |
| _zlib                     | zlib1g-dev         | sudo apt install zlib1g-dev         |
| No module named '_ctypes' | libffi-dev         | sudo apt install libffi-dev         |

指定安装路径并编译安装的命令如下所示。

```shell
cd Python-3.9.17
./configure --prefix=$HOME/.B/python39 --enable-shared --enable-optimizations CFLAGS=-fPIC
make -j8
make install
```

在$HOME/.B/python39/bin目录下，创建python和pip的软链接，如下所示。

```shell
cd $HOME/.B/python39/bin
ln -s python3.9 python
ln -s pip3 pip	
```

在用户的.bashrc文件中添加以下代码，以在每次启动shell时执行，配置bash的环境变量。需要注意的，如果源码安装多个版本的Python，则需要先将下述环境变量撤销，以免执行configure配置时导入错误版本的头文件，从而导致编译错误。

```shell
PYTHON_BIN=$HOME/.B/python39/bin
PYTHON_INCLUDE=$HOME/.B/python39/include
PYTHON_LIB=$HOME/.B/python39/lib
export PATH=$PYTHON_BIN:$PATH
export C_INCLUDE_PATH=$PYTHON_INCLUDE:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$PYTHON_INCLUDE:$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=$PYTHON_LIB:$LD_LIBRARY_PATH
export PYTHONUNBUFFERD=1
```

或者如果拥有sudo权限，可直接在/etc/ld.so.conf文件或/etc/ld.so.conf.d/xxx.conf文件中添加动态库的路径。

可以通过在用户家目录下，创建.pip/pip.conf文件，配置pip工具所使用的镜像源，添加清华的镜像源如下所示。

```
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
[install]
use-mirrors = true
mirrors = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
```

或者在使用pip命令时，临时指定所使用的镜像源，如下所示。

```shell
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple package-name
```

# 安装NVIDIA驱动和CUDA开发套件

在终端键入nvidia-smi，以查看是否安装NVIDIA显卡驱动，若未安装，则会提示设备硬件推荐驱动版本的安装命令。

```shell
nvidia-smi
```

```shell
Command 'nvidia-smi' not found, but can be installed with:
sudo apt install nvidia-driver-550
```

可以使用提示命令直接安装，也可以到NVIDIA官方提供的https://www.nvidia.com/Download/index.aspx地址下载对应版本的驱动程序包，也可下载其他版本的驱动程序。对于Ubuntu系统，还可使用ubuntu-drivers devices命令提示推荐版本。

下载完成后，在Linux平台下会得到类似于NVIDIA-Linux-x86_64-535.86.05.run的可执行程序，在终端执行sudo sh NVIDIA-Linux-x86_64-535.86.05.run以进入安装过程，根据提示选择所需功能，完成安装即可。下面给出手动安装的步骤。

注意，较旧版本的NVIDIA驱动、CUDA开发套件、cuDNN库是命令行式提示安装的，较新版本的时命令行式交互安装的，但其基本信息一致，注意辨别，根据需求选择即可。

## 1. 安装前准备

执行以下命令删除可能已安装的NVIDIA驱动，或执行/usr/bin/nvidia-uninstall以卸载NVIDIA驱动。

```shell
sudo apt-get remove --purge nvidia*
```

首先需要禁用系统自带的Nouveau开源驱动程序，编辑/etc/modprobe.d/blacklist-nouveau.conf配置文件，在最后添加如下几行命令。

```shell
blacklist nouveau
blacklist lbm-nouveau
options nouveau modeset=0
```

重启机器，如下所示。

```shell
sudo update-initramfs -u  # 更新内核并重新启动
sudo reboot
```

验证Nouveau是否禁用，在终端键入下述命令，若无任何结果显示，则说明已经禁用。

```shell
lsmod | grep nouveau
```

安装内核源码，以及内核开发包，以防在执行sh NVIDIA-Linux-x86_64-535.86.05.run的过程中提示"can not found kernel source tree"错误，如下所示。

```shell
sudo apt-get install linux-source               # 内核源码
sudo apt-get install linux-headers-$(uname -r)  # 内核开发包
```

内核开发包通常包含构建驱动程序所需的头文件和其他依赖项。

在某些情况下，即使安装了内核源码和开发包，仍然可能出现"can not found kernel source tree"错误，对此可手动创建一个符号链接，将内核源代码树链接到/usr/src/linux，如下所示。

```shell
sudo ln -s /usr/src/linux-headers-$(uname -r) /usr/src/linux
```

## 2. 安装NVIDIA驱动

进入纯字符命令行界面，关闭Linux的图形界面，如下所示。

```shell
sudo init 3
```

进入NVIDIA-Linux-x86_64-535.86.05.run所在目录，执行下述命令和选项，以进入安装过程，根据提示，完成安装即可。

```shell
chmod +x NVIDIA-Linux-x86_64-535.86.05.run
sudo sh NVIDIA-Linux-x86_64-535.86.05.run -no-x-check -no-nouveau-check --no-opengl-files
```

其中，选项--no-opengl-files表示不安装OpenGL文件。如果是双显卡（集显+独显），且用来显示的显卡并不是NVIDIA显卡，则不能安装OpenGL文件，否则会出现黑屏或者循环登录的问题，即用来显示的显卡的OpenGL库被覆盖。可以在此处指定，也可在安装过程中选择是否安装，下述CUDA安装过程同理。

安装完成后，使用如下命令，再启动Linux图形界面。

```shell
sudo init 5
```

可使用如下命令卸载NVIDIA驱动。

```shell
sudo /usr/bin/nvidia-uninstall
```

## 3. 安装CUDA开发套件

安装完NVIDIA驱动之后，使用nvidia-smi命令，可通过NVIDIA驱动查看显卡各种信息，其中右上角处的CUDA Version: 11.5提示表示当前驱动所能支持的最高CUDA版本，在安装CUDA时不应超过该版本。

到NVIDIA官方提供的https://developer.nvidia.com/cuda-toolkit-archive网址，下载所需版本的CUDA程序包。注意，可使用uname -a命令，查看当前Linux内核版本，以选择正确的CUDA安装包。如下所示。

```shell
wget https://developer.download.nvidia.com/compute/cuda/11.3.1/local_installers/cuda_11.3.1_465.19.01_linux.run
```

此处选择将CUDA安装到自定义路径，提前创建如下路径（必须使用绝对路径），若安装默认路径，跳过此步。

```shell
mkdir $HOME/.B/cuda113
mkdir $HOME/.B/cuda113/Samples
mkdir $HOME/.B/cuda113/Libs
```

因为之前安装NVIDIA驱动时，没有安装OpenGL文件，故这里也选择不安装OpenGL库。如下所示。

```shell
sudo chmod u+x cuda_11.3.1_465.19.01_linux.run
sudo sh cuda_11.3.1_465.19.01_linux.run --no-opengl-libs
```

有时会遇到GCC编译器不支持的错误，如下所示。

```shell
Failed to verify gcc version. See log at /tmp/cuda-installer.log for details.
cat /tmp/cuda-installer.log
[ERROR]: unsupported compiler version: 11.4.0. Use --override to override this check.
```

可以选择将GCC版本降到CUDA所支持的编译器版本，也可在安装时指定--override选项；推荐降级GCC版本，以避免在后续使用nvcc编译时，出现不支持的GCC版本问题。

执行.run安装程序后，对于较新版本的CUDA，会显示如下命令行交互式CUDA Installer界面。

```shell
┌──────────────────────────────────────────────────────────────────────────────┐
│ CUDA Installer                                                               │
│ - [ ] Driver                                                                 │
│      [ ] 465.19.01                                                           │
│ + [X] CUDA Toolkit 11.3                                                      │
│   [X] CUDA Samples 11.3                                                      │
│   [X] CUDA Demo Suite 11.3                                                   │
│   [X] CUDA Documentation 11.3                                                │
│   Options                                                                    │
│   Install                                                                    │
│                                                                              │
│ Up/Down: Move | Left/Right: Expand | 'Enter': Select | 'A': Advanced options │
└──────────────────────────────────────────────────────────────────────────────┘
```

可根据需要选择所需组件，由于前述已经安装了NVIDIA Driver驱动，此处不再选择。选择Options选项，进行自定义路径的配置，如下所示。

```shell
┌──────────────────────────────────────────────────────────────────────────────┐
│ Options                                                                      │
│   Driver Options                                                             │
│   Toolkit Options                                                            │
│   Samples Options                                                            │
│   Library install path (Blank for system default)                            │
│   Done                                                                       │
│                                                                              │
│ Up/Down: Move | Left/Right: Expand | 'Enter': Select | 'A': Advanced options │
└──────────────────────────────────────────────────────────────────────────────┘
```

选择Toolkit Options选项，Samples Options选项，Library install path选项，再选择Change Path选项，将其配置为之前创建的路径，如下所示。注意Toolkit Options选项卡下，其他创建符号链接的选项，若无需要（或已存在其他版本CUDA），也可取消，之后手动配置。

```shell
┌──────────────────────────────────────────────────────────────────────────────┐
│ Change Toolkit Install Path                                                  │
│ /home/bln/.B/cuda113/                                                        │
│                                                                              │
│ 'Enter': Finish                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

```shell
┌──────────────────────────────────────────────────────────────────────────────┐
│ Change Writeable Samples Install Path                                        │
│ /home/bln/.B/cuda113/Samples/                                                │
│                                                                              │
│ 'Enter': Finish                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

```shell
┌──────────────────────────────────────────────────────────────────────────────┐
│ Library install path (Blank for system default)                              │
│ /home/bln/.B/cuda113/Libs/                                                   │
│                                                                              │
│ 'Enter': Finish                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

配置完成后，选择Done进入CUDA Installer界面，选择Install开始安装，等待即可，完成后显示如下提示信息。

```
===========
= Summary =
===========

Driver:   Not Selected
Toolkit:  Installed in /storage/bln/.B/cuda113/
Samples:  Installed in /storage/bln/.B/cuda113/Samples/, but missing recommended libraries

Please make sure that
 -   PATH includes /storage/bln/.B/cuda113/bin
 -   LD_LIBRARY_PATH includes /storage/bln/.B/cuda113/lib64, 
 					  or, add /storage/bln/.B/cuda113/lib64 to /etc/ld.so.conf and run ldconfig as root

To uninstall the CUDA Toolkit, run cuda-uninstaller in /storage/bln/.B/cuda113/bin
***WARNING: Incomplete installation! This installation did not install the CUDA Driver. 
			A driver of version at least 465.00 is required for CUDA 11.3 functionality to work.
To install the driver using this installer, 
			run the following command, replacing <CudaInstaller> with the name of this run file:
    sudo <CudaInstaller>.run --silent --driver

Logfile is /tmp/cuda-installer.log
```

在用户的.bashrc文件中添加以下代码，以在每次启动shell时执行，配置bash的环境变量。

```shell
CUDA_BIN=$HOME/.B/cuda113/bin
CUDA_INCLUDE=$HOME/.B/cuda113/include
CUDA_LIB=$HOME/.B/cuda113/lib64
export PATH=$CUDA_BIN:$PATH
export C_INCLUDE_PATH=$CUDA_INCLUDE:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$CUDA_INCLUDE:$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=$CUDA_LIB:$LD_LIBRARY_PATH
```

使用如下命令验证CUDA是否安装成功。

```shell
nvcc --version
```

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Mon_May__3_19:15:13_PDT_2021
Cuda compilation tools, release 11.3, V11.3.109
Build cuda_11.3.r11.3/compiler.29920130_0
```

可使用如下命令卸载CUDA开发套件。

```shell
sudo /path/to/cuda-11.3/bin/cuda-uninstaller
```

## 4. 安装cuDNN库

选择与所安装CUDA版本对应的cuDNN版本，可从https://developer.nvidia.com/rdp/cudnn-archive下载。如下所示。

```shell
wget https://developer.nvidia.com/downloads/compute/cudnn/secure/8.9.3/local_installers/11.x/cudnn-linux-x86_64-8.9.3.28_cuda11-archive.tar.xz
```

安装cuDNN其实就是将cuDNN中的一些头文件和库文件，复制到已安装的CUDA目录当中，如下所示。

```shell
tar -xvf cudnn-linux-x86_64-8.9.3.28_cuda11-archive.tar.xz
cd cudnn-linux-x86_64-8.9.3.28_cuda11-archive
sudo cp include/* $HOME/.B/cuda113/include
sudo cp lib/* $HOME/.B/cuda113/lib64
```

修改cuDNN文件的权限，如下所示。

```shell
sudo chmod a+r $HOME/.B/cuda113/include/cudnn*.h
sudo chmod a+r $HOME/.B/cuda113/lib64/libcudnn*
```

> 注意，在Windows平台下，需要将cuDNN目录下的bin/cudnn\*.dll，include/cudnn\*.h，和lib/cudnn\*.lib文件都复制到CUDA相应目录下。

# 无图形界面安装Matlab

获得Matlab的R2019b_Linux.iso软件镜像，以及Matlab R2019b Linux64 Crack.tar.gz破解补丁。

可以将软件镜像挂载到文件系统，如下所示。因为要将Crack补丁拷贝到软件镜像目录下，而挂载的软件镜像目录是只读的，故推荐将软件镜像解压出来。

```shell
mkdir /media/R2019b
chmod -R 777 /media/R2019b
mount -o loop R2019b_linux.iso /media/R2019b
# 安装步骤，此处略，详细见下
umount /media/R2019b
```

创建软件镜像的暂存目录，使用7z解压R2019b_linux.iso软件镜像。

```shell
sudo apt install p7zip-full
mkdir $HOME/.B/R2019b  # 软件镜像的暂存目录
7z x R2019b_linux.iso -o'R2019b'
```

解压破解补丁，并将之移动到软件镜像目录下的Crack文件夹中。

```shell
tar -xvf 'Matlab R2019b Linux64 Crack.tar.gz'
mv 'Matlab R2019b Linux64 Crack' $HOME/.B/R2019b/Crack
```

创建软件的安装目录，并将配置文件复制到安装目录下。

```shell
mkdir $HOME/.B/matlab2019b  # 软件的安装目录
mkdir $HOME/.B/matlab2019b/etc
cp $HOME/.B/R2019b/installer_input.txt $HOME/.B/matlab2019b/etc
cp $HOME/.B/R2019b/activate.ini $HOME/.B/matlab2019b/etc
```

修改安装目录下的matlab2019b/etc/installer_input.txt安装配置文件，用##表示注释行，指定其中一些参数的值，内容如下所示。

```ini
## 软件安装目录
destinationFolder=/home/2022/bln/.B/matlab2019b
## 序列号，在目录 Crack/readme.txt 文件中，选择 standalone 中的序列号
fileInstallationKey=09806-07443-53955-64350-21751-41297
## 同意
agreeToLicense=yes
## 安装日志
outputFile=/tmp/mathwork_install.log
## 非GUI方式安装，静默模式
mode=silent
## 激活配置文件
activationPropertiesFile=/home/2022/bln/.B/matlab2019b/etc/activate.ini
## 证书文件
licensePath=/home/2022/bln/.B/R2019b/Crack/license_standalone.lic
```

其中，installer_input.txt文件最后面列出了Matlab所支持的ToolBox工具包，可使用##注释掉某一项表示不安装该软件包，默认情况下会安装所有软件包。

修改安装目录下的matlab2019b/etc/activate.ini激活配置文件，用#表示注释行，否则为非注释行，指定其中一些参数的值，内容如下所示。

```ini
# 静默模式
isSilent=true
# 离线激活
activateCommand=activateOffline
# 证书文件绝对路径
licenseFile=/home/2022/bln/.B/R2019b/Crack/license_standalone.lic
# 激活序列号，与前述相同
activationKey=09806-07443-53955-64350-21751-41297
# 证书文件所在目录
installLicenseFileDir=/home/2022/bln/.B/R2019b/Crack
# 证书文件名称
installLicenseFileName=license_standalone.lic
```

在软件镜像目录外，使用如下命令安装。为防止安装过程中软件镜像目录R2019b各种文件的访问权限问题，可将其全部改为rwx访问权限。

```shell
sudo $HOME/.B/R2019b/install -inputFile $HOME/.B/matlab2019b/etc/installer_input.txt
```

等待安装完毕，最后显示Successful即表示安装成功。

将证书文件拷贝到安装目录下的licenses文件夹中。

```shell
mkdir $HOME/.B/matlab2019b/licenses
cp $HOME/.B/R2019b/Crack/license_standalone.lic $HOME/.B/matlab2019b/licenses
```

备份安装目录中的libmwlmgrimpl.so库文件，将破解补丁中的库文件替换到原来的位置，并修改文件的访问权限。

```shell
cd $HOME/.B/matlab2019b/bin/glnxa64/matlab_startup_plugins/lmgrimpl
sudo tar -czf libmwlmgrimpl.so.bak.tgz libmwlmgrimpl.so
sudo rm -rf libmwlmgrimpl.so
sudo cp $HOME/.B/R2019b/Crack/R2019b/bin/glnxa64/matlab_startup_plugins/lmgrimpl/libmwlmgrimpl.so libmwlmgrimpl.so
sudo chmod 555 libmwlmgrimpl.so
```

执行安装目录bin下的激活脚本，如下所示。

```shell
$HOME/.B/matlab2019b/bin/activate_matlab.sh -propertiesFile $HOME/.B/matlab2019b/etc/activate.ini
```

显示Silent activation succeeded即表示激活成功。

在用户的.bashrc文件中添加以下代码，以在每次启动shell时执行，配置bash的环境变量。

```shell
MATLAB_PATH=$HOME/.B/matlab2019b/bin
export PATH=$MATLAB_PATH:$PATH
alias matlab='$HOME/.B/matlab2019b/bin/matlab -nodesktop -nodisplay'
```

假设当前目录下存在matlab_test.m脚本，可以先执行matlab命令，打开matlab命令行，然后输入当前目录下的脚本名称matlab_test可执脚本，也可使用如下命令非交互式的执行脚本。

```shell
matlab -batch matlab_test
```

