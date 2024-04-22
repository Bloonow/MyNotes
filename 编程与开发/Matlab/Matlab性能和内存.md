MATLAB性能和内存分析

以简单、易读的方式编写您的代码，特别是对于第一次实现。过早优化的代码可能带来无谓的复杂性，而不会显著改善性能。第一次实施应尽可能简单。此后，如果速度成为问题，请使用探查功能来找出瓶颈所在。

MATLAB会自动处理数据存储。但是，如果内存是问题所在，可以确定内存需求并应用相关技术来更有效地使用内存。

# 测量和探查代码

## （一）衡量代码性能

使用[`timeit`](https://ww2.mathworks.cn/help/matlab/ref/timeit.html)函数或秒表计时器函数[`tic`](https://ww2.mathworks.cn/help/matlab/ref/tic.html)和[`toc`](https://ww2.mathworks.cn/help/matlab/ref/toc.html)来计算代码运行所需的时间。

### 1. 计时函数

要测量运行函数所需的时间，请使用[`timeit`](https://ww2.mathworks.cn/help/matlab/ref/timeit.html)函数，timeit函数多次调用指定的函数，并返回测量结果的中位数。它采用要测量的函数的句柄并返回典型执行时间（以秒为单位）。

例如，用户定义了一个函数myComputeFunction来计算，它采用两个在工作区中定义的输入x和y，可以使用timeit计算执行函数所需的时间。

```matlab
f = @() myComputeFunction(x,y);  % handle to function
timeit(f)
```

### 2. 计算部分代码的时间

要计算某部分程序需要多长时间运行或者比较各部分程序的不同实现的速度，可使用秒表计时器函数[`tic`](https://ww2.mathworks.cn/help/matlab/ref/tic.html)和[`toc`](https://ww2.mathworks.cn/help/matlab/ref/toc.html)。调用tic可启动计时器，紧接着toc可读取已用时间。

```matlab
tic
   % The program section to time. 
toc
```

有时程序运行速度太快，导致tic和toc无法提供有用的数据。如果您的代码运行速度快于1/10秒，请考虑测量它在循环中运行的时间，然后求平均值以计算单次运行的时间。

### 3. cputime

建议使用timeit或tic和toc来度量代码的性能，这些函数会返回**墙上挂钟时间**。与tic和toc不同，timeit函数会调用您的代码多次，因此会考虑首次成本。

而`cputime`函数会测量总CPU时间并跨所有线程进行汇总，此测量值不同于timeit或tic/toc返回的挂钟时间，可能会造成误解。

例如，pause函数的CPU时间通常很小，但挂钟时间会考虑暂停MATLAB执行的实际时间，因此挂钟时间可能更长。再例如，如果您的函数均匀使用四个处理核，则CPU时间可能约是挂钟时间的四倍。

### 4. 有关测量性能的提示

在测量代码的性能时，请考虑以下提示：

- 计算足够大的一部分代码的时间。理想情况下，您进行计时的代码运行时间应该超过1/10秒。
- 将您要尝试计时的代码放在函数中，而不是在命令行或脚本内部对其计时。
- 除非您是尝试测量首次成本，否则请多次运行代码。使用timeit函数。
- 请不要在测量性能时执行clear all。有关详细信息，请参阅clear函数。
- 将您的输出分配给一个变量，而不是使其保留默认值ans。

## （二）探查您的代码以改善性能

通过探查，可以测量运行代码所需的时间和确定MATLAB在哪些位置最耗时。在确定哪些函数耗用大部分时间后，您可以对它们进行评估以确定可能的性能改进。您还可以探查您的代码以确定哪些代码行不运行。确定哪些代码行不运行在为代码开发测试时很有用，也可以作为调试工具来帮助隔离代码中的问题。

您可以使用MATLAB探查器Profiler工具，或以编程方式使用[`profile`](https://ww2.mathworks.cn/help/matlab/ref/profile.html)函数来以交互方式探查代码。如果您要探查并行运行的代码，为获得最佳结果，请使用Parallel Computing Toolbox并行探查器[Profiling Parallel Code](https://ww2.mathworks.cn/help/parallel-computing/profiling-parallel-code.html)。有关以编程方式探查代码的详细信息，请参阅profile函数。

### 1. 探查代码

要探查您的代码并改善其性能，请使用以下常规过程：

1. 对您的代码运行探查器。
2. 查看探查摘要结果。
3. 调查函数和单独的代码行。例如，您可能要调查占用了大量时间或调用最频繁的函数和代码行。
4. 保存探查结果。
5. 在您的代码中实施可能的性能改进。例如，如果循环中有一个load语句，您可以将load语句移到循环外以便仅调用一次。
6. 保存文件并运行`clear all`。再次运行探查器并将结果与原始结果比较。
7. 重复上述步骤，继续改进代码的性能。如果代码用大部分时间来调用少数内置函数，则表示您可能已对代码进行了最大程度的优化。

#### (1) 对代码运行探查器

这里写个solvelotka.m函数，它为MATLAB附带的Lotka-Volterra示例求猎物和捕食者数量的峰值，代码如下所示。

```matlab
% solvelotka.m
function [preypeaks,predatorpeaks] = solvelotka(t0, tfinal, y0)
    [~,y] = ode23(@lotka,[t0 tfinal],y0);
    
    preypeaks = calculatepeaks(y(:,1));
    predatorpeaks = calculatepeaks(y(:,2));
    
end

function peaks = calculatepeaks(A)
    [TF,P] = islocalmax(A);
    peaks = P(TF);
end
```

那么，要对一行代码运行探查器，可执行下列步骤。

1. 转至Apps选项卡，在MATLAB下点击探查器图标以打开探查器。或者也可以在命令行窗口中键入`profile viewer`。
2. 转至探查器选项卡，在探查部分，在编辑框中输入您要探查的代码。这里，在编辑框中输入以下语句以探查solvelotka函数，[preypeaks,predatorpeaks] = solvelotka(0,15,[20;20])。
3. 点击运行并计时。

要探查在编辑器中打开的代码文件，请在主窗口的编辑器选项卡的运行部分中，选择运行>运行并计时。探查器探查在当前“编辑器”选项卡中打开的代码文件，并在探查摘要中显示结果。

#### (2) 查看探查的摘要结果

探查完毕后，探查器会在探查摘要中显示结果。所探查的语句也会在命令行窗口中显示为已运行的状态。

<img src="Matlab性能和内存.assets/solvelotka profile viewer.png" style="zoom:50%;" />

| 列       | 描述                                                         |
| :------- | :----------------------------------------------------------- |
| 函数名称 | 由所探查的代码调用的函数的名称。                             |
| 调用     | 被探查的代码调用函数的次数。                                 |
| 总时间   | 在函数中花费的总时间，以秒为单位。函数所耗用的时间包括子函数所耗费的时间。探查器本身会耗用一些时间，该时间也包括在结果中。对于运行时间无足轻重的文件来说，总时间可以是零。 |
| 自用时间 | 函数所耗费的总时间，不包括任何子函数所耗用的时间（以秒为单位）。自用时间还包括探查过程产生的一些开销。 |
| 总时间图 | 以图形方式显示自用时间与总时间的对比情况。                   |

#### (3) 调查函数和单独的代码行

要在代码中找到潜在的改进机会，可以在函数表中寻找耗费了大量时间或调用最频繁的函数。点击函数名称可显示有关该函数的详细信息，包括有关单个代码行的信息。例如，点击solvelotka>calculatepeaks函数。探查器会显示该函数的详细信息。

<img src="Matlab性能和内存.assets/calculatepeaks profile viewer.png" style="zoom:50%;" />

在页面顶部当前函数的名称旁边，探查器显示父函数调用该函数的次数以及在该函数中花费的总时间。点击单个函数后，探查器将在下列部分显示其他信息。

| 部分             | 详细信息                                                     |
| :--------------- | :----------------------------------------------------------- |
| Flame火焰图      | 火焰图直观地显示MATLAB运行所探查的函数花费的时间。该图显示所探查的函数的层次结构，包括子函数（显示在当前函数上方）和父函数（显示在当前函数下方）。用户定义的函数显示为蓝色，MathWorks函数显示为灰色。将鼠标悬停在图中的条形上可查看实际的百分比和时间值以及完整函数名称。点击表示某函数的条形可显示关于该函数的详细信息。 |
| 父级             | 调用所探查函数的函数列表，包括父函数调用所探查函数的次数。点击列表中的函数名称可显示关于该函数的详细信息。 |
| 占用时间最长的行 | 列出所探查函数中耗用最长处理时间的代码行。点击代码行可在函数列表部分的函数其余代码的上下文中查看它。 |
| 子级             | 探查的函数所调用的所有函数列表。点击列表中的函数名称可显示关于该函数的详细信息。 |
| 代码分析器结果   | 所探查函数的问题和可能的改进的列表。                         |
| 范围结果         | 有关MATLAB在探查时所执行的函数中的代码行的代码覆盖率统计信息。要对您的代码执行其他代码覆盖率分析，请参阅[Collect Statement and Function Coverage Metrics for MATLAB Source Code](https://ww2.mathworks.cn/help/matlab/matlab_prog/collect-statement-and-function-coverage-metrics-for-matlab-source-code.html)。 |
| 函数列表         | 函数的源代码，如果它是MATLAB代码文件。对于每行代码，函数列表都包含以下列：每行代码的执行时间、MATLAB执行该代码行的次数、行号、函数的源代码。默认情况下，探查器高亮显示执行时间最长的代码行。高亮显示的颜色越深，执行代码行所耗用的时间越长。 |

### 2. 在命令行窗口探查多个语句

要在命令行窗口中探查多个语句，请执行下列步骤：

1. 转至命令行窗口，键入`profile on`。
2. 输入并运行您要探查的语句。
3. 运行所有语句后，键入`profile off`。
4. 通过键入`profile viewer`打开探查器，查看探查摘要结果。

# 改善性能

## （一）提升性能的方法

要提升代码的性能，请考虑使用这些方法。

### 1. 环境

请注意，共享计算资源的后台进程会降低MATLAB代码的性能。

### 2. 代码结构

在组织您的代码时，请注意以下几点：

- 使用函数代替脚本，函数的速度通常更快。
- 优先使用局部函数，而不是嵌套函数。当函数不需要访问主函数中的变量时，尤其应当使用这种方法。
- 使用模块化编程。要避免产生大文件或者包含不常使用代码的文件，请将代码拆分为简单的综合函数。这种做法可降低首次运行的成本。

### 3. 针对性能的编程做法

考虑使用以下编程做法可改进代码的性能。

- 预分配。您可以考虑预分配数组所需的最大空间量，而不用持续调整数组大小。有关详细信息，请参阅[预分配](https://ww2.mathworks.cn/help/matlab/matlab_prog/preallocating-arrays.html)。
- 向量化。请考虑使用MATLAB矩阵和向量运算，而不是编写基于循环的代码。有关详细信息，请参阅[向量化](https://ww2.mathworks.cn/help/matlab/matlab_prog/vectorization.html)。
- 将独立运算放在循环外。如果代码不使用每个for或while循环迭代进行不同计算，请将其移到循环外以避免冗余计算。
- 当数据类型更改时创建新变量。创建一个新变量，而不是将不同类型的数据分配给现有变量。更改现有变量的类或数组形状需要额外时间进行处理。
- 使用短路运算符。如果可能，请尽量使用短路逻辑运算符`&&`和`||`。短路运算符更有效，因为仅当第一个操作数不能完全确定结果时，MATLAB才会计算第二个操作数。
- 避免使用全局变量。尽量少使用全局变量是一种良好的编程做法，因为全局变量可能会降低MATLAB代码的性能。
- 避免重载内置函数。避免对任何标准MATLAB数据类重载内置函数。
- 避免使用“代码形式的数据”。如果有用于生成包含常量值变量的大段代码（例如，超过500行），请考虑构造变量并将其保存在文件中（例如.mat件或.csv文件）。然后，可以加载变量而不是执行代码来生成这些变量。
- 在后台运行代码。将[`parfeval`](https://ww2.mathworks.cn/help/matlab/ref/parfeval.html)与[`backgroundPool`](https://ww2.mathworks.cn/help/matlab/ref/parallel.backgroundpool.html)结合使用以在后台运行函数。您可以同时运行MATLAB中的其他代码，使您的App响应更快。有关详细信息，请参阅[Run Functions in the Background](https://ww2.mathworks.cn/help/matlab/matlab_prog/run-functions-on-threads.html#mw_045c3602-b005-477e-9ded-8c0d13acf448)。
- 在GPU上或以并行方式运行代码。如果您有Parallel Computing Toolbox许可证，可以通过将`gpuArray`数据传递给支持的函数来在GPU上运行代码，或使用`parfor`循环等方式并行运行代码。有关详细信息，请参阅[Choose a Parallel Computing Solution](https://ww2.mathworks.cn/help/parallel-computing/choosing-a-parallel-computing-solution.html)（Parallel Computing Toolbox）。

### 4. 有关特定MATLAB函数的提示

在编写性能关键代码时，请考虑有关特定MATLAB函数的以下提示。

- 请避免清除不是必须清除的代码。不要以编程方式使用`clear all`。有关详细信息，请参阅[`clear`](https://ww2.mathworks.cn/help/matlab/ref/clear.html)，它从工作区中删除项目、释放系统内存。
- 请避免使用查询MATLAB状态的函数，例如`inputname`、`which`、`whos`、`exist(var)`和`dbstack`。运行时自检会耗费大量计算资源。
- 请避免使用`eval`、`evalc`、`evalin`和`feval(fname)`等函数。尽可能使用到feval的函数句柄输入，如果您有函数句柄，不需要使用feval，因为直接调用函数句柄的结果是相等的。从文本间接计算MATLAB表达式会耗费大量计算资源。
- 请尽可能避免以编程方式使用`cd`、`addpath`和`rmpath`。在运行时更改MATLAB路径会导致重新编译代码。

## （二）预分配

每次经过for和while循环时，这些循环都会递增数据结构体的大小，这会对性能和内存的使用产生不利影响。反复重新调整数组大小往往需要MATLAB花费额外的时间来寻找更大的连续内存块，然后将数组移入这些块中。通常，您可以通过预分配数组所需的最大空间量来缩短代码的执行时间。

下面的代码显示了创建标量变量x，然后在for循环中逐步增加x大小所需的时间量。

```matlab
tic
x = 0;
for k = 2:1000000
   x(k) = x(k-1) + 5;
end
toc
% Elapsed time is 0.276065 seconds.
```

如果您为x预分配一个$1\times1000000$的内存块并将其初始化为零，则代码的运行速度更快，这是因为无需反复为不断增长的数据结构体重新分配内存。

```matlab
tic
x = zeros(1,1000000);
for k = 2:1000000
   x(k) = x(k-1) + 5;
end
toc
% Elapsed time is 0.013246 seconds.
```

对要初始化的数组类型使用适当的预分配函数，如下所列出的。

- 对于数值数组，使用`zeros`。
- 对于字符串数组，使用`strings`。
- 对于元胞数组，使用`cell`。
- 对于表数组，使用`table`。

当您预分配一个内存块来存储除double外的某类型的矩阵时，避免使用以下方法：

```matlab
A = int8(zeros(100));
```

该语句预分配了一个$100\times100$的int8矩阵，方法是先创建一个由double值组成的满矩阵，然后将每个元素转换为int8。而如果以int8值的形式，直接创建数组可节省时间和内存，如下所示：

```matlab
A = zeros(100,'int8');
```

此外，这里有一些关于预分配的示例。[重构和重新排列数组](https://ww2.mathworks.cn/help/matlab/math/reshaping-and-rearranging-arrays.html)，[为元胞数组预分配内存](https://ww2.mathworks.cn/help/matlab/matlab_prog/preallocate-memory-for-a-cell-array.html)，[使用分类数组访问数据](https://ww2.mathworks.cn/help/matlab/matlab_prog/access-data-using-categorical-arrays.html)，[预分配图形对象数组](https://ww2.mathworks.cn/help/matlab/creating_plots/preallocate-arrays.html)，[构造对象数组](https://ww2.mathworks.cn/help/matlab/matlab_oop/creating-object-arrays.html)。

## （三）向量化

### 1. 向量化的应用

MATLAB针对涉及矩阵和向量的运算进行了优化。修改“基于循环且面向标量”的代码以使用“MATLAB矩阵和向量运算”的过程称为向量化。向量化代码的运行速度通常比相应的、包含循环的代码更快。

#### (1) 实现代码向量化以进行常规计算

以下代码计算1001个从0到10之内的值的正弦：

```matlab
i = 0;
for t = 0:.01:10
    i = i + 1;
    y(i) = sin(t);
end
```

这是同一代码的向量化版本：

```matlab
t = 0:.01:10;
y = sin(t);
```

通常来说，第二个代码示例的执行速度比第一个快，是MATLAB的更高效的用法。

#### (2) 针对特定任务实行代码向量化

此代码计算某向量每五个元素的累加和：

```matlab
x = 1:10000;
ylength = (length(x) - mod(length(x),5)) / 5;
y(1:ylength) = 0;
for n = 5:5:length(x)
    y(n/5) = sum(x(1:n));
end 
```

使用向量化，您可以编写一个更精确的MATLAB过程，以下代码显示了完成此项任务的一种方法：

```matlab
x = 1:10000;
xsums = cumsum(x);
y = xsums(5:5:length(x)); 
```

### 2. 数组运算（逐元素操作）

数组运算符对数据集中的所有元素执行相同的运算，即**逐元素操作（element-wise operation）**，这些类型的运算用于重复计算。

例如，通过记录各圆锥体的直径D和高度H来收集这些圆锥体的体积V，如果只收集一个圆锥体的信息，则可以只计算该圆锥体的体积：

```matlab
V = 1/12 * pi * (D^2) * H;
```

现在，收集10000个圆锥体的信息，向量D和H均包含10000个元素，并且需要计算10000个体积。在绝大多数编程语言中，需要设置类似于以下MATLAB代码的循环：

```matlab
for n = 1:10000
   V(n) = 1/12 * pi * (D(n)^2) * H(n);
end
```

借助MATLAB，您可以使用与标量情形类似的语法对每个向量元素执行计算：

```matlab
% Vectorized Calculation
V = 1/12 * pi * (D.^2) .* H;
```

也即，在运算符`*`、`/`和`^`之前放置句点`.`，将其转换为数组运算符。

借助数组运算符，还能够组合不同维度的矩阵。这样自动的隐式扩展大小为1的维度，有助于对网格创建、矩阵和向量运算等进行向量化处理。

假设，要计算包含两个变量x和y的函数F，如$F(x,y)=x\exp(-x^2-y^2)$之类。要在x和y向量中的每个点组合处计算该函数，需要定义网格值。

对于此任务，应避免使用循环来循环访问点组合。在这种情况下，如果一个向量为列，而另一个向量为行，则当这些向量与数组运算符配合使用时，MATLAB将会自动构造网格。

在此示例中，x是一个$21\times1$向量，y是一个$1\times16$向量，因此该运算会通过扩展x的第2个维度和y的第2个维度来生成一个$21\times16$矩阵。

```matlab
x = (-2:0.2:2)';   % 21 ×  1
y = -1.5:0.2:1.5;  %  1 × 16
F = x.*exp(-x.^2-y.^2);  % 21 × 16
```

如果要显式创建网格，可以使用[`meshgrid`](https://ww2.mathworks.cn/help/matlab/ref/meshgrid.html)和[`ndgrid`](https://ww2.mathworks.cn/help/matlab/ref/ndgrid.html)函数。

### 3. 逻辑数组运算

批量处理数组的逻辑扩展，可实现（关系运算符）的**比较向量化**，MATLAB比较器接受向量输入并返回向量输出。

例如，假设在从10000个圆锥体中收集数据时，记录中有多个负值作为直径，可以使用>=运算符确定向量中的哪些值是有效的：

```matlab
D = [-0.2 1.0 1.5 3.0 -1.0 4.2 3.14];
D >= 0
%{
ans =
     0     1     1     1     0     1     1
%}
```

可以直接利用MATLAB的逻辑索引功能，来选择有效的圆锥体体积Vgood，其对应的D元素是非负的：

```matlab
V = 1/12 * pi * (D.^2) .* H;
Vgood = V(D >= 0);
```

MATLAB允许您分别使用`all`和`any`函数对整个向量的元素执行逻辑AND或OR运算。例如，如果D的所有值都小于零，则引发以下警告：

```matlab
if all(D < 0)
   warning('All values of diameter are negative.')
   return
end
```

MATLAB也可以比较两个大小兼容的向量，并允许施加更多限制。例如，下述代码查找V为非负且D大于H的所有值：

```matlab
V( (V >= 0) & (D > H) )
```

为便于比较，MATLAB含特殊值来表示溢出和未定义的运算符，例如`Inf`和`NaN`，而逻辑运算符`isinf`和`isnan`有助于针对这些特殊值执行逻辑测试。例如，从计算中排除NaN值的代码如下所示：

```matlab
x = [2 -1 0 3 NaN 2 NaN 11 4 Inf];
xvalid = x(~isnan(x))
```

注意，比较Inf==Inf返回true；但NaN==NaN始终返回false。

### 4. 矩阵运算

在使代码向量化时，通常需要构造一个具有特定大小或结构的矩阵。有多种方式可用来创建均匀矩阵。

例如，可能需要一个包含相等元素的$5\times5$矩阵：

```matlab
A = ones(5,5) * 10;
```

或者，可能需要一个包含重复值的矩阵：

```matlab
v = 1:5;
A = repmat(v,3,1)
%{
A =
     1     2     3     4     5
     1     2     3     4     5
     1     2     3     4     5
%}
```

函数`repmat`可以灵活地根据较小的矩阵或向量来构建矩阵，它通过重复输入矩阵来创建矩阵，类似于Numpy中的tile()函数。

### 5. 排序、设置和计数运算

在许多应用程序中，对向量的某个元素执行的计算取决于同一向量中的其他元素。例如，向量x可能表示一个集合set，如何循环访问一个集而不使用for或while循环并不明显。如果采用向量化代码，则此过程变得更加清晰而且语法也不再过于冗长。

#### 1. 消除冗余元素

有许多不同的方法可以用来查找向量的冗余元素。

一种方法涉及到函数`diff`，它会计算后一个元素与前一个元素之差，生成向量的元素个数比原向量少一个。在对向量元素进行排序后，当对该向量使用diff函数时，相等的邻接元素会产生零项。因为diff(x)生成的向量的元素数比x少一个，所以还必须添加一个不等于该集中的任何其他元素的元素，使用NaN即可始终满足此条件。最后，可以使用逻辑索引来选择该集中的唯一元素，如下所示。

```matlab
x = [2 1 2 2 3 1 3 2 1 3];
x = sort(x);
df = diff([x,NaN]);
y = x(df~=0)
%{
y =
     1     2     3
%}
```

此外，也可以使用`unique`函数完成同样的操作：

```
y=unique(x);
```

但注意，unique函数可能提供超出需要的功能并且降低代码的执行速度。

#### 2. 计算向量中的元素数

使用`find`函数可以返回一个向量中所有非零值的索引。

可以计算某元素在向量中的出现次数，而不只是返回x的集或子集。该向量排序后，可以使用find函数来确定diff(x)中零值的索引并显示元素的值在何处发生了改变。find函数中后续索引之间的差可指明特定元素的出现次数，如下所示。

```matlab
x = [2 1 2 2 3 1 3 2 1 3];
x = sort(x);
df = diff([x,max(x)+1]);
count = diff(find([1,df]))
y = x(find(df))
%{
count =
     3     4     3
y =
     1     2     3
%}
```

find函数不返回NaN元素的索引，可以使用isnan和isinf函数计算NaN和Inf值的数目，如下所示。

```
count_nans = sum(isnan(x));
count_infs = sum(isinf(x));
```

### 6. 向量化的常用函数

| 函数                                                         | 说明                                 |
| :----------------------------------------------------------- | :----------------------------------- |
| [`all`](https://ww2.mathworks.cn/help/matlab/ref/all.html)   | 确定所有数组元素均为非零元素还是true |
| [`any`](https://ww2.mathworks.cn/help/matlab/ref/any.html)   | 确定是否有任何数组元素非零           |
| [`cumsum`](https://ww2.mathworks.cn/help/matlab/ref/cumsum.html) | 累积和                               |
| [`diff`](https://ww2.mathworks.cn/help/matlab/ref/diff.html) | 差分和近似导数                       |
| [`find`](https://ww2.mathworks.cn/help/matlab/ref/find.html) | 查找非零元素的索引和值               |
| [`ind2sub`](https://ww2.mathworks.cn/help/matlab/ref/ind2sub.html) | 线性索引的下标                       |
| [`ipermute`](https://ww2.mathworks.cn/help/matlab/ref/ipermute.html) | N维数组的逆置换维度                  |
| [`logical`](https://ww2.mathworks.cn/help/matlab/ref/logical.html) | 将数值转换为逻辑值                   |
| [`meshgrid`](https://ww2.mathworks.cn/help/matlab/ref/meshgrid.html) | 二维和三维空间中的矩形网格           |
| [`ndgrid`](https://ww2.mathworks.cn/help/matlab/ref/ndgrid.html) | N维空间中的矩形网格                  |
| [`permute`](https://ww2.mathworks.cn/help/matlab/ref/permute.html) | 重新排列N维数组的维度                |
| [`prod`](https://ww2.mathworks.cn/help/matlab/ref/prod.html) | 数组元素的乘积                       |
| [`repmat`](https://ww2.mathworks.cn/help/matlab/ref/repmat.html) | 重复数组副本                         |
| [`reshape`](https://ww2.mathworks.cn/help/matlab/ref/reshape.html) | 重构数组                             |
| [`shiftdim`](https://ww2.mathworks.cn/help/matlab/ref/shiftdim.html) | 移动维度                             |
| [`sort`](https://ww2.mathworks.cn/help/matlab/ref/sort.html) | 对数组元素排序                       |
| [`squeeze`](https://ww2.mathworks.cn/help/matlab/ref/squeeze.html) | 删除单一维度                         |
| [`sub2ind`](https://ww2.mathworks.cn/help/matlab/ref/sub2ind.html) | 将下标转换为线性索引                 |
| [`sum`](https://ww2.mathworks.cn/help/matlab/ref/sum.html)   | 数组元素总和                         |

# 确定和降低内存要求

## （一）MATLAB如何分配内存

通过了解MATLAB如何分配内存，编写更高效使用内存的代码。本节提供关于MATLAB在处理变量时如何分配内存的信息。这些信息，就像关于MATLAB内部如何处理数据的任何信息一样，在以后的版本中可能会变更。

### 1. 为数组分配内存

将数值数组或字符数组分配给变量时，MATLAB会分配一个连续的内存块，并将数组数据存储在该内存块中。此外，还将有关数组数据的信息（如它的类和维度）存储在一个单独的小内存块中，称为**标头**。对于多数数组，存储标头所需的内存可忽略不计。然而，将大数据集存储在较少几个大数组中，比存储在较多几个小数组中可能更理想。这是因为较少几个数组需要较少的数组标头。

如果您向现有数组中添加新元素，MATLAB会按照使内存存储保持连续的方式扩展该数组。这通常需要查找新的足以容纳扩展后的数组的内存块。随后，MATLAB将该数组的内容从其原始位置复制到内存中这一新块中，向该块中的数组添加新元素，并释放原始数组在内存中的位置。

如果您从现有数组中删除元素，MATLAB会清除已删除的元素，然后调整数组的后序元素，使其在原始内存位置变得紧凑，来使内存存储连续。

#### (1) 复制数组

将数组分配给第二个变量时（例如，当执行B=A时），MATLAB不会立即分配新内存。此时，它会创建数组引用副本。只要不修改A和B引用的内存块的内容，就不需要存储多个数据副本。但是，如果使用A或B修改内存块的任何元素，MATLAB就会分配新内存，将数据复制到其中，然后修改所创建的副本。

在Windows系统上，[`memory`](https://ww2.mathworks.cn/help/matlab/ref/memory.html)函数可用于检查内存详细信息。要了解复制数组如何影响Windows系统的内存使用量，可创建一个函数memUsed，调用memory方法，以MB为单位返回MATLAB进程使用的内存量，如下所示。

```matlab
function y = memUsed
	usr = memory;
	y = usr.MemUsedMATLAB/1e6;
end
```

用memUsed以显示当前内存使用量。例如，创建一个$2000\times2000$数值数组，并观察内存使用量的变化，该数组使用大约32MB的内存。

```matlab
format shortG
memUsed
% ans = 3966.1
A = magic(2000);
memUsed
% ans = 3998.1
```

在B中制作A的副本，由于不需要使用数组数据的两个副本，MATLAB仅创建数组引用的一个副本，这不需要额外增加大量内存，如下所示。

```matlab
B = A;
memUsed
% ans = 3998.1
```

现在通过删除B的行数的一半来修改它，由于A和B不再指向同一数据，MATLAB必须为B分配一个单独的内存块。结果，MATLAB进程使用的内存量增加了B的大小，约为16MB（即A所需的32MB的一半）。

```matlab
B(1001:2000,:) = [];
memUsed
% and = 4014.1
```

### 2. 函数参数

在MATLAB的函数调用中，处理传递参数的方式与处理复制数组的方式相同。将变量传递给函数时，实际传递的是对该变量所表示数据的引用。只要被调用的函数未修改数据，调用者的变量和被调用函数中的变量就指向内存中的同一位置。如果被调用函数修改输入数据的值，则MATLAB将在内存中的新位置创建原始变量的副本，用修改后的值更新该副本，并将被调用函数中的输入参数指向此新位置。

例如，假设有函数myfun，它修改传递给它的数组的值，则MATLAB在内存中的新位置生成A的副本，将变量X设置为对此副本的引用，然后将X的一行设置为零；而调用者的A变量所引用的数组保持不变，如下所示。

```matlab
% myfun.m
function myfun(X)
    X(4,:) = 0;
    disp(X)
end
```

```matlab
A = magic(5);
myfun(A)
```

如果调用者需要被myfun函数修改后的值，则需要以被调用函数的输出形式返回更新的数组。

### 3. 数据类型和内存

MATLAB的各数据类型的内存要求不同。通过了解MATLAB如何处理各种数据类型，有助于减少代码使用的内存量。

#### (1) 数值数组

对整数来说，MATLAB分别对8位、16位、32位、64位有符号和无符号整数分配1、2、4、8个字节。

对浮点数来说，MATLAB以单精度single或双精度double格式表示浮点数，单精度single使用4个字节存储，双精度double使用8个字节存储。single数值的精度要低于double数值的精度。

在MATLAB中，double是默认的数值数据类型，它可为大多数计算任务提供足够的精度。

#### (2) 结构体和元胞数组

数值数组必须存储在连续内存块中，但结构体和元胞数组可以存储在不连续内存块中。对于结构体和元胞数组，MATLAB不仅为数组创建一个标头，还为结构体的每个字段及元胞数组的每个元胞创建一个标头。因此，存储结构体或元胞数组所需的内存量不仅取决于其包含的数据量，还取决于其构造方式。

例如，要存储$100\times50$的彩色RGB图片。

假设有标量结构体S1，它包含字段R、G、B，其中每个字段包含一个$100\times50$数组；则S1需要1个标头描述该标量结构体，3个标头描述每个字段唯一名称，3个标头描述每个字段，共需要7个标头。

```matlab
S1.R = zeros(100,50);
S1.G = zeros(100,50);
S1.B = zeros(100,50);
```

同样问题，假设有一个$100\times50$结构体数组S2，其中每个元素都有标量字段R、G、B；则S2需要1个标头描述该结构体数组，3个标头描述每个字段唯一名称，15000个标头描述每个元素的每个字段，共需要15004个标头。

```matlab
for i = 1:100
    for j = 1:50
        S2(i,j).R = 0;
        S2(i,j).G = 0;
        S2(i,j).B = 0;
    end
end
```

使用whos函数比较在64位系统上分配给S1和S2的内存量，尽管S1和S2包含相同的数据，但S1使用的内存明显更少。

```matlab
whos S1 S2
%{
  Name        Size              Bytes  Class     Attributes
  S1          1x1              120528  struct              
  S2        100x50            1800192  struct              
%}
```

#### (3) 复数数组

MATLAB使用复数的交错存储表示，其中实部和虚部一起存储在一个连续内存块中。如果创建复数数组的副本，然后仅修改该数组的实部或虚部，MATLAB会创建一个同时包含实部和虚部的数组。

有关内存中复数表示的详细信息，请参阅[MATLAB Support for Interleaved Complex API in MEX Functions](https://ww2.mathworks.cn/help/matlab/matlab_external/matlab-support-for-interleaved-complex.html)。

#### (4) 稀疏矩阵

使用稀疏存储来存储非零元素很少的矩阵是一种很好的做法。当一个满矩阵有少量非零元素时，将矩阵转换为稀疏存储通常会改善内存使用量和代码执行时间。可以使用[`sparse`](https://ww2.mathworks.cn/help/matlab/ref/sparse.html)函数将满矩阵转换为稀疏存储。在稀疏存储中，相同的数据使用的内存量要少得多。

例如，假设矩阵A是$1000\times1000$满存储单位矩阵，将B创建为A的稀疏副本，如下所示。

```matlab
A = eye(1000);
B = sparse(A);
whos A B
%{
  Name         Size                Bytes  Class     Attributes
  A         1000x1000            8000000  double              
  B         1000x1000              24008  double    sparse    
%}
```

### 4. 使用大数据集

当您处理大型数据集时，反复调整数组大小可能导致程序耗尽内存。如果您扩展数组使其超过其原始位置的可用连续内存，MATLAB必须创建该数组的副本，并将副本移至具有足够空间的内存块中。在此过程中，内存中有原始数组的两个副本。这会暂时使数组所需的内存量翻倍，并增加您的程序出现内存不足的风险。

您可以通过预分配数组所需的最大空间量来改善内存使用量和代码执行时间。有关详细信息，请参阅[预分配](https://ww2.mathworks.cn/help/matlab/matlab_prog/preallocating-arrays.html)。

## （二）高效使用内存的策略

减少程序中的内存使用，使用适当的数据存储，避免内存碎片化以及回收使用的内存。本节介绍在MATLAB中高效使用内存的几种方法。

### 1. 使用适当的数据存储

MATLAB提供了不同大小的数据类（例如double和uint8），因此您无需使用大型类存储较小的数据段。例如，与使用double相比，使用uint8类存储1000个无符号小整数值所用的内存少7KB。

#### (1) 使用相应的数值类

您应在MATLAB中使用的数值类取决于您的预期操作。默认类double可提供最佳精度，但存储每个元素需要8字节内存。

如果您计划执行复杂的数学运算（例如线性代数），则您必须使用浮点类，例如double或single。可使用single类执行的操作存在某些限制，但多数MATLAB数学运算都受支持。如果您只需执行简单的算术运算并将原始数据表示为整数，则您可以在MATLAB中使用整数类。

下面是数值类、内存要求（以字节为单位）及支持的运算的列表。

| 类（数据类型） | 字节 | 支持的运算         |
| :------------- | :--- | :----------------- |
| single         | 4    | 绝大多数的数学运算 |
| double         | 8    | 所有数学运算       |
| logical        | 1    | 逻辑/条件运算      |
| int8, uint8    | 1    | 算术和某些简单函数 |
| int16, uint16  | 2    | 算术和某些简单函数 |
| int32, uint32  | 4    | 算术和某些简单函数 |
| int64, uint64  | 8    | 算术和某些简单函数 |

#### (2) 减少存储数据时的开销

MATLAB数组（在内部作为`mxArrays`实现）需要一定的空间来将有关数据的元数据信息（例如类型、维度和属性）存储在内存中，每个数组大约需要104字节。仅当有大量（如数百或数千）较小的`mxArrays`（如标量）时，此开销才成问题。[`whos`](https://ww2.mathworks.cn/help/matlab/ref/whos.html)命令列出了变量所用的内存，但不包括此开销。

因为简单数值数组（包括一个`mxArray`）的开销最少，所以您应该尽可能使用它们。当数据太复杂而无法存储在简单数组（或矩阵）中时，您可以使用其他数据结构体。

结构体的每个字段需要类似的开销量。包含许多字段和较少内容的结构体具有较大的开销，因此应避免使用这样的结构体。由包含数值标量字段的结构体组成的大型数组所需的内存，要比具有包含较大数值数组的字段的结构体更多。

元胞数组由每个元素的单独`mxArrays`组成。因此，包含许多小元素的元胞数组具有较大的开销。

另请注意，虽然MATLAB将数值数组存储在连续内存中，但结构体和元胞数组则不然。有关详细信息，请参阅上一节[MATLAB 如何分配内存](https://ww2.mathworks.cn/help/matlab/matlab_prog/memory-allocation.html)。

#### (3) 将数据导入相应的MATLAB类

当使用[`fread`](https://ww2.mathworks.cn/help/matlab/ref/fread.html)读取二进制文件中的数据时，常见的错误是，仅指定该文件所存储数据的类，而不当读取数据后，其在工作区（内存）中组织时所使用的类。

因此，即使从文件以uint8类型读取数据，若不指定读出的数据如何存储在工作区中，则工作区会使用默认的double存储所读到的uint8数据，如下所示。

```matlab
fid = fopen('large_file_of_uint8s.bin', 'r');
a = fread(fid, 1e3, 'uint8');  %  Requires 8k
whos a
%{
  Name         Size            Bytes  Class    Attributes
  a         1000x1              8000  double
%}
a = fread(fid, 1e3, 'uint8=>uint8');  % Requires 1k
whos a
%{
  Name         Size            Bytes  Class    Attributes
  a         1000x1              1000  uint8
%}
```

#### (4) 尽可能使数组稀疏

如果您的数据包含许多零，请考虑使用稀疏数组，这样仅存储非零元素。以下示例比较稀疏存储和满存储的要求：

```matlab
A = eye(1000);        % Full matrix with ones on the diagonal
As = sparse(A);       % Sparse matrix with only nonzero elements
whos
%{
  Name         Size                Bytes  Class     Attributes

  A         1000x1000            8000000  double              
  As        1000x1000              24008  double    sparse  
%}
```

可以看到，该数组只需要大约24KB即可存储为稀疏数组，但要存储为满矩阵，则需要大约8MB。

通常，对于包含nnz个非零元素和ncol列的双精度型稀疏数组，所需的内存为$16\times\text{nnz}+8\times\text{ncol}+8$个字节（在64位计算机上）。根据该公式，可推测出MATLAB存储稀疏矩阵的格式为压缩稀疏列CSC存储。

请注意，MATLAB支持对稀疏数组执行大多数（但不是全部）数学运算。

### 2. 避免临时性的数据副本

避免创建不必要的临时性数据副本，以显著减少所需的内存量。

#### (1) 避免创建临时数组

避免创建大型临时变量，并在不再需要这些临时变量时清除它们。例如，以下代码创建由零组成的、存储为临时变量A的数组，然后将A转换为单精度：

```matlab
A = zeros(1e6, 1);
As = single(A);
```

使用一个命令来执行两个操作可更高效地使用内存：

```matlab
A = zeros(1e6, 1, 'single');
```

使用[`repmat`](https://ww2.mathworks.cn/help/matlab/ref/repmat.html)函数、数组预分配和[`for`](https://ww2.mathworks.cn/help/matlab/ref/for.html)循环是“处理非双精度数据而不需要内存中的临时存储”的其他方法。

#### (2) 使用嵌套函数减少传递的参数

处理大型数据集时，注意MATLAB会创建输入变量的临时副本（如果被调用函数修改其值），这会暂时使存储数组所需的内存翻倍，从而导致MATLAB在没有足够内存时生成错误。在此情形下使用较少的内存的一种方法是使用嵌套函数。嵌套函数共享所有外部函数的工作区，为嵌套函数提供对其通常范围之外的数据的访问权。

在如下示例中，嵌套函数setrowval可直接访问外部函数myfun的工作区，从而无需在函数调用中传递变量副本。当setrowval修改A的值时，它在调用者的工作区中修改它，无需使用额外内存为所调用函数存储一个单独数组，且无需返回A的修改后的值，如下所示。

```matlab
function myfun
	function setrowval(row, value)
    	A(row,:) = value;
    end
    
    A = magic(500);
    setrowval(400, 0)
    disp('The new value of A(399:401,1:10) is')
    A(399:401, 1:10)
end
```

### 3. 回收使用的内存

增加可用内存量的一种简单方法是清除不再使用的大型数组。

如果您的程序生成非常大量的数据，请考虑定期将数据写入磁盘，在保存该部分数据后，使用[`clear`](https://ww2.mathworks.cn/help/matlab/ref/clear.html)函数从内存中删除变量并继续生成数据。

当您重复或以交互方式处理非常大的数据集时，请首先清除旧变量以为新变量腾出空间。否则，MATLAB需要等大小的临时存储才能覆盖此变量，例如。

```matlab
a = rand(1e5);
b = rand(1e5);
% Out of memory. Type "help memory" for your options.
```

```matlab
clear a
a = rand(1e5);  % New array 
```

## （三）避免不必要的数据副本

MATLAB可以在通过值传递函数输入参数时，应用内存优化。

### 1. 将值传递给函数

当使用输入参数调用函数时，MATLAB会将值从调用者的工作区复制到被调用函数的参数变量中。但MATLAB会应用各种技术，避免在非必要时复制这些值。

MATLAB没有像C++之类的语言那样提供定义值引用的方法，但MATLAB允许多个输出和多个输入参数，因此您知道有哪些值要输入到函数中，以及要从函数中输出哪些值。

#### (1) 传入时复制

如果函数未修改输入参数，则MATLAB不会复制输入变量所包含（指向）的值。

例如，假设您将一个大型数组传递给函数。

```matlab
A = rand(1e7, 1);
B = f1(A);
```

函数f1将输入数组X中的每个元素乘以1.5，并将结果赋给变量Y。

```matlab
function Y = f1(X)
	Y = X .* 1.5;  % X is a shared copy of A
end
```

由于此函数并未修改输入值，因此局部变量X和调用者工作区中的变量A共享数据。执行f1后，赋给A的值不变，调用者工作区中的变量B包含按元素相乘的结果。输入通过值进行传递，但是，调用f1时没有制作副本。

该示例的另一个函数如下。函数f2会修改输入变量的本地副本，从而导致本地副本与输入A不共享。现在，函数中的X值在调用者工作区中是输入变量A的独立副本。当f2将结果返回到调用方的工作区时，局部变量X将被销毁。

```matlab
A = rand(1e7, 1);
B = f2(A);
```

```matlab
function Y = f2(X)
	X = X .* 1.5;  % X is an independent copy of A
	Y = X;         % Y is a shared copy of X
end
```

#### (2) 传递表达式作输入

您可以将一个函数的返回值用作另一个函数的输入参数。

例如，使用rand函数的返回值直接作为函数f2的输入参数。

```matlab
B = f2(rand(1e7, 1));
```

保存rand返回值的唯一变量是函数f2的工作区中的临时变量X。

在调用者的工作区中，**不存在**这些值的共享副本或独立副本。直接传递函数输出可以节省在被调用函数中创建输入值副本所需的时间和内存。当输入值不会再次使用时，可以使用此方法。

#### (3) 就地赋值

当您不需要保留原始输入值时，可以将函数的输出赋给与输入相同的变量。

```matlab
A = f2(A);
```

就地赋值发生在前面介绍的“传入时复制”行为之后。MATLAB在某些条件下可以应用内存优化，请参考以下示例。

函数canBeOptimized函数在变量A中生成一个很大的随机数数组，然后它调用局部函数fLocal，传递A作为输入，并将局部函数的输出赋给相同的变量名称。

```matlab
function canBeOptimized
	A = rand(1e7, 1);
	A = fLocal(A);
end

function X = fLocal(X)
	X = X .* 1.5;
end
```

由于对局部函数的调用A=fLocal(A)将输出赋给变量A，因此MATLAB在执行函数的过程中不需要保留A的原始值，故对fLocal内的X所做的修改不会产生数据副本（在原来A上修改即可）。赋值X=X.*1.5只是就地修改X，不会为乘法结果分配新数组。如此，消除局部函数中的副本可以节省内存并提高大型数组的执行速度。

存在几项其他限制。在函数引发错误时可能会用到变量的情况下，MATLAB不能应用内存优化。因此，这种优化不适用于脚本、命令行、对[`eval`](https://ww2.mathworks.cn/help/matlab/ref/eval.html)的调用以及`try/catch`代码块内的代码。此外，MATLAB在调用函数的执行过程中，当原始变量可直接访问时，不会应用内存优化。例如，如果fLocal是嵌套函数，则MATLAB无法应用优化，因为它会与父函数共享变量。最后，当指定的变量声明为全局变量或持久变量时，MATLAB也不会应用内存优化。

注意，当MATLAB对赋值语句应用就地优化时，赋值左侧的变量会设置为一种临时状态，这种状态使得左侧变量在MATLAB执行赋值语句的右侧语句之前都不可访问。如果MATLAB在调试器中停止，而此时还未将语句右侧的执行结果赋给变量，这时检查左侧的变量就会报错，指示该变量不可用。

### 2. 为什么使用传值语义

MATLAB在向函数传递参数以及从函数返回值时使用传值语义。在某些情况下，传值会在被调用函数中生成原始值的副本。但是，传值语义也有一些好处。

当调用函数时，您知道输入变量不会在调用者工作区中被修改。因此，不需要只是为了防止这些值可能被修改而在函数内或在调用位置创建输入副本。只有赋给返回值的变量会被修改。

此外，如果通过引用传递变量的函数中发生错误，可以避免出现损坏工作区变量的可能性。

### 3. 句柄对象

有一些特殊的对象称为句柄。保存同一个句柄副本的所有变量都可以访问和修改同一个底层对象。在特定的情况下，即当对象表示物理对象（例如窗口、绘图、设备或人）而不是数学对象（如数字或矩阵）时，句柄对象很有用。

句柄对象从[`handle`](https://ww2.mathworks.cn/help/matlab/ref/handle-class.html)类派生而来，该类提供事件（event）和监听方法（listener）、析构函数方法以及动态属性支持等函数。有关值和句柄的详细信息，请参阅[句柄类和值类的比较](https://ww2.mathworks.cn/help/matlab/matlab_oop/comparing-handle-and-value-classes.html)和[Which Kind of Class to Use](https://ww2.mathworks.cn/help/matlab/matlab_oop/which-kind-of-class-to-use.html)。

## （四）解决内存不足错误

当MATLAB无法分配请求的内存时，对错误进行故障排除。

### 1. 问题与可能的解决方案

当您的代码处理大量数据或不能高效使用内存时，MATLAB可能会因数组大小不合理或内存不足而生成错误。MATLAB具有内置防护机制，可防止创建过大的数组。例如，以下代码会导致错误，因为MATLAB无法创建包含请求的元素数的数组。

```matlab
A = rand(1e9);
% Requested array exceeds the maximum possible variable size.
```

默认情况下，MATLAB可使用100%的计算机RAM（不包括虚拟内存）来为数组分配内存，如果数组大小超过该阈值，则MATLAB生成错误。例如，以下代码尝试创建一个大小超过最大数组大小限制的数组。

```matlab
B = rand(1e6);
% Requested 1000000x1000000 (7450.6GB) array exceeds maximum array size preference (63.7GB).
% This might cause MATLAB to become unresponsive.
```

如果在MATLAB工作区预设项中关闭数组大小限制，则尝试创建一个不合理的大型数组可能导致MATLAB耗尽内存，或可能由于内存分页过多（即在RAM和磁盘之间移动内存页），导致MATLAB甚至您的计算机没有响应。

```matlab
B = rand(1e6);
% Out of memory.
```

可能的解决方案。无论您因何种原因遇到内存限制，MATLAB都提供了相应的解决方案，您可以根据自己的情况和目标进行选择。例如，您可以改进代码利用内存的方式，利用专用数据结构体（如数据存储和tall数组），利用计算集群中的池化资源，或调整您的设置和预设项。注意，此处介绍的解决方案是针对MATLAB的。要优化系统范围的内存性能，请考虑为您的计算机添加更多物理内存 (RAM) 或在操作系统级别进行调整。

有多种具体的解决方案，其中一些方案已在前述几节中详细介绍过，此处仅介绍一些新的策略。

### 2. 仅加载需要的数据

解决内存问题的一种方法是，只将大型数据集中所需的数据导入到MATLAB中。从数据库等源中导入时，数据集大小通常不是问题，因为您可以显式搜索匹配查询的元素。但在加载大型简单文本或二进制文件时这是个常见问题。

[`datastore`](https://ww2.mathworks.cn/help/matlab/ref/datastore.html)函数允许您以增量方式处理大型数据集。在您需要一次只将数据集的某些部分加载到内存中时，datastore数据存储很有帮助。由于datastore同时支持本地和远程数据位置，因此您处理的数据不需要位于您用于分析这些数据的计算机上，有关详细信息，请参阅[处理远程数据](https://ww2.mathworks.cn/help/matlab/import_export/work-with-remote-data.html)。

要创建数据存储，请提供文件名，或包含一系列具有相似格式的文件的目录。例如，对于单个文件，使用以下方式。

```matlab
ds = datastore("path/to/file.csv");
```

或者对于一个文件夹中的一系列文件，使用以下方式。

```matlab
ds = datastore("path/to/folder/");
```

您也可以使用通配符`*`来选择特定类型的所有文件。

```matlab
ds = datastore("path/to/*.csv");
```

数据存储支持多种文件格式（表格数据、图像、电子表格等），有关详细信息，请参阅[Select Datastore for File Format or Application](https://ww2.mathworks.cn/help/matlab/import_export/select-datastore-for-file-format-or-application.html)。

除了数据存储之外，MATLAB还提供其他几个函数来加载部分文件，下表按所处理的文件类型摘要显示了这些函数。

| 文件类型              | 部分加载                                                     |
| :-------------------- | :----------------------------------------------------------- |
| MAT文件               | 通过对使用[`matfile`](https://ww2.mathworks.cn/help/matlab/ref/matlab.io.matfile.html)函数创建的对象进行索引来加载部分变量，有关详细信息，请参阅[在MAT文件中保存和加载部分变量](https://ww2.mathworks.cn/help/matlab/import_export/load-parts-of-variables-from-mat-files.html)。 |
| 文本                  | 使用[`textscan`](https://ww2.mathworks.cn/help/matlab/ref/textscan.html)函数可通过仅读取选定的列和行来访问大型文本文件的一部分。如果您使用`textscan`指定行数或重复格式数字，MATLAB会提前计算所需的确切内存量。 |
| 二进制                | 您可以使用低级别二进制文件I/O函数（例如[`fread`](https://ww2.mathworks.cn/help/matlab/ref/fread.html)）访问具有已知格式的任何文件的一部分。对于未知格式的二进制文件，请尝试通过[`memmapfile`](https://ww2.mathworks.cn/help/matlab/ref/memmapfile.html)函数使用内存映射。 |
| 图像、音频、视频和HDF | 许多MATLAB函数都支持从这些类型的文件中加载数据，使您可以选择读取部分数据。有关详细信息，请参阅[支持的导入和导出的文件格式](https://ww2.mathworks.cn/help/matlab/import_export/supported-file-formats-for-import-and-export.html)中列出的函数参考页。 |

### 3. 使用tall数组

tall数组帮助您处理太大而无法放入内存的数据集，MATLAB一次处理一小块数据，并在后台自动执行所有的数据分块和处理。在转换为tall数组后，MATLAB可免于生成整个数组的临时副本，而是以较小的分块处理数据，这让您能够对数组执行各种运算，而不会耗尽内存。

主要有两种方式使用tall数组，如下所述。

如果大型数组可放入内存，但在尝试执行计算时内存不足，则您可以将该数组转换为tall数组。

```matlab
t = tall(A);
```

如果大型数组可放入内存，但这些数组消耗了太多内存而无法在计算过程中容纳数据副本，则您可以使用这种方法进行处理。例如，如果您有8GB内存和一个5GB矩阵，将该矩阵转换为tall数组可让您在不耗尽内存的情况下对矩阵执行计算。有关此用法的示例，请参阅[`tall`](https://ww2.mathworks.cn/help/matlab/ref/tall.tall.html)。

如果您有基于文件或文件夹的数据，您可以创建一个数据存储，然后基于该数据存储创建一个tall数组。

```matlab
ds = datastore("path/to/file.csv");
t = tall(ds);
```

这种方法可让您利用MATLAB中tall数组的全部功能。数据可以有任意数量的行，并且MATLAB不会耗尽内存。

要了解有关tall数组的更多信息，请参阅[使用tall数组处理无法放入内存的数据](https://ww2.mathworks.cn/help/matlab/import_export/tall-arrays.html)。

### 4. 使用多台计算机的内存

如果您有计算机集群，则可以使用分布式数组（要求Parallel Computing Toolbox），利用集群中所有计算机的总内存来执行计算。根据您的数据能否放入内存，有不同的对应方式在并行池的工作进程之间对数据进行分区。有关详细信息，请参阅[Distributing Arrays to Parallel Workers](https://ww2.mathworks.cn/help/parallel-computing/distributing-arrays-to-parallel-workers.html)（Parallel Computing Toolbox）。

### 5. 调整设置和预设项

一般情况下，重写代码是提高内存性能最有效的方法。但是，如果您无法对代码进行更改，以下解决方案可能会帮助您提供所需的内存量。

不带Java虚拟机启动MATLAB，或减小Java堆大小。如果不带Java虚拟机（JVM）软件启动MATLAB，或减小Java堆大小，则可以增大可用的工作区内存。要在不使用JVM的情况下启动MATLAB，请使用命令行选项`-nojvm`。有关如何减小Java堆大小的信息，请参阅[Java 堆内存预设](https://ww2.mathworks.cn/help/matlab/matlab_external/java-heap-memory-preferences.html)。

使用`-nojvm`会带来一定的损失，因为您会失去一些依赖于JVM的功能，例如桌面工具和图形。启动MATLAB时指定`-nodesktop`选项并不会节省大量内存。

调整数组大小限制。如果您遇到数组大小超过最大数组大小预设项的错误，可以在MATLAB中调整此数组大小限制。有关调整数组大小限制的信息，请参阅[工作区和变量预设项](https://ww2.mathworks.cn/help/matlab/matlab_env/set-workspace-and-variable-preferences.html)。仅当要创建的数组超过当前最大数组大小限制但又不是太大而无法放入内存时，此解决方案才有帮助。即使您关闭数组大小限制，尝试创建一个不合理的大型数组也可能导致MATLAB耗尽内存，或由于内存分页过多而导致MATLAB甚至您的计算机没有响应。

