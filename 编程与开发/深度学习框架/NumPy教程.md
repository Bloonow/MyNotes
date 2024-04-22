[toc]

NumPy（Numerical Python）是Python语言的一个扩展程序库，支持大量的维度数组与矩阵运算，此外也针对数组运算提供大量的数学函数库。

NumPy通常与SciPy（Scientific Python）和Matplotlib（绘图库）一起使用，这种组合广泛用于替代MatLab，是一个强大的科学计算环境，有助于我们通过Python学习数据科学或者机器学习。

SciPy是一个开源的Python算法库和数学工具包。SciPy包含的模块有最优化、线性代数、积分、插值、特殊函数、快速傅里叶变换、信号处理和图像处理、常微分方程求解和其他科学与工程中常用的计算。

Matplotlib是Python编程语言及其数值数学扩展包NumPy的可视化操作界面。它为利用通用的图形用户界面工具包，如Tkinter、wxPython、Qt、GTK+向应用程序嵌入式绘图提供了应用程序接口（API）。

## （一）NumPy数据类型

numpy支持的数据类型比Python内置的类型要多很多，基本上可以和C语言的数据类型对应上，其中部分类型对应为Python内置的类型。numpy的数值类型实际上是`dtype`对象的实例，并对应唯一的字符，包括np.bool_，np.int32，np.float32，等等。

dtype对象是使用以下语法构造的，`numpy.dtype(object, align, copy)`。

- object，要转换为的数据类型对象，如numpy.int32、numpy.bool_等。int8、int16、int32、int64四种数据类型可以使用字符串'i1'、'i2'、'i4'、'i8'代替，每一个内建类型都有一个唯一定义它的字符代码。使用`<`表示小端数据模式，`>`表示大端数据模式。也可以是结构化数据类型。
- align，如果为true，填充字段使其类似C的结构体。
- copy，复制dtype对象，如果为false，则是对内置数据类型对象的引用。

使用dtye的结构化类型，一个例子如下：

```python
student = np.dtype([('name', 'S20'), ('age', 'i1'), ('marks', 'f4')])	# 类似结构体，每个二元组指定一个字段
a = np.array([('abc', 21, 50), ('xyz', 18, 75)], dtype=student)
print(student, a, sep='\n')
''' 输出：
[('name', 'S20'), ('age', 'i1'), ('marks', '<f4')]
[(b'abc', 21, 50.) (b'xyz', 18, 75.)]
'''
```

## （二）NumPy字节交换

在几乎所有的机器上，多字节对象都被存储为连续的字节序列。字节顺序，是跨越多字节的程序对象的存储规则。

- 小端模式：数据的低字节保存在内存的低地址中，数据的高字节保存在内存的高地址中，这种存储模式将地址的高低和数据位权有效地结合起来，高地址部分权值高，低地址部分权值低。
- 大端模式：数据的低字节保存在内存的高地址中，数据的高字节保存在内存的低地址中，这样的存储模式有点儿类似于把数据当作字符串顺序处理：地址由小向大增加，而数据从高位往低位放；这和我们的阅读习惯一致。

`numpy.ndarray.byteswap()`，该函数将ndarray中每个元素中的字节进行大小端转换。

## （三）NumPy数组属性

NumPy数组的维数称为秩（rank），秩就是轴的数量，即数组的维度，一维数组的秩为1，二维数组的秩为2，以此类推。在NumPy中，每一个线性的数组称为是一个轴（axis），也就是维度（dimensions）。很多时候可以声明axis。

NumPy的数组中比较重要ndarray对象属性有：

- `ndarray.ndim`，秩，即轴的数量或维度的数量。
- `ndarray.shape`，数组的维度，返回一个元组，这个元组的长度就是维度的数目，即ndim属性(秩)；元组的各项从左到右为从第一维到最后一维上元素的个数。对于矩阵，n行m列。可以用合适的元组对象赋值给ndarray.shape属性以调整数组各维度形状；NumPy也提供了`reshape()`函数来调整数组大小。
- `ndarray.size`，数组（末端维度上）元素的总个数，相当于.shape返回的元组中，各项的积。
- `ndarray.dtype`，ndarray对象的元素类型。
- `ndarray.itemsize`，ndarray对象中每个元素的大小，以字节为单位。
- `ndarray.flags`，ndarray对象的内存信息。
- `ndarray.real`，ndarray元素的实部。
- `ndarray.imag`，ndarray元素的虚部。
- `ndarray.data`，包含实际数组元素的缓冲区，由于一般通过数组的索引获取元素，所以通常不需要使用这个属性。

## （四）ndarray对象

ndarray对象是齐次N维数组对象，它是一系列同类型数据的集合，以0下标为开始进行集合中元素的索引。创建一个ndarray对象只需要调用`numpy.array(object, dtype=None, copy=True, order=None, subok=False, ndmin=0)`方法即可。

- object，数组或嵌套的数列、列表。
- dtype，数组元素的数据类型，可选。
- copy，对象是否需要复制，可选。
- order，数组在内存中存储的样式，C为行方向优先，F为列方向优先，A为任意方向（默认原方向）。
- subok，默认返回一个与基类类型一致的数组。
- ndmin，指定生成数组的最小维度。

例子如：

```python
import numpy as np
a = np.array([[1, 2], [3, 4]])
print(a)
''' 输出：
[[1 2]
 [3 4]]
'''
```

ndarray对象是用于存放同类型元素的多维数组。ndarray中的每个元素在内存中都有相同存储大小的区域。ndarray内部由以下内容组成：

- 一个指向数据（内存或内存映射文件中的一块数据）的指针。
- 数据类型或dtype，描述在数组中的固定大小值的格子。
- 一个表示数组形状（shape）的元组，表示各维度大小的元组。
- 一个跨度元组（stride），其中的整数指的是为了前进到当前维度下一个元素需要"跨过"的字节数。

### 1. 创建ndarray数组对象

此外，ndarray数组除了可以使用底层ndarray构造器来创建外，也可以通过以下几种方式来创建。

`numpy.empty(shape, dtype=float, order='C')`，该方法用来创建一个指定形状（shape）、数据类型（dtype）且未初始化（元素为随机值）的数组。

`numpy.zeros(shape, dtype=float, order='C')`，该方法创建指定大小的数组，数组元素以0来填充。

`numpy.ones(shape, dtype=None, order='C')`，该方法创建指定形状的数组，数组元素以1来填充。

`ndarray.copy(order)`，该方法用于从一个ndarray拷贝一个新的数组。

### 2. 从已有的数组创建ndarray

`numpy.asarray(obj, dtype=None, order=None)`，其中obj是任意形式的输入参数，可以是列表、列表的元组、元组、元组的元组、元组的列表、多维数组。

`numpy.frombuffer(buffer, dtype=float, count=-1, offset=0)`，该方法可用于实现动态数组。接受buffer输入参数（可以是任意对象），以流的形式读入转化成ndarray对象；count表示读取的数据数量，默认为-1，读取所有数据；offset表示读取的起始位置，默认为0。值得注意的是，buffer是字符串的时候，Python 3.X默认str是Unicode类型，所以要转成bytestring在原str前加上b。

`numpy.fromiter(iterable, dtype, count=-1)`，该方法从可迭代对象中建立ndarray对象，返回一维数组。

### 3. 从数值范围创建ndarray

`numpy.arange(start, stop, step, dtype)`，该方法根据start与stop指定的范围以及step设定的步长，生成一个ndarray。start为起始值，默认为0；stop为终止值但不包含；setp为步长，默认为1；dtype为返回ndarray的元素类型，如果没有提供，则会使用输入数据的类型。

`numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)`，该函数用于创建一个一维数组，数组是一个等差数列构成的。

- start，序列的起始值。
- stop，序列的终止值，如果endpoint为True，该值包含于数列中。
- num，要生成的等步长的样本数量，默认为50。
- endpoint，该值为True时，数列中包含stop值，反之不包含，默认是True。
- retstep，如果为True时，生成的数组中会显示间距，反之不显示。
- dtype，ndarray元素的数据类型。

`numpy.logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None)`，该函数用于创建一个于等比数列。开始项是base^start^，结束项是base^stop^。设根据start、stop、num能生成一系列值value，设$value=\log_{base}X$，则X就是所生成的目标元素的值。

## （四）索引ndarray对象

### 1. 切片和索引

ndarray对象的内容可以通过索引或切片来访问和修改，与Python中list的切片操作一样。ndarray数组可以基于[0:n)的下标进行索引，切片对象可以通过内置的slice函数，并设置start、stop、step参数进行，从原数组中切割出一个新数组。也可以通过冒号分隔切片参数start:stop:step来进行切片操作。

对于多维数组而言，在切片时使用逗号`,`隔开，从左到右依次为对第一维直到最后一维的索引或切片。一个例子如：

```python
a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
print(a[1:, 1:2, 0])
''' 输出：
[[ 7]
 [11]]
'''
```

使用`...`用来省略指定为`:`的连续相邻维度的切片，如对于一个三维的数组，`[...,1]`等价于`[:,:,1]`。

```python
"""
1 2 3
4 5 6
7 8 9
"""
A = np.array([[1,2,3],
              [4,5,6],
              [7,8,9]])
print(A[1])   # [4,5,6]
print(A[:,1]) # [2,5,8]
```

- 对于在NumPy中存储的**二维矩阵**，它的**第一维索引的是矩阵的行索引，第二维索引是矩阵的列索引**。

### 2. 高级索引

NumPy比一般的Python序列提供更多的索引方式。除了用整数和切片的索引外，数组可以由整数数组索引、布尔索引及花式索引。可以借助切片`:`或`...`与索引数组组合。

对于二维ndarray数组来说，整数数组索引需要在最外层方括号中有两个参数，分别是行数组和列数组，每个行和列对应位置上的值都构成以一个索引，这些索引所取到的元素，就是结果。所取结果的形式可以有行和列参数的形式确定。如下：

```python
x = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
rows = [[0, 3], [3, 0]]
cols = [[0, 2], [0, 2]]
print(x[rows, cols])
''' 输出：
[[ 0 11]
 [ 9  2]]
'''
```

- 行和列中分别将[0, 3]和[0, 2]放在一个方括号中，表明它们对应的结果也放在一个方括号中，[x[0, 0], x[3, 2]]；其余的同理。

布尔索引通过布尔运算（如：比较运算符）来获取符合指定条件的元素的数组。可以使用`~`取补运算符来实现反面。以下实例演示如何从数组中过滤掉非复数元素、使用了`~`来过滤NaN。

```python
a = np.array([1, 2 + 6j, 5, 3.5 + 5j])
print(a[np.iscomplex(a)])

b = np.array([np.nan, 1, 2, np.nan, 3, 4, 5])
print(b[~np.isnan(b)])

''' 输出：
[2. +6.j 3.5+5.j]
[1. 2. 3. 4. 5.]
'''
```

花式索引指的是利用整数数组进行索引，根据索引数组的值作为目标数组的某个轴的下标来取值。花式索引跟切片不一样，它总是将数据复制到新数组中。它可以传入顺序索引数组、倒序索引数组（负数数组）、传入多个索引数组（要使用`numpy.ix_()`），例如：

```python
x=np.arange(32).reshape((8,4))
print (x[np.ix_([1,5,7,2],[0,3,1,2])])
```

- 与整数数组索引不同的是，使用np.ix_的索引数组不是行和列位置上的一一对应，而是行和列取笛卡尔积的结果，即每个行数组元素会对应所有列数组元素。

## （五）NumPy广播（Broadcast）

如果两个数组a和b形状相同，即满足a.shape==b.shape，那么a*b的结果就是a与b数组对应位相乘。这要求维数相同，且各维度的长度相同。例如：

```python
a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])
print(a * b)
# 输出：[ 10  40  90 160]
```

广播（Broadcast）是numpy对不同形状（shape）的数组进行数值计算的方式，对数组的算术运算通常在相应的元素上进行。当运算中的2个数组的形状不同时，numpy将自动触发广播机制。广播的规则如下，若条件不满足，抛出"ValueError: frames are not aligned"异常。

- 让所有输入数组都向其中形状最长的数组看齐，形状中不足的部分都通过在前面加1补齐。
- 输出数组的形状是输入数组形状的各个维度上的最大值。
- 当输入数组的某个维度的长度为1时，沿着此维度运算时都用此维度上的第一组值。
- 如果输入数组的某个维度和输出数组的对应维度的长度相同或者其长度为1时，这个数组能够用来计算，否则出错。

例如：

```python
a = np.array([[0, 0, 0],
              [10, 10, 10],
              [20, 20, 20],
              [30, 30, 30]])
b = np.array([1, 2, 3])
print(a + b)
''' 输出：
[[ 1  2  3]
 [11 12 13]
 [21 22 23]
 [31 32 33]]
'''
```

使用`numpy.tile(ndarray, tuple)`方法会以重复的方式广播数组，基本规则是：第二个元组参数，从左到右依次维第一维到最后一维，其值表示对相应维度上的ndarray数组重复多少次。如果ndarray小于tuple指定的维度，数组自动扩维；如果tuple小于数组原来的维度，则在tuple之前用1补齐到相同维度。

```python
a = np.array([1, 2])
aa = np.tile(a, (1, 2))
print(aa)
# 输出：[[1 2 1 2]]
```

## （六）NumPy迭代数组

NumPy迭代器对象`numpy.nditer`提供了一种灵活访问一个或者多个数组元素的方式。迭代器最基本的任务的可以完成对数组元素的访问。注意：访问的是最末维度上的基本元素，而不是当前维度上的下一维度元素。

### 1. 控制遍历顺序

生成一个迭代器对象可以用方法`numpy.nditer(ndarray, order='C')`。值得注意的是，它迭代数组不是使用标准C或者Fortran顺序，选择的顺序是和数组内存布局一致的，这样做是为了提升访问的效率，默认是行序优先（row-major order，或者说是C-order）。可以用`F`指定order为Fortran order，即是列序优先。

```python
for x in np.nditer(a, order='F')	# Fortran order，即是列序优先；
for x in np.nditer(a.T, order='C')	# C order，即是行序优先；ndarray.T表示转置矩阵
```

### 2. 修改数组中元素的值

nditer对象有另一个可选参数`op_flags`。默认情况下，nditer将视待迭代遍历的数组为只读对象（read-only），为了在遍历数组的同时，实现对数组元素值得修改，必须指定read-write或者write-only的模式。

```python
for x in np.nditer(a, op_flags=['readwrite']):
    x[...] = 2 * x
```

### 3. 使用外部循环

nditer类的构造器拥有flags参数，它可以接受下列值：

- c_index，可以跟踪C顺序的索引。
- f_index，可以跟踪Fortran顺序的索引。
- multi_index，每次迭代可以跟踪一种索引类型。
- external_loop，给出的值是具有多个值的一维数组，而不是零维数组，即将末维合并成一维。

```python
a = np.arange(0, 60, 5).reshape(3, 4)
print(a, '\n')
for x in np.nditer(a, flags=['external_loop'], order='F'):
    print(x, end=", ")
''' 输出：
[[ 0  5 10 15]
 [20 25 30 35]
 [40 45 50 55]] 

[ 0 20 40], [ 5 25 45], [10 30 50], [15 35 55], 
'''
```

### 4. 广播迭代

如果两个数组是可广播的，nditer组合对象能够同时迭代它们。

```python
for x, y in np.nditer([a, b])		# a、b被迭代成所共有的最大的大小
```

## （七）NumPy数组操作

NumPy中包含了一些函数用于处理数组，大概可分为以下几类：修改数组形状、翻转数组、修改数组维度、连接数组、分割数组、数组元素的添加与删除等。这些函数有的是numpy的方法，有的是ndarray的方法，有的两者都有，使用时注意区分。

### 1. 修改数组形状

`numpy.reshape(arr, newshape, order='C')`，该函数可以在不改变数据的条件下修改形状。

- arr，要修改形状的数组。
- newshape，整数或者整数数组，新的形状应当兼容原有形状。
- order，'C'为按行，'F'为按列，'A'为原顺序，'k'为元素在内存中的出现顺序。

`numpy.ndarray.flat`，该属性方法是一个数组元素迭代器，它迭代的是基本元素。

`numpy.ndarray.flatten(order='C')`，该函数返回一维数组，即展开数组，它是原数组元素迭代后的拷贝，对拷贝所做的修改不会影响原始数组。

使用flatten()函数，可以将多维数组展平成一维数组，而如果想要在第一维上保持不变，而将剩下的维数展平，可以使用如下方法。

```python
a = np.arange(0, 50, 1).reshape((5, 5, 2))
b = a.flatten()
c = a.reshape((a.shape[0], np.prod(a.shape[1:])))
```

`numpy.ravel(a, order='C')`，该函数展平的数组元素，顺序通常是"C风格"，返回的是数组视图（View），修改视图会影响原始数组。

`numpy.squeeze(a, axis=None)`，压缩数组，若不指定axis，则将所有只含一个元素的维度移除掉，如(3,1,5)压缩后为(3,5)。

### 2. 翻转数组

`numpy.transpose(arr, axes)`，该函数用于对换数组的维度。

- arr，要操作的数组。
- axes，整数列表，对应维度，通常所有维度都会对换。

`numpy.ndarray.T`，属性方法，求原数组（矩阵）的转置，类似于numpy.transpose()。

`numpy.rollaxis(arr, axis, start=0)`，函数向后滚动特定的轴到一个特定位置。改变基本元素的索引顺序。

- arr，数组。
- axis，要向后滚动的轴，其它轴的相对位置不会改变。
- start，默认为零，表示完整的滚动。该参数指定滚动到特定位置。

```python
a = np.ones((3, 4, 5, 6))
print(np.rollaxis(a, 2).shape)
# 输出：(5, 3, 4, 6)
```

`numpy.swapaxes(arr, axis1, axis2)`，该函数用于交换数组的两个轴。

### 3. 修改数组维度

`numpy.broadcast(ndarray, ndarray)`，用于模仿广播的对象，它返回一个对象，该对象封装了将一个数组广播到另一个数组的结果。它拥有iterator属性，基于自身组件的迭代器元组。

`numpy.broadcast_to(array, shape, subok)`，该函数将数组广播到新形状。它在原始数组上返回只读视图。它通常不连续。如果新形状不符合NumPy的广播规则，该函数可能会抛出ValueError。

`numpy.expand_dims(arr, axis)`，函数通过在指定位置插入新的轴来扩展数组形状。arr为要被插入的原数组，axis为新轴插入的位置。

`numpy.squeeze(arr, axis)`，函数从给定数组的形状中删除一维的条目。arr为要被删除的数组，axis为整数或整数元组，用于选择形状中一维条目的子集。

### 4. 连接数组

`numpy.concatenate((a1, a2, ...), axis)`，函数用于沿指定轴连接相同形状的两个或多个数组，在指定维数上进行操作，不扩展维数。

`numpy.stack((a1, a2, ...), axis)`，函数用于沿新轴连接数组序列。axis，数组中的轴，输入数组沿着它来堆叠，它将每个参数作为一个整体，所有参数堆叠在一起，并将它们外面套一层数组，扩展维数。

`numpy.hstack((a1, a2, ...))`是numpy.stack函数的变体，它沿着第二个轴堆叠来生成数组。

`numpy.vstack((a1, a2, ...))`是numpy.stack函数的变体，它沿着第一个轴堆叠来生成数组。

### 5. 分割数组

`numpy.split(ary, indices_or_sections, axis)`，函数沿特定的轴将数组分割为子数组。

- ary，被分割的数组。
- indices_or_sections，果是一个整数，就用该数平均切分，如果是一个数组，为沿轴切分的位置（左开右闭）。
- axis，沿着哪个维度进行切向，默认为0，第0轴即横向切分。为1时，第1轴即纵向切分。

`numpy.hsplit(ary, indices_or_sections)`，函数用于水平分割数组，通过指定要返回的相同形状的数组数量来拆分原数组。

`numpy.vsplit(ary, indices_or_sections)`，函数沿着垂直轴分割，其分割方式与hsplit用法相同。

### 6. 数组元素的添加与删除

`numpy.resize(arr, shape)`，函数返回指定大小的新数组。如果新数组大小大于原始大小，则包含原始数组中的元素的副本。shape为返回数组的新形状。

`numpy.append(arr, values, axis=None)`，函数在数组的末尾添加值。追加操作会分配整个数组，并把原来的数组复制到新数组中。此外，输入数组的维度必须匹配否则将生成ValueError。

- arr，输入数组。
- values，要向arr添加的值，需要和arr形状相同（除了要添加的轴）。
- axis，默认为None。当axis无定义时，是横向加成，返回总是为一维数组。当axis有定义的时候，为向第axis轴添加元素，形状要相同。

`numpy.insert(arr, index, values, axis)`，函数在给定索引之前，沿给定轴在输入数组中插入值。如果值的类型转换为要插入，则它与输入数组不同。插入没有原地的，函数会返回一个新数组。此外，如果未提供轴，则输入数组会被展开。

- arr，输入数组。
- index，在其之前插入值的索引。
- values，要插入的值。
- axis，沿着它插入的轴，如果未提供，则输入数组会被展开。

`numpy.delete(arr, index, axis)`，函数返回从输入数组中删除指定子数组的新数组。与insert()函数的情况一样，如果未提供轴参数，则输入数组将展开。index可以为切片、整数或者整数数组，表明要从输入数组删除的子数组。

`numpy.unique(arr, return_index, return_inverse, return_counts)`，函数用于去除数组中的重复元素。

- 输入数组，如果不是一维数组则会展开。
- return_index，如果为True，返回的是一个新列表，它的元素是去重后所保留的元素在旧列表中的位置（下标），并以列表形式储。
- return_inverse，如果为true，返回旧列表元素在新列表中的位置（下标），并以列表形式储。
- return_counts，如果为True，返回去重数组中的元素在原数组中的出现次数。

## （八）NumPy常见运算

### 1. 位运算

NumPy中"bitwise_"开头的函数是位运算函数，补码形式。NumPy位运算包括以下几个函数：

- `bitwise_and(a, b)`，对数组中整数的二进制形式执行位与运算。
- `bitwise_or(a, b)`，对数组中整数的二进制形式执行位或运算。
- `invert()`，函数对数组中整数以二进制数的补码进行位取反运算，注意带符号数的运算。$\sim n=-n-1$。
- `left_shift()`，函数将数组元素的二进制形式向左移动到指定位置，右侧附加相等数量的0。
- `right_shift()`，函数将数组元素的二进制形式向右移动到指定位置，左侧附加相等数量的0（无符号右移）。

### 2. 三角函数

NumPy提供了标准的三角函数：`numpy.sin()`、`numpy.cos()`、`numpy.tan()`、`numpy.arcsin()`、`numpy.arccos()`、`numpy.arctan()`等，以及标准的π即`numpy.pi`。上述函数的参数和返回值都是弧度制，`numpy.degrees()`函数将弧度制的值转换为角度制的值。

### 3. 舍入函数

`numpy.around(a, decimals)`，函数返回指定数字的四舍五入值。decimals为舍入的小数位数，默认值为0（整数），如果为负，整数将四舍五入到小数点左侧的位置。

`numpy.floor()`，返回小于或者等于指定表达式的最大整数，即向下取整。

`numpy.ceil()`，返回大于或者等于指定表达式的最小整数，即向上取整。

### 4. 比较运算符

可以对ndarray数组使用`>`、`<`、`==`等比较运算符进行操作，它返回的是一个ndarray的布尔数组，数组的每个位置上的值为True或False，代表原数组该位置上的元素是对于所指定条件是否成立，成立即为True，否则为False。

```python
a = np.array([1, 2, 3, 4, 5, 6])
b = a % 2 == 0
print(b)
# 输出：[False True False True False True]
```

### 5. 算术函数

NumPy算术函数包含且不限于简单的加减乘除，需要注意的是数组必须具有相同的形状或符合数组广播规则，两个数组对应位置上的元素做相关操作。如下：`numpy.add()`、`numpy.subtract()`、`numpy.multiply()`、`numpy.divide()`。也可以直接使用`+`、`-`、`*`、`/`运算符。

此外Numpy也包含了其他重要的算术函数。

`numpy.reciprocal()`，函数返回参数逐元素的倒数。

`numpy.power()`，函数将第一个输入数组中的元素作为底数，计算它与第二个输入数组中相应元素的幂。

`numpy.mod()`，计算输入数组中相应元素的相除后的余数。函数`numpy.remainder()`也产生相同的结果。

## （九）NumPy常用函数

### 1. 统计函数

NumPy提供了很多统计函数，用于从数组中查找最小元素，最大元素，百分位标准差和方差等。

`numpy.amin(a, axis)`，用于计算数组中的元素沿指定轴的最小值。

`numpy.amax(a, axis)`，用于计算数组中的元素沿指定轴的最大值。

`numpy.ptp(a, axis)`，函数计算数组中元素最大值与最小值的差（最大值-最小值）。

`numpy.sum(a, axis)`，函数计算数组中所有元素的和。

`numpy.percentile(a, q, axis)`，百分位数是统计中使用的度量，在axis轴上，求百分位数q的值。关于百分位数值，值为p的百分位数是这样一个值，它使得至少有p%的数据项小于或等于这个值，且至少有(100-p)%的数据项大于或等于这个值。若已知p=50，求百分位数值：将原序列排序。遍历，直到某个值x，对它有百分之p的数小于等于x，百分之(100-p)的数大于等于x，则x就是p的百分位数值。p位50时，x实际上就是中位数。

`numpy.median(a, axis)`，函数用于计算数组a中元素的中位数（中值）。

`numpy.mean(a, axis)`，函数返回数组中元素的算术平均值。如果提供了轴，则沿其计算。算术平均值是沿轴的元素的总和除以元素的数量。

`numpy.average(a, axis, weights)`，函数根据在另一个数组中给出的各自的权重计算数组中元素的加权平均值。该函数可以接受一个轴参数。如果没有指定轴，则数组会被展开。

`numpy.var(a)`，统计中的方差（样本方差）是每个样本值与全体样本值的平均数之差的平方值的平均数即$mean(x-x.mean())^2$。

`numpy.std(a)`，标准差是一组数据平均值分散程度的一种度量，标准差是方差的算术平方根。

### 2. 排序函数

NumPy提供了多种排序的方法。这些排序函数实现不同的排序算法，每个排序算法的特征在于执行速度，最坏情况性能，所需的工作空间和算法的稳定性。

`numpy.sort(a, axis, kind='quicksort', order)`，函数返回输入数组的排序副本。

- axis，沿着它排序数组的轴，如果没有数组会被展开，沿着最后的轴排序；axis=0对列排序，axis=1对行排序。
- kind，默认为'quicksort'（快速排序）。此外还有'mergesort'（归并排序）、'heapsort'（堆排序）。
- order，如果数组包含字段，则是要排序的字段。

`numpy.argsort(arr)`，函数返回的是一个数组，它的值依次是原来数组元素从小到大的索引值。第一个为最小元素的索引值，第二个为次小元素的索引值，依次类推。

`numpy.lexsort((arr1, arr2, ...))`，用于对多个序列进行排序。多个数组相对应的位置上构成一个元组，不同数组从当这个元组不同的列，先根据最后一个列排序，再根据前一个，依次向前直到最后一个。它的实际效果是：对元组排序，arr1相同时根据arr2排序，arr2相同时根据arr3，依次类推。

`numpy.msort(a)`，数组按第一个轴（列）排序，返回排序后的数组副本。np.msort(a)相等于np.sort(a, axis=0)。

`sort_complex(a)`，对复数按照先实部后虚部的顺序进行排序。

`partition(a, kth[, axis, kind, order])`，指定一个数，对数组进行分区。kth为基数，小于kth的在左侧，大于kth的在右侧；kth也可以是一个区间（用元组或列表指定），小于左界的在左侧，大于右界的在右侧，位于区间中的在中间。

`argpartition(a, kth[, axis, kind, order])`，先得到原数组元素从小到大的索引序列，然后对这个序列基于kth进行划分。

### 3. 条件刷选函数

`numpy.argmax(a, axis=None)`，沿给定轴返回最大元素的索引。

`numpy.argmin(a, axis=None)`，沿给定轴返回最小元素的索引。

`numpy.nonzero()`，函数返回输入数组中非零元素的索引。

`numpy.where(cond, x, y)`，函数返回输入数组中满足给定条件的元素的索引。

- cond表示选择条件，如果没有x、y时，则输出原数组中满足条件元素的索引（坐标），以元组的形式给出。通常原数组有多少维，输出的tuple中就包含几个数组，分别对应符合条件元素的各维坐标。
- 如果指定x和y，则符合条件cond的位置指定为x，不符合条件的位置指定为y。

```python
a = np.array([0.8, 0.7, 0.9, 0.01, 1.0, 0.3, 0.66, 0.8, 0.7, 0.9])
arg = np.where(a <= 0.7)
print(arg)
# 输出：[1, 3, 5, 6, 8]
```

`numpy.extract()`，函数根据某个条件从数组中抽取元素，返回满条件的元素。

```python
x = np.arange(9.).reshape(3, 3)
print(x, '\n')
print(np.extract(np.mod(x, 2) == 0, x))
''' 输出：
[[0. 1. 2.]
 [3. 4. 5.]
 [6. 7. 8.]] 

[0. 2. 4. 6. 8.]
'''
```

### 4. 字符串函数

Numpy提供了对dtype为numpy.string\_或numpy.unicode\_的数组执行向量化字符串操作的函数，它们基于Python内置库中的标准字符串函数。这些函数在字符数组类（`numpy.char`）中定义。

主要有：`add`、`multiply`、`center`、`capitalize`、`title`、`lower`、`upper`、`split`、`splitlines`、`strip`、`join`、`replace`、`encode`、`decode`。

## （十）NumPy副本和视图

简单的赋值不会创建数组对象的副本。相反，它使用原始数组的相同`id()`来访问它。id()返回Python对象的通用标识符。此外，一个数组的任何变化都反映在另一个数组上。

副本是一个数据的完整的拷贝，如果我们对副本进行修改，它不会影响到原始数据，物理内存不在同一位置。它一般在发生在：Python序列的切片操作，调用deepCopy()函数；调用ndarray的copy()函数产生一个副本。

`ndarray.copy()`，函数创建一个副本，深拷贝。对副本数据进行修改，不会影响到原始数据，它们物理内存不在同一位置。

Python中的切片操作会创建原始数据的副本，对视图的操作不会影响到原始数组。

视图是数据的一个别称或引用，通过该别称或引用亦便可访问、操作原有数据，但原有数据不会产生拷贝。如果我们对视图进行修改，它会影响到原始数据，物理内存在同一位置。它一般发生在：numpy的切片操作返回原数据的视图；调用ndarray的view()函数产生一个视图。

`ndarray.view()`，该方法会创建一个新的数组对象，该方法创建的新数组的维数更改不会更改原始数据的维数。但是数组中数据的改变，会影响到原数组中的数据。

在使用numpy库的切片操作时，它会创建原始数据的视图，对视图的修改数据会影响到原始数组，注意与Python中的不同。

## （十一）NumPy常用库

### 1. 矩阵库（matrix）

NumPy中包含了一个矩阵库`numpy.matlib`，该模块中的函数返回的是一个矩阵，而不是ndarray对象。一个N×M的矩阵是一个N行（row）M列（column）元素排列成的矩形阵列，二维数组。

矩阵总是二维的，而ndarray是一个n维数组。两个对象都是可互换的。可以使用`numpy.array()`、`numpy.asarray()`、`numpy.matrix()`、`numpy.asmatrix()`方法等。

`numpy.matlib.empty(shape, dtype, order)`，函数返回一个新的矩阵，元素的值为未初始化的随机值。

- shape，定义新矩阵形状的整数或整数元组。
- dtype，可选，数据类型。
- order，'C'（行序优先）或者'F'（列序优先）。

`numpy.matlib.zeros(shape)`，函数创建一个以0填充的矩阵。

`numpy.matlib.ones(shape)`，函数创建一个以1填充的矩阵。

`numpy.matlib.eye(n, M, k, dtype)`，函数返回一个矩阵，对角线元素为1，其他位置为零。

- n，返回矩阵的行数。
- M，返回矩阵的列数，默认为N，此时是对角线为（左上到右下）1的n阶方阵。
- K，对角线的索引，默认为0，从[0,0]开始；为1时指定的是为0的右侧一条；为负数时指定的是左侧。
- dtype，数据类型。

`numpy.matlib.identity(n, dtype)`，函数返回给定大小的单位矩阵。

`numpy.matlib.rand(n, m)`，函数创建一个给定大小的矩阵，数据是随机填充的。

对向量、矩阵、张量进行矩阵乘法、对应元素乘法、点积，可以使用如下方法。

```python
x = np.array([[1, 2, 3], [3, 4, 5]])
# 向量、矩阵乘法
y1 = x @ x.T
y2 = np.matmul(x, x.T)
# 对应元素相乘，结果形状与原来相同，元素为对应位置元素相乘
z1 = x * x
z2 = np.multiply(x, x)
# 求点积
t1 = np.dot(x.flatten(), x.flatten())
```

值得注意的是，对于`numpy.dot(x,y)`函数。如果x和y是两个向量，则dot()执行的是这两个向量的内积，它的结果是一个数。如果x和y中有一个是矩阵或都是矩阵，则dot()执行的就是矩阵乘法（需要满足矩阵乘法对左右两个矩阵的要求），此时与`@`和`numpy.matmul()`相同，结果是一个矩阵或向量。

另外，如果要计算一个列向量乘一个行向量所得到的矩阵，不能用上述方法，因为在NumPy中，一维的向量数组既可以看成列的，也可以看成行的，即**NumPy中对于一维的向量没有行或列的概念**，故对于两个向量使用上面的dot()、matmul()函数，都是执行内积操作，得到的结果都是一个数。此时要使用`numpy.outer(x,y)`函数，它表示的是求外积操作，可以用来求列向量与行向量相乘生成的矩阵，且前一个向量x表示列向量，后一个向量y表示行向量。

### 2. 线性代数

NumPy提供了线性代数函数库`numpy.linalg`，该库包含了线性代数所需的所有功能。主要函数有：

|     函数      |              描述              |
| :-----------: | :----------------------------: |
|     `dot`     | 两个数组的点积，即元素对应相乘 |
|    `vdot`     |         两个向量的点积         |
|    `inner`    |         两个数组的内积         |
|   `matmul`    |        两个数组的矩阵积        |
| `determinant` |          数组的行列式          |
|    `solve`    |        求解线性矩阵方程        |
|     `inv`     |      计算矩阵的乘法逆矩阵      |

### 3. 随机数

NumPy的随机数库是`numpy.random`。

`numpy.random.random(shape)`，函数用于生成指定形状的用随机数（从0到1）填充的ndarray。

## （十二）NumPy其他知识

NumPy可以读写磁盘上的文本数据或二进制数据。NumPy为ndarray对象引入了一个简单的文件格式：.npy文件。.npy文件用于存储重建ndarray所需的数据、shape、dtype和其他信息。常用的IO函数有：

- `load()`和`save()`函数是读写文件数组数据的两个主要函数，默认情况下，数组是以未压缩的原始二进制格式保存在扩展名为.npy的文件中。
- `savze()`函数用于将多个数组写入文件，默认情况下，数组是以未压缩的原始二进制格式保存在扩展名为.npz的文件中。
- `loadtxt()`和`savetxt()`函数处理正常的文本文件（如.txt等）。
