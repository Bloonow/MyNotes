[toc]

C#源代码文件的后缀名是.cs。

# 一、保留字

## （一）标识符和关键字

标识符是用来识别类、变量、函数或任何其它用户定义的项目。在C#中，标识符必须以字母、下划线`_`或`@`开头，后面可以跟一系列的字母、数字、下划线和@。

关键字是C#编译器预定义的保留字，这些关键字不能用作标识符，但是可以在关键字前面加上@字符作为前缀，将其当做标识符。

在C#中，有些关键字在代码的上下文中有特殊的意义，如get和set，这些被称为上下文关键字（contextual keywords）。另外as关键字可以实现向下类型转换，is关键字可以判断一个对象是否属于一个类型。

### 1. using

`using`关键字的用法主要有如下四种。

```c#
using NameSpace.SubNameSpace;    // 引入命名空间

using static System.Math;
double pi = PI;        // 指定无需指定类型名称即可访问其静态成员的类型

using Project = PC.MyCompany.Project;    // 起别名

using (Font fontA = new Font(/*xxx*/), fontB = new Font(/*xxx*/))
{
    // use fontA and fontB
    // 代码段结束时，自动调用fontA和fontB的Dispose方法，释放实例。
    // using语句支持初始化多个变量，但前提是这些变量的类型必须相同，且都实现了IDisposable接口
    // 如果多个变量类型不同，则可以用IDisposable声明类型
}
```

### 2. 访问修饰符

一个访问修饰符定义了一个类成员的范围和可见性，C#支持的访问修饰符如下所示。

- public，所有对象都可以访问。
- private，对象本身在对象内部可以访问。
- protected，只有该类对象及其子类对象可以访问。
- internal，同一个程序集的对象可以访问。该访问说明符允许一个类将其成员变量和成员函数暴露给当前程序中的其他函数和对象；换句话说，带有internal访问修饰符的任何成员可以被定义在该成员所定义的应用程序内的任何类或方法访问。
- protected internal，访问限于当前程序集或派生自包含类的类型。
- new，新建类/方法，表示由基类/基类的方法继承而来，隐藏了从起父类中继承而来的同名成员。
- sealed，密封类/方法，说明该类不能再派生新类，该方法不能被重写。
- readonly，只读字段，对它的赋值只能在声明的同时进行，或者通过构造函数进行，其值并不要求在编译时给出；而const要求成员的值要在编译时能计算。
- partial，可以将类、结构或接口的定义拆分到两个或多个源文件中，在类声明前添加partial关键字即可。

未指定访问修饰符则采用默认，类成员默认是private，接口和枚举默认是public。

### 3. 预处理指令

预处理器指令指导编译器在实际编译开始之前对信息进行预处理。所有的预处理器指令都是以#开始，且在一行上，只有空白字符可以出现在预处理器指令之前；预处理器指令不是语句，它们不以分号`;`结束；一个预处理器指令必须是该行上的唯一指令。与C/C++不同，C#的预处理指令不是用来创建宏的。

下表列出了C#中可用的预处理器指令。

| 预处理指令 | 描述                                                         |
| :--------- | :----------------------------------------------------------- |
| #define    | 它用于定义一系列成为符号的字符。这样，通过使用符号作为传递给#if指令的表达式，表达式将返回true |
| #undef     | 它用于取消定义符号                                           |
| #if        | 它用于测试符号是否为真，可以使用如!、==、!=、&&、\|\|运算符。如果为真，编译器会执行#if和下一个指令之间的代码 |
| #else      | 它用于创建复合条件指令，与#if一起使用                        |
| #elif      | 它用于创建复合条件指令                                       |
| #endif     | 指定一个条件指令的结束                                       |
| #line      | 它可以修改编译器的行数以及（可选地）输出错误和警告的文件名，使用default恢复默认的行号 |
| #error     | 它允许从代码的指定位置生成一个错误                           |
| #warning   | 它允许从代码的指定位置生成一级警告                           |
| #region    | 它可以在使用Visual Studio Code Editor的大纲特性时，指定一个可展开或折叠的代码块 |
| #endregion | 它标识着#region块的结束                                      |
| #pragma    | 它可以抑制或还原指定的编译警告，与命令行选项不同，该指令可以在类或方法级别执行，对抑制警告的内容和抑制的时间进行更精细的控制。如#pragma warning disable 169取消编号为169的警告（字段未使用的警告），使用#pragma warning restore 169恢复编号为169的警告 |

### 4. try和catch

异常是在程序执行期间出现的问题，异常提供了一种把程序控制权从某个部分转移到另一个部分的方式。C#异常处理时建立在四个关键词之上的try、catch、finally和throw，如下所示。

```c#
try {
    throw new Exception("xxx");
} catch (ExceptionType eName) {
    /* handle or rethrow */
} finally {
}
```

C#异常是使用类来表示的，C#中的异常类主要是直接或间接地派生于System.Exception类。如System.ApplicationException类，它支持由应用程序生成的异常，程序员定义的异常都应派生自该类；和System.SystemException类，它是所有预定义的系统异常的基类。下表列出了一些派生自Sytem.SystemException类的预定义的异常类。

| 异常类                            | 描述                                         |
| :-------------------------------- | :------------------------------------------- |
| System.IO.IOException             | 处理I/O错误                                  |
| System.IndexOutOfRangeException   | 处理当方法指向超出范围的数组索引时生成的错误 |
| System.ArrayTypeMismatchException | 处理当数组类型不匹配时生成的错误             |
| System.NullReferenceException     | 处理当依从一个空对象时生成的错误             |
| System.DivideByZeroException      | 处理当除以零时生成的错误                     |
| System.InvalidCastException       | 处理在类型转换期间生成的错误                 |
| System.OutOfMemoryException       | 处理空闲内存不足生成的错误                   |
| System.StackOverflowException     | 处理栈溢出生成的错误                         |

### 5. unsafe

当一个代码块使用unsafe修饰符标记时，C#允许在unsafe的作用域中使用指针变量，unsafe可以标记一个类、方法、代码块等。不安全代码或非托管代码是指使用了指针变量的代码块。

对于C#来说，可以使用指针的ToString()方法检索存储在指针变量所引用位置的数据，如下所示。

```c#
unsafe
{
    int a = 20;
    int* p = &a;
    Console.WriteLine(p->ToString());
}
```

在C#中，数组名称和一个指向与数组数据具有相同数据类型的指针是不同的变量类型，即int* p和int[] p是不同的类型。对于指针变量p来说，可以对其运用+运算符，因为它在内存中不是固定的，但是数组地址在内存中是固定的，所以不能增加数组p。

因此，如果需要使用指针变量访问数组数据，可以像通常在C/C++中所做的那样，使用fixed关键字来固定指针。下面的实例演示了这点。

```c#
int[] arr = { 10, 100, 1000 };
fixed (int* p = arr)
{
    for (int i = 0; i < 3; ++i)
    {
        Console.WriteLine(*(p + i));
    }
}
```

由于C#中声明的变量在内存中的存储受垃圾回收器管理，因此一个变量（例如一个大数组）有可能在运行过程中被移动到内存中的其他位置；如果一个变量的内存地址会变化，那么指针也就没有意义了。解决方法就是使用fixed关键字来固定变量位置不移动。

而且，在unsafe不安全环境中，可以通过stackalloc在堆栈上分配内存，因为在堆栈上分配的内存不受内存管理器管理，因此其相应的指针不需要固定。如下所示。

```c#
int* ptr = stackalloc int[1024];
```

另外，需要注意的是，为了编译不安全代码，必须切换到命令行编译器指定`/unsafe`命令行。例如为了编译包含不安全代码的名为Prog.cs的程序，需要如下命令。

```shell
csc /unsafe Prog.cs
```

如果使用的是Visual Studio IDE，那么需要在项目属性>生成中选择启用不安全代码。

## （二）运算符

### 1. 可空类型修饰符 ?

引用类型可以使用空引用表示一个不存在的值，而值类型通常不能表示为空。为了使值类型也可为空，就可以使用可空类型，即用可空类型修饰符`?`来表示，表现形式为`Type?`。它其实是System.Nullable泛型结构的缩写形式，也就是用到T?时编译器编译时会将其编译成System.Nullable的形式。如下。

```c#
int? a = null;
Nullable<int> b = null;
```

### 2. 空合并运算符 ??

用于定义可空类型和引用类型的默认值。如果此运算符的左操作数不为 null，则此运算符将返回左操作数，否则返回右操作数。如下。

```c#
object obj = a ?? b;
```

- 当a为null时则返回b，a不为null时则返回a本身。

空合并运算符为右结合运算符，即操作时从右向左进行组合的，它可以嵌套使用。如a??b??c的形式按a??(bb??c)计算，当a为null时返回b，当b也为null时返回c，即返回从左至右第一个非null的，或者最后一个数的值。

### 3. null检查运算符 ?.

它是对象取成员运算符`.`的加强版，当对象为null的时候，使用`?.`不会报错，而是返回null，其返回类型实际上是原来类型的Nullable\<Type\>泛型。如下所示。

```c#
Point p = new Point(3, 4);
Console.WriteLine(p?.X.GetType() == typeof(int?));    // true
```

## （三）特性

特性（Attribute）是用于在运行时传递程序中各种元素（如类、方法、结构、枚举、组件等）的行为信息的声明性标签，通过使用特性可以向程序添加声明性信息，一个声明性标签是通过放置在它所应用的元素前面的方括号`[]`来描述的。C#中的特性有点类似于Java中的注解。

特性用于添加元数据，如编译器指令和注释、描述、方法、类等其他信息，.Net框架提供了预定义的和自定义的两种特性。要规定一个特性，其名称和值是在方括号内规定的，放置在它所应用的元素之前。

```c#
[AttributeType(positional_parameters, name_parameter = value, ...)]
SomeElement
```

其中，positional_parameters规定必需的信息，name_parameter规定可选的信息。

### 1. 预定义特性

.Net框架提供了三种预定义特性，即AttributeUsage、Conditional、Obsolete。

预定义特性AttributeUsage描述了如何使用一个自定义特性类，它在自定义特性的时候用到，规定了特性可应用到的项目的类型。它在命名空间System中。

```c#
[AttributeUsage(
   validon,
   AllowMultiple=aBoolean,
   Inherited=aBoolean
)]
```

- 参数validon，规定特性可被放置的语言元素，它是枚举器AttributeTargets的值的组合，默认值是AttributeTargets.All。其他的多种组合可以使用|连接起来，如AttributeTargets.Class|AttributeTargets.Field等。
- 可选属性AllowMultiple，使用一个布尔值，true表示该特性是多用的，默认值是false，即表示单用的。
- 可选属性Inherited，使用一个布尔值，true表示该特性可被派生类继承，默认值是false，即表示不被继承。

预定义特性Conditional标记了一个条件方法，其执行依赖于指定的预处理标识符，它会引起方法调用的条件编译，取决于指定的值，如Debug或Trace。

利用Conditional属性，程序可以定义条件方法。Conditional属性通过测试条件编译符号来确定适用的条件。当运行到一个条件方法调用时，是否执行该调用，要根据出现该调用时是否已定义了此符号来确定。如果定义了此符号，则执行该调用；否则省略该调用（包括对调用的参数的计算）。使用Conditional是封闭#if和#endif内部方法的替代方法，它更整洁、更别致，减少了出错的机会。

Conditional在命名空间System.Diagnostics中，一个例子如下。

```c#
#define MyDebug
using System.Diagnostics;
/* xxx */
[Conditional("MyDebug")]
public static void Message(string msg)
{
    Console.WriteLine(msg);
}
```

预定义特性Obsolete标记了不应被使用的程序实体，它可以指示编译器丢弃某个特定的目标元素。例如，当一个新方法被用在一个类中，但是仍要保持类中已废弃的旧方法，则可以通过显示一个应该使用新方法，而不是旧方法的消息，来把它标记为 Obsolete（过时的）。它在命名空间System中。

```c#
[Obsolete(message, iserror)]
```

- 参数message，是一个字符串，描述项目为什么过时以及该替代使用什么。
- 参数iserror，是一个布尔值。如果该值为true，编译器应把该项目的使用当作一个错误，默认值是false（编译器生成一个警告）。

使用的例子如下所示。

```c#
[Obsolete("This function was obsolete, use the new one instead.", false)]
public void SomeFunction() { /* xxx */ }
```

### 2. 创建自定义特性

.Net框架允许创建自定义特性，用于存储声明性的信息，且可在运行时被检索；该信息根据设计标准和应用程序需要，可与任何目标元素相关。创建并使用自定义特性包含四个步骤：声明自定义特性、构建自定义特性、在目标程序元素上应用自定义特性、通过反射访问特性。

一个新的自定义特性应派生自System.Attribute类，并使用AttributeUsage特性注解这个自定义的类。每个特性必须至少有一个构造函数，前面提到，一个特性可以拥有必需的定位（positional）参数和可选的命名（named）参数，必需的定位（positional）参数应通过构造函数传递。

一个自定义特性和使用它的例子如下所示。

```c#
[AttributeUsage(AttributeTargets.Class |
    AttributeTargets.Constructor |
    AttributeTargets.Field |
    AttributeTargets.Method |
    AttributeTargets.Property,
    AllowMultiple = true)]
class MyInfo : System.Attribute
{
    private int codeNo;
    private string developer;
    public string message;
    public MyInfo(int cn, string dev)
    {
        this.codeNo = cn;
        this.developer = dev;
    }
}

[MyInfo(01, "Akame", message = "This is the Main point.")]
class MainClass
{ /* xxx */ }
```

# 二、数据及类型

## （一）类型和常量

### 1. 概述

C#的变量可以大致分为三种类型，值类型（Value types）、引用类型（Reference types）、指针类型（Pointer types）。可以使用`typeof()`方法，或者一个类型的`GetType()`方法获取类型消息。

值类型变量可以直接分配给一个值，它们是从类System.ValueType中派生的。值类型直接包含数据，比如bool、byte、char、decimal、double、float、int、long、sbyte、short、uint、ulong、ushort等。

引用类型指的是一个内存位置。内置的引用类型有object**、**dynamic和string。

**对象（object）类型**是C#通用类型系统（Common Type System，CTS）中所有数据类型的终极基类，object是System.Object类的别名。所以对象（object）类型可以被分配任何其他类型（值类型、引用类型、预定义类型或用户自定义类型）的值，但是在分配值之前，需要先进行类型转换。当一个值类型转换为对象类型时，则被称为装箱；另一方面，当一个对象类型转换为值类型时，则被称为拆箱。

**动态数据类型**变量中可以存储任何类型的值，这些变量的类型检查是在运行时发生的。声明动态类型的语法如下。

```c#
dynamic variable_name = value;
```

动态类型与对象类型相似，但是对象类型变量的类型检查是在编译时发生的，而动态类型变量的类型检查是在运行时发生的。

**字符串（string）类型**允许给变量分配任何字符串值，字符串（string）类型是System.String类的别名，它是从对象（object）类型派生的。普通字符串（string）类型的值通过引号声明，如"Hello World"；也可以在引号前加@符号表示逐字字符串，它会将转义字符当做普通字符串处理，也允许一个字符串多行并且换行符及缩进空格都计算在字符串长度之内，如`@"Hello \n World"`。

指针类型变量存储另一种类型的内存地址。C#中的指针与C或C++中的指针有相同的功能，但是因为裸指针不安全，故不提倡使用。

### 2. 类型转换

类型转换从根本上说是类型铸造，或者说是把数据从一种类型转换为另一种类型。在C#中，类型铸造有两种形式，即隐式类型转换和显示类型转换。

C#提供了下列内置的类型转换方法，它们都是Convert工具类的方法，如ToBoolean、ToByte、ToChar、ToDateTime、ToDecimal、ToDouble、ToInt16、ToInt32、ToInt64、ToSbyte、ToSingle、ToType、ToUInt16、ToUInt32、ToUInt64。

除Convert工具类的方法，相应的类型如int、double等，还有对应的TryParse、Parse方法，用来解析用字符串表示的值。

### 3. 函数调用的传参类型

对于函数的参数而言，普通类型是按值传递的；而引用类型参数是按引用传递的（实际上是不同的栈中位置保存了同一个内存地址值的引用）。

如果想让按值传递的普通类型参数按引用传递，则可以使用`ref`关键字，注意函数定义和调用都要使用，如下所示。

```c#
void func(ref int x) { x = 10; }
int a = 50;        // 一定要初始化分配空间，否则引用出错
func(ref a);
```

普通的函数，return只能从函数中返回一个值，但是可以使用输出参数来从函数中返回多个值，输出参数会把方法输出的数据赋给自己，其他方面与引用参数相似。可以使用`out`声明，注意函数定义和调用都要使用，如下所示。

```c#
void foo(out int x) { x = 15; }
int b;        // 提供给输出参数的变量不需要赋值，但在函数中一定要赋值初始化
foo(out b);
```

- 另外需要注意的是，ref型参数可以传入函数中，即有进有出；而out型参数无法将数据传入函数中，只出不进。
- 重载方法时若两个方法的区别仅限于一个参数类型为ref另一个方法中为out，编译器会报错。

### 4. 常量

整数常量可以是十进制、八进制或十六进制的常量。前缀指定基数0x或0X表示十六进制，0表示八进制，没有前缀则表示十进制；整数常量也可以有后缀，可以是U和L的组合，其中U和L分别表示unsigned和long。后缀可以是大写或者小写，多个后缀以任意顺序进行组合。

一个浮点常量是由整数部分、小数点、小数部分和指数部分组成，可以使用小数形式或者指数形式来表示浮点常量。使用小数形式表示时，必须包含小数点、指数或同时包含两者；使用指数形式表示时，必须包含整数部分、小数部分或同时包含两者；有符号的指数是用e或E表示的。

字符常量是括在单引号里，且可存储在一个简单的字符类型变量中。一个字符常量可以是一个普通字符（例如'x'）、一个转义序列（例如'\t'）或者一个通用字符（例如'\u02C0'）。

字符串常量是括在双引号""里，或者是括在@""里，字符串常量包含的字符与字符常量相似。

常量可分为静态常量（编译时常量）和动态常量（运行时常量）。

- 静态常量，在编译时就确定了值，必须在声明时就进行初始化且之后不能进行更改，可在类和方法中定义。它使用const来定义。
- 动态常量，在运行时确定值，只能在声明时或构造函数中初始化，只能在类中定义。它使用`readonly`来定义。

在取值永久不变或者对程序性能要求非常苛刻时可以使用const常量，除此之外的其他情况都应该优先采用readonly常量。

## （二）数组

### 1. 概述

在C#中声明一维数组，C#编译器会根据数组类型隐式初始化每个数组元素为一个默认值。声明数组可以使用如下的语法。

```c#
Type[] arrayName = new Type[n] { /* xxx */ };
```

可以使用for循环来遍历每个数组元素，也可以使用一个foreach语句来遍历数组，该语句不会区分数组分为几维，而是按照其在内存中的存储顺序，从头到尾遍历数组中的全部元素。

```c#
foreach(Type a in arrayName) { /* xxx */ }
```

C#支持多维数组，多维数组又称为矩形数组。它们的声明和访问方式类似，与其他C++、Java等语言有稍微语法不同，下面以二维数组为例。

```c#
Type[,] twoDimension = new Type[m,n] { /* xxx */ };
Type aValue = twoDimension[i,j];
```

C#还支持交错数组，它的每一行的长度是可以不一样的。它是数组的数组，本质上是一维数组。

```c#
Type[][] aArray = new Type [m][];
aArray[i] = new Type[ni];
aArray[j] = new Type[nj];
```

### 2. 参数数组

C#参数数组通常用于传递未知数量的参数给函数，定位可以类比其他语言中的可变参数列表。在使用数组作为形参时，C#提供了`params`关键字，使调用数组为形参的方法时，既可以传递数组实参，也可以传递一组数组元素。

```c#
void func(params Type[] arr);
Type[] ay = new Type[] { /* xxx */ };
func(ay);    // 可以直接使用数组作为实参
func(t1, t2, t3);    // 也可以使用可变数量的参数作为实参
```

- 带params关键字的参数类型必须是一维数组，不能使用在多维数组上。
- 不允许和ref、out同时使用。
- 带params关键字的参数必须是最后一个参数，并且在方法声明中只允许一个params关键字。
- 不能仅使用params来使用重载方法。
- 没有params关键字的方法的优先级高于带有params关键字的方法的优先级。

## （三）封装结构

### 1. 结构

在C#中定义一个结构体，需要使用`struct`语句。在C#中的结构与传统的C/C++中的结构不同，C#中的结构有以下特点。

- 结构可带有方法、字段、索引、属性、运算符方法和事件。结构成员不能指定为abstract、virtual或protected。
- 结构可定义构造函数，但不能定义析构函数。但不能为结构定义无参构造函数，无参构造函数（默认）是自动定义的，且不能被改变，即结构不能声明默认的构造函数。
- 与类不同，结构不能继承其他的结构或类。结构不能作为其他结构或类的基础结构。结构可实现一个或多个接口。
- 当使用new操作符创建一个结构对象时，会调用适当的构造函数来创建结构。与类不同，结构可以不使用new操作符即可被实例化。如果不使用new操作符，只有在所有的字段都被初始化之后，字段才被赋值，对象才被使用。

- 结构体中声明的字段无法赋予初值，类可以。结构体的构造函数中，必须为结构体所有字段赋值，类的构造函数无此限制。

另外比较重要的就是，结构是值类型，其存储在栈中；而类是引用类型，类的对象是存储在堆空间中。

### 2. 枚举

枚举是一组命名整型常量，默认情况下，第一个枚举符号的值是0，枚举使用enum关键字声明。C#枚举是值类型，换句话说，枚举包含自己的值，且不能继承或传递继承。如下所示。

```c#
enum Days { Sun, Mon, tue, Wed, thu, Fri, Sat };
```

### 3. 类

#### (1) 成员

类的析构函数是类的一个特殊的成员函数，当类的对象超出范围时执行。析构函数的名称是在类的名称前加上一个波浪形`~`作为前缀，它不返回值，也不带任何参数。析构函数用于在结束程序（比如关闭文件、释放内存等）之前释放资源。析构函数不能继承或重载。

可以使用static关键字把类成员定义为静态的，静态变量可在成员函数或类的定义外部进行初始化，也可以在类的定义内部初始化静态变量。

#### (2) 继承

一个类可以派生自一个父类或多个接口，使用`:`表示继承，父类或多个接口之间可以使用逗号分隔，这意味着它可以从基类或接口继承数据和函数。父类对象应在子类对象创建之前被创建，可以在成员初始化列表中进行父类的初始化。

```c#
class Father {
    public Father(int fatherAge) {}
}
class Son : Father {
    public Son(int fatherAge, int sonAge) : base(fatherAge) {}
}
```

C#不支持多重继承，但可以使用接口来实现多重继承。

C#的类与C++中的类有些许相似，如构造方法、析构方法、this、virtual关键字等。但也有一个不同的地方，如一个子类重写了父类的virtual方法，需要在子类的方法的返回类型前加上`override`关键字修饰；与之对应的是`new`关键字修饰子类方法或成员，它的作用是隐藏父类中的方法（实际上还是存在）。如果想在子类中访问被new关键字隐藏掉的父类成员，可以使用基类访问表达式，即`base.SomeField`。关于重写与隐藏的详细关系可以参看《C++高级编程》。

使用关键字abstract创建抽象类或抽象方法；在类定义前面放置关键字`sealed`可以将类声明为密封类，它不能被继承，类似于Java中的final类。抽象类不能被声明为sealed。

值得注意的是，virtual和abstract都是用来修饰父类的，通过覆盖父类的定义，让子类重新定义。

- virtual修饰的方法必须有实现（哪怕是空实现），而abstract修饰的方法一定不能实现。
- virtual可以被子类重写，而abstract必须被子类重写。
- 如果类成员被abstract修饰，则该类前必须添加abstract，因为只有抽象类才可以有抽象方法。
- 无法创建abstract类的实例，只能被继承无法实例化。

#### (3) 重载运算符

可以在类中重定义或重载C#中内置的运算符，通过关键字operator后跟运算符的符号来定义的，operator关键字用于在类或结构声明中声明运算符。

与其他函数一样，重载运算符有返回类型和参数列表；且运算符只能采用值参数，不能采用ref或out参数。运算符声明可以采用下列四种形式之一。

```c#
public static ResultType operator UnaryOperator(OpType oper);
public static ResultType operator BinaryOperator(OpType oper, OpType2 oper2);
public static implicit operator ConvTypeOut(ConvTypeIn oper);
public static explicit operator ConvTypeOut(ConvTypeIn oper);
```

- ResultType表示运算符的结构类型，OpType是运算符参数的类型，oper是形式参数。
- UnaryOperator是一元运算符，可以为+、-、!、~、++、--、true、false。
- BinaryOperator是二元运算符，可以为+、-、*、/、%、&、|、^、<<、>>、==、!=、>、<、>=、<=。另外OpType和OpType2中至少有一个必须是封闭类型，即运算符所属的类型，或理解为自定义的类型。
- ConvTypeOut是类型转换运算符的目标类型、ConvTypeIn是类型转换运算符的输入类型。转换运算符的ConvTypeOut和ConvTypeIn中正好有一个必须是封闭类型，即转换运算符只能从它的封闭类型转换为其他某个类型，或从其他某个类型转换为它的封闭类型。

需要注意的是，前两种形式声明了用户定义的重载内置运算符的运算符，并非所有内置运算符都可以被重载，不可重载的有&&、||、+=、-=、*=、/=、%=、=、.、?:、new、is、sizeof、typeof。

C#要求成对重载比较运算符。如果重载了==，则也必须重载!=，否则产生编译错误。同时，比较运算符必须返回bool类型的值，这是与其他算术运算符的根本区别。

C#不允许重载=运算符，但如果重载例如+运算符，编译器会自动使用+运算符的重载来执行+=运算符的操作。

#### (4) 扩展方法

C#允许在string、int、DataRow、DataTable等类型的基础上为其增加实例方法，使用时不需要修改或编译类型本身的源代码。需要注意的是，只有静态类才可以添加扩展方法，且扩展方法必须是静态的。

```c#
static class MyClass
{
    public static void Main()
    {
        string str = "Hello, ";
        str = str.MyAddMethod("World!");
        Console.WriteLine(str);
    }
    public static string MyAddMethod(this string srcStr, string addStr)
    {
        return srcStr + addStr;
    }
}
```

### 4. 接口

接口定义了属性、方法和事件，这些都是接口的成员，接口只包含了成员的声明；接口中的成员缺省是public访问权限，但不能显式地用访问修饰符指定。C#使用`interface`来定义接口，并且接口名通常使用I开头。

另外需要注意的一些事项如下。

- C#接口的成员不能有public、protected、internal、private等修饰符。接口成员不能有new、static、abstract、override、virtual修饰符。
- 接口中只能包含方法、属性（接口内不能有字段变量）、事件和索引器的组合。使用属性如下，其他如事件、索引器见之后的笔记。

```C#
interface IBook
{
    string BookName { get; set; }
}
```

## （四）正则表达式

基本的正则表达式语法可以参看《C++高级编程》，这里仅记述C#提供的进行正则操作的一些类库。

### 1. Regex类

`System.Text.RegularExpressions.Regex`类用于表示一个正则表达式，它的构造函数接受一个模式字符串。下面列出了Regex类中一些常用的方法。如需了解Regex类的完整的属性列表，可参阅微软的C#文档。

```c#
public bool IsMatch(string input);
public bool IsMatch(string input, int startat);
public static bool IsMatch(string input, string pattern);
```

- 指示Regex构造函数中指定的正则表达式是否在指定的输入字符串中找到匹配项。可以从字符串中指定的开始位置startat开始。

```c#
public MatchCollection Matches(string input);
public MatchCollection Matches(string input, int startat);
public static MatchCollection Matches(string input, string pattern);
```

- 在指定的输入字符串中搜索正则表达式的所有匹配项。

```c#
public string Replace(string input, string replacement);
public string Replace(string input, string replacement, int count, int startat);
public static string Replace(string input, string pattern, string replacement);
```

- 在指定的输入字符串中，把所有匹配正则表达式模式的所有匹配的字符串替换为指定的替换字符串。可用count指定要替换的次数，用startat指定要开始的位置。

```c#
public string[] Split(string input);
public string[] Split(string input, int count, int startat);
public static string[] Split(string input, string pattern);
```

- 把输入字符串分割为子字符串数组，根据指定的正则表达式模式定义的位置进行分割。

### 2. 示例

```c#
Regex reg = new Regex(@"boy");
string str = reg.Replace("I am a boy.", "girl");
Console.WriteLine(str);
```

## （五）Lambda表达式

lambda表达式的基本形式如下。

```c#
(Type parameters) => expression;
(Type parameters) => { expressions; }
```

- 可选参数类型声明：不需要声明参数类型，编译器可以统一识别参数值。如：`(x, y) => x + y;`。
- 仅有一个参数且省略参数类型时，则可以省略`()`；没有参数时，则不能省略括号。

Lambda表达式可以被转换为委托类型，这要求它们拥有相同的返回类型、参数类型和个数。

## （六）泛型

泛型（Generic）允许延迟编写类或方法中的编程元素的数据类型的规范，直到实际在程序中使用它的时候。换句话说，泛型允许编写一个可以与任何数据类型一起工作的类或方法。

在使用时，可以通过数据类型的替代参数编写类或方法的规范；当编译器遇到类的构造函数或方法的函数调用时，它会生成代码来处理指定的数据类型。

C#的泛型和C++、Java中泛型类似，都是通过占位符来指定类型参数，多个类型参数之间可以使用逗号分隔；C#的泛型除了可以用于泛型类、泛型方法，还可以用于泛型委托等，如下。

```c#
class MyGenericClass<T> { /* xxx */ }
void MyGenericFunc<T>(paramList);
delegate T MyGenericDelegate<T>(T n);
```

在声明泛型方法、泛型类的时候，可以给泛型加上一定的约束来满足我们特定的一些条件。其基本格式如下所示。

```c#
class MyClass<T> where conditions { /* xxx */ }
```

上面的where condition是对类型参数T的一种限定，主要是如下几种情况中的一种或几种的组合（多个组合约束用逗号间隔）。

- `where T : struct`，类型参数T必须是值类型，可以指定除Nullable以外的任何值类型。
- `where T : class`，类型参数T必须是引用类型，包括任何类、接口、委托或数组类型。
- `where T : new()`，类型参数T必须具有无参数的公共构造函数，当与其他约束一起使用时，new()约束必须最后指定。
- `where T : BaseClassName`，类型参数T必须是指定的基类或派生自指定的基类。
- `where T : InterfaceName`，类型参数T必须是指定的接口或实现指定的接口，可以指定多个接口约束，约束接口也可以是泛型的。
- `where T : U`，类型参数T可以是其他类型参数指定的类型或其子类，如class MyGenClass<T, U> where T : U。

使用泛型是一种增强程序功能的技术。它有助于最大限度地重用代码、保护类型的安全以及提高性能。并且支持创建泛型集合类；如.NET框架类库在System.Collections.Generic命名空间中包含了一些新的泛型集合类。可以创建自己的泛型接口、泛型类、泛型方法、泛型事件和泛型委托；并且可以对泛型类进行约束以访问特定数据类型的方法。泛型数据类型中使用的类型的信息可在运行时通过使用反射获取。

# 三、输入输出

## （一）类库基础介绍

### 1. 概述

System.IO命名空间有各种不同的类，用于执行各种文件操作，如创建和删除文件、读取或写入文件，关闭文件等。

- BinaryReader、BinaryWriter，以二进制流读取原始数据、和以二进制格式写入原始数据。
- StreamReader、StreamWriter，用于从字节流中读取字符、和用于向一个流中写入字符。
- StringReader、StringWriter，用于读取字符串缓冲区、和用于写入字符串缓冲区。
- File，有助于处理文件；FileInfo，用于对文件执行操作；FileStream，用于文件中任何位置的读写。
- Directory，有助于操作目录结构；DirectoryInfo，用于对目录执行操作；Path，对路径信息执行操作。
- DriveInfo，提供驱动器的信息。
- MemoryStream，用于随机访问存储在内存中的数据流。

C#提供了使用各种目录和文件相关的类来操作目录和文件的方式，比如DirectoryInfo类和FileInfo类。DirectoryInfo提供了各种用于创建、移动、浏览目录和子目录的方法；FileInfo提供了用于创建、复制、删除、移动（MoveTo方法也可以用于重命名文件）、打开文件的属性和方法，且有助于FileStream对象的创建。

### 2. 文件流FileStream

System.IO命名空间中的`FileStream`类有助于文件的读写与关闭，该类派生自抽象类Stream。通过创建一个FileStream对象来创建一个新的文件，或打开一个已有的文件，创建FileStream对象的语法如下。

```c#
string filepath = @"E:\Cache\test.txt";
FileStream fs = new FileStream(filepath, FileMode.Enumerator, FileAccess.Enumerator, FileShare.Enumerator);
```

- FileMode枚举表示各种打开文件的方法，其枚举的成员有：
  - Append，打开一个已有的文件，并将光标放置在文件的末尾。如果文件不存在，则创建文件。
  - Create，创建一个新的文件。如果文件已存在，则删除旧文件，然后创建新文件。
  - CreateNew，指定操作系统应创建一个新的文件。如果文件已存在，则抛出异常。
  - Open，打开一个已有的文件。如果文件不存在，则抛出异常。
  - OpenOrCreate，指定操作系统应打开一个已有的文件。如果文件不存在，则用指定的名称创建一个新的文件打开。
  - Truncate，打开一个已有的文件，文件一旦打开，就将被截断为零字节大小。然后我们可以向文件写入全新的数据，但是保留文件的初始创建日期。如果文件不存在，则抛出异常。
- FileAccess枚举用来表示对文件的读写操作，其成员有Read、Write、ReadWrite。
- FileShare用来表示其他进程对该文件的操作权限，其枚举成员有：
  - Inheritable，允许文件句柄可由子进程继承。Win32不直接支持此功能。
  - None，谢绝共享当前文件。文件关闭前，打开该文件的任何请求（由此进程或另一进程发出的请求）都将失败。
  - Delete，允许随后删除文件。
  - Read，允许随后打开文件读取。如果未指定此标志，则文件关闭前，任何打开该文件以进行读取的请求（由此进程或另一进程发出的请求）都将失败。但是，即使指定了此标志，仍可能需要附加权限才能够访问该文件。
  - Write，允许随后打开文件写入。如果未指定此标志，则文件关闭前，任何打开该文件以进行写入的请求（由此进程或另一进过程发出的请求）都将失败。但是，即使指定了此标志，仍可能需要附加权限才能够访问该文件。
  - ReadWrite，允许随后打开文件读取或写入。如果未指定此标志，则文件关闭前，任何打开该文件以进行读取或写入的请求（由此进程或另一进程发出）都将失败。但是，即使指定了此标志，仍可能需要附加权限才能够访问该文件。

一个示例如下。

```c#
string filepath = @"E:\Cache\test.txt";
FileStream fs = new FileStream(filepath, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.Read);
for (int i = 1; i <= 20; i++)
{
    fs.WriteByte((byte)i);
}
fs.Position = 0;    // 重置文件流指针的位置到文件头
for (int i = 0; i <= 20; i++)
{
    Console.Write(fs.ReadByte() + " ");
}
fs.Close();
Console.WriteLine();
```

### 3. 文件属性操作

`File`类与`FileInfo`都能实现文件属性的操作，主要是静态方法与示例方法的区别。

```c#
string filepath = @"E:\Cache\test.txt";

// use File
Console.WriteLine(File.GetAttributes(filepath));
File.SetAttributes(filepath, FileAttributes.Hidden | FileAttributes.ReadOnly);  // 隐藏与只读
Console.WriteLine(File.GetAttributes(filepath));

// use FilInfo
FileInfo fi = new FileInfo(filepath);
Console.WriteLine(fi.Attributes.ToString());
fi.Attributes = FileAttributes.Hidden | FileAttributes.ReadOnly;    // 隐藏与只读
Console.WriteLine(fi.Attributes.ToString());

// 只读与系统属性，删除时会提示拒绝访问
fi.Attributes = FileAttributes.Archive;
Console.WriteLine(fi.Attributes.ToString());
```

### 4. 文件路径

文件和文件夹的路径操作都在Path类中；另外还可以用Environment类，里面包含环境和程序的信息。

```c#
string filepath = @"E:\Cache\test.txt";
string dirpath = @"E:\Cache";
string str;

// 文件名
str = Path.GetFileName(filepath);
str = Path.GetFileNameWithoutExtension(filepath);  // 没有扩展名的文件名
str = Path.GetFileName(dirpath);
// 文件扩展名
str = Path.GetExtension(filepath);
// 更改扩展名
str = Path.ChangeExtension(filepath, ".dat");

// 获得当前路径
str = Environment.CurrentDirectory;
// 根目录
str = Path.GetPathRoot(filepath);

// 文件或文件夹所在目录
str = Path.GetDirectoryName(filepath);
str = Path.GetDirectoryName(dirpath);
// 绝对路径
str = Path.GetFullPath(filepath);
str = Path.GetFullPath(dirpath);
                                                                      
// 生成路径
str = Path.Combine(new string[] { @"E:\", "Cache", "SubDir", "a.c" }));
// 生成随即文件夹名或文件名
str = Path.GetRandomFileName();

// 创建磁盘上唯一命名的零字节的临时文件并返回该文件的完整路径
str = Path.GetTempFileName();
// 返回当前系统的临时文件夹的路径
str = Path.GetTempPath();

// 文件名中无效字符
char[] chs1 = Path.GetInvalidFileNameChars();
// 路径中无效字符
char[] chs2 = Path.GetInvalidPathChars();
```

## （二）文本文件的读写

`StreamReader`和`StreamWriter`类用于文本文件的数据读写，这些类从抽象基类Stream继承，Stream支持文件流的字节读写。

StreamReader类继承自抽象基类TextReader，表示读取器读取一系列字符，下面列出StreamReader类中一些常用的方法。如需查看完整的方法列表，请访问微软的C#文档。

```c#
public override int Read();    // 从输入流中读取下一个字符，并把字符位置往前移一个字符
public override string ReadLine();    // 从当前流中读取一行字符并将数据作为字符串返回
public override int Peek();    // 返回下一个可用的字符，但不使用它
public override void Close();    // 关闭StreamReader对象和基础流，并释放任何与读取器相关的系统资源
```

StreamWriter类继承自抽象类TextWriter，表示写入器写入一系列字符，下面列出了StreamWriter类中一些常用的方法。

```c#
public override void Write(char value);    // 把一个字符写入到流，其他基本类型有类似方法
public virtual void WriteLine();    // 把行结束符写入到文本字符串或流
public override void Flush();    // 清理当前写入器的所有缓冲区，使得所有缓冲数据写入基础流
public override void Close();    // 关闭当前的StreamWriter对象和基础流
```

下面给出一个例子，它按行从一个文件中读取字符打印，并将其存入到另一个文件中。

```c#
string srcFilePath = @"E:\Cache\test.txt";
string toFilePath = @"E:\Cache\to.txt";
StreamReader sr = new StreamReader(srcFilePath);
FileStream fs = new FileStream(toFilePath, FileMode.Create, FileAccess.ReadWrite, FileShare.Read);
StreamWriter sw = new StreamWriter(fs);

// 按行从原文件中读取字符串，打印并将其存入一个新文件中
string lineStr;
while ((lineStr = sr.ReadLine()) != null)
{
    Console.WriteLine(lineStr);
    sw.WriteLine(lineStr);
}
sr.Close();
sw.Close();
```

另外需要注意的是，在读取中文时由于字符集等设置的问题，可能出现乱码，则可以使用`System.Text.Encoding`的静态成员或者静态方法，显式指定读取器或写入器的字符集编码方式，如下。

```c#
StreamReader sr = new StreamReader(filePath, Encoding.UTF8);
```

## （三）二进制文件的读写

`BinaryReader`和`BinaryWriter`类用于二进制文件的读写。BinaryReader和BinaryWriter类用于从文件读取和写入二进制数据，它们的构造函数接收一个FileStream对象。

这两个二进制读取和写入的类，提供了一个系列类似的Read和Write方法，它们通过和不同的类型进行组合，从而形成相应的读写方法。对于这些方法需要注意的是它们读取或写入的数据是何种形式，并且操作后根据读取的字节数和Encoding，对流的指针移动的影响。这些信息都被在这些方法的注释中详细列出，这里就不在赘述了。

# 四、C#高级编程

## （一）属性

属性（Property）是类、结构、接口的命名（named）成员。类或结构中的成员变量或方法称为域（Field），属性是域的扩展，且可使用相同的语法来访问。属性不会确定存储位置，相反，它们具有可读写或计算它们值的访问器（accessors），它们使用访问器让私有域的值可被读写或操作。

属性的访问器包含有助于获取（读取或计算）、设置（写入）属性的可执行语句，访问器声明可包含一个get或set访问器，或同时包含二者；也可以自动实现属性（即简写为只有get和set），但必须有get访问器，如果只有set访问器，则不能简写。

需要注意的是，自动实现的属性，将没有与之相绑定的字段域（即使属性和字段名称相似），而是由程序自动维护其存储空间。

```c#
class Student
{
    private string name;
    // 声明类型为string的Name属性，它和字段name绑定
    public string Name
    {
        get
        {
            return name;
        }
        set
        {
            name = value;
        }
    }
    // int类型的Age属性，它没有相绑定的字段，而是由程序自动维护存储空间
    public int Age { get; set; }
}
```

也可以使用lambda表达式简写属性的实现，并且能够手动指定属性对应的字段，如下。

```c#
class Student
{
    private string birthPlace;
    public string BirthPlace { get => birthPlace; set => birthPlace = value; }
}
```

抽象类可拥有抽象属性，这些属性应在派生类中被实现。

## （二）索引器

索引器（Indexer）允许一个对象可以像数组一样使用下标的方式来访问。当为类定义一个索引器时，该类的行为就会像一个虚拟数组（virtual array）一样，程序员可以使用数组访问运算符`[]`来访问该类的的成员。

索引器的行为的声明在某种程度上类似于属性，它也使用get和set访问器来定义索引器。但是，属性返回或设置一个特定的数据成员，而索引器返回或设置对象实例的一个特定值；换句话说，它把实例数据分为更小的部分，并索引每个部分，获取或设置每个部分。

定义一个索引器定义的时候不带有名称，但带有this关键字，它指向对象实例。下面的例子通过类的成员数组演示了这个概念，除此之外对于多个同类型成员变量，可以在索引器中使用switch和case语句来实现。

```c#
class Contents
{
    public static int size = 10;
    private string[] nameList = new string[size];
    public Contents()
    {
        for (int i = 0; i < size; nameList[i++] = "N.A.") ;
    }
    // 一维索引器，它由返回类型，this关键字，索引类型和形参，以及get/set访问器构成
    public string this[int index] 
    {
        get 
        {
            string tmp;
            if (index >= 0 && index <= size - 1)
            {
                tmp = nameList[index];
            }
            else
            {
                tmp = "";
            }
            return tmp;
        }
        set
        {
            if (index >= 0 && index <= size - 1)
            {
                nameList[index] = value;
            }
        }
    }
}
```

对于上面给出的示例代码，使用的示例如下所示。

```c#
Contents contents = new Contents();
contents[0] = "Hello";
contents[2] = "World";
contents[4] = "!";
contents[6] = "Good";
contents[8] = "Bye";
for (int i = 0; i < Contents.size; Console.WriteLine(contents[i++])) ;
```

值得注意的是，索引器可被重载，而且声明的时候也可带有多个参数，且每个参数可以是不同的类型，没有必要让索引器必须是整型的，C#允许索引器可以是其他类型，例如字符串类型。对于上面给出的例子，使用字符串作为索引的索引器如下。

```c#
class Contents
{
    /* xxx */
    public int this[string name]
    {
        get
        {
            int index = 0;
            while (index < size)
            {
                if (nameList[index].Equals(name))
                {
                    return index;
                }
                index++;
            }
            return index;
        }
    }
}
```

## （三）委托

### 1. 委托

C#中的委托（Delegate）类似于C/C++中函数的指针，委托是存有对某个方法的引用的一种引用类型变量，引用可在运行时被改变。所有的委托都派生自System.Delegate类。委托特别用于实现事件和回调方法。

委托声明决定了可由该委托引用的方法，委托可指向一个与其具有相同标签的方法，声明委托的语法如下。

```c#
delegate ReturnType DelegateTypeName(parameterList);
```

一旦声明了委托类型，委托对象必须使用new关键字来创建，且与一个特定的方法有关。下面的一个例子演示了委托的声明和使用方法。

```c#
// 声明一个委托类型，它有一个string参数，并且返回int类型
public delegate int MyDelegateType(string str);

public static int MyGetLength(string str)
{
    Console.WriteLine(str + " 's length is: " + str.Length);
    return str.Length;
}

public static void Main()
{
    MyDelegateType md = new MyDelegateType(MyGetLength);    // 委托对象指向某个具体的方法
    md("Hello Delegate!");      // 使用委托对象来调用方法
}
```

相同类型的委托对象，可以使用`+`运算符进行合并，一个合并委托依次调用它所合并的两个委托；`-`运算符可用于从合并的委托中移除某个委托。使用委托的合并，可以创建一个委托被调用时要调用的方法的调用列表，这称为委托的多播（multicasting），也叫组播。下面的程序演示了委托的多播。

```c#
public delegate void MyStringOperation(string str);
public void MyStringPrinter(string str)
{
    Console.WriteLine(str);
}
public void MyStringLengthGetter(string str)
{
    Console.WriteLine(str.Length);
}
public void MyStringUpper(string str)
{
    Console.WriteLine(str.ToUpper());
}
public static void Main()
{
    MainClass mc = new MainClass();
    MyStringOperation mso = new MyStringOperation(mc.MyStringPrinter);
    mso += mc.MyStringLengthGetter;     // 使用+运算符，组合多个函数，形成委托的多播
    mso += mc.MyStringUpper;
    mso("Hello World!");
}
```

委托类型可以作为函数的参数进行传递，类似于用函数指针做参数，它可以很方便地用来实现函数的回调操作。

### 2. 匿名委托方法

委托是用于引用与其具有相同标签的方法。匿名方法（Anonymous methods）提供了一种传递代码块作为委托参数的技术，匿名方法是没有名称只有主体的方法，在匿名方法中不需要指定返回类型，它是从方法主体内的return语句推断的。

匿名方法是通过使用delegate关键字创建委托实例来声明的，一个例子如下。

```c#
public delegate int IntStringDelegate(string str);    // 委托类型
public static void Main()
{
    IntStringDelegate isd = delegate (string str)
    {
        // 这个是一个匿名方法体，它没有名字，且返回值类型由return语句推断
        Console.WriteLine(str.Length);
        return str.Length;
    };
    isd("Hello World");
}
```

## （四）事件

事件（Event）基本上说是一个用户操作，如按键、点击、鼠标移动等等，或者是一些提示信息，如系统生成的通知；应用程序需要在事件发生时响应事件，例如中断。C#中使用事件机制实现线程间的通信。

C#中的事件本质上就是使用`event`关键字修饰的委托对象。事件使用发布-订阅（publisher-subscriber）模型。

事件在类中声明且生成，且通过使用同一个类或其他类中的委托与事件处理程序关联。包含事件的类用于发布事件，被称为发布器（publisher）类。发布器是一个包含事件和委托定义的对象，事件和委托之间的联系也定义在这个对象中，发布器类的对象调用这个事件，并通知其他的对象。

其他接受该事件的类被称为订阅器（subscriber）类。订阅器是一个接受事件并提供事件处理程序的对象，在发布器类中的委托调用订阅器类中的方法，即是事件处理程序。

一个简单的例子如下。

```c#
delegate void EventHandler();
class MyComponent
{
    public event EventHandler AEventHandler; // 实际上是一个委托对象
    public void OnSomeEventFire(object sender)
    {
        Console.WriteLine("The evnet results from: " + sender.ToString());
        if (AEventHandler != null)
        {
            // 触发事件，它将自动调用AEventHandler委托对象注册的函数
            AEventHandler();
        }
        else
        {
            Console.WriteLine("Event not fire.");
        }
    }
}
class MyClass
{
    public void foo()
    {
        MyComponent aButton = new MyComponent();
        aButton.AEventHandler += () => Console.WriteLine("User clicked the button.");
        aButton.OnSomeEventFire(this);
    }
    public static void Main()
    {
        MyClass mc = new MyClass();
        mc.foo();
    }
}
```

## （五）反射

反射指程序可以访问、检测和修改它本身状态或行为的一种能力。程序集包含模块，而模块包含类型，类型又包含成员，反射则提供了封装程序集、模块和类型的对象。使用反射可以动态地创建类型的实例，将类型绑定到现有对象，或从现有对象中获取类型，然后可以调用类型的方法或访问其字段和属性。

反射的有点如下：反射提高了程序的灵活性和扩展性；降低耦合性，提高自适应能力；它允许程序创建和控制任何类的对象，无需提前硬编码目标类。

同时它也有一些缺点：使用反射基本上是一种解释操作，用于字段和方法接入时要远慢于直接代码；因此反射机制主要应用在对灵活性和拓展性要求很高的系统框架上，普通程序不建议使用。使用反射会模糊程序内部逻辑，反射绕过了源代码的技术，因而会带来维护的问题，反射代码比相应的直接代码更复杂。

在C#中，反射允许在运行时查看特性（attribute）信息；允许审查集合中的各种类型，以及实例化这些类型；允许延迟绑定的方法和属性（property）；允许在运行时创建新类型，然后使用这些类型执行一些任务。

C#的反射库位于命名空间System.Reflection中，可以利用一些如MemberInfo、MethodInfo、ParameterInfo、FieldInfo等类和位于System空间中的Type类（它的对象可以通过typeof操作符获得），来实现各种反射操作。它们都尤其是Type类提供了一系列的Get方法操作，用来获得目标的各种信息。一个简单的例子如下。

```c#
Type type = typeof(myObject);
// 遍历方法的特性
foreach (MethodInfo m in type.GetMethods())
{
    foreach (Attribute a in m.GetCustomAttributes(true))
    {
        MyClassType mct = a as MyClassType;
        if (null != mct)
        {
            Console.WriteLine("/* xxx */");
        }
    }
}
```

# 五、多线程

线程的概念这里不再赘述了，详细可以看操作系统的有关笔记，这里只是给出C#线程库的操作简介。

线程生命周期开始于`System.Threading.Thread`类的对象被创建时，结束于线程被终止或完成执行时。线程的生命周期中一般可以分为如下几个状态：

- 未启动状态，当线程实例被创建但Start方法未被调用时的状况。
- 就绪状态，当线程准备好运行并等待CPU周期时的状况。
- 不可运行状态，有几种情况下线程是不可运行的，如已经调用Sleep方法、已经调用Wait方法、通过IO操作阻塞。
- 死亡状态，当线程已完成执行或已中止时的状况。

当C#程序开始执行时，主线程自动创建；使用Thread类创建的线程被主线程的子线程调用，可以使用Thread.CurrentThread属性访问当前执行的线程。

## （一）Thread类介绍

下面列出了Thread类的一些常用的属性。

| 属性               | 描述                                                         |
| :----------------- | :----------------------------------------------------------- |
| CurrentThread      | 获取当前正在运行的线程                                       |
| CurrentCulture     | 获取或设置当前线程的区域性                                   |
| CurrentUICulture   | 获取或设置资源管理器使用的当前区域性，以便在运行时查找区域性特定的资源 |
| CurrentPrincipal   | 获取或设置线程的当前负责人（对基于角色的安全性而言）         |
| CurrentContext     | 获取线程正在其中执行的当前上下文                             |
| ExecutionContext   | 获取一个ExecutionContext对象，该对象包含有关当前线程的各种上下文的信息 |
| IsAlive            | 获取一个值，该值指示当前线程的执行状态                       |
| IsBackground       | 获取或设置一个值，该值指示某个线程是否为后台线程             |
| IsThreadPoolThread | 获取一个值，该值指示线程是否属于托管线程池                   |
| ManagedThreadId    | 获取当前托管线程的唯一标识符                                 |
| Name               | 获取或设置线程的名称                                         |
| Priority           | 获取或设置一个值，该值指示线程的调度优先级                   |
| ThreadState        | 获取一个值，该值包含当前线程的状态                           |

下面列出了Thread类的一些常用的方法。

```c#
public void Start();
public static void Sleep(int millisecondsTimeout);
public static void SpinWait(int iterations);
public void Abort();
public static void ResetAbort();
public void Interrupt();
public void Join();
public bool Join(TimeSpan timeout);
public static bool Yield();
```

- Start方法开始一个线程。
- Sleep方法让线程暂停一段时间。
- SpinWait方法，导致线程等待由iterations参数定义的时间量。
- Abort方法，在调用此方法的线程上引发ThreadAbortException，以开始终止此线程的过程；调用此方法通常会终止线程。ResetAbort方法，用来取消为当前线程请求的Abort。
- Interrupt方法，用来中断处于WaitSleepJoin线程状态的线程。
- Join方法，在继续执行标准的COM和SendMessage消息泵处理期间，阻止调用线程，直到由该实例表示的线程终止，或经过了指定时间为止。对于带有timeout参数的方法，如果线程已终止，则返回true；如果timeout参数指定的时间量已过之后还未终止线程，则返回false。
- Yield方法，导致调用线程执行准备好在当前处理器上运行的另一个线程，由操作系统选择要执行的线程。如果操作系统转而执行另一个线程，则返回true，否则为false。

```c#
public static LocalDataStoreSlot AllocateDataSlot();
public static LocalDataStoreSlot AllocateNamedDataSlot(string name);
public static void FreeNamedDataSlot(string name);
```

- Local方法，在所有的线程上分配未命名、或者已命名为name的数据槽。
- Free方法，为进程中的所有线程消除名称与槽之间的关联。
- 为了获得更好的性能，请改用以System.ThreadStaticAttribute特性标记的字段。

```c#
public static LocalDataStoreSlot GetNamedDataSlot(string name);
public static object GetData(LocalDataStoreSlot slot);
public static void SetData(LocalDataStoreSlot slot, object data);
```

- LocalDataStoreSlot方法，用来查找命名的数据槽。
- GetData方法，用来在当前线程的当前域中从当前线程上指定的槽中检索值。
- SetData方法，在当前正在运行的线程上为此线程的当前域在指定槽中设置数据。
- 为了获得更好的性能，请改用以System.ThreadStaticAttribute特性标记的字段。

```c#
public static AppDomain GetDomain();
public static int GetDomainID();
```

- 返回当前线程正在其中运行的当前域，或者返回唯一的应用程序域标识符。

```c#
public static void BeginCriticalRegion();
public static void EndCriticalRegion();
```

- BeginCriticalRegion方法，用来通知宿主执行将要进入一个代码区域，在该代码区域内线程中止或未经处理异常的影响可能会危害应用程序域中的其他任务。
- EndCriticalRegion方法，用来通知主机执行将要进入一个代码区域，在该代码区域内线程中止或未经处理异常的影响限于当前任务。

```c#
public static void BeginThreadAffinity();
public static void EndThreadAffinity();
```

- BeginThreadAffinity方法，用来通知宿主托管代码将要执行依赖于当前物理操作系统线程的标识的指令。
- EndThreadAffinity方法，用来通知宿主托管代码已执行完依赖于当前物理操作系统线程的标识的指令。

## （二）简单线程操作

### 1. 使用Thread类

创建一个线程Thread对象的构造函数，需要一个ThreadStart对象作为参数，其中ThreadStart是一个委托，它标识的是返回void的传入空参数的函数；如果想要传入参数，需要使用ParameterizedThreadStart委托对象作为Thread的构造参数，这个委托标识的是返回void的传入一个object的函数。演示它们的例子如下所示。

```c#
public static void aFunc()
{
    Console.WriteLine("No params function, runing at {0} Thread.", 
                      Thread.CurrentThread.ManagedThreadId);
}

public static void bFunc(object data)
{
    Console.WriteLine("Has one param function, runing at {0} Thread.",
                      Thread.CurrentThread.ManagedThreadId);
    string str = data as string;
    Console.WriteLine("The param is string Type, and it's {0}.", str);
}

public static void Main(string[] args)
{
    // 无参数线程
    ThreadStart ts = new ThreadStart(aFunc);
    new Thread(ts).Start();
    
    // 这种写法与上面一样，它是C#的语法糖，由编译器把它转换成上面的形式
    new Thread(aFunc).Start();

    // 有参数线程
    ParameterizedThreadStart pts = new ParameterizedThreadStart(bFunc);
    new Thread(pts).Start("Hello World!");

    Thread.Sleep(1000);        // 睡眠主线程1000ms
    Console.WriteLine("Main() over");
}
```

可以使用Abort()方法来销毁线程，它通过抛出ThreadAbortException在运行时中止线程，这个异常不能被捕获（即使捕捉了也不会执行catch后的代码块），当然如果有finally块，控制会被首先送至finally块中。一个演示的例子如下所示。

```c#
public static void aFunc()
{
    try
    {
        double x = 1.5;
        for (int i = 0; i < 5; ++i)
        {
            Console.WriteLine(i * x);
            Thread.Sleep(100);
        }
    }
    catch (ThreadAbortException tae)
    {
        Console.WriteLine("Caught ThreadAbortException at catch()");
    }
    finally
    {
        Console.WriteLine("at finally");
    }
    Console.WriteLine("aFunc() over");
}

public static void Main(string[] args)
{
    Thread t = new Thread(aFunc);
    t.Start();

    Thread.Sleep(300);  // 主线程睡眠一段时间
    t.Abort();

    Console.WriteLine("Main() over");
}

// 输出如下
0
1.5
3
Caught ThreadAbortException at catch()
at finally
Main() over
```

### 2. 其他线程操作

C#在4.0以后一共有3种创建线程的方式：

1. Thread类，自己创建的独立的线程，优先级高，需要使用者自己管理，即上面所示。
2. ThreadPool线程池（位于System.Threading命名空间），有.Net自己管理，只需要把需要处理的方法写好，然后交给.Net Framework，后续只要方法执行完毕，则自动退出。
3. Task（位于System.Threading.Tasks命名空间）是4.0以后新增的线程操作方式，类似ThreadPool，但效率测试比ThreadPool略高，Task对多核的支持更为明显，所以在多核的处理器中，Task的优势更为明显。

这里给出演示后两种方式的例子。

```c#
public static void aFunc(object tag)
{
    Console.WriteLine("aFunc() at {0} Thread", 
                      Thread.CurrentThread.ManagedThreadId);
}

public static void Main(string[] args)
{
    // 使用线程池
    ThreadPool.QueueUserWorkItem(aFunc, new object());

    // 使用Task方式创建线程
    Task.Factory.StartNew(aFunc, new object());
}
```

# 六、集合与常用基础类

## （一）集合

集合（Collection）类是专门用于数据存储和检索的类，这些类提供了对栈（stack）、队列（queue）、列表（list）和哈希表（hash table）的支持，大多数集合类实现了相同的接口。各种常用的集合类位于System.Collections命名空间中，下面对它们进行详细的说明。

### 1. 动态数组 ArrayList

ArrayList代表了可被单独索引的对象的有序集合。它基本上可以替代一个数组，但与数组不同的是，它允许使用索引在指定的位置添加和移除项目，动态数组会自动重新调整它的大小。它也允许在列表中进行动态内存分配、增加、搜索、排序各项。

ArrayList类的一些常用的属性如下。Capacity表示ArrayList可以包含的元素个数；Count表示实际包含的元素个数；IsFixedSize表示ArrayList是否具有固定大小；Item[Int32]获取或设置指定索引处的元素；SyncRoot获取一个对象用于同步访问ArrayList。

ArrayList类的一些常用的方法如下。Add方法用于在ArrayList的末尾添加一个对象；Remove方法从ArrayList中移除第一次出现的指定对象。

这里给出一个ArrayList存储不同类型时的排序例子。

```c#
// 自定义比较器
class MyCompare : IComparer
{
    public int Compare(object x, object y)
    {
        // 自定义比较规则：
        // 如果x，y都是int，则按正常流程比较
        // 如果其中一个不是int，则认为不是int的值最小
        // 如果都不是int，则认为它们相等
        if (x is int)
        {
            if (y is int)
            {
                if ((int)x < (int)y) return -1;
                if ((int)x == (int)y) return 0;
                return 1;
            }
            return 1;
        }
        if (y is int) return -1;
        return 0;
    }
}

class MainClass
{
    public static void Main()
    {
        ArrayList al = new ArrayList();
        al.Add(3);
        al.Add(7);
        al.Add("abc");
        al.Add(4);
        al.Add(1);
        al.Sort(new MyCompare());

        foreach (object obj in al) Console.Write("{0} ", obj);
        Console.WriteLine();
    }
}
```

### 2. 哈希表 Hashtable

Hashtable类代表了一系列基于键的哈希代码组织起来的键-值对，它使用键来访问集合中的元素。

Hashtable类的一些常用的属性如下。Count表示Hashtable中包含的键-值对个数；Item[int32]获取或设置与指定的键相关的值；Keys获取一个ICollection，包含Hashtable中的键；Values获取一个ICollection，包含Hashtable中的值；对于上面获取的ICollection对象，可以使用foreach语句遍历。

Hashtable类的一些常用的方法如下。Add方法向Hashtable添加一个带有指定的键和值的元素；Remove从中移除带有指定的键的元素；ContainsKey判断是否包含指定的键；ContainsValue判断是否包含指定的值。

### 3. 排序列表 SortedList

SortedList类代表了一系列按照键来排序的键-值对，这些键值对可以通过键和索引来访问。集合中的各项总是按键值排序。

排序列表是数组和哈希表的组合，它包含一个可使用键或索引访问各项的列表；如果使用索引访问各项，则它是一个动态数组（ArrayList），如果使用键访问各项，则它是一个哈希表（Hashtable）。

SortedList类的一些常用的属性如下。Capacity、Count、Item[int32]、Keys、Values。

SortedList类的一些常用的方法如下。Add、Remove、ContainsKey、ContainsValue；GetKey方法获取SortedList的指定索引处的键；GetByIndex获取指定索引处的值；GetKeyList、GetValueList方法分别获取SortedList中的键和值。

### 4. 堆栈 Stack

Stack代表了一个后进先出的对象集合。

它的Count属性获取Stack中包含的元素个数。Push方法向Stack的顶部添加一个对象；Pop方法移除并返回在Stack的顶部的对象；Peek方法返回在Stack的顶部的对象，但不移除它；Contains方法判断某个元素是否在Stack中。

### 5. 队列 Queue

Queue代表了一个先进先出的对象集合。

它的Count属性获取Queue中包含的元素个数。Enqueue方法向Queue的末尾添加一个对象；Dequeue方法移除并返回在Queue的开头的对象；Contains方法判断某个元素是否在Queue中。

### 6. 点列阵 BitArray

BitArray类管理一个紧凑型的位值数组，它使用布尔值来表示，其中true表示位是开启的（1），false表示位是关闭的（0）。当需要存储位且事先不知道位数时，可以使用点阵列，它支持使用整型索引从点阵列集合中访问各项，索引从零开始。

BitArray类的一些常用的属性如下。Count获取BitArray中包含的元素个数；Item[int32]获取或设置BitArray中指定位置的位的值；Length获取或设置BitArray中的元素个数。

BitArray类的一些常用的方法如下。And方法对当前的BitArray中的元素和指定的BitArray中的相对应的元素执行按位与操作；Or方法执行按位或操作；Xor方法执行按位异或操作；Not方法取反；Set、SetAll方法把BitArray中指定位置的位或全部的位设置为指定的值；Get方法获取BitArray中指定位置的位的值。

### 7. 集合空间接口

Icollection、Ilist、Ienumerable是集合空间里最常用的接口，大多数集合类通常都实现了这些接口。

Icollection接口是集合空间里最基础的接口，它定义了一个集合最基本的操作，所有集合类均实现了这个接口。

Ienumerable接口只有一个GetEnumerator()方法，它得到一个Ienumerator枚举接口，它可以循环访问集合中的元素，它有方法Current、MoveNext、Reset。

Ilist接口提供索引的方式访问其元素，定义了访问集合元素的索引器，除此还定义了修改，添加Add、Insert，移除Remove、RemoveAt、Clear等操作。

## （二）常用基础类

### 1. Math类

System.Math类提供了若干实现不同标准数学函数的静态方法，这里列出了其常用的数学函数。

- 绝对值函数，`Abs()`。
- 以弧度为单位的标准三角函数，`Sin()`、`Cos()`、`Tan()`。
- 以弧度为单位的标准反三角函数，`ASin()`、`ACos()`、`ATan()`、`ATan2()`。
- 标准双曲函数，`Sinh()`、`Cosh()`、`Tanh()`。
- 最大值最小值函数，`Max()`、`Min()`。
- 不小于指定数的最小整数，`Celling()`。
- 不大于指定数的最大整数，`Floor()`。
- 指定数的四舍五入的值，`Round()`。
- 返回数字整数部分，`Truncate()`。
- 自然对数或以10为底的对数，`Log()`、`Log10()`。
- 指数函数，`Exp()`。
- 指定数的乘方，`Pow()`。
- 指定数的符号值，负数为-1，零为0，正数为1，`Sign()`。
- 指定数的平方根，`Sqrt()`。
- 返回两数相除的余数，`IEEERemainder()`。

### 2. DataTime和TimeSpan类

使用System.DateTime类可以完成日期与时间数据的处理，在一个DateTime变量中，可以使用Year、Month、Day、Hour、Minute、Second、Millisecond等属性分别获相应的数据信息；除此还有DayOfYear、DayOfWeek、TimeOfDay等属性获取对应信息。

DataTime类还有如下公共静态属性用来获得当前日期或时间，以及一些实用的方法。

```c#
public static DateTime Now { get; }        // 当前日期和时间的对象
public static DateTime UtcNow { get; }    // 当前UTC日期和时间的对象
public static DateTime Today { get; }    // 当天日期的一个对象，其时间组成部分设置为00:00:00
public static bool IsLeapYear(int year);    // 判断指定的年份是否为闰年
public static int DaysInMonth(int year, int month);        // 返回指定年和月中的天数
```

TimeSanp类表示一个时间间隔，范围在Int64.MinValue到Int64.MaxValue之间。它的Milliseconds、Seconds、Minutes、Hours、Days属性表示当前System.TimeSpan结构所表示的时间间隔的对应时间部分；它的TotalMilliseconds、TotalSeconds、TotalMinutes、TotalHours、TotalDays属性表示当前System.TimeSpan结构对应时间间隔表示的总秒数、总天数等。

### 3. Random类

System.Random类用来产生随机数，它的默认构造方法Random()使用依赖于时间的默认种子值，还有一个构造方法Random(int Seed)可以指定种子。Random类的常用方法如下。

```c#
public virtual int Next();    // 在[0, System.Int32.MaxValue)之间的随机整数
public virtual int Next(int minValue, int maxValue);    // 在[minValue, maxValue)之间的随机整数
public virtual int Next(int maxValue);    // 在[0, maxValue)之间的随机整数
public virtual double NextDouble();        // 在[0.0, 1.0)之间的双精度浮点数
protected virtual double Sample();        // 在[0.0, 1.0)之间的双精度浮点数
```

### 4. string类

System.string类是引用类型的一种，表示一个Unicode字符序列，一个字符串可存储约231个Unicode字符。需要注意的是，string类的索引函数是只读的，其各种操作方法不是修改字符串本身，而是生成新的字符串（占用新的内存空间，旧的没用引用的字符串会等待CLR回收），显然如果这种操作非常多，对内存的消耗是很大的。

使用string.Format()方法可以将字符串格式化（Console.WriteLine()按格式输出会调用调用string的Format方法），它要使用格式参数，其一般形式如`{N[,M][:format]}`，其中N是以0为起始编号的、将被替换的参数号码；M是一个可选整数，表示最小宽度值，M为负数表示左对齐，M为正表示右对齐，若M大于实际参数的长度，则用空格填充；若实际参数比占位符长，则输出实际参数。format是一个可选参数，其含义见下面。

- `C`，按金额形式输出，如WriteLine("{0:C}", 50)输出￥50。
- `D`，按整数输出。
- `E`，用科学计数格式输出。
- `F#`，小数点后位数固定，其中#是数字表示小数位数，如WriteLine("{0:F3}", 23.4)输出23.400。
- `N#`，输出带有千位分隔符的数字，其中#是数字表示小数位数，如WriteLine("{0:N2}", 19800)输出19,800.00。
- `P#`，百分比格式分出，其中#是数字表示小数位数，如WriteLine("{0:P0}", 80)输出80%。
- `0`，数字或0占位符（若占位符比实际参数长，则用0填充；若实际参数比占位符长，则输出实际参数），如WriteLine("{0:0000.000}", 456.78)输出0456.780。
- `#`，数字占位符（若占位符比实际参数长，则不占位；若实际参数比占位符长，则输出实际参数），如WriteLine("{0:####.000}", 456.78)456.780。
- `.`，小数点，如WriteLine("{0:#.0}", 123)输出123.0。
- `,`，数字分隔符，如WriteLine("{0:#,###.000}", 456.78)输出3,456.780。
- `%`，百分号，如WriteLine("{0:0.00%}", 0.80)输出80.00%。

除此之外，string类和对象还拥有一些常用的操作方法，如Compare、Equals、IsNullOrEmpty、IndexOf、LastIndexOf、Insert、Remove、Replace、Split、ToCharArray、Substring、ToUpper、ToLower、TrimStart、TrimEnd、Trim、IsNumber等。

### 5. StringBuilder类

System.Text.StringBuilder类可以对字符串进行动态管理，其操作不再生成一个新实例，它允许直接在原字符串内存空间修改字符串本身的内容，动态地分配占用的内存空间大小；在字符串处理操作比较多的情况下，使用StringBuilder类可以显著提高系统性能。

StringBuilder类与String类的有很多相似操作方法，通过StringBuilder的ToString方法就可以获得其中的字符串。除此之外，StringBuilder类还提供了Capacity和MaxCapacity属性，分别表示字符串的初始容量和最大容量；还有Append、AppendLine、AppendFormat向其实例的尾端追加字符串、追加一行字符串、按格式追加字符串。

### 6. Array类

在.NET Framework环境中，并不能直接创建System.Array类型的变量，但是所有的数组都可以隐式地转换为Array类型，这样就可以在数组中使用Array类定义的一系列属性和方法，常见的如下。

- 属性Rank，用于获取数组的维数（又称秩）。
- 属性Length，用于获取数组所有维数中元素的总和。
- `GetLength(dimension)`，于获取指定dimension维中元素的个数，其中dimension从0开始，它表示维数指定是从[i,j,k,...]左端开始数起，所表示的对应维中元素的个数。
- `GetLowerBound(dimension)`和`GetUpperBound(dimension)`方法分别用于获取指定dimension维的第一个元素和最后一个元素的索引。
- `System.Array.Sort(array)`，对指定数组升序排序。
- `System.Array.Reverse(array)`，对数组元素进行逆序，即首尾倒置。
- `System.Array.Copy(srcArr, destArr, length)`，将srcArr复制到destArr中。
- `System.Array.IndexOf(array, value)`，返回指定数组中与value匹配元素的索引。
- `System.Array.BinarySearch(array, value)`，对已排序的数组进行二分查找，返回value元素的索引，若没有对应元素则返回一个负值。
- `System.Array.Clear(array, index, length)`，将数组中的一系列元素置0、false、null等，具体取决于元素类型。

### 7. 并行计算Parallel类

自.NET 4.0起，C#引入了Parallel类，对并行开发提供支持。Parallel类提供了Parallel.For、Parallel.ForEach、Parallel. Invoke等方法，其中Parallel.Invoke用于并行调用多个任务。

# 七、Windows窗体应用

C#的图形库，由Component作为基类，它启用应用程序之间共享的对象；Component派生出Control，Control定义控件的基类，控件是带有可视化表示形式的组件；Control向下派生出一系列组件（如Button、TextBox），同时派生出ContainerControl，可用作其他控件的容器的控件提供焦点管理功能；ContainerControl派生出Form，表示组成应用程序的用户界面的窗口或对话框，用户自定义的窗体类通常继承自Form。

可以在Visual Studio IDE中新建项目“Windows 窗体应用（.NET Framework）”，开发环境将会自动创建一个空的Windows窗体应用，及其所需的文件。初始创建后有三个文件Program.cs（含有程序的Main方法入口）、Form1.cs（可视化窗体设计器与代码）、Form1.Designer.cs（窗体设计器自动生成的代码文件）。

除了上述方法，也可以新建一个空项目，并在解决方案的引用中添加对System.Windows.Forms.dll、System.Drawing.dll等动态库的引用，然后就可以手动编写Windows窗体应用的代码了，如下所示。

```c#
using System.Drawing;
using System.Windows.Forms;

public static void Main()
{
    Button b = new Button();
    b.Text = "Click me";
    b.Location = new Point(200, 100);
    b.Size = new Size(100, 50);
    b.Click += delegate (object sender, EventArgs e)
    {
        MessageBox.Show("Hello World!");
    };

    Form f = new Form();
    f.Text = "MyFormTitle";
    f.Size = new Size(500, 250);
    f.Controls.Add(b);

    Application.Run(f);
}
```

## （一）控件的基本概念与操作

.NET提供了许多控件类，如窗体Form类、文本框TextBox类，它们都处于System.Windows.Forms名称空间中。每一个控件类都具有自己的属性、方法、和能够响应的外部事件。

### 1. 常用属性

下面列出了大多数控件都具有的常用属性。

- `Name`，获取或设置控件的名称，名称是控件的标识，任何控件都具有该属性。
- `Text`，获取或设置控件的标题文字。
- `Width`、`Height`，获取或设置控件的宽度、高度，即大小。
- `Size`，获取或设置控件的大小，其类型是System.Drawing.Size结构体，它有Height和Width属性。
- `Left`、`Top`，获取或设置控件的位置。
- `Location`，获取或设置控件的位置，即窗体左上角的坐标值，其类型是System.Drawing.Point结构体，它有X和Y属性。
- `Visible`，获取或设置控件是否可见。
- `ForeColor`、`BackColor`，获取或设置控件的前景色、背景色。
- `Font`，获取或设置控件的字体。
- `BorderStyle`，获取或设置控件的边框。
- `AutoSize`，获取或设置控件是否自动调整大小。
- `Anchor`，获取或设置控件的哪些边缘锚定到其容器边缘，使控件的位置相对窗体的某一边固定，改变窗体的大小时，控件的位置会随之改变，但其与窗体的相对距离不变。其值是System.Windows.Forms.AnchorStyles枚举类型，他有None、Top、Bottom、Left、Right五个值，使用多个时用`|`将它们合起来。
- `Dock`，获取或设置控件停靠到父容器的哪一个边缘，无论窗体的大小如何变化，控件总是会自动调整大小和位置以保持停靠不变。其值是System.Windows.Forms.DockStyle枚举类型，它有None、Top、Bottom、Left、Right、Fill六个值，同一时刻只能有一个值生效。
- `TabIndex`，获取或设置控件的Tab键顺序。
- `TextAlign`，获取或设置文本对齐方式。
- `Cursor`，获取或设置鼠标移到控件上时，被显示的鼠标指针的类型。

### 2. 鼠标事件

通过鼠标触发的事件称为鼠标事件。鼠标通常有左中右三个按钮，它的基本操作方式主要有按下、松开、单击、双击、移动等几种，这些操作均能触发相应的事件，下表列出了鼠标的操作及其所触发的事件。

| 事件名称    | 操作                               | 事件参数类型   |
| :---------- | ---------------------------------- | -------------- |
| Click       | 单击鼠标左键时发生                 | EventArgs      |
| DoubleClick | 双击鼠标左键时发生                 | EventArgs      |
| MouseEnter  | 鼠标进入控件的可见部分时发生       | EventArgs      |
| MouseLeave  | 鼠标离开控件的可见部分时发生       | EventArgs      |
| MouseHover  | 鼠标在控件内保持静止一段时间后发生 | EventArgs      |
| MouseDown   | 鼠标指针在控件上并按下鼠标键时发生 | MouseEventArgs |
| MouseUp     | 鼠标指针在控件上并松开鼠标键时发生 | MouseEventArgs |
| MouseMove   | 移动鼠标时发生                     | MouseEventArgs |

其中MouseEventArgs事件参数类型含有属性：

- `Button`，获取曾按下的是哪个鼠标按钮。其值是System.Windows.Forms.MouseButtons枚举类型，它有None、Left、Right、Middle等值。
- `Clicks`，获取按下并松开鼠标按钮的次数。
- `X`、`Y`，获取鼠标在产生鼠标事件时的x、y坐标。

鼠标的某个操作有可能会触发一系列事件，如单击鼠标操作实际上会触发三个事件，即MouseDown、MouseUp、和Click事件。

下面是一个鼠标事件的例子。

```c#
aButton.MouseUp += delegate (object sender, MouseEventArgs e)
{
    if (e.Button == MouseButtons.Left)
    {
        MessageBox.Show("Left");
    }
    else if (e.Button == MouseButtons.Right)
    {
        MessageBox.Show("Right");
    }
};
```

### 3. 键盘事件

当用户按下键盘的某个键时，如果该键有对应的ASCII值，就会发生KeyPress事件，随后便发生KeyDown事件；如果该键没有对应的ASCII值（如Shift、Ctrl、Alt、F1、F2等键），就只发生KeyDown事件。当用户松开键盘的某个按键时，就会发生KeyUp事件。下表列出了键盘操作所触发的事件。

| 事件名称 | 操作                         | 事件参数类型      |
| :------- | :--------------------------- | :---------------- |
| KeyPress | 按下一个ASCII键时被触发      | KeyPressEventArgs |
| KeyDown  | 按下键盘上任意键时被触发     | KeyEventArgs      |
| KeyUp    | 松开键盘上任意一个键时被触发 | KeyEventArgs      |

其中KeyPressEventArgs事件参数类型含有属性：

- `KeyChar`，获取或设置与按下的键对应的字符。
- `Handled`，获取或设置一个值，该值指示该KeyPress事件是否被处理过，若为true则表示被处理过，该事件将不会再继续传递被默认控件处理程序处理。

其中KeyEventArgs事件参数类型含有属性：

- `Shift`、`Control`、`Alt`，获取一个值，该值指示是否曾按下Shift、Ctrl、Alt键。
- `Handled`，获取或设置一个值，该值指示是否处理过此事件，为true则绕过该控件的默认处理，为false则将沿事件传递给默认控件处理程序。
- `KeyCode`，获取事件KeyDown或KeyUp的键盘代码，其值是System.Windows.Forms.Keys枚举类型，它标识了键盘的所有按键。
- `KeyData`，获取事件KeyDown或KeyUp的键盘数据，其值是System.Windows.Forms.Keys枚举类型，表示已按下的键代码，同时包含了Shift、Ctrl、Alt键的修饰符标志。
- `KeyValue`，获取事件KeyDown或KeyUp的键盘值，用整数表示KeyCode属性。
- `Modifiers`，获取事件KeyDown或KeyUp的修饰符标志，其值是System.Windows.Forms.Keys枚举类型，标志按下的Shift、Ctrl、Alt键，可包含一个或多个修饰符标志。

下面是一个键盘事件的例子。

```c#
aForm.Load += (object sender, EventArgs e) =>
{
    aForm.KeyPreview = true;    // 窗体优先接受键盘事件
};
aForm.KeyDown += (object sender, KeyEventArgs e) =>
{
    if (e.Control && e.KeyCode == Keys.V)
    {
        MessageBox.Show("禁止使用【Ctrl+V】粘贴");
        Clipboard.Clear();      // System.Windows.Forms.Clipboard类操控系统粘贴板
    }
};
aTextBox.KeyPress += (sender, e) =>
{
    if (e.KeyChar < '0' || e.KeyChar > '9')
    {
        e.Handled = true;   // 只允许输入0~9的数字
    }
};
```

## （二）窗体

窗体通常是一个矩形的屏幕显示区域，它可以是标准窗口、多文档界面（MDI）窗口和对话框等。C#的System.Windows.Forms.Form类组成应用程序的用户界面的窗口或对话框，它派生自ContainerControl类，ContainerControl可用作其他控件的容器的控件提供焦点管理功能。用户可以通过继承Form类来实现自定义窗体。

窗体由以下四部分组成：

- 标题栏，显示该窗体的标题，标题的内容由该窗体的Text属性决定。
- 控制按钮，提供窗体最小化、最大化、关闭的控制。
- 边界，限定窗体的大小，可以有不同样式的边界。
- 窗口区，这是窗体的主要部分，应用程序的其他对象可放在上面。

### 1. 窗体的属性

窗体的属性决定了窗体的外观，除前述的公共属性外，下面列出了窗体特有的常用属性。

- `Controls`，获取包含在窗体内的控件的集合，其值是System.Windows.Forms.Control.ControlCollection类，它有Add、Clear、Find等方法以及索引器，用来访问集合中的控件。
- `KeyPreview`，表示在将键盘事件传递到具有焦点控件前，窗体是否接收此事件。

- `BackgroundImage`，窗体的背景图像，其值是System.Drawing.Image类型。
- `Icon`，窗体的图标，其值是System.Drawing.Icon类型。
- `ControlBox`，窗体上是否显示控制菜单，默认为true显示控制菜单。
- `FormBorderStyle`，窗体边界的类型，它会影响标题栏及其上按钮的显示，其值是System.Windows.Forms.FormBorderStyle枚举类型，它有七个值。
- `MinimizeBox`、`MaximizeBox`，窗体上是否显示最小化、最大化按钮，属性默认值为true。
- `Opacity`，窗体的透明度，其值为1.00（默认值）时窗体完全不透明，其值为0时，窗体完全透明。
- `ShowInTaskbar`，表示窗体是否出现在任务栏中。
- `StartPosition`，表示运行时窗体的起始位置，其值是System.Windows.Forms.FormStartPosition枚举类型，它有Manual（此时由Location属性确定）、CenterScreen、WindowsDefaultLocation、WindowsDefaultBounds、CenterParent五个值。
- `TopMost`，表示窗体是否应显示为应用程序的最顶层窗体。
- `WindowState`，表示窗体运行时正常、最小化、最大化的三种状态，其值是System.Windows.Forms.FormWindowState枚举类型，它有Normal、Minimized、Maximized三个值。

其中FormBorderStyle属性是System.Windows.Forms.FormBorderStyle枚举类型，它有七个值：

- `None`，无边框，可以改变大小。
- `Fixed3D`，固定的三维边框效果，不允许改变窗体大小，可以包含控制菜单、最大化、最小化按钮。
- `FixedDialog`，固定的对话框，不允许改变窗体大小，可以包含控制菜单、最大化、最小化按钮。
- `FixedSingle`，固定的单线边框，不允许改变窗体大小，可以包含控制菜单、最大化、最小化按钮。
- `Sizable`（默认值），双线边框，可重新设置窗体的大小，可以包含控制菜单、最大化、最小化按钮。
- `FixedToolWindow`，固定的工具窗口，不允许改变窗体大小，只带有标题栏和关闭按钮；需手动确保System.Windows.Forms.Form.ShowInTaskbar属性为false。
- `SizableToolWindow`，可调整大小的工具窗口，只带有标题栏和关闭按钮；需手动确保System.Windows.Forms.Form.ShowInTaskbar属性为false。

创建一个窗体并显示的例子如下。

```c#
aButton.MouseClick += delegate (object sender, MouseEventArgs e)
{
    Form nForm = new Form();
    nForm.Text = "Pen Box";
    nForm.ShowInTaskbar = false;
    nForm.FormBorderStyle = FormBorderStyle.FixedToolWindow;
    nForm.Show();
};
```

### 2. 窗体的方法

窗体方法定义了窗体的行为，下面列出了窗体一些常用方法。

- `Show()`，以非模式窗口显示窗体，不关闭此窗体的情况下，可以操作其他窗体。
- `ShowDialog()`，以模式窗口显示窗体，并返回一个值，以标识用户在该窗体中选择了哪个按钮。必须关闭模式窗体才能操作其他窗体。
- `Activate()`，激活窗体使其获得焦点。
- `CenterToScreen()`，使窗体居中。
- `Close()`，关闭窗体，并从内存中清除。
- `Dispose()`，关闭窗体，并释放其使用的所有资源。
- `Hide()`，隐藏窗体，不从内存中清除。
- `Refresh()`，强制控件使其工作区无效并立即重绘自己和所有子控件。
- `Update()`，引起控件重绘其工作区内的无效区域，常用于图形刷新。
- `Contains(Control ctl)`，判断指定控件是否为指定窗体的子控件。
- `GetNextControl()`，按照窗体上子控件的Tab顺序向前或向后检索下一个控件。

### 3. 窗体事件

窗体事件定义了如何同窗体进行交互，窗体能响应所有的鼠标事件和键盘事件，还能响应其他一些事件。除了前面介绍的公共事件，下面列出了其他常见的窗体事件。

- `Load`，在将窗体装入内存时发生，该事件过程主要用来进行一些初始化操作。
- `Activate`，当窗体变为活动窗体时发生。
- `DeActivate`，当窗体由活动状态变成不活动状态时，则该事件发生。
- `FormClosing`、`FormClosed`，用户关闭窗体时，事件分别在窗体关闭前、关闭后发生。
- `Paint`，在窗体重新绘制时发生。
- `ReSize`，在窗体初次装载或用户改变其大小后发生。
- `Move`，移动窗体时发生。

## （三）基本控件

.NET Framework提供了众多的控件类，这些控件大都有常用的属性、事件、方法，一些基本的常用的控件类如下。

| 控件                             | 说明                         | 主要属性                                                     | 主要方法                                                     | 主要事件                                 |
| -------------------------------- | ---------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ---------------------------------------- |
| Label                            | 标签用于显示文本或图片       | AutoSize、Image、ImageList、ImageIndex、Text                 |                                                              |                                          |
| LinkLabel                        | 连接标签用来显示网址链接     | LinkArea、Links、Text                                        |                                                              | LinkClicked                              |
| TextBox                          | 文本框控件                   | PasswordChar、MaxLength、MultiLine、ScrollBars、Text、SelecledText | Clear、Select、AppendText                                    | TextChanged、KeyDown、KeyPress           |
| Button                           | 按钮控件                     | Enable、Image、ImageList、ImageIndex、Text                   |                                                              | Click                                    |
| CheckBox、RadioBox               | 复选框、单选按钮             | Checked、CheckState、Image、ImageList、ImageIndex            |                                                              | CheckedChanged、Click、CheckStateChanged |
| ListBox                          | 列表框                       | Items、MultiColumn、SelectedIndex、SelectedItem、SelectionMode、Text | ClearSelected、Items.Add、Items.Clear、Items.Insert、Items.Remove | SelectedlndexChanged                     |
| ComboBox                         | 组合框，结合了列表框和文本框 | 同ListBox、DropDownStyle                                     | 同ListBox                                                    | 同ListBox                                |
| CheckedListBox                   | 复选列表框，扩展了列表框     | 同ListBox、CheckcdOnClick                                    | 同ListBox、GetItemCheckState、SetItemCheckState              | 同ListBox                                |
| NumericUpDown                    | 数值调节控件                 | Increment、Maximum、Minimum、Value、InterceptArrowKeys       |                                                              | ValueChanged                             |
| DomainUpDown                     | 文本值调节控件               | Items、SelectedIndex、SelectedItem、Sorted、Wrap             |                                                              | SelectedItemChanged                      |
| HScrollBar、VScrollBar、TrackBar | 滚动条控件、滑块控件         | Maximum、Minimum、Value、SmallChange、LargeChange            |                                                              | Scroll、ValueChanged                     |
| ProgressBar                      | 进度条                       | Maximum、Minimum、Value                                      | Increment、PerformStep                                       |                                          |
| Timer                            | 定时器控件                   | Enable、Interval                                             | Start、Stop、Dispose                                         | Tick                                     |
| DateTimePicker                   | 日期时间选择控件             | Format、MaxDate、MinDate、Text、Value                        |                                                              |                                          |
| MonthCalendar                    | 日期范围选择控件             | MaxDate、MinDate、MaxSelectionCount、ShowToday、TodayDate    |                                                              |                                          |
| PictureBox                       | 专门用来显示图片的控件       | Image、ImageList、ImageIndex、SizeMode                       |                                                              |                                          |
| ToolTip                          | 显示提示信息的控件           | Active、AutomaticDelay、AutoPopDelay、InitialDelay           | SetToolTip(cotr, "message")                                  |                                          |

## （四）对话框

MessageBox，消息框通常用来显示一些提示警告信息，使用`DialogResult MessageBox.Show(string text[, string caption, MessageBoxButtons buttons, MessageBoxIcon icon, MessageBoxDefaultButton defaultButton])`方法。

- 枚举`MessageBoxButtons`类型的值有，OK、OKCancel、AbortRetryIgnore、YesNoCancel、YesNo、RetryCancel。
- 枚举`MessageBoxIcon`类型的值有，None、Hand、Stop、Error、Question、Exclamation、Warning、Asterisk、Information。
- 枚举`MessageBoxDefaultButton`类型的值有，Button1、Button2、Button3。
- 返回的枚举`DialogResult`类型的值有，None、OK、Cancel、Abort、Retry、Ignore、Yes、No。

C#的通用对话框派生自System.Windows.Forms.CommonDialog类，它们主要的方法是DialogResult ShowDialog()，其常用属性如下。

| 控件                           | 说明                     | 主要属性                                                     |
| ------------------------------ | ------------------------ | ------------------------------------------------------------ |
| OpenFileDialog、SaveFileDialog | 文件打开、保存文件对话框 | AddExtension、DefaultExt、FileName、FileNames、InitialDirectory、RestoreDirectory、Title、ShowHelp |
| ColorDialog                    | 颜色对话框               | AllowFullOpen、AnyColor、Color、CustomColors、CustomColors、ShowHelp |
| FontDialog                     | 字体对话框               | AllowVectorFonts、AllowVerticalFonts、Color、Font、MaxSize、MinSize、ShowApply、ShowColor |
| PrintDialog                    | 打印对话框               | AllowPrintToFile、Document、PrintToFile                      |

## （五）容器类控件

容器类控件能够包容其他控件。

| 控件            | 说明                                                         | 主要属性                                                     | 主要方法                                                     |
| --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| GroupBox、Panel | 组框、面板主要用来对控件进行分组                             | Controls                                                     | Controls.Add、Controls.Remove                                |
| ImageList       | 不可见的控件，主要用于存储图像，为ToolStrip、ListView、TreeView提供图像 | Images、ImageSize                                            | Images.Add、Images.Remove                                    |
| ListView        | 采用大图标、小图标、列表、详细信息、标题五种形式显示一个条目列表 | CheckBoxes、Columns、Items、MultiSelect、SelectedItems、View、LargeImageList、SmallImageList | SelectedIndexChanged                                         |
| TreeView        | 以树形结构展示数据，其节点是TreeNode类型                     | ShowLines、CheckBoxes、ImageList、Nodes、SelectedNode        | AfterSelect                                                  |
| #TreeNode       | 作为TreeView节点的类型                                       | Checked、PrevNode、NextNode、Nodes、FirstNode、LastNode、FullPath、ImageIndex、SelectedImageIndex、Text | Collapse、Expand、ExpandAll、GetNodeCount、Add、Remove、Clear |
| TabControl      | 用来管理一组TabPage对象，使用该控件可以创建“选项卡对话框”    | Alignment、ImageList、MultiLine、SelectedIndex、SelectedTab、TabCount、TabPages | TabPages.Add、TabPages.Remove                                |
| SplitContainer  | 分隔条控件，它是一个由拆分器组成的控件，拆分器将控件分为可左右调节大小的两部分 | IsSplitterFixed、SplitterWidth、Orientation                  |                                                              |

## （六）菜单

菜单主要分为两种，即主菜单`MenuStrip`类和快捷菜单`ContextMenuStrip`。

由MenuStrip控件创建的菜单项可以是MenuItem、ComboBox、TextBox，其中只有MenuItem可有子菜单。MenuStrip的主要属性有：Text、TextAlign、Name、Checked、CheckOnClick、Enabled、Visible、ToolTipText、DropDownItems（子菜单）；主要事件有Click。

快捷菜单ContextMenuStrip又称右键菜单，它创建的菜单项可以是ComboBox、TextBox、Separator。在使用时需要将ContextMenuStrip对象cms绑定到持有它的窗体上，使用窗体的ContextMenuStrip属性，即cntr.ContextMenuStrip=cms。

## （七）工具栏与状态栏

工具栏`ToolStrip`类和状态栏`StatusStrip`类都是Windows应用程序中常用的控件。

工具栏是一个显示一组动作、命令或功能的组件，一般说来，工具栏中的组件都是带图标的按钮，可以使用户更加方便地选择所需的功能。其主要属性有：AllowItemReorder、Checked、CheckOnClick、Enabled、Enabled、Enabled、Size、Text、Visible；主要事件有ItemClicked，其事件参数类型是ToolStripItemClickedEventArgs，该事件类型有属性ClickedItem.Name，可以在switch语句中执行分支。

## （八）多文档界面

多文档界面（Multiple Document Interface，MDI）由父窗口和子窗口组成。MDI父窗口是子窗口的容器，MDI子窗口显示各自的文档，所有子窗体均显示在MDI父窗体的工作区中，用户可以改变子窗体大小或移动子窗体，但被限制在MDI父窗体中。

MDI父窗体是一个从Form类派生的窗体类，首先创建一个普通窗体，然后将IsMdiContainer属性设置为true即可，此时该窗体就成为一个MDI父窗体。MDI子窗体就是一个常规的Form派生类，在显示它之前，把其MdiParent属性设置为要包含它的MDI主窗体。

MDI父窗体的主要属性有：IsMdiContainer、MdiChildren、ActiveMdiChild（当前活动的MDI子窗体）。MDI子窗体的主要属性有：IsMdiChild、MdiParent。常用的多窗体方法有：Show、ShowDialog、Close、Dispose、Form.LayoutMdi(MidLayout value)。主要事件有：Form.MdiChildActivate，当激活或关闭一个MDI子窗体时将发生该事件。

# 八、图形图像编程

GDI+（Graphics Device Interface Plus）提供了二维矢量图形、文本、图像处理、区域、路径以及图形数据矩阵等方面的类，其中，图形类Graphics是GDI+接口中的一个核心类，许多绘图操作都可用它完成。

GDI+所有图形图像处理功能都包含在如下名称空间中：

- System.Drawing，提供了基本图形功能，主要有Graphics类、Bitmap类、Brush的各种派生类、Font类、Icon类、Image类、Pen类、Color类等。
- System.Drawing.Drawing2D，提供了高级的二维和矢量图形功能，主要有梯度型画刷、Matrix类（用于定义几何变换）和GraphicsPath类。
- System.Drawing.Design，包含扩展设计时用户界面逻辑和绘制的类。
- System.Drawing.Imaging，提供了高级GDI+图像处理功能。
- System.Drawing.Text，提供了高级GDI+字体和文本排版功能。
- System.Drawing.Printing，提供了与打印相关的服务。

GDI+图形图像处理功能，经常使用的一些结构有如下。

- Point、PointF，属性有，X、Y。
- Rectangle、RectangleF，属性有，X、Y、Width、Height、Top、Left、Bottom、Right、Location、Size。
- Size、SizeF，属性有，Width、Height。
- Color，创建方法有Color.FromArgb(int a, int r, int g, int b)，也可使用预定义的常见颜色。

## （一）Graphics类

### 1. 创建Graphics对象

创建Graphics对象方法有三种：

1. 从当前窗体获取对Graphics对象的引用，并作为该窗体的成员。调用某控件或窗体的CreateGraphics方法来获取对Graphics对象的引用，该对象表示该控件或窗体的绘图表面。如果想在已存在的窗体或控件上绘图，则可使用此方法。

```c#
public class MyForm
{
    private Graphics graphics;
    public void foo() 
    {
        graphics = CreateGraphics();
        // ...
        graphics.Dispose();
    }
}
```

2. 在窗体或控件的Paint事件中，直接引用事件对象的Graphics对象。每一个窗体或控件都有一个Paint事件，该事件的参数中包含了当前窗体或控件的Graphics对象，在为窗体或控件创建绘制代码时，一般使用此方法来获取对图形对象的引用。

```c#
private void aFormPaintEventHandler(object sender, PaintEventArgs e)
{
    Graphics g = e.Graphics;
    // ...
    g.Dispose();
}
```

3. 从Bitmap类或其子类的对象创建Graphics对象，使用Graphics.FromImage(Bitmap b)方法，此方法在需要更改已存在的图像时十分有用。

```c#
public class MyForm
{
    private Bitmap bitmap;
    private Graphics graphics;
    public void foo()
    {
        bitmap = new Bitmap();
        this.BackgroundImage = bitmap;
        graphics = Graphics.FromImage(bitmap);
        // ...
        graphics.Dispose();
    }
}
```

需要注意的是，上述第一种把窗体或控件本身作为Graphics对象来画图的，画出来的图像是暂时的，如果当前窗体被切换或被其他窗口覆盖，这些图像就会消失。为使图像永久地显示可使用第二种方法，不过它只适合于显示的图像固定不变的情况。要使画出的图像能自动刷新，可使用第三种方法，即建立一个Bitmap对象，在其上绘制图像后，将其赋给窗体或控件的Bitmap对象，这样绘制出的图就能自动刷新，不需要程序来重绘图像。

### 2. Graphics类绘图方法

Graphics类中提供了许多绘图方法，如下所示。

- Clear，使用一种指定的颜色填充整个绘图表面。
- DrawLine，绘制直线；DrawLines，绘制多条直线。
- DrawCurve，绘制曲线；DrawClosedCurve，绘制闭合曲线；DrawArc，绘制圆弧。
- DrawBezier，绘制三维贝塞尔曲线；DrawBeziers，基于Point数组绘制一系列三维贝塞尔曲线。
- DrawRectangle，绘制矩形；DrawRectangles，绘制Rectangle数组中给出的多个矩形。
- DrawEllipse，绘制椭圆；DrawPolygon，绘制多边形。
- Drawicon，绘制图标；DrawImage，绘制图像。
- DrawPie(pen, rect, startAngle, sweepAngle)，绘制饼图（angle以角度为单位）。
- DrawPath，绘制路径。
- DrawString，在指定位置以指定字体显示字符串。
- FillRectangle，填充矩形；FillRectangles，填充多个矩形；FillClosedCurve，填充闭合曲线。
- FillEllipse，填充楠圆；FillPolygon，填充多边形；FillRegion，填充一个区域。
- FillPie，填充饼图。
- FillPath，填充路径。
- GetHDC，返回与Graphics相关联的设备句柄。
- ReleaseHDC，释放设备句柄。

上面一系列方法中，画单个图形的方法DrawXXX的参数通常表示一个图形，画多个图形的方法DrawXXXs的参数通常表示多个图形。上面这些方法的参数，通常第一个参数是Pen（对应的FillXXX方法的第一个参数是Brush），如果需要的话第二个参数是Rectangle表示在一个矩形上画图形，如果不需要则从第二个参数开始就是表示图形的参数。对于表示图形的一些参数Point、Rectangle来说，它们可以用(x,y)、(x,y,width,height)相互替换。

由于图像对象非常占用系统资源，所以在不用这些对象时，要及时使用Dispose方法释放占用的资源，否则将会严重影响系统的性能。

## （二）Pen类

在GDI+中，可使用笔（Pen）对象和画刷（Brush）对象绘制或填充图形、文本和图像。

画笔Pen对象的常用属性如下：

- `Brush`，获取或设置用于确定此Pen对象属性的Brush对象。
- `Color`，获取或设置此Pen对象的颜色。
- `Width`，获取或设置此Pen对象的宽度。
- `StartCap`、`EndCap`，获取或设置用于通过此Pen对象绘制的直线起点、终点的帽样式
- `DashStyle`，获取或设置用于通过此Pen对象绘制的线的样式。它是Drawing2D.DashStyle枚举类型，其值有Solid实线、Dot点构成的直线、Dash短划线构成的直线、DashDot、DashDotDot、Custom自定义。

画刷Brush对象，决定如何填充图形形状（如矩形、椭圆形、扇形、多边形、封闭路径）内部，它有五个派生类：SolidBrush(Color)单色画刷、TextureBrush(Image)纹理画刷、HatchBrush阴影画刷、LinearGradientBrush、线性渐变画刷、PathGradientBrush路径渐变画刷。

## （三）图像显示与保存

除了PictureBox控件可以显示图像之外，System.Drawing.Bitmap类对象也可以显示图像文件，且Bitmap对象还能把绘制的图形保存到文件中；Bitmap类是Image的子类。可以使用GDI+显示多种格式的图像，如Jpeg、Png、Bmp、Gif、Tiff、Emf等。

图像可以使用Image对象表示，可以用它的`Image.FromFile(string filename)`、`Image.FromStream(stream)`从文件和流中加载一个图像，它的`Image.Save(string filename)`保存图像。

显示和保存图像可以按如下步骤进行。

```c#
Bitmap aBitmap = new Bitmap(aFilename);
aGraphics.DrawImage(aBitmap, aPoint);
aBitmap.Save(aSaveFilename, System.Drawing.Imaging.ImageFormat.Jpeg);
```

# 九、数据库

ADO.NET包含两个核心组件，数据提供程序（Data Provider）主要负责数据访问，DataSet主要负责数据的操作。

数据提供程序的核心类库有XXConnection、XXCommand、XXDataReader、XXDataAdapter，其中XX根据数据库源不同，可以为Sql、OleDb、ODBC、Oracle四种之一，在使用它们访问数据库时，需要添加引用相应数据库提供的驱动程序。

.NET数据提供程序组件中包含的主要对象如下：

- XXConnection，表示与特定数据库的连接。
- XXCommand，数据命令对象，执行用于返回数据、修改数据、运行存储过程以及发送或检索参数信息的数据库命令。
- XXParameter，为Command对象提供参数。
- XXDataAdapter，数据适配器，该对象是连接DataSet对象和数据库的桥梁，DataAdapter使用Command对象在数据库中查询数据，并将数据加载到DataSet中，对DataSet中数据的更改也由其更新回数据库中，使数据库与数据集数据保持一致。
- XXCommandBuilder，为XXDataAdapter对象创建命令属性或将从存储过程派生的参数信息填充到Command对象的Parameters集合中。
- XXDataReader，数据读取对象，从数据库中读取记录集。
- XXTransaction，事务对象，实现事务操作。

DataSet又称为数据集，是支持ADO.NET断开式、分布式数据方案的核心对象，它允许从数据库中检索到的数据存放在内存中，可以用于多种不同的数据源，并提供一致的关系编程模型。DataSet包含一个或多个DataTable对象。DataTable对象由数据行（DataRow）和数据列（DataColumn），以及主键、外键、约束（Constrain）和有关DataTable对象中数据的关系信息组成。

## （一）直接访问数据库

下述的一些类都在相应数据库提供的驱动程序的命名空间里面。

使用Connection对象间建立一个到数据库的连接，其构造方法接受一个表示连接源的字符串，不同的数据库使用的连接字符串不同（如MySQL是`"Database=aDbName;Data Source=aIp;port=aPort;User Id=aUserName;Password=pw"`），它的Connection.Open方法打开这个连接，它的Connection.Close方法用于关闭这个连接。

直接访问模式是利用命令类Command直接对数据库进行操作，使用它一般有两种方法，一是直接执行SQL语句或存储过程完成数据的增删改，二是执行Command.ExecuteReader方法返回一个DataReader对象，将数据读到数据读取器DataReader中，再通过数据读取器来查询数据。

使用`Command([string cmdText, Connection conn, Transaction tran])`来构造一个Command对象，其中cmdText是初始化查询文本。Command的主要属性如下。

- `CommandText`，获取或设置要对数据源执行的Transact-SQL语句、表名、存储过程等。
- `CommandType`，获取或设置一个CommandType枚举值，该值指示如何解释CommandText属性；CommandType枚举有如下值，Text表示SQL文本命令（默认）、StoredProcedure表示CommandText是一个存储过程的名称、TableDirect表示CommandText是一个表的名称。
- `Parameters`，参数集合ParameterCollection，用于设置Command的参数，向SQL命令传递数据，执行参数査询。
- `CommandTimeout`，获取或设置在终止执行命令的尝试并生成错误之前的等待时间（毫秒）。
- `Connection`，获取或设置Command实例使用的Connection。
- `Transaction`，获取或设置将在其中执行Command的Transaction。

Command的主要方法如下。

- `CreateParameter()`，创建Parameter对象的新实例。
- `Dispose()`，关闭有关对象，释放资源。
- `ExecuteNonQuery()`，执行SQL语句，并返回受影响的行数。这里的SQL语句通常是create、insert、delete、update、以及其他没有返回值的语句。
- `ExecuteScalar()`，执行SQL查询，并返回查询所返回的结果集中第一行的第一列（一个Object对象），忽略其他列或行。这里的SQL语句通常是count、avg、min、max、sum等聚合函数，这些函数返回的都是单行单列的结果集。
- `ExecuteReader()`，将cmdTxt发送到Connection并生成一个DataReader，它能够尽可能快地对数据库进行査询并得到结果。DataReader对象是一个简单的数据集，用于从数据源中检索只读、仅向前数据集，读取数据速度快，常用于检索数据量较大的场合。
- `ExecuteXmlReader()`，将cmdTxt发送到Connection并生成一个XmlReader对象。

执行SQL查询命令通常使用Command.ExecuteReader方法，它返回一个DataReader对象（继承于DbDataReader）。DataReader类型常用的方法、常用的属性、索引器如下。

- `bool Read()`，使读取器前进到结果集中的下一条记录，如果存在更多行则返回true，否则为false。
- `bool NextResult()`，在读取一批语句的结果时，使读取器前进到下一个结果，如果存在更多结果集则返回true，否则为false。
- `bool IsDBNull(int ordinal)`，返回ordinal（从0开始的列索引）指定的列中是否包含System.DBNull或丢失的值。
- `string GetName(int ordinal)`，返回ordinal指定的列字段的名称。
- `int GetOrdinal(string name)`，返回列名是name的列的序号。
- `Type GetFieldType(int ordinal)`，返回ordinal指定的列字段的类型。
- `string GetDataTypeName(int ordinal)`，返回ordinal指定的列字段类型的名称。
- `object GetValue(int ordinal)`，返回当前行中，由ordinal指定的列的值。
- `int GetValues(object[] values)`，将当前行中所有列的值赋给values，并返回values的个数。
- `FieldCount`，该属性获取当前行中的列数。

- `this[int ordinal]`，该索引器根据列序号，获取当前行中的指定列的值。
- `object this[string name]`，该索引器根据列名，获取当前行中的指定列的值。

Parameter参数实际上就是cmdTxt中的占位字符串，格式为`@ParaName`，其中Parameter类型常用的属性如下。

- `ParameterName`，参数的名称，与在参数化SOL中出现的参数名要对应。
- `Direction`，指示参数类型，它是一个ParameterDirection枚举类型，其值有，Input表示只输入，Output表示只输出，InputOutput表示可输入输出，ReturnValue表示存储过程返回值。
- `SqlDbType`，参数的数据类型。
- `Value`，获取或设置参数的值。
- `Size`，获取或设置列中的数据的最大大小（以字节为单位），仅影响输入的参数值。
- `IsNulLabel`，指示参数是否接受空值。
- `SourceColumn`，获取或设置映射到的源列System.Data.DataSet的名称。
- `SourceVersion`，确定参数值使用的是原始值还是当前值Value参数的值。

Command的参数存储在它的Parameters属性中，该它是一个ParameterCollection类型的集合，它的主要操作有Add、Remove、以index和paraName为键的索引器。

一个较为完整例子如下。

```c#
string connStr = "Database=TEST;Data Source=localhost;port=3306;User Id=root;Password=xxxxxx";
MySqlConnection conn = new MySqlConnection(connStr);
conn.Open();

string cmdTxt = "SELECT * FROM Student where sName = @name";
MySqlCommand cmd = new MySqlCommand(cmdTxt, conn);
cmd.CommandType = System.Data.CommandType.Text;
cmd.CommandTimeout = 30;

MySqlParameter para = new MySqlParameter();
para.ParameterName = "@name";
para.Value = "Bloonow";
para.MySqlDbType = MySqlDbType.String;
para.Direction = System.Data.ParameterDirection.Input;
cmd.Parameters.Add(para);

MySqlDataReader dr = cmd.ExecuteReader();
while (dr.Read())
{
    int cnt = dr.FieldCount;
    for (int i = 0; i < cnt; ++i)
    {
        string str = dr.GetName(i) + " : " + dr[i];
        Console.WriteLine(str);
    }
}

conn.Close();
```

通过命令Command对象也可以调用数据库系统中的存储过程，通常使用Command.ExecuteNonQuery方法执行存储过程，只需将Command.ommandType属性设置为CommandType.StoredProcedure，并将Command.CommandText属性设置为存储过程名，如果命令采用参数则要设置参数。需要注意的是，这里用作在CommandText作参数的占位符名，要与数据库定义的存储过程中声明的参数名一样；如果存储过程中有返回参数，要使用相应的参数与之对应，并设置Parameter.Direction属性为ParameterDirection.Output，在执行完后可以直接使用返回参数对象Value属性访问到返回的值。

## （二）数据集DataSet模式

在命名空间System.Data里面。

DataSet作为数据库的临时数据容器，可以实现数据库的断开式访问，可以一次性将需要的数据装入DataSet中，等操作完成后一次性更新到数据库中。DataSet的数据源并不一定是关系数据库，还可以是文本、XML文件等，DataSet都提供了一致的编程模型。

DataAdapter又被称为数据适配器，用于从数据源中检索数据并充填数据集中表，并用于将数据集中数据的更改解析回数据源，达到数据库更新的目的。可以使用构造方法`DataAdapter([Command selectCommand, Connection connection])`来创建一个DataAdapter。

DataAdapter常用属性和方法如下。

- `SelectCommand`，该属性获取或设置用来选择数据源中的记录的Command命令。
- `InsertCommand`，该属性获取或设置用来将新记录插入到数据源的Command命令。
- `DeleteCommand`，该属性获取或设置用于从数据集中删除记录的Command命令。
- `UpdateCommand`，该属性获取或设置用于更新数据源中的记录的Command命令。
- `Fill(DataSet dataSet)`，方法用于初始化DataSet。
- `Update(DataRow[] dataRows)`、`Update(DataTable dataTable)`、`Update(DataSet dataSet)`，方法用于写回更改后的数据。

Update方法本质是为添加到DataTable中的行执行insert命令，为被修改的行执行update命令，为被删除的行执行delete命令，这些命令是由XXCommandBuilder对象自动生成的，因此在对数据集操作完进行更新Update之前，要创键一个XXCommandBuilder对象，即`XXCommandBuilder cb = new XXCommandBuilder(da);`。

DataSet的常用属性和方法如下。

- `DataSetName`，获取或设置的当前名称。
- `Tables`，获取集合中包含的表，其类型是DataTableCollection表集合，其实现了相应的索引器、Add、Remove、Clear方法。
- `Relations`，获取关系链接表，并允许导航从父表到子表的集合。
- `Clear()`，清除DataSet的所有表中删除所有行的任何数据。
- `DataTableReader CreateDataReader(params DataTable[] dataTables)`，返回指定表的DataTableReader，它继承于DbDataReader，常用方法见上面。

DataSet的Tables属性返回的是表的集合DataTableCollection，它是一个DataTable对象的集合，DataTableCollection提供了常用的索引器和方法，如下。

- `this[string name]`，获取具有指定名称的DataTable表。
- `this[int index]`，根据索引获取集合中对应的DataTable表。
- `DataTable Add(string name)`，创建一个名为name的新DataTable表，将其添加到集合，并返回这个新表。
- `Remove(DataTable table)`、`Remove(string name)`，删除集合中的表。
- `Clear()`，清除集合中的所有表。

DataTable表示内存中数据的一个表，它的常用属性、方法、事件如下。

- `TableName`，该表的名称。
- `Constraints`，获取此表是由约束的集合，其类型是ConstraintCollection列集合，其实现了相应的索引器、Add、Remove、Clear方法。
- `Columns`， 获取属于此表的列的集合，其类型是DataColumnCollection列集合，其实现了相应的索引器、Add、Remove、Clear方法。
- `Rows`，获取属于此表的行的集合，其类型是DataRowCollection列集合，其实现了相应的索引器、Add、Remove、Clear方法。
- `DataRow NewRow()`，创建一个与表相同结构的新行，并返回。
- `DataRow[] Select(string filterExpression[, string sort])`，用筛选条件，获取所有的匹配的DataRow的数组；筛选条件请参阅DataView RowFilter Syntax [C#]。
- `DataRowChangeEventHandler`、`DataColumnChangeEventHandler`、`DataTableClearEventHandler`，事件的委托类型，其事件有RowChanging、RowChanged等。

DataRow表示行中的数据，它的常用索引器、方法如下。

- `this[DataColumn column]`，按列获取数据。
- `this[string columnName]`，按列名获取数据。
- `this[int columnIndex]`，按列的索引获取数据。
- `void Delete()`，删除行。
- `SetNull(DataColumn column)`，置该行的某列为空。
- `IsNull(DataColumn column)`、`IsNull(string columnName)`、`IsNull(int columnIndex)`，判断该行的某列是否为空。

ADO.NET用System.Data.DataView的表示视图，它支持排序和筛选，它有三个常用属性，即Sort、RowFilter、RowStateFilter。

# 托管内存

在C#编程语言所开发的应用程序中，所使用的内存称为托管内存（managed memory）；当C#程序运行时，会向操作系统申请一块专用内存，即托管内存。在C#语言开发的程序中，所声明的变量，不论是常量还是变量，都位于托管内存。C#的托管内存，是具有自身管理功能的，并使用垃圾回收器（GC）工具，判断程序所声明的内存是否仍在使用，来实现对托管内存的管理。

C#程序所使用的内存，都叫托管内存，而C#程序不使用的内存，即是非托管内存。其它许多语言并没有托管内存的概念，例如C++语言，没有专门的内存管理机制。所以，在C#程序和其它语言开发的程序进行交互时，或使用C#进行混合开发时，需要使用C#语言访问非托管内存。