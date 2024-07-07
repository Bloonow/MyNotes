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

其中，一些以c作为前缀的头文件，是C++提供的对相应的C标准标头的包装实现。其中，<hash_map>，<hash_set>并不是ISO C++标准的一部分，但因为有着广泛的应用场景，编译器通常会提供实现，但它们并不位于std命名空间中，而是位于stdext命名空间或\_\_gnu\_cxx命名空间。

可按照任何顺序包括标准标头，可多次包括一个标准标头，或包括定义了相同宏或相同类型的多个标准标头。声明内不能包括标准标头。

如果需要某种类型的定义，则C++库标头会包含任何其他C++库标头。但是，用户应该始终显式包含编译单元中所需的任何C++库标头，以免弄错其实际依赖项。而C标准标头从不包含其他标准标头。库中的每个函数都在标准标头中声明，与C标准库不同，C++标准库不会提供屏蔽宏。一个屏蔽宏是指与某个声明函数的名称和作用都相同的宏定义，用于屏蔽对某个函数的调用。

除C++库标头中的operator new()和operator delete()以外的所有名称都在std命名空间中定义，或者在std命名空间内的嵌套命名空间中定义。在某些转换环境（包括C++库标头）中，使用using关键字，可以将在std命名空间中声明的外部名称提升到全局命名空间当中；否则，标头不会将任何库名称引入当前命名空间中。C++标准要求C标准标头对命名空间std中的所有外部名称进行声明，然后将它们提升至全局命名空间中。但在某些转换环境中，C标准标头不包含命名空间声明，所有名称都直接在全局命名空间中进行声明。
