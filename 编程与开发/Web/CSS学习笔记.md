# 一、CSS简介

CSS是指层叠样式表（Cascading Style Sheets），样式定义如何显示HTML元素。样式通常存储在样式表中，把样式添加到HTML 4.0中，是为了解决内容与表现分离的问题。

在HTML文档中使用样式表有多种方法，如内联样式、内部样式表、外部引用等，可以参看《HTML学习笔记》，而使用外部样式表可以极大提高工作效率，外部样式表通常存储在CSS文件中，多个样式定义可层叠为一个。

样式表定义如何显示HTML元素，就像HTML中的字体标签和颜色属性所起的作用那样。样式通常保存在外部的.css文件中，只需要编辑一个简单的CSS文档就可以改变所有页面的布局和外观。

注释是用来解释代码用的，并且可以随意编辑它，浏览器会忽略它。CSS注释与C语言类似，以`/*`开始，以`*/`结束。

# 二、CSS规则

## （一）CSS声明

CSS规则由两个主要的部分构成：选择器，以及一条或多条声明。CSS声明又称为CSS实例，每条声明由一个属性和一个值组成，属性（property）是所要设置的样式属性（style attribute），每个属性有一个值，属性和值被冒号分开。声明总是以`attr:value`的形式存在，并以分号`;`结束，一组声明总以大括号`{}`括起来。如下。

```css
h1 { color:blue; font-size:12px; }
```

- 上面的h1就是选择器，花括号中的就是一组声明。

## （二）CSS选择器

如果要在html标签中设置CSS样式，可以使用选择器，css选择器有如下几种。

### 1. 各种选择器

ID选择器（ID selector，IS）：使用#标识selector，语法格式：`#S{...}`（S为选择器名）。id选择器可以为标有特定id的HTML元素指定特定的样式，HTML元素以id属性来设置id选择器，CSS中id选择器以`#`来定义。如下所示。

```html
<head>
<style>
/* 定义一个id为para1的id选择器 */
#para1 { text-align:center; color:red; }
</style>
</head>
<body>
<p id="para1">Hello World!</p>
<p>这个段落不受该样式的影响</p>
</body>
```

- id属性不要以数字开头，数字开头的ID在Mozilla/Firefox浏览器中不起作用。
- 在同一HTML文档中，所有元素的id不能重复，因而CSS定义的id选择器在同一个HTML文档中只能使用一次。

类选择器（class selector，CS）：使用.标识selector，语法格式：`.S{...}`（S为选择器名）。class选择器用于描述一组元素的样式，class选择器有别于id选择器，class可以在多个元素中使用。class选择器在HTML中以class属性表示，在CSS中class选择器以一个点`.`号显示。如下所示。

```html
<head>
<style>
.myCenter
{
	text-align:center;
}
</style>
</head>
<body>
<h1 class="myCenter">标题居中</h1>
<p class="myCenter">段落居中</p> 
</body>
```

当然也可以在CSS定义class选择器的时候，通过指定特定HTML标签的前缀，如`p.classSelector`，来让特定的HTML元素使用class。如在以下实例中，所有的p元素使用myCenter的class选择器让该元素的文本居中。

```html
<head>
<style>
p.myCenter
{
	text-align:center;
}
</style>
</head>
<body>
<h1 class="myCenter">这个标题不受影响</h1>
<p class="myCenter">这个段落居中对齐</p> 
</body>
```

- class选择器的第一个字符不能使用数字，否则它无法在Mozilla/Firefox中起作用。

元素选择器（element selector，ES）：又叫标签选择器，使用标签名作为selector名，语法格式：`S{...}`（S为选择器名）。以html标签作为css修饰所用的选择器。如下所示。

```html
<head>
<style>
h1 { color:red; }	/* 适用于所有h1的一级标题 */
</style>
</head>
<body>
<h1>第一标题</h1>
<h1>第二标题</h1>
</body>
```

包含选择器（package selector，PS）：指定目标选择器必须处在某个选择器对应的元素内部，语法格式：`A B{...}`和`.A B{...}`（A、B为HTML元素/标签，表示对处于A中的B标签有效）。如下所示。

```html
<head>
<style>
p{
  color:red;
}
div p{
  color:yellow;
}
</style>
</head>
<body>
<p>红色文本</p>
<div>
  <p>黄色文本</p>
</div>
</body>
```

子选择器（sub-selector，SS）：类似于PS，指定目标选择器必须处在某个选择器对应的元素内部，两者区别在于PS允许“子标签”甚至“孙子标签”及嵌套更深的标签匹配相应的样式，而SS强制指定目标选择器作为“包含选择器对应的标签”内部的标签，语法格式：`A>B{...}`和`.A>B{...}`（A、B为HTML元素/标签）。如下所示。

```html
<head>
<style>
div>p{
  color:red;
}
</style>
</head>
<body>
<div>
  <p>红色文本</p>
  <table>
    <tr>
      <td>
        <p>不是红色文本</p>
      </td>
    </tr>
  </table>
</div>
</body>
```

兄弟选择器（brother selector，BS）：BS是CSS3.0新增的一个选择器，语法格式：`A~B{...}`（A、B为HTML元素/标签），表示A标签匹配selector的A，B标签匹配selector的B时，B标签匹配样式。另外相邻选择器`A+B{...}`可能与这个类似。如下所示。

```html
<head>
<style>
div~p{
  color:red;
}
</style>
</head>
<body>
<div>
  <p>不是红色文本</p>
  <div>匹配第div，即A，但不是红色文本</div>
  <p>匹配p，即B，红色文本</p>
</div>
</body>
```

通用选择器，语法形式为：`*{属性:属性值}`，它的作用是匹配html中的所有元素标签。还有一种选择器，即直接在标签内部写css代码，即之前所说的内联样式。

上面所讨论的css选择器有修饰上的优先级，当都指定是，将应用最高优先级指定的样式。它们的优先级为：内联样式>id选择器>class选择器>标签选择器。

### 2. 分组和嵌套

在样式表中有很多具有相同样式的元素，为尽量减少代码，可以使用分组选择器，每个选择器用逗号分隔，如下所示。

```css
h1, h2, p { color:green; }
```

嵌套选择器，可能适用于选择器内部的选择器的样式，如：

- `p{}`，为所有\<p\>元素指定一个样式。
- `.marked{}`，为所有class="marked"的元素指定一个样式。
- `.marked p{}`，为所有class="marked"元素内的\<p\>元素指定一个样式。
- `p.marked{ }`，为所有class="marked"的\<p\>元素指定一个样式。

### 3. 权重

一个元素，如果可以被多个选择器选中，则计算这些选择器的权重（id，类，标签），值为类型的个数，不能进阶，高权重被使用。如果权重一样走，则使用距离元素较近的那个选择器。

若元素不能被选中，则使用最近的一个父级元素的选择器。

属性可以加一个!important，它只提升该属性的权重，不提升该选择器的权重，被该选择器选中的元素的子元素不被影响。

# 三、应用CSS样式

## （一）CSS创建

### 1. 样式表的种类

当读到一个样式表时，浏览器会根据它来格式化HTML文档。插入样式表的方法有三种，即外部样式表（External style sheet）、内部样式表（Internal style sheet）、内联样式（Inline style）。

当样式需要应用于很多页面时，可以使用外部样式表，可以通过改变一个文件来改变整个站点的外观；每个页面使用`<link>`标签链接到样式表，\<link\>标签在（文档的）头部，如下所示。

```html
<head>
<link rel="stylesheet" type="text/css" href="mystyle.css">
</head>
```

- 浏览器会从文件mystyle.css中读到样式声明，并根据它来格式文档。
- 外部样式表可以在任何文本编辑器中进行编辑，文件不能包含任何的html标签，样式表文件应该以`.css`扩展名进行保存。

当单个文档需要特殊的样式时，就应该使用内部样式表；可以使用`<style>`标签在文档头部定义内部样式表，如上面选择器中的各例子。

当样式仅需要在一个元素上应用一次时，可以使用内联样式，它是直接在相关的标签内使用样式（style）属性，style属性可以包含任何CSS属性。示例如下。

```html
<p style="color:sienna;margin-left:20px">这是一个段落。</p>
```

需要注意的是，如果某些属性在不同的样式表中被同样的选择器定义，那么属性值将从更具体的样式表中被继承过来。 

### 2. 多重样式优先级

优先级是浏览器是通过判断哪些属性值与元素最相关以决定并应用到该元素上的；优先级仅由选择器组成的匹配规则决定的。

一般而言，按照优先级从高到低的顺序为：内联样式>ID选择器>伪类选择器=类选择器=属性选择器>标签（元素、类型）选择器=伪标签选择器>通用选择器。

实际上优先级就是分配给指定的CSS声明的一个权重，它由匹配的选择器中的每一种选择器类型的数值决定。内联样式权重1000，ID选择器权重100，类选择器权重10，元素选择器权重1，它们的规则如下：

- 选择器都有一个权值，权值越大越优先。
- 当权值相等时，后出现的样式表设置要优于先出现的样式表设置。
- 创作者的规则高于浏览者，即网页编写者设置的CSS样式的优先权高于浏览器所设置的样式。
- 继承的CSS样式不如后来指定的CSS样式。
- 在同一组属性设置中标有!important规则的优先级最大。

一个例外是使用`!important`，如`color:red !important;`。当!important规则被应用在一个样式声明中时，该样式声明会覆盖CSS中任何其他的声明，无论它处在声明列表中的哪里；!important规则与优先级毫无关系。一些经验法则如下：

- Always，要优化考虑使用样式规则的优先级来解决问题而不是!important；
- Only，只在需要覆盖全站或外部css（例如引用的ExtJs或者YUI）的特定页面中使用!important；
- Never，永远不要在全站范围的css上使用!important；
- Never，永远不要在自己的插件中使用!important。

## （二）常用CSS属性

CSS背景属性用于定义HTML元素的背景，CSS属性定义背景有下面的几种效果。

background-color，属性定义了元素的背景颜色，颜色可以使用六位十六进制数、RGB、颜色名称。页面的背景颜色使用在body的选择器中，如下所示。

```css
body {background-color:#B0C4DE;}
```

background-image，属性描述了元素的背景图像，默认情况下，背景图像进行平铺重复显示，以覆盖整个元素实体。

background-repeat，属性设置图片的平铺属性。使用repeat-x表示图像在水平方向平铺；repeat-y表示图像在垂直方向平铺；no-repeat表示不平铺。

background-attachment，属性设置背景图像是否固定或者随着页面的其余部分滚动。可选值有scroll（默认）背景图片随着页面的滚动而滚动；fixed，不会滚动；local，背景图片会随着元素内容的滚动而滚动。

background-position，属性用来改变图像在背景中的位置，可以指定top、bottom、left、right等值，可同时指定多个值并使用空格或`|`间隔。

页面背景图片设置的例子如下。

```css
body {
    background-image:url('paper.gif');
    background-repeat:no-repeat;
    background-position:right top;
}
```

可以看到，页面的背景颜色通过了很多的属性来控制，为了简化这些属性的代码，可以将这些属性合并在同一个属性中，即背景颜色的简写属性为background:bg-color bg-image position/bg-size bg-repeat bg-origin bg-clip bg-attachment initial inherit;，一个例子如下所示。

```css
body{background:#ffffff url('img_tree.png') no-repeat right top;}
```

