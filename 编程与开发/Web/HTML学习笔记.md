[toc]

# 一、概述

[菜鸟教程的参考手册](https://www.runoob.com/tags/html-reference.html)

超文本标记语言（Hyper Text Markup Language，HTML）是一种用于创建网页的标准标记语言，HTML不是一种编程语言，而是一种标记语言，标记语言是一套标记标签（markup tag）。HTML使用标记标签来描述网页，HTML文档包含了HTML标签及文本内容，HTML文档也叫做web 页面。

Web浏览器（如谷歌浏览器，Internet Explorer，Firefox，Safari）是用于读取HTML文件，并将其作为网页显示。浏览器并不是直接显示的HTML标签，但可以使用标签来决定如何展现HTML页面的内容给用户。

值得注意的是，我们无法确定 HTML 被显示的确切效果。屏幕的大小，以及对窗口的调整都可能导致不同的结果。对于 HTML，无法通过在 HTML 代码中添加额外的空格或换行来改变输出的效果。当显示页面时，浏览器会移除源代码中多余的空格和空行，所有连续的空格或空行都会被算作一个空格，HTML代码中的所有连续的空行（换行）也被显示为一个空格。

## （一）HTML标签和示例

```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Bloonow's Note</title>
</head>
<body>
    <h1>My First Heading</h1>
    <p>My first paragraph.</p>
</body>
</html>
```
- `<!DOCTYPE html>`，声明HTML的版本，现在默认为html 5。
- `<html>`，元素是 HTML 页面的根元素。
- `<head>`，元素包含了文档的元（meta）数据，`<meta charset="utf-8">`定义网页编码格式为utf-8。
- `<title>`，元素描述了文档的标题，用浏览器打开时的文件名（浏览器标签名）。
- `<body>`，元素包含了可见的页面内容主体，只有`<body>`区域的内容才会在浏览器中显示。
- `<h1>` ~ `<h6>`，用于生成一级到六级标题。搜索引擎使用标题为网页的结构和内容编制索引，故不要仅仅是为了生成粗体或大号的文本而使用标题。
- `<p>`，定义一个段落，HTML可以将文档分割为若干段落。通常其他一些标签都需要存在于段落标签中。
- `<hr>`，在 HTML 页面中创建水平线。
- `<!--  -->`，其中的内容为注释，浏览器会忽略注释，也不会显示它们。
- `<br/>`，该标签插入一个换行，可以在不产生一个新段落的情况下进行换行，它是一个空的HTML元素。由于关闭标签没有任何意义，因此它没有结束标签。

## （二）HTML元素、属性

上述列出的为HTML标记标签，简称为HTML标签（HTML tag），它们是由尖括号包围的关键词，如`<html>`，不区分大小写（但推荐小写）。

标签通常是成对出现的，如`<br>`和`</br>`，第一个为开始标签，第二个为结束标签，它们也被称为开放标签和闭合标签（opening tag、closing tag）。因为关闭标签是可选的，无结束标签在浏览器中也能正常显示，但不要依赖这种做法，忘记使用结束标签会产生不可预料的结果或错误。

HTML元素以开始标签起始，以结束标签终止。元素的内容是开始标签与结束标签之间的内容。大多数 HTML 元素可拥有属性。

某些 HTML 元素具有空内容（empty content），空元素在开始标签中进行关闭（以开始标签的结束而结束），如`<br/>`定义换行。反斜杠是可选的，它是参考了xml更严谨的定义所引进的。

HTML属性是可以在元素中添加附加信息。HTML元素一般在开始标签设置属性，属性总是以名/值对的形式出现。属性值应该始终被包括在引号内，通常使用双引号，也可以用单引号。 在某些个别的情况下，比如属性值本身就含有双引号，那么就必须使用单引号。

最常用HTML元素的属性：

- `class`，为html元素定义一个或多个类名（classname，类名从样式文件引入）。
- `id`，定义元素的唯一id。
- `style`，规定元素的行内样式（inline style）。
- `title`，描述了元素的额外信息 （作为工具条使用）。

## （三）HTML样式CSS

CSS（Cascading Style Sheets，层叠样式表）是在 HTML 4 开始使用的，为了更好的渲染HTML元素。CSS 可以通过内联样式、内部样式表、外部引用添加到HTML文档中。CSS属性的形式是`name:value;`类似的名值对和半角分号。

### 1. 内联样式

当特殊的样式需要应用到个别元素时，就可以使用内联样式，它使得特殊的样式可以应用到指定的个别元素。使用内联样式的方法是在相关的标签中使用样式`style`属性，样式属性可以包含任何 CSS 属性。例如：`<p style="color:blue; ">蓝色段落</p>`。

一些常见的CSS属性如下：

- `color:blue;`，（字体）颜色。
- `margin-left:20px;`，左外边距。
- `background-color:red;`，背景颜色。早期的背景色属性是使用标签的`bgcolor`属性定义的。
- `font-family:arial;`，字体。现在通常不再使用\<font>标签。
- `font-size:20px;`，字体大小。
- `text-align:center;`，文本对齐方式。取代了旧标签\<center>。

对于大部分标签，修改父级标签，子级标签特性也会改变。但某些标签确无法通过修改父级标签来改变子级标签特性，如\<a>标签，修改其颜色特性，必须直接修改\<a>标签的特性才可。

### 2. 内部样式表

当单个文件需要特别样式时，就可以使用内部样式表。使用方法是在HTML文档头部\<head>区域使用\<style>元素来包含CSS。

```html
<head>
    <style type="text/css">
        body {
            background-color: yellow;
        }

        p {
            color: blue;
        }
    </style>
</head>
```

### 3. 外部引用

当样式需要被应用到很多页面的时候，外部样式表将是理想的选择。使用外部样式表，可以通过更改一个文件来改变整个站点的外观。

```html
<head>
<link rel="stylesheet" type="text/css" href="mystyle.css">
</head>
```

# 二、HTML标签列述

## （一）HTML格式化标签

### 1. 文本格式化标签

- `<b>`，定义粗体（blot）文字。
- `<strong>`，定义强调语气，它的效果可以是通过加粗来实现，同`<b>`，也可以用css等指定其他强调样式。
- `<i>`，定义斜体（italics）字。
- `<em>`，定义着重文字，它的效果可以是通过斜体实现，同`<i>`，也可以用css等指定其他着重样式。
- `<samll>`，定义小号字。
- `<del>`，定义删除字。
- `<ins>`，定义插入字。
- `<sup>`，定义上标字。
- `<sub>`，定义下标字。

### 2. 计算机输出格式标签

这些标签常用于显示计算机/编程代码。

- `<code>`，定义计算机代码。
- `<samp>`，定义计算机代码样本。
- `<var>`，定义变量。
- `<kbd>`，定义键盘码。
- `<pre>`，定义预格式文本。

### 3. 引文、引用、定义标签

- `<address>`，定义地址，默认斜体，可用来写署名等。
- `<abbr title="all text">`，定义缩写。在某些浏览器中，当把鼠标移至缩略词语上时，title属性可用于展示表达的完整版本。`<acronym>`，标签也是缩写，但已经淘汰不用了。
- `<bdo dir="rtl">`， 定义文字方向，此例中的`rtl`表示元素内容中的文字从右到左显示。此外还有`ltr`、`auto`。
- `<q>`，定义短的引用语，默认用双引号括起来。
- `<blockquote>`，定义一个引用块。
- `<cite>`，定义引用、引证。
- `<dfn>`，定义一个定义项目，默认样式是斜体。

## （二）HTML常用标签

### 1. \<head>标签元素

\<head>元素包含了所有的头部标签元素。在\<head>元素中可以插入脚本（scripts），样式文件（CSS），及各种meta元信息。可以添加在头部区域的元素标签有：`<title>`、`<style>`、`<meta>`、` <link>`、`<script>`、`<noscript>`、`<base>`。

- `<title>`，标签定义了不同文档（浏览器工具栏）的标题；也是当网页添加到收藏夹时，显示在收藏夹中的标题；也是显示在搜索引擎结果页面的标题。它在HTML/XHTML文档中是必须的。

- `<base>`，标签描述了基本的链接地址/链接目标，该标签作为HTML文档中所有的链接标签的默认链接，如果该文档中的一个连接没有指定前部分，会默认加上base标签href属性的值。它同\<a>标签一样也有`href`、`target`等属性。

- `<link>`，标签定义了文档与外部资源之间的关系，通常用于链接到样式表。属性有`rel`、`type`、`href`。也可以在左侧显示logo等图片，如`<link rel="shortcut icon" href="图片url">`。

- `<style>`，标签定义了HTML文档的样式文件引用地址，在元素中也可以直接添加样式来渲染HTML文档。

- `<script>`，标签用于加载脚本文件，如：JavaScript。

- `<meta>`，meta标签描述了一些基本的元数据，这些元数据不显示在页面上，但会被浏览器解析。meta元素通常用于指定网页的描述，关键词，文件的最后修改时间，作者，和其他元数据。元数据可以使用于浏览器（如何显示内容或重新加载页面），搜索引擎（关键词），或其他Web服务。

```html
<!-- 为搜索引擎定义关键词 -->
<meta name="keywords" content="HTML, CSS, XML, XHTML, JavaScript">

<!-- 为网页定义描述内容 -->
<meta name="description" content="免费 Web & 编程 教程">

<!-- 定义网页作者 -->
<meta name="author" content="My Name">

<!-- 每30秒钟刷新当前页面 -->
<meta http-equiv="refresh" content="30">
```

### 2. HTML连接

HTML使用超级链接与网络上的另一个文档相连，点击链接可以从一张页面跳转到另一张页面。可以根据href的内容指定其他功能，如发送邮件等。

`<a href="address">元素内容</a>`，该标签来设置超文本链接。

- `href`，属性来描述链接的地址。
- `target`，属性值`_blank`指定链接将在新窗口打开，属性值`_top`跳出框架（覆盖当前页进行跳转）。
- 请始终将正斜杠添加到子文件夹，即最后以`/`结尾。如果最后没有，服务器会自动添加正斜杠到这个地址，然后创建一个新的请求，这会产生两次HTTP请求。

值得注意的是，超链接元素内容不一定是文本，它可以是一个字，一个词，或者一组词，也可以是一幅图像，可以点击这些内容来跳转到新的文档或者当前文档中的某个部分。

默认情况下，一个未访问过的链接显示为蓝色字体并带有下划线，访问过的链接显示为紫色并带有下划线，点击链接时，链接显示为红色并带有下划线。如果为这些超链接设置了 CSS 样式，展示样式会根据 CSS 的设定而显示。

创建页面内连接，可以使用HTML连接的`id`属性，用于在一个HTML文档创建书签标记，并在其他地方使用`#`进行跳转。书签不会以任何特殊的方式显示，它在HTML文档中是不显示的，仅仅是一个书签标记，所以对于读者来说是隐藏的。例如：

```html
<!-- 在某个地方用id创建一个书签标记，在其他的地方使用这个id进行跳转 -->
<a id="title_1">这是第一个标题</a>

<!-- 在同一页面内跳转 -->
<a href='#title_1'>返回第一个标题</a>

<!-- 从另一个页面跳转 -->
<a href='https://创建书签的页面的地址url#title_1'>返回第一个标题</a>
```

### 3. HTML图像

#### (1) 引用图像

在HTML中，图像由`<img src="图片url">`标签定义，它是一个空标签，只包含属性，没有元素内容和闭合标签。浏览器会将图像显示在文档中图像标签出现的地方。可以在`<a>`标签中的内容元素中插入`<img>`使用图片连接。

- `src`，属性即是source，它的值是图像的URL地址。URL指存储图像的位置，如果名为apple.jpg的图像位于`www.bloonow.com`的images目录中，那么其 URL 为`http://www.bloonow.com/images/apple.jpg`。
- `alt`，属性用来为图像定义一串预备的可替换的文本，替换文本属性的值是用户定义的。在浏览器无法载入图像时，替换文本属性告诉读者失去的信息，此时，浏览器将显示这个替代性的文本而不是图像。
- `width`、`height`，属性用于设置图像的高度与宽度，默认单位为像素。
- `border`，属性指定图片内边界宽度，默认单位为像素。
- `align`，该属性指定图片对齐方式，其值有`top`、`middle`、`bottom`（默认）。在HTML4中该属性已废弃，HTML5已不支持，可以使用CSS代替。
- `float`，该属性指定浮动图像，其值`left`使图片浮动在文本的左边，`right`为右边。它是CSS属性，HTML4已废弃，HTML5已不支持，可以使用CSS代替。

值得注意的是，加载图片是需要时间的，慎用图片。加载页面时，要注意插入页面图像的路径，如果不能正确设置图像的位置，浏览器无法加载图片，图像标签就会显示一个破碎的图片。

#### (2) 图像映射 map

在一个图像\<img>标签中，可以使用属性`usemap=#map_name`来为该图片指定到一个映射。

然后使用`<map>`标签指定具体的映射关系（与img标签同级），它的`name`属性的值，就是在img中使用usemap所要指定的值。

在\<map>标签的元素内容中，使用`<area>`标签指定具体的映射格式，\<area>标签有以下属性：

- `shape`，指定区域的形状，有：`rect`（矩形）、`circle`（圆形）、`poly`（多边形）等。
- `coords`，指定区域的坐标，对应上述形状，它的属性值分别可为：`x1,y1,x2,y2`、`x,y,r`、`x1,y1,x2,y2,...`等。
- `alt`、`href`，一些之前叙述过的属性。

```html
<img src="planets.gif" width="145" height="126" alt="Planets" usemap="#planetmap">

<map name="planetmap">
  <area shape="rect" coords="0,0,82,126" alt="Sun" href="sun.htm">
  <area shape="circle" coords="90,58,3" alt="Mercury" href="mercur.htm">
  <area shape="circle" coords="124,58,8" alt="Venus" href="venus.htm">
</map>
```

### 4. HTML表格

表格由`<table>`标签来定义，每个表格均有若干行，每行被分割为若干单元格。数据单元格可以包含文本、图片、列表、段落、表单、水平线、表格等等。表格三要素`<table>`、`<tr>`、`<th>`缺一不可。

```html
<table border="1">
    <caption>Title of Table</caption>
    <tr>
        <th>Header 1</th>
        <th>Header 2</th>
    </tr>
    <tr>
        <td>row 1, cell 1</td>
        <td>row 1, cell 2</td>
    </tr>
    <tr>
        <td>row 2, cell 1</td>
        <td>row 2, cell 2</td>
    </tr>
</table>
```

- `<table>`，定义表格。一些属性如下：
  - `border`，指定边框宽。如果不定义边框属性或值为0，表格将不显示边。
  - `cellpadding`，指定单元格内边与数据内容的间距。
  - `cellspacing`，指定单元格内边与外边之间的距离。
  - `width`、`height`，指定表格的总宽度、高度。
  - `bordercolor`，表格边框的颜色。
  - `bgcolor`，表格整体的背景色，如`#fff`。
- `<caption>`，该标签定义表格的标题。
- `<tr>`，该标签定义表格的行。一些属性如下：
  - `bgcolor`，行的颜色。
  - `align`，文字的水平对齐方式，参数有`left`、`center`、`right`。
  - `valign`，文字的垂直对齐方式，参数有`top`、`middle`、`bottom`。
- `<th>`，该标签定义表格的表头，大多数浏览器会把表头显示为粗体居中的文本。它可在同一行中（同一个tr中），也可以在同一列中（不同tr中的但同一位置上）。
- `<td>`，该标签定义表格的列（数据单元格），它的元素内容是表格数据（table data），即数据单元格的内容。
  - \<th>、\<td>常用的属性如下：
  - `width`、`height`，单元格的宽度、高度，设置后对当前一列、行的单元格都有影响。
  - `align`、`valign`，水平、垂直对齐方式，参数如前述。
  - `rowspan`、`colspan`，属性定义一个单元格跨的行数、列数，注意保证对应单元格数量一致。
- `<col>`，定义用于表格列的属性。
- `<colgroup>`，定义表格列的组。
- `<thead>`，定义表格的页眉。
- `<tbody>`，定义表格的主体。
- `<tfoot>`，定义表格的页脚。

### 5. HTML列表

列表项内部可以使用段落、换行符、图片、链接以及其他列表等等。

无序列表（unordered list），使用`<ul>`标签定义，在它的元素内容中使用`<li>`指定某一项，可嵌套。

- `style="list-style-type:disc"`，\<ul>标签的属性，指定项的样式，`disc`为实心圆点（默认）、`circle`为空心圆点、`square`为实心方点。

有序列表（ordered list），使用`<ol>`标签定义，在它的元素内容中使用`<li>`指定某一项，可嵌套。

- `start="1"`，\<ol>标签的属性，指定第一个开始的序号，序号默认从1开始递增。
- `type="A"`，\<ol>标签的属性，指定序号的形式，`A`大写字母列表、`a`小写字母列表、`I`罗马数字列表、`i`小写罗马数字列表，默认为阿拉伯数字列表。

自定义列表（define list），使用`<dl>`标签定义，不仅仅是一列项目，还是项目及其注释的组合。在它的元素内容中使用`<dt>`标签指定一个项，用`<dd>`标签指定一个定义或描述。

```html
<dl>
<dt>Coffee</dt>
<dd>- black hot drink</dd>
<dt>Milk</dt>
<dd>- white cold drink</dd>
</dl>
```

### 6. HTML框架

通过使用框架，可以在同一个浏览器窗口中显示不止一个页面。

```html
<iframe src="https://www.baidu.com/" width="200" height="200" frameborder="0"></iframe>
```

- `height`和`width`属性用来定义iframe标签的高度与宽度，属性默认以像素为单位，也可以指定其按比例显示，如`"80%"`。
- `frameborder`属性用于定义iframe表示是否显示边框，设置属性值为"0"表示移除iframe的边框。
- 有些老旧的浏览器不支持\<iframe\>标签，可以把需要的文本放置在\<iframe\>和\</iframe\>之间。

iframe可以显示一个目标链接的页面，目标链接的属性必须使用iframe的属性，如下实例：

```html
<iframe name="iframe_a"></iframe>
<p><a href="https://www.baidu.com/" target="iframe_a">BaiDu</a></p>
```

- 有些属性在HTML 5中已经被废弃乃至不再支持，如name属性，可以使用id属性代替。具体可翻阅参考。

## （三）HTML布局

大多数HTML元素被定义为块级元素或内联元素。块级元素在浏览器显示时，通常会以新行来开始和结束，如\<h1>、\<p>、\<ul>、\<table>等；内联元素在显示时通常不会以新行开始，如\<b>、\<td>、\<a>、\<img>等。

由于创建高级的布局非常耗时，使用模板是一个快速的选项，可以使用一些预先构建好的网站布局，并优化它们。CSS 用于对元素进行定位，或者为页面创建背景以及色彩丰富的外观。值得注意的是，虽然我们可以使用HTML table标签来设计出漂亮的布局，但是table标签是不建议作为布局工具使用的，毕竟表格不是布局工具。

### 1. \<div>标签

`<div>`元素是用于分组 HTML 元素的块级元素。它的主要属性有：

- `id`，可以指定该\<div>块的标识。
- `style=""`，其值可以是若干CSS风格的名值对，指定各种属性。

标签`<div>`元素是块级（block-level）元素，定义了文档的区域，它可用于组合其他HTML元素的容器。\<div>元素没有特定的含义，除此之外，由于它属于块级元素，浏览器会在其前后显示折行。如果与CSS一同使用，\<div>元素可用于对大的内容块设置样式属性。

\<div>元素的另一个常见的用途是文档布局，它取代了使用表格定义布局的老式方法。注：使用\<table>元素进行文档布局不是表格的正确用法，\<table>元素的作用是显示表格化的数据。

### 2. \<span>标签

标签`<span>`元素是内联（inline）元素，用来组合文档中的行内元素，可用作文本的容器。\<span>也没有特定的含义。当与CSS一同使用时，\<span>元素可用于为部分文本设置样式属性。

## （四）HTML表单

表单是一个包含表单元素的区域，表单元素内容通常是但不局限于输入元素，从而允许用户在表单中输入内容，比如：文本域（textarea）、下拉列表、单选框（radio-buttons）、复选框（checkboxes）等等。

标签`<form>`用来设置表单，表单本身并不可见，它的元素内容中可以包含各种元素。

- `action="html_form_action.php"`，表单的动作属性定义了目的文件的文件名。由动作属性定义的这个文件通常会对接收到的输入数据进行相关的处理。当表单中只有\<input>时，要指定action属性，可为空。
- `method="get"`，属性，它的值还有`post`等。
- `enctype="text/plain"`，资源类型属性。

### 1. 输入标签

标签`<input>`是多数情况下被用到的表单的输入标签。主要属性如下：

- `name`，为控件命名，以备后台程序 ASP、PHP 使用。

- `value`，提交数据到服务器的值（后台程序PHP使用）。

- `checked`，可对单选框、复选框的type使用，当设置checked="checked"时，该选项默认被选中。

- `size`，大小，如type为text时指定文本的长度。

- `type`，定义输入类型。大多数经常被用到的输入类型如下：

  - `type="text"`，文本域（Text Fields），当用户要在表单中键入字母、数字等内容时，就会用到文本域。在大多数浏览器中，文本域的默认宽度是 20 个字符。

  - `type="password"`，密码字段，字符不会明文显示，而是以星号或圆点替代。

  - `type="radio"`，单选框选项，即单选按钮（Radio Buttons）。同一组的单选按钮，name 取值一定要一致，当用户点击一个单选按钮时，它就会被选中，其他同名（name属性相同）的单选按钮就不会被选中。

```html
<form>
    <input type="radio" name="sex" value="male">Male<br>
    <input type="radio" name="sex" value="female">Female
</form>
```

  - `type="checkbox"`，复选框（Checkboxes），用户需要从若干给定的选择中选取一个或若干选项。

  - `type="submit"`，当用户单击确认按钮时，表单的内容会被传送到另一个文件。
  
  - `type="button"`，创建按钮，可以用\<input>标签的value属性对按钮上的文字进行自定义。

### 2. 下拉选择标签

标签`<select>`，用来设置简单的下拉标签，它有`name`属性。在它的元素内容中添加`<option>`标签来列举每一个选项，它有`value`属性。在众多option中，为某一个指定`selected`属性，可将其设为默认选项。

```html
<form action="">
<select name="cars">
<option value="volvo">Volvo</option>
<option value="saab">Saab</option>
<option value="fiat" selected>Fiat</option>
<option value="audi">Audi</option>
</select>
</form>
```

### 3. 文本域

标签`<textarea>`，用来创建文本域（多行文本输入控件）。用户可在文本域中写入文本。可写入字符的字数不受限制。它有属性`rows`、`cols`等。

```html
<textarea rows="10" cols="30">
文本框内容。
</textarea>
```

### 3. 边框

标签`<fieldset>`，定义了一组相关的表单元素，并使用外框包含起来。标签`<legend>`可以用来定义\<fieldset>元素的标题。

```html
<form action="">
<fieldset>
<legend>Personal information:</legend>
Name: <input type="text" size="30"><br>
E-mail: <input type="text" size="30"><br>
</fieldset>
</form>
```

### 4. 其他标签

- `<label>`，定义了 \<input> 元素的标签，一般为输入标题，它有`for`属性，一般指向input标签的`id`属性。
- `<optgroup>`，定义选项组。
- `<button>`，定义一个点击按钮。
- `<datalist>`，指定一个预先定义的输入控件选项列表。
- `<keygen>`，定义了表单的密钥对生成器字段。
- `<output>`，定义一个计算结果。

## （五）HTML颜色

每个颜色由一个十六进制符号来定义，这个符号由红色、绿色、蓝色的值组成（RGB），每种颜色的最小值是00，最大值是FF，三种颜色红、绿、蓝的组合从0到255，一共有1600万种不同颜色（256 x 256 x 256）。

如#FF0000就表示<font color="#FF0000">红色</font>，也可以使用rgb(255,0,0)来表示红色。

另外，RGBA的意思是（Red-Green-Blue-Alpha）它是在RGB上扩展了`alpha`通道，来对颜色值设置透明度，alpha的值为从0（完全透明）到1（完全不透明），如使用rgba(255,0,0,0.5)来指定透明的红色。通常为了省略，alpha的0.x可以直接写为.x。

### 1. 颜色名

为了使用方便，141个颜色名称是在HTML和CSS颜色规范定义的（17标准颜色，再加124）。17标准颜色包括：黑色，蓝色，水，紫红色，灰色，绿色，石灰，栗色，海军，橄榄，橙，紫，红，白，银，蓝绿色，黄色。

其具体表格因为太多，这里篇幅有限，不再列出，可以参考[颜色名](https://www.runoob.com/tags/html-colorname.html)或者其他列表。

### 2. 颜色值

颜色由红(R)、绿(G)、蓝(B)组成，颜色值由十六进制来表示RGB，可以是三位或者六位16进制数，三位数表示法为#RGB，转换为六位数表示为#RRGGBB。如三位的红色#F00转换为六位的是#FF0000。

## （六）HTML符号

### 1. HTML字符实体

在HTML中，某些字符是预留的（如小于号\<和大于号\>，直接使用它们会被浏览器当成标签），因而预留字符必须使用字符实体（character entities）来表示，一些在键盘上找不到的字符也可以使用字符实体来替换。

字符实体可用实体名或者数字来表示：`&entityName;`或`&#entityNumber;`。使用实体名的好处是易于记忆；坏处是，浏览器也许并不支持所有实体名称（对实体数字的支持却很好）。

如小于号（\<）、大于号（\>）、和号（\&）、引号（\"）为`&lt;`、`&gt;`、`&amp;`、`&quot;`，实体编号为`&#60;`、`&#62;`、`&#38;`、`&#34;`。

HTML中的常用字符实体是不间断空格（Non-breaking Space），它的实体名为`&nbsp;`。因为浏览器总是会截短HTML页面中的空格，如在文本中写10个空格，在显示该页面之前，浏览器会删除它们中的9个。如需在页面中增加空格的数量，需要使用`&nbsp;`字符实体。

发音符号是加到字母上的一个"glyph(字形)"，一些变音符号如尖音符 ̀和抑音符 ́。变音符号可以出现字母的上面和下面，或者字母里面，或者两个字母间。变音符号可以与字母、数字字符的组合来使用。使用实例等请查阅参考。

### 2. HTML的URL

URL（Uniform Resource Locators，统一资源定位器）是一个网页地址，可以由字母组成，如"baidu.com"，或互联网协议（IP）地址。

Web浏览器通过URL从Web服务器请求页面，当点击HTML页面中的某个链接时，对应的\<a\>标签指向万维网上的一个地址，一个统一资源定位器（URL）用于定位万维网上的文档。

URL的语法规则如下：

```
scheme://host.domain:port/path/filename
```

- `scheme`，定义因特网服务的类型。常见的类型有：
  - `http`，超文本传输协议，用于以http://开头的普通网页，不加密。
  - `https`，安全超文本传输协议，安全网页，加密所有信息交换。
  - `ftp`，文件传输协议，用于将文件下载或上传至网站。
  - `file`，本计算机上的文件。
- `host`，定义域主机，http的默认主机是www
- `domain`，定义因特网域名，比如baidu.com
- `:port`，定义主机上的端口号，http的默认端口号是80
- `path`，定义服务器上的路径，如果省略，则文档必须位于网站的根目录中。
- `filename`，定义文档/资源的名称。

URL只能使用ASCII字符集来通过因特网进行发送。由于URL常常会包含ASCII集合之外的字符，URL必须转换为有效的ASCII格式。URL编码使用`%XX`即百分号后跟两位16进制数来替换非ASCII字符。URL不能包含空格，URL编码通常使用`+`来替换空格，而JavaScript函数将空格编码成`%20`。

其余更多的编码可以参考[HTML URL编码](https://www.runoob.com/tags/html-urlencode.html)参考手册。

## （七）HTML脚本

脚本使HTML页面具有更强的动态和交互性，通过`<script>`标签定义客户端脚本（如JavaScript），\<script\>元素既可包含脚本语句，也可通过src属性指向外部脚本文件。

```html
<script>
    document.write("Hello World!")
</script>
<noscript>抱歉，浏览器不支持JavaScript!</noscript>
```

- `<noscript>`标签提供无法使用脚本时的替代内容，如浏览器禁用脚本，或浏览器不支持客户端脚本时。只有在浏览器不支持脚本或者禁用脚本时，才会显示\<noscript\>元素中的内容。
- \<noscript\>元素可包含普通HTML页面的body元素中能够找到的所有元素。

JavaScript最常用于图片操作，表单验证以及内容动态更新；JavaScript可以直接在HTML输出，也可以进行事件响应，也可以处理HTML的样式等等。

有关JavaScript的内容，请参阅专门的资料。