> **《python和pygame游戏开发指南》**
>
> ***Making Games With Python and Palme***
>
> ***[美]Ai Sweigart 著，李强 译，2015.12第一版***

[TOC]

# 一、初识pygame

```python
import pygame, sys
from pygame.locals import *

pygame.init()	#初始化pygame模块
DISPLADISYSURF = pygame.display.set_mode((400,300))
pygame.display.set_caption('Hello World')	#显示窗口顶部标题文字
while True:		#main game loop
	for event in pygame.event.get():
		if event.type == QUIT:
			pygame.quit()
			sys.exit()
	pygame.display.update()
```

导入pygame模块时，也会自动导入位于pygame模块之中的所有模块，如pygame.images和pygame.mixer.music等，不需要再用其他的import语句来导入这些位于该模块之中的模块。

针对pygame.locals使用这种形式的import语句，是因为pygame.locals包含了几个常量变量，它们前面不需要用pygame.locals做前缀，也可以很容易识别出是pygame.locals模块中的常量变量。包括`.xxx`事件。

第5行调用了`pygame.display.set_mode()`函数，它返回了用于该窗口的pygame.Surface对象，其实参为两个整数的元组值（若创建一个800 × 600的窗口，则左上角为 (0, 0)，右下角为 (799, 599) ）。返回的pygame.Surface对象（为了简便起见，我们称其为Surface对象），存储在一个名为DISPLADISYSURF的变量中。`pygame.display.set_icon()`是一个Pygame函数，它负责设置窗口的标题栏的图标。仅有一个Surface参数，理想图像大小为32像素×32像素，尽管也可以使用其他大小的图像。

第7行的while True循环，为main game loop（游戏主循环），其中的代码做如下3种事情：1、处理事件，2、更新游戏状态，3、在屏幕上绘制游戏状态。

第12行调用了`pygame.dispaly.update()`函数，它把pygame.dispaly.set_mode()所返回的Surface对象绘制到屏幕上。

在`def main()`里定义`global`，如FPSCLOCK，DIS等初始化，在 while True: 中调用`runGame()`，定义runGame()。

# 二、显示相关

> 构造函数（constructor function）是可以和常规函数一样调用的函数，只不过其返回值是一个新的对象。例如，`pyagem.Rect()`和`pygame.Surface()`都是pygame中的构造函数，它们分别返回一个新的Rect对象和Surface对象。

## （一）Surface对象是表示一个矩形的2D图像的对象

可以通过pygame绘制函数，来改变Surface对象的像素，然会再显示到屏幕上。窗口的边框、标题栏和按钮并不是Surface对象的一部分。特别是pygame.display.set_mode()返回的Surface对象叫做显示Surface（display Surface）。绘制到显示Surface对象上的任何内容，当调用pygame.display.update()函数的时候，都会显示到窗口上。(在一个Surface对象上绘制（该对象只存在于计算机内存之中），比把一个Surface对象绘制到计算机屏幕上要快很多。这是因为修改计算机内存比修改显示器上的像素要快很多。）程序经常要把几个不同的内容绘制到一个Surface对象中。在游戏循环的本次选代中，一旦将一个Surface对象上的所有内容都绘制到了显示Surface对象上（这叫作一帧），这个显示Surface对像就会绘制到屏幕上。

`pygame.Surface((width, height), flag = 0, depth = 0, masks = None)`用以返回一个 Surface对象，非显示 Surface对象。Surface的`copy()`方法将返回一个新的Surface对象，这个新的对象具有相同的需要绘制的图像，但是，它们是两个不同的Surface。在调用了copy()方法之后，如果要使用`blit()`或者Pygame的绘制方法在一个Surface对象上绘制，这将不会修改另一个Surface对象上的图像。

`pygame.image.load("path")`加载一个image图像对象。

Surface对象的`get_width()`和`get_height()`函数返回其宽度和高度。

### 1. Surface对象的旋转、翻转、缩放

`pygame.transform.rotate()`，第一个参数为要旋转的Surface对象，第二个参数为施转的角度（逆时针为正，采用角度制，若参数大于360，则自动减至360内的角）。该函数返回一个新的Surface对象（注：旋转后的Surface对象可能比之前的Surface对象要大，如正方形选择45°则变成菱形，而由于Rect为正方形，故其范围为相切于菱形四个顶点的正方形）。

为什么旋转的Surface存储在一个单独的变量中，而不是直接覆盖原Surface呢？原因有两个：

1. 首先，施转一幅2D图像并不完美。施转总是近似的。持续旋转，图像会变得越来越糟糕，因为细小的扭曲会叠加在一起。（除非将图像旋转90度的倍数，这种情况下，图像不会有任何扭曲）。
2. 其次，旋转后的Surface会比之前稍大，若旋转已经旋转的Surface，那么旋转后的surface会更大，若持续施转下去，最终Surface会变得太大，以至于pygame无法处理它，程序会崩溃并得到一条错误：pygame.error: width or height is too large.

`pygame.transform.flip()`，第一个参数为要翻转的Surface对象，第二个参数为布尔值，是否水平翻转，第三个参数是布尔值，是否
垂直翻转。该函数这回一个新的Surface对象。

`pygame.transform.scale()`，第一个参数为原Surface对象，第二、三个参数分别为缩放后的宽，缩放后的高。该函数返回一个新的Surface对象。

### 2. Surface对象的颜色

Surface对象的`fill()`方法填充颜色Color对象。关于pygame.Color对象。例：`(255, 128, 192, 64) == pygame.Color(255, 128, 192, 64)`的结果是True。即使二者是不同的数据类型，但如果它们表示相同的颜色的话，一个Color对象等同于一个3或4个整数的元组（即可通用，来表示颜色）。

Pygame中，使用3个整数的元组来表示颜色 (0~255, 0~255, 0~255)，额外地还可以通过第4个整数（0~255）参数来表示透明效果。这个值叫做alpha值（Aphra value），255表示完全不透明。注：为了使用透明颜色来进行绘制，必须使用Surface对象的`convert_alpha()`方法创建一个Surface对象。例：`anotherSurface = DISPLAYSURF.convert_alpha()`。一且在该Surface对象上绘制的内容存储到了anotherSurface中，随后anotherSurface可以“复制”，（blit，也就是copy）到DISPLAYSURF中，以便它可以显示在屏幕上。（在非convert_alpha()返回的对象上，不能使用透明颜色。)

创建一个Surface对象的pygame.PixelArray对象（简称PixelArray对象）。创建一个Surface对象的PixelArray对象，将会“锁定”该Surface对象。而当一个Surface对象锁定时，仍能在其上调用绘制函数，但不能使用`blit()`方法在其上绘制诸如PNG或JPG这样的图像。（如果想查看一个Surface对象是否锁住，可使用其`get_locked()`方法，锁定则返回True，否则返回False。返回的PixelArray对象可以用两个索引访问，即`PixelArray[x][y] = Color对象`，从而设置单个像素的颜色。完成绘制后，使用`del`语句册除PixelArray对象，以解除锁定。

### 3. 关于使用Surface对象的convert_alpha()方法实现闪烁功能

```python
import pygame,sys
from pygame.locals import *
pygame.init()
BackgroundColor = (0, 0, 0)	# 白色
DIS = pygame.display.set_mode((200, 200))
flashSurface = pygame.Surface((100, 100))
flashSurface = flashSurface.convert_alpha() # 转化为透明Surface对象
speed = 1 # 调节参数，用来控制闪烁的频率
while True: # 主循环
    for event in pygame.event.get():
        if event.type == QUIT:  # 检查退出
            pygame.quit()
            sys.exit()
    for start, end, step in ((0, 255, 1), (255, 0, -1)):
        for alpha in range(start, end, step * speed):
            DIS.fill(BackgroundColor)
            flashSurface.fill((255, 0, 0, alpha)) # 该例中为红色闪烁
            DIS.blit(flashSurface, (50, 50))
            pygame.display.update()
    DIS.fill(BackgroundColor)    
    pygame.display.update()
```

## （二）关于Rect对象

Pygame有两种方法来表示矩形区域。第一种是4个整数的元组。(x, y, width, height) 即（左上角的x坐标，左上角的y坐标，矩形的宽度（以像素为单位），矩形的高度）。第二种方法是作为一个`pygame.Rect()`对象（简称Rect对象）。例：`(10, 20, 200, 300) == pygame.Rect(10, 20, 200, 300)`的结果为True，使用第二种方法可以用Rect对象的属性自动计算矩形的坐标，也可给坐标属性赋值。

Rect对象有一个`collidepoint()`方法，也可以传递x，y坐标给它，如果该坐标位于Rect对象的区域之中（也就是说发生了碰撞），它会返回True。

`aRect = aSurface.get_rect()`：调用Surface对象的get_rect()方法，返回一个Rect对象，其有宽度width和高度height，默认x和y为0（即left和top为0）。注：这个Rect对象和Surface对象是相关联的，该Rect对象的属性值发生改变，Surface所显示是位置也会发生改变。`aRect.center = (xValue, yValue)`通过修改Rect对象的属性之一，来设置其位置。

以下列出pygame.Rect对象所提供的所有属性：

|  属性   |         表示的含义         |    属性     |     表示的含义     |
| :-----: | :------------------------: | :---------: | :----------------: |
|  left   |  矩形的左边的x坐标的int值  |   topleft   |    (left, top)     |
|  right  |        右边的x坐标         |  topright   |    (right, top)    |
|   top   |        顶部的y坐标         | bottomleft  |   (left, bottom)   |
| bottom  |        底部的y坐标         | bottomright |  (right, bottom)   |
| centerx |        中央的x坐标         |   midleft   |  (left, centery)   |
| centery |        中央的y坐标         |  midright   |  (right, centery)  |
|  width  |      矩形宽度的int值       |   midtop    |   (centerx, top)   |
| height  |      矩形高度的int值       |  midbottom  | (centerx, bottom)  |
|  size   | 两整数元组 (width, height) |   center    | (centerx, centery) |


## （三）基本绘制函数

Pygame提供了几个不同的函数，用于一个Surface对象上绘制不同的形状。这些形状包括矩形、圆形、椭圆形、线条或单个像素，通常称为图元（drawing primitives）。

`fill(pygame.Color)`是pygame.Surface对象的一个方法，它会用Color来填充整个Surface对象。

`pygame.draw.polygon(Surface, Color, pointList, width)`多边形：pointList参数是一个(x, y)坐标元组所组成的元组或列表。封闭（即最后一个顶点与第一个顶点自动连接）。width参数：若省略或为0，表示用Color填充；否则为线条的宽度。注：所有的`pygame.draw`绘制函数的最后都有可选的width参数，且其工作方式相同。

`pygame.draw.line(Surface, Color, start_point, end_point, width)`绘制直线。

`pygame.draw.linse(Surface, Color, closed, pointList, width)`绘制折线（连续）：其中参数closed为布尔类型，True表示封闭，False表示不封闭。

`pygame.draw.circle(Surface, Color, center_point, radius, width)`绘制圆。

`pygame.draw.ellipse(Surface, Color, bounding_rectangle, width)`绘制椭圆：其中bounding_rectangle表示边界矩形，为一个4整数元组(x, y, width, height)或pygame.Rect对象。

`pygame.draw.rect(Surface, Color, rectangle_tuple, width)`绘制矩形：其中rectangle_tuple为4整数元组(x, y, width, height)或pygame.Rect对象。

## （四）字体及抗锯齿

`Tmpf = pygame.font.Font('字体类型路径', 大小整数)`，若无法打开字体可用`.SysFont('字体类型', 大小整数)`：创建一个pygame.font.Font字体对象。

`Surf = Tmpf.render('内容', True, Color[, Color])`：调用Font对象的render()方法，返回一个Surface对象，并将文本绘制于其上。第二个布尔参数，True表示抗锯齿，False表示不抗锯齿（基本绘制函数也可以使用该参数来达到抗锯齿的效果）。第四个可选参数为文字填充色，省略则为透明。将带有文本的Surface对象用blit()方法复制到显示Surface对象上，在调用pygame.display.update()，使其显示在屏幕上（可以用其Rect对象来设置显示位置）。

# 三、控制相关

## （一）关于pygame.event.Event对象（简称为Event对象）

任何时候，当用户做了诸如按下一个按键或者鼠标移动到程序窗口之上等几个动作之一 ，pygame库就会创建一个pygame.event.Event对象来记录这个动作，也就是“事件”（这种叫做Event对象，存在于event模块中，该模块本身位于pygame模块中）。我们可以调用`pygame.event.get()`函数来搞清楚发生了什么事件，该函数发生Event事件的一个列表。这个列表包含了自上次调用pygame.event.get()函数之后所发生的所有事件（或者，如果从来没有调用过pygame.event.get()，会包括自程序启动以来所发生的所有事件）。列表按事件发生的顺序排序。如果没有事件发生，将返回空列表。第8行的for循环，会遍历pygame.event.get()返回的Event对象的列表。

Pygame内部拥有自己的列表数据结构，它创建该列表并且会添加所产生的Event对象，这个数据结构叫作事件列队（event queue）。当使用pygame.event.get()无参数时，返回整个列表。也可以给pygame.event.get()传递如QUIT这样的一个常量，以使其只返回内部事件列表的QUIT事件。剩下的事件仍然留在事件列队中，等待下一次的pygame.event.get()调用。（注：pygame.event.get()返回列表后，内部列队会被清空（有参数时，del参数事件））。也可以使用`pygame.event.post(event)`来向pygame事件列队未尾放入事件（手动放入）。而且Pygame事件列队最多只能存储127个Event对象，若程序没有足够频繁地调用pygame.event.get()，并且队列填满了，那么发生的任何新的事件都不能够再添加到事件列队中。

Event对象有一个名为type的成员变量（member variable，也叫属性，attributes或properties），它告诉我们对象表示何种事件。针对pygame.locals模块中的每一种可能的类型，Pygame都有一个常量变量。第9行检查Event对象的type是否等于常量QUIT。（由于使用了from pygame.locals import *形式的import语句，主要输入QUIT就可以，而不必输入pygame.locals.QUIT)。

每个事件都设有一个标识符（identifier)，为一个int数，为Event对象的type属性。其中有，1：鼠标移出/入窗口范围；2、3：分别为按下/松开键盘按键；4：鼠标在窗口内移动；5、6：分别为鼠标的按下/松开操作（左右键都是，按下滚动滚轮也是）。可以通过shell窗口中输出`pygame.xxx`来查看其int值（注：xxx事件有返回属性）。

例如：`if event.type == pygame.MOUSEMOTION:  x,y = event.pos`是正确的。因为pygame.MOUSEMOTION返回三个属性，即`.pos`，`.rel`和`.buttons`，而若直接使用`event.pos`则为错误。因为 .pos 属性是由MOUSTMOTION事件返回的，其他事件类似。

关于鼠标事件所返回的pos和button属性，MOUSEMOTION返回pos、rel、buttons，MOUSEBUTTONDOWN和MOUSEBUTTONUP返回pos、button。示例代码如下：

```python
if event.type == MOUSEMOTION:
    print(event.pos)	# 输出当前鼠标坐标
    print(event.rel)	# 输出该次使用event.rel相对于上次使用所产生的相对位移
    print(event.buttons)	# 输出一个三元组 (左键是否按下, 滚轮是否按下, 右键是否按下)，1为按下，0为未按，滚轮滚动不是按下
if event.type == MOUSEBUTTONDOWN or event.type == MOUSEBUTTONUP:
    print(event.pos)	# 同上，输出当前鼠标坐标
    print(event.button)		# 输出类型，整数，1为左键，2为滚轮，3为右键，4为滚轮向上，5为滚轮向下
```

事件及返回属性如下表：

|     事件      |       属性        |      事件       |       属性       |
| :-----------: | :---------------: | :-------------: | :--------------: |
|     QUIT      |       none        |   ACTIVEEVENT   |   gain、state    |
|    KEYDOWN    | unicode、key、moe |      KEYUP      |     key、mod     |
|  MOUSEMOTION  | pos、rel、buttons | MOUSEBUTTONDOWN |   pos、button    |
| MOUSEBUTTONUP |    pos、button    |  JOYAXISMOTION  | joy、axis、value |
| JOYBALLMOTION |  joy、ball、rel   |  JOYHATMOTION   | joy、hat、value  |
| JOYBUTTONDOWN |    joy、button    |   JOYBUTTONUP   |   joy、button    |
|  VIDEORESIZE  |    size、w、h     |   VIDEOEXPOSE   |       none       |
|   USEREVENT   |       none        |                 |                  |

关于KEYDOWN，一直按住键盘上的键只会产生一个KEYDOWN事件，即在按下的瞬间。

## （二）帧数率和pyagme.time.Clock对象

pygame.time.clock对象可以帮助我们确保程序以某一个最大的FPS运行。Clock对象将会在游戏循环的每一次选代上都设置一个小小的暂停，从而确保游戏程序不会运行得太快。如果没有这些暂停，游戏程序可能会按照计算机所能够运行的速度去运行，这对玩家来说往往太快了，并且计算机越快，它们运行游戏也就越快。在游戏循环中调用一个Clock对象的`tick()`方法，可以确保不管计算机有多快，游戏都按相同的速度运行。每次游戏循环的最后，在调用了`pygame.display.update()`之后，应调用Clock对象的tick()方法。根据前一次调用tick()之后经过了多长时间，来计算需要暂停多长时间。tick()参教可为浮点数。

`pygame.time.wait(ms)`，暂停游戏ms毫秒。

用`pygame.image.load()`和`blit()`绘制图像。很多游戏中都有图像（也叫作精灵，sprite)。Pygame能够从PNG、JPG、GIF和BMP图像文件中，将图像加载到Surface对象上。将守符串（路径参数）传递给pygame.image.load()函数。pygame.image.load()函数调用将会这回一个Surface对象，图像已绘制于其上，这个Surface对象将会是和显示Surface对象不同的另一个Surface对象，因此必须将图像的Surface对象复制到显示Surface对象上。位图复制（Blitting）就是将一个Surface的内容绘制到另一个Surface之上，通过Surface对象的blit()方法来完成。blit()方法有两个参数。第一个是源Surface对象，这是要复制的内容，第二个是2整数元组，表示图像左上角应复制到的x坐标和y坐标。注：不能复制当前“锁定”的一个Surface对象。

## （三）播放声音

首先调用`pygame.mixer.Sound()`构造函数，来创建一个pygame.mixer.Sound对象（简称 Sound对象）。接受一个字符串参数（声音文件路径）。pygame可加载WAV、MP3或OGG文件，通过调用Sound对象的`play()`和`stop()`方法来实现声音的播放与停止，在调用play()之后，程序会立即执行，不会等待声音播放完成。                                                                                                                                                                                                                                                                                                                                                                                                                             

可以通过time模块中的`time.sleep()`方法来实现挂起多长时间。（Pygame一次只能加载一个作为背景音乐的声音文件。调用`pygame.mixer.music.load()`函数加载背景音乐，参数为路径，可以是WAV、MP3或MIDI格式，函数无返回。若播放背亲音乐，调用`pygame.mixer.music.play(-1, 0.0)`函数，其中参数-1表示循环播放，若为其他非负数，则表示播放几遍，参数0.0表示从头开始播放，为其他正数，则表示从头播放到第几秒。停止背景音乐播放，调用函数`pygame.mixer.music.stop()`来实现。

## （四）相机（屏幕显示区域）

坐标和游戏世界坐标的转换。删除相机外较远的元素，以释放内存。相机延迟（相机更新之前玩象所能移动的像素）。记录游戏世界中物体的位置，根据它们属性中的movex和movey的值来移动，若为正数，则向右或向下移动；若为负数，则向上或向左移动。相机移动，通过背景移动来完成。活动区域，在相机周围所围成的区域。

| 活动区域 |     活动区域     | 活动区域 |
| :------: | :--------------: | :------: |
| 活动区域 | 相机区域，即屏幕 | 活动区域 |
| 活动区域 |     活动区域     | 活动区域 |

- 相机工作机制，Squirrel Eat Squirrel为例

游戏世界的坐标义，相机原点在游戏世界里的坐标 Camerax，Cameray。游戏世界的XY坐标永远地，持续地变大或变小，游戏世界的原点就是游戏世界坐标 (0, 0) 所在的位置。由于在屏幕上显示的是相机区域，因此我们需要记录相机的原点，位于游戏世界坐标中的何处。由于相机看到的内容显示在玩家的屏幕之上，”相机“坐标和”像素“坐标是相同的，要得到对象的像素坐标（在该例中对象为松鼠和草），即松鼠出现在屏幕的什么位置，接收游戏坐标，并减去相机的原点在游戏世界中坐标，得到松鼠的屏幕坐标。

- 活动区域，用来表示游戏世界的一个区域，即相机视图加上其四周的相机视图那么大的区域。可用函数判断是否位于活动区域，例：

```python
# 是否位于活动区域之外
def isOutsideActiveArea(camerax, cameray, obj):
    boundsLeftEdge = camerax - WindowsWidth
    boundsTopEdge = cameray - WindowsHeight
    boundsRect = pygame.Rect(boundsLeftEdge, boundsTopEdge, 3 * WindowsWidth, 3 * WindowsHeight)
    objRect = pygame.Rect(obj.x, obj.y, obj.width, obj.height)
    return not boundsRect.colliderect(objRect)
```

- 当我们创建新的敌人或草对象的时候，我们不想将其创建在相机视图之中，从而使它们看上去好像不知道从哪里跳出来的，但也不想离开相机太远，因为那样的话，它们根本不可能出现在相机视图中，在活动区域内，相机视图外是一个很好的选择。创建代码示例：

```python
# 活动一个位于相机外的随机坐标
def getRandomOffCameraPos(camerax, cameray, objWidth, objHeight):
    cameraRect = pygame.Rect(camerax, cameray, WindowsWidth, WindowsHeight)
    while True:
        x = random.randit(camerax - WindowsWidth, camerax + 2 * WindowsWidth)
        y = random.randit(cameray - WindowsHeight, cameray + 2 * WindowsHeight)
        objRect = pygame.Rect(x, y, objWidth, objHeight)
        if not objRect.colliderect(cameraRect):
            return x, y

# 创建对象
def makeNewObject(camerax, cameray):
    obj = pygame.image.load("path")
    obj.Rect = obj.get_rect()
    obj.Rect.topleft = getRandomOffCameraPos(camerax, cameray, obj.Rect.width, obj.Rect.height)
    return obj
```

- 在程序开始处，可以设置NUMOBJECT常量，从而确保任何时候在活动区域内都有足够多的草对象和松鼠对象。例如：
```python
while len(objList) < NUMOBJECT:
    objList.append(makeNewObjcet(camerax, cameray))
```

- 此外，当松鼠和草对象超出了活动区域之外时，它们已经太远了，可以删除它们已不再用太多的内存。太远的对象是不再需要的，因为它们再次回到相机视图的可能性很小。

```python
# 逆向遍历，防止越界
for i in range(len(objList) - 1, -1, -1):
    if isOutsideActiveArea(camerax, cameray, objList[i]):
        del objList[i]
```

- 相机延迟以及移动相机视图。当玩家移动时候，相机的位置 (camerax cameray) 需要更新。我们把相机更新之前玩家所能够移动的像素称为“相机延迟”（camera slack)，初始化时可设置常量CAMERASLACK表示延足距离。注：我们想要移动相机而不是玩家。例：

```python
if camerax + half_WindowsWidth - CAMERASLACK > playerCenterx:
    camerax = playerCenterx - (half_WindowsWidth - CAMERASLACK)
elif camerax + half_WindowsWidth + CAMERASLACK < playerCenterx:
    camerax = playerCenterx - (half_WindowsWidth + CAMERASLACK)
if cameray + half_WindowsHeight - CAMERASLACK < playerCentery:
    cameray = playerCentery - (half_WindowsHeight - CAMERASLACK)
elif cameray + half_WindowsHeight + CAMERASLACK > playerCentery:
    cameray = playerCentery - (half_WindowsHeight + CAMERASLACK)
```

- 显示在屏幕上，Rect对象的x，y值要从游戏世界坐标转换成像素坐标：

```python
objRect.topleft = (objRect.x - camerax, objRect.y - cameray)
DIS.blit(obj, objRect)
```

- 如果按下向上箭头键或向下箭头键（或者其对等的WASD键，那么该方向键的移动变量（moved.up，moved.down等）应该设置为True，并且相反方向的移动变量应该设置为False。如果玩家松开了任何的箭头方向键或者WASD键，代码应该将对应方向的移动变量设置为False。