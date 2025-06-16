# 一、TensorFlow概述

以最常见的有监督学习（supervised learning）为例，一个模型可用$\hat{y}=f(x,\theta)$表示，其中$x,y$分别为输入和输出，$\theta$为模型参数，确定一个损失函数$L(y,\hat{y})$，以及一批数据$X$和相对应的标签$Y$。此时，我们希望可以有一个程序库，可以实现如下功能。

- 用计算机程序表示向量、矩阵、张量等数学概念，并方便地进行运算。
- 方便地建立模型$\hat{y}=f(x,\theta)$和损失函数$L(y,\hat{y})=L(y,f(x,\theta))$；给定输入$x_0\in X$，对应的标签$y_0\in Y$，和当前迭代轮次的参数值$\theta_0$，能够方便地计算出模型预测值$\hat{y}_0=f(x_0,\theta_0)$，并计算损失函数的值$L_0=L(y_0,\hat{y}_0)=L(y_0,f(x_0,\theta_0))$。
- 当给定$x_0,y_0,\theta_0$时，自动计算损失函数$L$对模型参数$\theta$的偏导数，即$\theta_0'=\dfrac{\part L}{\part\theta}\Big|_{x=x_0,y=y_0,\theta=\theta_0}$，而无需人工推导求导结果。这意味着，这个程序库需要支持某种意义上的“符号计算”，能够记录运算的全过程，这样才能根据链式法则进行反向求导。
- 根据所求出的偏导数$\theta_0'$，方便地调用一些优化方法更新当前迭代轮次的模型参数$\theta_0$，得到下一轮次的模型参数$\theta_1$，例如梯度下降法中，$\theta_1=\theta_0-\alpha\theta_0'$，其中$\alpha$为学习率。

更抽象一些说，这个程序库需要做到以下两点。

- 对数学概念和运算的程序化表达。
- 对于任意可导函数$f(x)$，可以求在自变量$x=x_0$时的梯度$\nabla f\big|_{x=x_0}$，即“符号计算”的能力。

TensorFlow可以为以上的这些需求提供完整的解决方案。具体而言，TensorFlow包含以下特性，主要分为训练流程和部署流程。

- 数据的处理：使用tf.data和TFRecord可以高效地构建和预处理数据集，构建训练数据流。同时可以使用TensorFlow Datasets快速载入常用的公开数据集。
- 模型的建立与调试：使用即时执行模式和著名的神经网络高层API框架Keras，结合可视化工具TensorBoard，简易、快速地建立和调试模型。也可以通过TensorFlow Hub方便地载入已有的成熟模型。
- 模型的训练：支持在CPU、GPU、TPU上训练模型，支持单机和多机集群并行训练模型，充分利用海量数据和计算资源进行高效训练。
- 模型的导出：将模型打包导出为统一的SavedModel格式，方便迁移和部署。
- 服务器部署：使用TensorFlow Serving在服务器上为训练完成的模型提供高性能、支持并发、高吞吐量的API接口。
- 移动端和嵌入式设备部署：使用TensorFlow Lite将模型转换为体积小、高效率的轻量化版本，并在移动端、嵌入式端等功耗和计算能力受限的设备上运行，支持使用GPU代理进行硬件加速，还可以配合Edge TPU等外接硬件加速运算。
- 网页端部署：使用TensorFlow.js，在网页端等支持JavaScript运行的环境上运行模型，支持使用WebGL进行硬件加速。

从TensorFlow 2.1开始，在使用Python的pip安装tensorflow时，同时包含GPU的支持，无需通过特定的pip包tensorflow-gpu安装GPU版本。可以使用tensorflow-cpu包安装仅支持CPU的TensorFlow版本。

> 需要注意，在Windows平台上，TensorFlow 2.10是支持GPU的最新版本，故如果要在Windows平台上使用支持GPU的TensorFlow，其版本不得超过2.10。

需要注意，若要为TensorFlow启用GPU支持，必须在系统中安装以下NVIDIA软件：(1)NVIDIA GPU驱动程序，(2)CUDA工具包，(3)由CUDA附带的CUPTI工具包，(4)cuDNN SDK工具包，(5)TensorRT软件包。具体的软件版本要求，可以参考https://www.tensorflow.org/install/gpu?hl=zh-cn#software_requirements网址。在安装完成后，还需将其添加至环境变量中，包括NVIDIA驱动程序目录下的bin，CUDA目录下的bin;include;lib64，位于CUDA目录下extras/CUPTI/lib64的CUPTI库，以及后续安装的cuDNN路径。

安装TensorFlow 2.x后，可使用如下测试代码验证是否安装成功。

```python
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 设置TensorFLow的LOG等级，以过滤一些低等级信息

print(tf.config.list_physical_devices())
"""
[PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU'), PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
"""

A = tf.constant([[1, 2], [3, 4]])
B = tf.constant([[5, 6], [7, 8]])
C = tf.matmul(A, B)
print(C)
"""
tf.Tensor(
[[19 22]
 [43 50]], shape=(2, 2), dtype=int32)
"""
```

# 二、TensorFlow基础

本书基于TensorFlow的即时执行模式。在TensorFlow 1.x版本中，必须在导入TensorFlow库后调用tf.enable_eager_execution()函数才能启用即时模式。在TensorFlow 2.x版本中，默认为即时执行模式，无需调用tf.enable_eager_execution()函数，反之，使用tf.compat.v1.disable_eager_execution()函数可关闭即时执行。

## （一）张量与运算

TensorFlow使用张量tf.Tensor作为数据的基本单位，在概念上等同于多维数组，可以使用它来描述数学中的标量（零维数组）、向量（一维数组）、矩阵（二维数组）、张量（多维数组）等。示例如下。

```python
# 定义一个随机数（标量）
random_float = tf.random.uniform(shape=())
# 定义一个有2个元素的零向量
zero_vector = tf.zeros(shape=(2))
# 定义两个2×2的常量矩阵
A = tf.constant([[1., 2.], [3., 4.]])
B = tf.constant([[5., 6.], [7., 8.]])
```

张量的重要属性是形状、类型和值，它们分别可以通过张量的shape、dtype属性和numpy()方法获得。示例如下。

```python
print(A.shape)
print(A.dtype)
print(A.numpy())  # 将tf.Tensor张量的值转为一个NumPy数组
"""
(2, 2)
<dtype: 'float32'>
[[1. 2.]
 [3. 4.]]
"""
```

TensorFlow的大多数API函数会根据输入的值自动推断张量中元素的类型（一般默认为tf.float32），也可通过dtype参数来指定元素类型，例如tf.zeros(shape=(2,), dtype=tf.int32)所示。

TensorFlow中有大量的操作（operation），可以通过已有的张量运算得到新的张量。示例如下。

```python
C = tf.add(A, B)
D = tf.matmul(A, B)
print(C, D, sep='\n')
"""
tf.Tensor(
[[ 6.  8.]
 [10. 12.]], shape=(2, 2), dtype=float32)
tf.Tensor(
[[19. 22.]
 [43. 50.]], shape=(2, 2), dtype=float32)
"""
```

## （二）自动求导机制

在机器学习中，经常需要计算函数的导数。TensorFlow提供了强大的自动求导机制来计算导数，可使用tf.GradientTape上下文管理器类，在该上下文中的计算会被自动记录计算步骤，之后可以方便地使用tf.GradientTape.gradient()方法计算导数。

方法就算函数在某个值处的导数。示例如下。

```python
x = tf.constant([[1., 2.], [3., 4.]])
y = tf.constant([[1.], [2.]])
w = tf.Variable(initial_value=[[1.], [2.]])
b = tf.Variable(initial_value=1.)

with tf.GradientTape() as tape:
    out = tf.square(tf.matmul(x, w) + b - y)
    loss = tf.reduce_sum(out)  # 求和，可指定axis

w_grad, b_grad = tape.gradient(target=loss, sources=[w, b])
print(loss, w_grad, b_grad, sep='\n')
"""
tf.Tensor(125.0, shape=(), dtype=float32)
tf.Tensor(
[[ 70.]
 [100.]], shape=(2, 1), dtype=float32)
tf.Tensor(30.0, shape=(), dtype=float32)
"""
```

这里的w和b是一个变量tf.Variable，与普通张量一样，该变量同样具有形状、类型和值这三种属性。使用变量需要初始化，可以在tf.Variable()中指定initial_value参数来设置初始值。变量与普通张量的一个重要区别是，它默认能够被TensorFlow的自动求导机制求导，因此经常用于定义机器学习模型的参数。

tf.GradientTape是一个自动求导的记录器，其中的变量和计算步骤都会被自动记录，因此可以使用tf.GradientTape.gradient()方法计算导数。

TensorFlow中有大量的张量操作API，包括数学运算、张量形状操作，例如tf.reshape()、切片和连接，例如tf.concat()等，可以通过查阅TensorFlow的官方API文档进一步了解。

## （三）示例：线性回归

下面考虑一个实际问题，若已知某市每年的房价，希望通过对该数据进行线性回归分析，即使用线性模型$y=ax+b$来拟合上述数据，此处a和b是待求的参数。

首先定义数据，进行基本的归一化操作。

```python
x_raw = np.array([2015, 2016, 2017, 2018, 2019, 2020], dtype=np.float32)
y_raw = np.array([12000, 14000, 15000, 16500, 17500, 19000], dtype=np.float32)
x = (x_raw - x_raw.min()) / (x_raw.max() - x_raw.min())
y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())
```

接下来使用梯度下降法来求线性模型中参数$a,b$的值，所采用的损失函数为均方误差$\min_{a,b}L(a,b)=\min_{a,b}\sum\limits_{i=1}^N(ax_i+b-y_i)^2$，其关于参数$a,b$的偏导数为
$$
\begin{align}
\frac{\part L}{\part a} &= 2\sum_{i=1}^N (ax_i+b-y_i)x_i \\
\frac{\part L}{\part b} &= 2\sum_{i=1}^N (ax_i+b-y_i)
\end{align}
$$
由于均方误差取均值的系数$\dfrac{1}{N}$在训练过程中一般为常数（批次batch_size的大小），而对损失函数乘以常数等价于调整学习率，因此在具体实现时通常不写在损失函数中。

> 注意，其实线性回归是有解析式的，这里使用梯度下降法只是为了展示TensorFlow的运作方式。

对于简单的模型，使用常规的科学计算库或者工具就可以求解，这里先使用NumPy这一通用的科学计算库来实现梯度下降法，如下所示。

```python
a, b = 0, 0
num_epoch = 10000
lr = 1e-3
for e in range(num_epoch):
    y_pred = a * x + b
    grad_a, grad_b = 2 * np.dot(y_pred - y, x), 2 * np.sum(y_pred - y)
    # update parameters
    a, b = a - lr * grad_a, b - lr * grad_b

print(a, b, sep='\n')
"""
0.9591714862483188
0.04422438431531203
"""
```

可以看到，这需要手动求函数关于参数的偏导数，以及手动根据求导结果更新参数。而当偏导公式和更新方法非常复杂时，手动更新便不可取。

TensorFlow的即时执行模式与上述NumPy的运行方式十分类似，但它提供了硬件加速运算（GPU支持）、自动求导、优化器等一系列对深度学习非常重要的功能。下面将展示如何使用TensorFlow计算线性回归。

```python
x = tf.constant(value=x)
y = tf.constant(value=y)
a = tf.Variable(initial_value=0.)
b = tf.Variable(initial_value=0.)

num_epoch = 10000
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
for e in range(num_epoch):
    with tf.GradientTape() as tape:
        y_pred = a * x + b
        loss = tf.reduce_sum(tf.square(y_pred - y))
    grad_a, grad_b = tape.gradient(loss, [a, b])
    optimizer.apply_gradients(grads_and_vars=zip([grad_a, grad_b], [a, b]))

print(a, b, sep='\n')
"""
<tf.Variable 'Variable:0' shape=() dtype=float32, numpy=0.9591617>
<tf.Variable 'Variable:0' shape=() dtype=float32, numpy=0.044228815>
"""
```

这里，使用tf.keras.optimizers.SGD(learning_rate=1e-3)声明了一个梯度下降优化器（optimizer），可以根据计算出的求导结果更新模型参数，从而最小化某个特定的损失函数，具体使用方式是调用其optimizer.apply_gradients(grads_and_vars)方法，该方法需要提供grads_and_vars参数，指定一个迭代器，该迭代器返回形如(param,grad)的数据对。

在实际应用中，模型往往要复杂得多，通常会编写并实例化一个模型类model=Model()，然后使用y_pred=model(x)调用该模型，使用model.variables()属性方法获取模型参数。

# 三、TensorFlow模型建立与训练

## （一）层与模型

在TensorFlow中，推荐使用Keras（对应tf.keras模块）构建模型，它是一个广为流行的高级神经网络API抽象，简单、快速且灵活，目前已内置在TensorFlow中。

Keras有两个重要的概念，层（tf.keras.layers.Layer）和模型（tf.keras.Model）。层将各种计算流程和变量进行了封装（例如基本的全连接层、卷积层、池化层等）；而模型将各种层进行组织和连接，并封装成一个整体，描述了如何将输入的数据通过各种层及运算得到输出，在需要调用的时候，直接使用a_model(x)即可。Keras在tf.keras.layers模块中内置了大量深度学习中常用的预定义层，也允许自定义层。其基本继承关系如下。

```python
# tf.Module implemented in C/C++

@keras_export("keras.layers.Layer")
class Layer(tf.Module): ...

@keras_export("keras.Model", "keras.models.Model")
class Model(base_layer.Layer): ...

class Functional(training_lib.Model): ...

@keras_export("keras.Sequential", "keras.models.Sequential")
class Sequential(functional.Functional): ...
```

Keras模型以类的形式呈现，可以通过继承tf.keras.Model这个Python类来自定义模型，在继承类中，需要重写\_\_init\_\_()构造函数和call(input)模型调用两个方法，同时也可以根据需要增加自定义的方法。示例如下。

```python
class MyModel(tf.keras.Model):
    def __init__(self, in_feats, out_feats, **kwargs):
        super(MyModel, self).__init__()
        # 此处添加初始化代码，并声明call()方法中会用到的层
        self.layer1 = tf.keras.layers.BuiltInLayer(...)
        self.layer2 = MyCustomLayer(...)

    def call(self, input):
        # 此处添加模型调用的代码，处理输入并返回输出
        x = self.layer1(input)
        output = self.layer2(x)
        return output
```

继承tf.keras.Model后，同时可以使用父类的若干方法和属性，例如在实例化类model = MyModel()后，可以通过model.variables这一属性直接获得模型中的所有参数变量，而无需手动一个一个获取模型的参数变量。

## （二）示例：多层感知器

从一个最简单的多层感知机（Multilayer Perceptron，MLP），即多层全连接神经网络开始，介绍TensorFlow的模型编写方式。在这一部分，依次进行以下步骤：(1)数据集获取及处理，(2)模型构建，(3)模型训练，(4)模型评估。

这里，使用多层感知机完成MNIST手写体数字图片数据集的分类任务。

### 1. 数据集获取及处理：tf.keras.datasets

先进行预备工作，实现一个简单的MNISTLoader类来读取MNIST数据集数据，这里使用tf.keras.datasets类快速载入MNIST数据集。

```python
class MNISTLoader():
    def __init__(self):
        mnist = tf.keras.datasets.mnist
        (self.train_data, self.train_label), (self.test_data, self.test_label) = mnist.load_data()
        # MNIST中的图像默认为uint8（0-255的数字），以下代码将其归一化到0-1之间的浮点数，并在最后增加一维作为颜色通道
        self.train_data = np.expand_dims(self.train_data.astype(np.float32) / 255.0, axis=-1)    # [60000, 28, 28, 1]
        self.test_data = np.expand_dims(self.test_data.astype(np.float32) / 255.0, axis=-1)      # [10000, 28, 28, 1]
        self.train_label = self.train_label.astype(np.int32)    # [60000]
        self.test_label = self.test_label.astype(np.int32)      # [10000]
        self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0]

    def get_batch(self, batch_size):
        # 从数据集中随机取出batch_size个元素并返回
        index = np.random.randint(0, self.num_train_data, batch_size)
        return self.train_data[index, :], self.train_label[index]
```

在使用tf.keras.datasets.mnist时将从网络上自动下载MNIST数据集，并放置于$HOME/.keras/datasets目录中。

### 2. 模型构建：tf.keras.Model和tf.keras.layers

多层感知机的模型类使用tf.keras.Model和tf.keras.layers构建实现，该模型输入一个向量，输出10维的向量，每个分量表示对应概率。

```python
class MLP(tf.keras.Model):
    def __init__(self):
        super(MLP, self).__init__()
        # Flatten层，将除了第一维（batch_size）之外的维度展平
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(
            units=100, 
            activation=tf.keras.activations.relu,
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros'
        )
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        # inputs = [batch_size, 28, 28, 1]
        x = self.flatten(inputs)    # [batch_size, 784]
        x = self.dense1(x)          # [batch_size, 100]
        x = self.dense2(x)          # [batch_size, 10]
        output = tf.nn.softmax(x)
        return output
```

其中，tf.keras.layers.Dense即为全连接层，units参数表示输出张量的维度，activation参数表示激活函数（默认为无），use_bias参数表示是否使用偏置向量，kernel_initializer参数和bias_initializer参数分别指定权重矩阵和偏置向量的初始化方式。

### 3. 模型训练：tf.keras.losses和tf.keras.optimizer

要进行一个训练，需要该训练配置的超参数，此处定义如下。

```python
num_epochs = 10
batch_size = 64
learning_rate = 0.001
```

然后，实例化模型和数据读取类，并实例化一个tf.keras.optimizer的优化器，这里使用常用的Adam优化器，如下所示。

```python
model = MLP()
data_loader = MNISTLoader()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
```

然后迭代进行以下步骤：从DataLoader中随机取一批训练数据；将这批数据送入模型，计算出模型的预测值；将模型预测值与真实值进行比较，计算损失函数loss；计算损失函数关于模型变量的梯度；将求出的梯度值传入优化器，使用优化器的apply_gradients()方法更新模型参数以最小化损失函数。

```python
num_batches = int(data_loader.num_train_data // batch_size * num_epochs)
for batch_index in range(num_batches):
    x, y = data_loader.get_batch(batch_size)
    with tf.GradientTape() as tape:
        y_pred = model(x)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
        loss = tf.reduce_mean(loss)
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
    # logging info
    print(f'batch {batch_index}, loss: {loss.numpy()}')
```

这里使用的是tf.keras.losses.sparse_categorical_crossentropy()交叉熵损失函数，设共有$C$个类别，则其接受的y_pred参数是C维的向量，每个分量表示对应类别的概率，而y_true是表示真值类别的整数。此外，还有一个tf.keras.losses.categorical_crossentropy()交叉熵损失函数，其接受的y_pred是C维向量，且y_true也是C维的one-hot向量，仅对应真值类别的分量有效。

### 4. 模型评估：tf.kears.metrics

最后，使用测试集评估模型的性能。这里使用tf.keras.metrics.SparseCategoricalAccuracy评估器来评估模型在测试集上的性能，该评估器能够对模型预测的结果与真实结果进行比较，并输出预测正确的样本数占总样本数的比例。

首先在测试数据集上进行迭代，每次通过评估器的update_state()方法向评估器传入y_pred和y_true两个参数，即模型预测出的结果和真实结果。评估器具有内部变量来保存当前评估指标相关的参数数值，例如当前已传入的累计样本数和当前预测正确的样本数。迭代结束后，使用评估器的result()方法输出最终的评估指标值，例如预测正确的样本数占总样本数的比例。

```python
sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
num_batches = int(data_loader.num_test_data // batch_size)
for batch_index in range(num_batches):
    start_index, end_index = batch_index * batch_size, (batch_index + 1) * batch_size
    y_pred = model.predict(data_loader.test_data[start_index: end_index])
    sparse_categorical_accuracy.update_state(y_true=data_loader.test_label[start_index: end_index], y_pred=y_pred)
print(f'test accuracy: {sparse_categorical_accuracy.result()}')
"""
test accuracy: 0.972155
"""
```

## （三）卷积神经网络

卷积神经网络（Convolutional Neural Network，CNN）是一种结构类似于人类或动物的视觉系统的人工神经网络，包含一个或多个卷积层（Convolutional Layer）、池化层（Pooling Layer）和全连接层（Fully-connected Layer）。

### 1. 使用Keras实现卷积神经网络

卷积神经网络的一个示例实现如下所示。

```python
class CNN(tf.keras.Model):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,             # 卷积核数目
            kernel_size=[5, 5],     # 感受野大小
            padding='same',         # padding策略，使用same方式（保持特征图大小不变，周围补0值）或vaild方式
            activation=tf.keras.activations.relu   # 激活函数
        )
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.keras.activations.relu
        )
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.flatten = tf.keras.layers.Reshape(target_shape=(7 * 7 * 64,))
        self.dense1 = tf.keras.layers.Dense(units=1024, activation=tf.keras.activations.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        # inputs = [batch_size, 28, 28, C]
        x = self.conv1(inputs)    # [batch_size, 28, 28, 32]
        x = self.pool1(x)         # [batch_size, 14, 14, 32]
        x = self.conv2(x)         # [batch_size, 14, 14, 64]
        x = self.pool2(x)         # [batch_size, 7, 7, 64]
        x = self.flatten(x)       # [batch_size, 7 * 7 * 64]
        x = self.dense1(x)        # [batch_size, 1024]
        x = self.dense2(x)        # [batch_size, 10]
        output = tf.nn.softmax(x)
        return output
```

将上一节中的模型由MLP替换为CNN，训练后进行评估，得到如下精度。

```
test accuracy: 0.989083
```

可以发现准确率相较于前节的多层感知机有非常显著的提高。事实上，通过改变模型的网络结构（比如加入Dropout层防止过拟合），准确率还有进一步提升的空间。

### 2. 使用Keras中预定义的卷积神经网络结构

tf.keras.applications中有一些预定义好的经典卷积神经网络结构，如VGG16、VGG19、ResNet、MobileNet等，可以直接调用这些经典的卷积神经网络结构，甚至载入预训练的参数，而无需手动定义网络结构。

例如，可以使用以下代码来实例化一个MobileNetV2网络结构。

```python
model = tf.keras.applications.MobileNetV2()
```

对于一些模型，其中的某些层，例如BatchNormalization层，其在训练和评估时的行为是不同的。因此，在训练模型时，需要手动设置训练状态，告诉模型处于训练阶段，可以通过tf.keras.backend.set_learning_phase(True)方法进行设置，也可以在调用模型时通过为training参数传入True来设置。

以下展示一个例子，使用MobileNetV2网络在tf_flowers五分类数据集上进行训练，同时将classes设置为5，对应于5分类的数据集。为了代码的简短高效，在该示例中使用了TensorFlow Datasets和tf.data载入和预处理数据。

```python
import tensorflow as tf
import tensorflow_datasets as tfds

num_epoch = 5
batch_size = 50
learning_rate = 0.001

dataset = tfds.load('tf_flowers', split=tfds.Split.TRAIN, as_supervised=True)
dataset = dataset.map(lambda img, label: (tf.image.resize(img, (224, 224)) / 255.0, label)).shuffle(1024).batch(batch_size)
model = tf.keras.applications.MobileNetV2(weights=None, classes=5)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
for e in range(num_epoch):
    for images, labels in dataset:
        with tf.GradientTape() as tape:
            labels_pred = model(images, training=True)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=labels, y_pred=labels_pred)
            loss = tf.reduce_mean(loss)
        print(f'loss {loss.numpy()}')
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))
    # logging info
    print(f'loss {loss.numpy()}')
```

## （四）循环神经网络

循环神经网络（Recurrent Neural Network，RNN）是一种适宜于处理序列数据的神经网络，被广泛用于语言模型、文本生成、机器翻译等。

这里，使用RNN来进行尼采风格文本的自动生成，这个任务的本质其实预测一段英文文本的接续字母的概率分布。比如，有以下句子

```
I am a studen
```

这个句子（序列）一共有13个字符（包含空格），当阅读到这个由13个字符组成的序列后，根据人的经验，可以预测出下一个字符很大概率是t。

希望建立这样一个模型，逐个输入一段长为seq_length的序列，输出这些序列接续的下一个字符的概率分布，从下一个字符的概率分布中采样作为预测值，然后迭代式地生成下两个字符，下三个字符等等，即可完成文本的生成任务。

首先，实现一个简单的DataLoader类来读取文本，并以字符为单位进行编码。设字符种类数为num_chars个，则每种字符赋予一个0到num_chars-1之间的唯一整数编号i，可采用one-hot编码表示。

```python
class DataLoader():
    def __init__(self):
        path = tf.keras.utils.get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
        with open(path, encoding='utf-8') as f:
            self.raw_text = f.read().lower()
        self.chars = sorted(list(set(self.raw_text)))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        self.text = [self.char_indices[c] for c in self.raw_text]

    def get_batch(self, seq_length, batch_size):
        seq = []
        next_char = []
        for i in range(batch_size):
            start_index = np.random.randint(0, len(self.text) - seq_length)
            seq.append(self.text[start_index: start_index + seq_length])
            next_char.append(self.text[start_index + seq_length])
        return np.array(seq), np.array(next_char)    # [batch_size, seq_length], [batch_size, ]
```

接下来进行模型的实现，在\_\_init\_\_()方法中实例化一个常用的tf.keras.layers.LSTMCell单元，以及一个线性变换用的全连接层。

```python
class RNN(tf.keras.Model):
    def __init__(self, num_chars, batch_size, seq_length):
        super(RNN, self).__init__()
        self.num_chars = num_chars
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.cell = tf.keras.layers.LSTMCell(units=256)
        self.dense = tf.keras.layers.Dense(units=self.num_chars)

    def call(self, inputs, from_logits=False):
        # inputs = [batch_size, seq_length]
        inputs = tf.one_hot(inputs, depth=self.num_chars)    # [batch_size, seq_length, num_chars]
        # 获得 RNN 的初始状态
        state = self.cell.get_initial_state(batch_size=self.batch_size, dtype=tf.float32)
        for t in range(self.seq_length):
            # 通过当前输入和前一时刻的状态，得到输出和当前时刻的状态
            output, state = self.cell(inputs[:, t, :], state)
        logits = self.dense(output)
        if from_logits:
            return logits
        else:
            return tf.nn.softmax(logits)
```

可以看到，在call()调用方法中，对序列中的字符进行One-Hot操作，即将序列中的每个字符的编码i均变换为一个num_char维向量，其第i位为1，其余均为0，变换后的序列张量形状为[seq_length, num_chars]。然后，初始化RNN单元的状态，保存至变量state，接下来，将序列从头到尾依次送入RNN单元，即在t时刻，将上一个时刻t-1的RNN单元状态state和序列的第t个元素inputs[t, :]送入RNN单元，得到当前时刻的输出output和RNN单元状态。取RNN单元最后一次的输出，通过全连接层变换到num_chars维，即作为模型的输出。

定义一些超参数，及训练过程如下所示。

```python
num_batches = 1000
seq_length = 40
batch_size = 50
learning_rate = 1e-3

data_loader = DataLoader()
model = RNN(num_chars=len(data_loader.chars), batch_size=batch_size, seq_length=seq_length)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
for batch_index in range(num_batches):
    x, y = data_loader.get_batch(seq_length, batch_size)
    with tf.GradientTape() as tape:
        y_pred = model(x)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
        loss = tf.reduce_mean(loss)
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
    # logging info
    print(f'batch {batch_index}, loss: {loss.numpy()}')
```

关于文本生成的过程有一点需要特别注意。之前一直使用tf.argmax()函数，将对应概率最大的值作为预测值。然而对于文本生成而言，这样的预测方式过于绝对，会使得生成的文本失去丰富性。于是，可使用np.random.choice()函数按照生成的概率分布取样。这样，即使是对应概率较小的字符，也有机会被取样到。同时，加入一个temperature参数控制分布的形状，参数值越大则分布越平缓（最大值和最小值的差值越小），生成文本的丰富度越高；参数值越小则分布越陡峭，生成文本的丰富度越低。该逻辑实现为RNN的predict()方法，如下所示。

```python
class RNN(tf.keras.Model):
    /* ... */
    def predict(self, inputs, temperature=1.):
        batch_size, _ = tf.shape(inputs)
        # 调用训练好的RNN模型，预测下一个字符的概率分布
        logits = self(inputs, from_logits=True)
        # 使用带 temperature 参数的 softmax 函数获得归一化的概率分布值
        prob = tf.nn.softmax(logits / temperature).numpy()
        # 使用 np.random.choice 函数，在预测的概率分布 prob 上进行随机取样
        return np.array([np.random.choice(self.num_chars, p=prob[i, :]) for i in range(batch_size.numpy())])
```

通过这种方式进行“滚雪球”式的连续预测，即可得到生成文本，如下所示。

```python
pred_seq_len = 150
diversity = 1.0    # 丰富度，即temperature
print(f'diversity = {diversity}')
x, _ = data_loader.get_batch(seq_length, 1)
for t in range(pred_seq_len):
    y_pred = model.predict(x, diversity)    # 预测下一个字符的编号
    print(data_loader.indices_char[y_pred[0]], end='', flush=True)    # 输出预测的字符
    # 将预测的字符接在输入 X 的末尾，并截断 X 的第一个字符，以保证 X 的长度不变
    x = np.concatenate([x[:, 1:], np.expand_dims(y_pred, axis=1)], axis=-1)
print('\n')
```

## （五）深度强化学习

强化学习（Reinforcement learning，RL）强调如何基于环境而行动，以取得最大化的预期利益。

这里，使用深度强化学习玩CartPole（倒立摆）游戏。倒立摆是控制论中的经典问题，在这个游戏中，一根杆的底部与一个小车通过轴相连，而杆的重心在轴之上，因此是一个不稳定的系统。在重力的作用下，杆很容易倒下。而我们则需要控制小车在水平的轨道上进行左右运动，以使得杆一直保持竖直平衡状态。

这里使用OpenAI推出的Gym中的CartPole游戏环境，和Gym的交互过程很像是一个回合制游戏。首先需要获得游戏的初始状态（比如杆的初始角度和小车位置），然后在每个回合，需要在当前可行的动作中选择一个并交由Gym执行（比如向左或者向右推动小车，每个回合中二者只能择一），Gym在执行动作后，会返回动作执行后的下一个状态和当前回合所获得的奖励值。例如，选择向左推动小车并执行后，小车位置更加偏左，而杆的角度更加偏右，Gym将新的角度和位置返回；而如果杆在这一回合仍没有倒下，Gym同时返回给一个小的正奖励。这个过程可以一直迭代下去，直到游戏终止（比如杆倒下）。

在Python中，Gym的基本调用方法如下所示。

```python
import gym

env = gym.make('CartPole-v1')       # 实例化一个游戏环境，参数为游戏名称
state = env.reset()                 # 初始化环境，获得初始状态
while True:
    env.render()                    # 对当前帧进行渲染，绘图到屏幕
    action = model.predict(state)   # 假设我们有一个训练好的模型，能够通过当前状态预测出这时应该进行的动作
    # 让环境执行动作，获得执行完动作的下一个状态，动作的奖励，游戏是否已结束以及额外信息
    next_state, reward, done, info = env.step(action)
    if done:                        # 如果游戏结束则退出循环
        break
```

那么，我们的任务就是训练出一个模型，能够根据当前的状态预测出应该进行的一个好的动作。粗略地说，一个好的动作应当能够最大化整个游戏过程中获得的奖励之和，这也是强化学习的目标。以CartPole游戏为例，目标是希望做出合适的动作使得杆一直不倒，即游戏交互的回合数尽可能地多，而回合每进行一次，都会获得一个小的正奖励，回合数越多则累积的奖励值也越高。因此，最大化游戏过程中的奖励之和与最终目标是一致的。

以下代码展示了如何使用深度强化学习中的Deep Q-Learning方法来训练模型。首先，引入TensorFlow、Gym和一些常用库，并定义一些模型超参数。

```python
import tensorflow as tf
import numpy as np
import gym
import random
from collections import deque

num_episodes = 500              # 游戏训练的总episode数量
num_exploration_episodes = 100  # 探索过程所占的episode数量
max_len_episode = 1000          # 每个episode的最大回合数
batch_size = 32                 # 批次大小
learning_rate = 1e-3            # 学习率
gamma = 1.                      # 折扣因子
initial_epsilon = 1.            # 探索起始时的探索率
final_epsilon = 0.01            # 探索终止时的探索率
```

然后，使用tf.keras.Model建立一个Q函数网络（Q-network），用于拟合Q-Learning中的Q函数，这里使用较简单的多层全连接神经网络进行拟合，该网络输入当前状态，输出各个动作下的Q-value值，在CartPole下为2维，即向左和向右推动小车。

```python
class QNetwork(tf.keras.Model):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(units=24, activation=tf.keras.activations.relu)
        self.dense2 = tf.keras.layers.Dense(units=24, activation=tf.keras.activations.relu)
        self.dense3 = tf.keras.layers.Dense(units=2)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

    def predict(self, inputs):
        q_values = self(inputs)
        return tf.argmax(q_values, axis=-1)
```

最后，在主程序中实现Q-Learning算法。

```python
if __name__ == '__main__':
    # 实例化一个游戏环境，参数为游戏名称
    env = gym.make('CartPole-v1')
    model = QNetwork()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # 使用一个 deque 作为 Q Learning 的经验回放池
    replay_buffer = deque(maxlen=10000)
    epsilon = initial_epsilon
    for episode_id in range(num_episodes):
        # 初始化环境，获得初始状态
        state = env.reset()
        # 计算当前探索率
        epsilon = max(
            initial_epsilon * (num_exploration_episodes - episode_id) / num_exploration_episodes, 
            final_epsilon
        )
        for t in range(max_len_episode):
            env.render()                              # 对当前帧进行渲染，绘图到屏幕
            if random.random() < epsilon:             # epsilon-greedy 探索策略，以 epsilon 的概率选择随机动作
                action = env.action_space.sample()    # 选择随机动作（探索）
            else:
                # 选择模型计算出的 Q Value 最大的动作
                action = model.predict(np.expand_dims(state, axis=0)).numpy()
                action = action[0]

            # 让环境执行动作，获得执行完动作的下一个状态，动作的奖励，游戏是否已结束以及额外信息
            next_state, reward, done, info = env.step(action)
            # 如果游戏Game Over，给予大的负奖励
            reward = -10. if done else reward
            # 将(state, action, reward, next_state)的四元组（外加 done 标签表示是否结束）放入经验回放池
            replay_buffer.append((state, action, reward, next_state, 1 if done else 0))
            # 更新当前 state
            state = next_state

            if done:
                # 游戏结束则退出本轮循环，进行下一个 episode
                print("episode %4d, epsilon %.4f, score %4d" % (episode_id, epsilon, t))
                break

            if len(replay_buffer) >= batch_size:
                # 从经验回放池中随机取一个批次的四元组，并分别转换为 NumPy 数组
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = map(
                    np.array,
                    zip(*random.sample(replay_buffer, batch_size))
                )

                q_value = model(batch_next_state)
                y = batch_reward + (gamma * tf.reduce_max(q_value, axis=1)) * (1 - batch_done)    # 计算 y 值
                with tf.GradientTape() as tape:
                    # 最小化 y 和 Q-value 的距离
                    loss = tf.keras.losses.mean_squared_error(
                        y_true=y,
                        y_pred=tf.reduce_sum(model(batch_state) * tf.one_hot(batch_action, depth=2), axis=1)
                    )
                grads = tape.gradient(loss, model.variables)
                # 计算梯度并更新参数
                optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
```

对于不同的任务（或者说环境），需要根据任务的特点，设计不同的状态以及采取合适的网络来拟合Q函数。例如，如果考虑经典的打砖块游戏，每一次执行动作（挡板向左、向右或不动），都会返回一个210×160×3的RGB图片，表示当前屏幕画面。为了给打砖块游戏这个任务设计合适的状态表示，有以下分析。

- 砖块的颜色信息并不是很重要，画面转换成灰度也不影响操作，因此可以去除状态中的颜色信息（即将图片转为灰度表示）；
- 小球移动的信息很重要，如果只知道单帧画面而不知道小球往哪边运动，即使是人也很难判断挡板应当移动的方向。因此，必须在状态中加入表征小球运动方向的信息。一个简单的方式是将当前帧与前面几帧的画面进行叠加，得到一个210×160×X（其中X为叠加帧数）的状态表示；
- 每帧的分辨率不需要特别高，只要能大致表征方块、小球和挡板的位置以做出决策即可，因此对于每帧的长宽可做适当压缩。

而考虑到需要从图像信息中提取特征，使用CNN作为拟合Q函数的网络将更为适合，由此，将上面的QNetwork更换为CNN网络，并对状态做一些修改，即可用于玩一些简单的视频游戏。

## （六）Keras Pipeline

以上示例均使用Keras的Subclassing API建立模型，即通过对tf.keras.Model类进行扩展以自定义新模型，同时手动编写训练和评估模型的流程。这种方式灵活度高，且与其他流行的深度学习框架（如PyTorch、Chainer）共通，是推荐的方法。

不过有的时候，只需要建立一个结构相对简单和典型的神经网络，并使用常规的手段进行训练。这时，Keras也提供了另一套更为简单高效的内置方法来建立、训练和评估模型。

### 1. Keras Sequential与Functional API模式建立模型

最典型和常用的神经网络结构是将一堆层按特定顺序叠加起来，tf.keras.models.Sequential类接受一个网络层的列表，并将它们自动首尾相连，形成模型。

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Softmax()
])
```

不过，这种层叠结构并不能表示任意的神经网络结构。为此，Keras提供了Functional API接口，帮助建立更为复杂的模型，例如多输入/多输出或存在参数共享的模型。其使用方法是将层作为可调用的对象，并返回张量，并将输入向量和输出向量提供给tf.keras.Model的inputs和outputs参数，示例如下。

```python
inputs = tf.keras.Input(shape=(28, 28, 1))
x = tf.keras.layers.Flatten()(inputs)
x = tf.keras.layers.Dense(units=100, activation=tf.keras.activations.relu)(x)
x = tf.keras.layers.Dense(units=10)(x)
outputs = tf.keras.layers.Softmax()(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

print(type(inputs), inputs, type(outputs), outputs, type(model), model, sep='\n')
"""
<class 'keras.engine.keras_tensor.KerasTensor'>
KerasTensor(type_spec=TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32, name='input_1'), 
            name='input_1', description="created by layer 'input_1'")
<class 'keras.engine.keras_tensor.KerasTensor'>
KerasTensor(type_spec=TensorSpec(shape=(None, 10), dtype=tf.float32, name=None), 
            name='softmax/Softmax:0', description="created by layer 'softmax'")  
<class 'keras.engine.functional.Functional'>
<keras.engine.functional.Functional object at 0x0000023F2ADABF10>
"""
```

### 2. 使用tf.keras.Model的compile(),fit(),evaluate()方法训练和评估模型

当模型建立完成后，可通过tf.keras.Model.compile()方法配置训练过程，其接受oplimizer优化器、loss损失函数、metrics评估函数三个重要参数，如下所示。

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=[tf.keras.metrics.sparse_categorical_accuracy, ]
)
```

接下来，可以使用tf.keras.Model.fit()方法训练模型，其接受x训练数据、y标签数据、batch_size批次大小、epochs训练迭代次数、validation_data验证数据五个重要参数，如下所示。

```python
model.fit(
    x=data_loader.train_data,
    y=data_loader.train_label,
    batch_size=batch_size,
    eopchs=num_epochs,
    validation_data=data_loader.valid_data
)
```

最后，使用tf.keras.Model.evaluate()方法评估训练效果，其接受x测试数据、y测试数据标签两个重要参数。

```python
loss_val, metric_val = model.evaluate(
    x=data_loader.test_data,
    y=data_loader.test_label
)
```

## （七）自定义层、损失函数和评估指标

其实，不仅可以继承tf.keras.Model编写自定义的模型类，也可以继承tf.keras.layers.Layer编写自定义的层。

### 1. 自定义层

自定义层需要继承tf.keras.layers.Layer类，并重写\_\_init\_\_()、build()和call()三个方法。如下所示，展示如何实现一个全连接层。

```python
class MyLinearLayer(tf.keras.layers.Layer):
    def __init__(self, units):
        super(MyLinearLayer, self).__init__()
        # 初始化代码
        self.units = units

    def build(self, input_shape):
        # input_shape 是一个 TensorShape 类型对象，表示输入的形状
        # 该形状是第一次运行call()方法时的，其inputs参数的形状，这可使得模型参数的形状自适
        # 如果已经可以完全确定输入的形状，也可以在__init__部分创建模型参数变量
        self.w = self.add_weight(
            name='w',
            shape=[input_shape[-1], self.units],
            initializer=tf.keras.initializers.glorot_normal()
        )
        self.b = self.add_weight(
            name='b',
            shape=[self.units],
            initializer=tf.zeros_initializer()
        )

    def call(self, inputs):
        # 模型调用的代码，处理输入并返回输出
        y_pred = tf.matmul(inputs, self.w) + self.b
        return y_pred
```

在即时执行模式下，模型中各层的初始化和变量的建立是在模型第一次被调用时，即第一次调用call()时才进行的，这可以根据输入张量的形状自动确定参数变量的形状，无须手动指定。

### 2. 自定义损失函数和评估指标

自定义损失函数需要继承tf.keras.losses.Loss类，重写call()方法，输入真实值y_true和模型预测值y_pred，输出模型预测值和真实值之间通过自定义的损失函数计算出的损失值。下面的示例为均方差损失函数。

```python
class MyMeanSquaredError(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_pred - y_true))
```

自定义评估指标需要继承tf.keras.metrics.Metric类，并重写\_\_init\_\_()、update_state()和result()三个方法。下面的示例对前面用到的SparseCategoricalAccuracy评估指标类做了一个简单的重实现。

```python
class MySparseCategoricalAccuracy(tf.keras.metrics.Metric):
    def __init__(self):
        super(MySparseCategoricalAccuracy, self).__init__()
        self.total = self.add_weight(name='total', dtype=tf.int32, initializer=tf.zeros_initializer())
        self.count = self.add_weight(name='count', dtype=tf.int32, initializer=tf.zeros_initializer())

    def update_state(self, y_true, y_pred, sample_weight=None):
        values = tf.cast(tf.equal(y_true, tf.argmax(y_pred, axis=-1, output_type=tf.int32)), tf.int32)
        self.total.assign_add(tf.shape(y_true)[0])
        self.count.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.count / self.total
```

# 四、TensorFlow常用模块

## （一）变量的保存与恢复：tf.train.Checkpoint

很多时候，希望在模型训练完成后能将训练好的参数变量保存起来，在需要使用模型的其他地方载入模型和参数，就能直接得到训练好的模型。不幸的是，TensorFlow的变量类型ResourceVariable并不能被Python的pickle模块序列化存储。

好在TensorFlow提供了tf.train.Checkpoint这一强大的变量保存与恢复类，可以使用其save()和restore()方法将TensorFlow中所有包含Checkpointable State的对象进行保存和恢复，具体而言，tf.Variable、tf.keras.Layer、tf.keras.Model、tf.keras.optimizer实例都可以被保存。

> 注意，Checkpoint只保存模型的参数，而不保存模型的计算过程（计算图），因此一般用于在具有模型源代码的时候恢复之前训练好的模型参数。如果需要导出模型（无需源代码也能运行模型），需要将之导出为特定格式的模型，如ONNX模型。

tf.train.Checkpoint类的使用方法非常简单，声明一个实例即可，其构造函数参数为\*\*kwargs键值对类型，键名可以随意取（在恢复时仍需用到），值为要保存的对象。然后调用tf.train.Checkpoint.save()方法即可保存指定对象，该方法接受一个file_prefix参数，表示要保持到的路径名称，这是完整的路径，包括目录、文件名、格式后缀。

例如，此处要保存一个自定义模型及其优化器，可按如下形式构造Checkpoint实例，并保存对象。

```python
my_model = MLP()
my_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
# 模型训练
checkpoint = tf.train.Checkpoint(model=my_model, optimizer=my_optimizer)
checkpoint.save(file_prefix='./saved/model.ckpt')
```

> 注意，只有模型被调用model.call()方法，其参数才会真正被创建（分配内存），或手动调用其model.build(input_shape)方法强制创建，只有被创建的参数才会被保存。通常来说，无需担心此问题，因为对模型训练肯定会调用call()方法。

注意，其中file_prefix指定所要保存的路径，该路径所在目录即被tf.train.Checkpoint用于保存对象的目录，若是在该目录下首次调用Checkpoint.save()方法，会在file_prefix所指定路径的目录中，生成一个checkpoint文本文件，其中声明了所保存对象的各种信息。

若指定file_prefix路径为'./saved/model.ckpt'，则真正的对象数据其实是保存到诸如'./saved/model.ckpt-1.index'文件和'./saved/model.ckpt-1.data-00000-of-00001'文件当中。其中的'-1'表示所保存对象的保存编号，因为同一对象可能会被保存多次。每次调用Checkpoint.save()方法都会使得编号自增一，保存为新的.index文件和.data文件，这样最新一次保存的模型就会被保存到最大编号所对应的文件中。

可以调用tf.train.Checkpoint.restore()方法恢复所保存的对象，其接受一个save_path参数，表示要加载的路径名称，该路径需要指定具体的保存编号，即诸如'./saved/model.ckpt-1'之类的路径。加载对象的示例如下所示。

若某个目录saved下存在多次保存的对象文件，如从model.ckpt-1到model.ckpt-10，通常需要加载最新保存的对象，此时可使用辅助的tf.train.latest_checkpoint()方法，且接受checkpoint_dir参数，表示用于保存对象的目录。例如，tf.train.latest_checkpoint('./saved')会返回目录'./saved'下的最新对象的路径名称，若编号最大的对象为model.ckpt-10，则该方法就会返回表示路径的'./saved/model.ckpt-10'字符串。

```python
# using saved model
saved_model = MLP()
# 模型参数必须被构建，才能正确加载，Checkpoint支持延迟构建，即先调用restore()再调用call()
saved_model.build(input_shape=[64, 28 * 28])  # 此处示例后续没有使用模型，故手动构建以避免错误
saved_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
checkpoint = tf.train.Checkpoint(model=saved_model, optimizer=saved_optimizer)
latest_path = tf.train.latest_checkpoint(checkpoint_dir='./saved')    # './saved/model.ckpt-10'
checkpoint.restore(save_path=latest_path)
print(saved_model, saved_optimizer)
```

tf.train.Checkpoint与以前版本常用的tf.train.Saver相比，强大之处在于其支持在即时执行模式下“延迟”恢复变量。具体而言，当调用checkpoint.restore()后，但模型中的变量还没有被建立的时候，Checkpoint可以等到变量被建立的时候再进行数值的恢复。另外，tf.train.Checkpoint同时也支持图执行模式。

即时执行模式下，模型中各个层的初始化和变量的建立是在模型第一次被调用的时候才进行的，这意味着当模型刚刚被实例化的时候，其实里面还一个变量都没有，这时候使用以往的方式去恢复变量数值是一定会报错的。例如，调用tf.keras.Model.save_weight()方法保存模型参数，并在后续使用时实例化模型并立即调用其load_weight()方法，就会出错，只有当调用一遍model(x)之后，或调用model.build(input_shape=x.shape)之后，再运行load_weight()方法才能得到正确的结果。

在模型的训练过程中，往往每隔一定步数保存一个Checkpoint并进行编号，很多时候会有这样的需求。例如，(1)在长时间的训练后，程序会保存大量的Checkpoint，但只想保留最新的几个Checkpoint并清除之前旧的文件；(2)保存Checkpoint默认从1开始编号，每次自增1，但希望使用其他的编号方式，例如使用当前batch_index的作为文件编号。

这时，可以使用TensorFlow的tf.train.CheckpointManager来实现以上需求，具体而言，在定义Checkpoint后接着定义一个CheckpointManager，其构造函数接受max_to_keep参数，用于指定同一对象所保存文件的最大个数。并使用CheckpointManager.save()方法进行保存，其接受checkpoint_number参数指定保存的文件编号。如下所示。

```python
checkpoint = tf.train.Checkpoint(model=my_model, optimizer=my_optimizer)
checkpoint_manager = tf.train.CheckpointManager(
    checkpoint=checkpoint, 
    directory='./saved', 
    checkpoint_name='model.ckpt', 
    max_to_keep=3
)
checkpoint_manager.save(checkpoint_number=batch_index)
```

其中，max_to_keep指定只为对象保存3个最新文件，而之前的旧文件会被删除，checkpoint_number指定当次保存文件的编号。

最后展示一个实例，训练一个多层全连接网络，并保存和使用，具体的MLP和MNISTLoader实现见前述章节。

```python
def train(model, optimizer, data_loader, num_epochs, batch_size):
    checkpoint = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(checkpoint, directory='.\saved', max_to_keep=3)

    num_batches = int(data_loader.num_train_data // batch_size * num_epochs)
    for batch_index in range(1, num_batches):
        x, y = data_loader.get_batch(batch_size)
        with tf.GradientTape() as tape:
            y_pred = model(x)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
        
        # logging info
        print('batch %d: loss %f' % (batch_index, loss.numpy()))
        if batch_index % 100 == 0 or batch_index == num_batches - 1:
            # 使用CheckpointManager保存模型参数到文件并自定义编号
            path = manager.save(checkpoint_number=batch_index)
            print('model saved to %s' % path)

def test(model, data_loader):
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(tf.train.latest_checkpoint('./saved'))
    y_pred = np.argmax(model.predict(data_loader.test_data), axis=-1)
    print('test accuracy: %f' % (sum(y_pred == data_loader.test_label) / data_loader.num_test_data))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MNIST training configure.')
    parser.add_argument('--mode', default='train', help='train or test')
    parser.add_argument('--num_epochs', default=5)
    parser.add_argument('--batch_size', default=50)
    parser.add_argument('--learning_rate', default=0.001)
    args = parser.parse_args()

    if args.mode == 'train':
        model = MLP()
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
        data_loader = MNISTLoader()
        train(model, optimizer, data_loader, args.num_epochs, args.batch_size)
    elif args.mode == 'test':
        model = MLP()
        data_loader = MNISTLoader()
        test(model, data_loader)
```

## （二）TensorBoard可视化

有时，希望查看模型训练过程中各个参数的变化情况，例如损失函数loss的值。虽然可以通过命令行输出来查看，但有时显得不够直观，而TensorBoard就是一个帮助将训练过程可视化的工具。

### 1. 实时记录参数变化情况

使用tf.summary.create_file_writer()实例化一个SummaryWriter记录器，其语法如下所示。

```python
tf.summary.create_file_writer(
    logdir,
    max_queue=None,
    flush_millis=None,
    filename_suffix=None,
    name=None,
    experimental_trackable=False
)
```

其中，logdir参数指定tensorboard用于存放日志的目录。

创建一个记录器的示例如下所示。

```python
summary_writer = tf.summary.create_file_writer('./tblog')
print(summary_writer)
"""
<tensorflow.python.ops.summary_ops_v2._ResourceSummaryWriter object at 0x0000017F7647B400>
"""
```

接下来，当需要记录训练过程中的参数时，通过with上下文管理语句，来指定希望使用的记录器，例如summary_writer.as_default()，并对需要记录的参数使用tf.summary.xxx()方法记录，其语法如下所示。

```python
tf.summary.scalar(name, data, step=None, description=None)
tf.summary.text(name, data, step=None, description=None)
tf.summary.image(name, data, step=None, max_outputs=3, description=None)
tf.summary.audio(name, data, sample_rate, step=None, max_outputs=3, encoding=None, description=None)
```

其中，name参数指定要记录数据的名称，data参数是要被记录的数据，step参数指定步骤编号。每运行一次如tf.summary.scalar()之类的记录方法，记录器就会向记录文件中写入一条记录。

一般情况下，需要记录的是标量（如loss值），步骤编号step设置为batch_index编号，整体框架如下所示。

```python
summary_writer = tf.summary.create_file_writer('./tblog')
# 开始模型训练
for batch_index in range(num_batches):
    # 训练代码，当前batch_index的损失值放入变量loss中
    with summary_writer.as_default():
        tf.summary.scalar('loss', loss, step=batch_index)
```

当要对训练过程实时可视化时，在代码目录打开终端，键入如下命令。

```shell
tensorboard --logdir=./tblog
```

```
Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
TensorBoard 2.10.1 at http://localhost:6006/ (Press CTRL+C to quit)
```

然后使用浏览器访问命令行程序所输出的网址，通常是ttp://localhost:6006/，即可访问TensorBoard的可视界面。默认情况下，TensorBoard每30秒更新一次数据，不过也可以点击右上角的刷新按钮手动刷新。

需要注意的是，如果重新训练，需要删除掉记录文件夹内的信息并重启TensorBoard；或者建立一个新的记录文件夹并开启TensorBoard，并将--logdir参数设置为新建立的文件夹。

### 2. 查看Graph和Profile信息

除此以外，可以在训练时使用tf.summary.trace_on(True)开启Trace追踪，此时TensorFlow会将训练时的大量信息，例如计算图的结构，每个操作所耗费的时间等记录下来。在训练完成后，使用tf.summary.trace_export()将记录结果输出到文件。其语法结构如下所示。

```python
tf.summary.trace_on(graph=True, profiler=False)
tf.summary.trace_off()
tf.summary.trace_export(name, step=None, profiler_outdir=None)
```

trace_on()方法必须在即时执行模式下调用。启用后，TensorFlow运行时将收集信息，这些信息稍后可以导出并由TensorBoard使用。跟踪在整个TensorFlow运行时过程中都会处于激活状态，并影响所有执行线程。要停止跟踪并导出收集的信息，使用tf.summary.trace_export()方法，要只停止跟踪而不导出，使用tf.summary_trace_off()方法。

整体框架如下所示。

```python
# 开启Trace，可以记录图结构和profile信息
tf.summary.trace_on(graph=True, profiler=True)
# 模型训练
with summary_writer.as_default():
    tf.summary.trace_export(name='model_trace', step=0, profiler_outdir=log_dir)    # 保存Trace信息到文件
```

之后，就可以在TensorBoard的浏览器界面中选择“Profile”，以时间轴的方式查看各操作的耗时情况，如果使用@tf.function建立了计算图，也可以点击“Graphs”查看图结构。

> 在TensorFlow 2.3及后续版本中，需要使用
>
> ```shell
> pip install -U tensorboard-plugin-profile
> ```
>
> 安装独立的TensorBoard Profile插件以使用Profile功能。

需要注意的是，在较新的TensorFlow 2.10版本，使用tf.summary.trace_on()方法启用跟踪，以及使用tf.summary.trace_export()方法停止跟踪，已被启用。可以分别使用tf.profiler.experimental.start(logdir=logdir)方法和tf.profiler.experimental.stop(save=True)方法替代。

### 3. 查看模型训练情况的示例

```python
def train(model, optimizer, data_loader, num_epochs, batch_size, log_dir):
    checkpoint = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(checkpoint, directory='./saved', max_to_keep=3)

    summary_writer = tf.summary.create_file_writer(log_dir)
    tf.summary.trace_on(graph=True, profiler=True)
    # tf.profiler.experimental.start(logdir=log_dir)

    num_batches = int(data_loader.num_train_data // batch_size * num_epochs)
    for batch_index in range(num_batches):
        x, y = data_loader.get_batch(batch_size)
        with tf.GradientTape() as tape:
            y_pred = model(x)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

        with summary_writer.as_default():
            tf.summary.scalar('loss', loss, step=batch_index)

        if batch_index % 100 == 0 or batch_index == num_batches - 1:
            print('batch %d: loss %f' % (batch_index, loss.numpy()))
            path = manager.save(checkpoint_number=batch_index)
            print('model saved to %s' % path)

    with summary_writer.as_default():
        # 保存Trace信息到文件
        tf.summary.trace_export(name='model_trace', step=0, profiler_outdir=log_dir)
        # tf.profiler.experimental.stop(save=True)
```

