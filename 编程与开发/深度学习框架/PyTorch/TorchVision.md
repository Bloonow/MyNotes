[toc]

TorchVision库是PyTorch项目的一部分，PyTorch是一个开源的机器学习框架。`torchvision`包由流行的数据集、模型架构和用于计算机视觉的常见图像转换组成。

# Transforming and augmenting images

本节内容若无特殊说明，均在`torchvision.transforms`模块的命名空间中。

变换（transforms）是torchvision.transforms模块中常见的图像变换，它们可以使用`torchvision.transforms.Compose`链接（chained）在一起。

大多数变换类都有等效的函数，函数式变换（Functional Transforms）可对变换进行细粒度控制。如果必须构建一个更复杂的变换流水线（transformation pipeline），这是很有用的，例如对分割任务来说。

大多数变换同时接受PIL图像和tensor图像，而有些变换只接受PIL图像，有些只接受tensor图像。转换（Conversion）可用于转换成PIL图像，或将PIL图像转换成别的，或进行类型dtype和范围range的转换。

变换可以接受一批tensor图像，一批tensor图像是一个(B,C,H,W)形状的tensor，其中B是批中的图像数量，C是通道的数量，H和W是图像的高度和宽度。

tensor图像值的期望范围由tensor的dtype隐式定义。浮点型tensor图像的值应在[0,1)中，整型tensor图像的值应在[0,MAX_DTYPE]中，其中MAX_DTYPE是该整型dtype可以表示的最大值。

随机变换（Randomized transformations）会对同一批量中的所有图像应用相同的变换，但在不同的调用中会产生不同的变换。对于跨调用的可重复转换，可以使用函数式转换。

## 1. Transforms scriptability

若要编写可脚本化（用于JIT即时编译）的变换，需要使用`torch.nn.Sequential`代替Compose对变换进行链接。如下一个示例。

```python
transforms = torch.nn.Sequential(
    transforms.CenterCrop(10),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
)
scripted_transforms = torch.jit.script(transforms)
```

应确保只使用可脚本化的变换，即只使用torch.Tensor，不用lambda函数或PIL.Image图像。对于任何用于`torch.jit.script`的自定义变换，它们应该从torch.nn.Module派生。

## 2. Geometry

```python
class Resize(torch.nn.Module):
    def __init__(self, size, interpolation, max_size, antialias) -> None
```

将输入图像调整为给定大小。如果图像是torch.Tensor，它应该具有[...,H,W]形状，其中...表示任意数量的前缀维度。

注意，基于图像类型，输出图像可能不同。当下采样时，PIL图像和tensor图像的插值略有不同，因为PIL应用了抗锯齿（antialiasing），这可能会导致网络性能的显著差异。因此，最好使用相同的输入类型来训练模型。另外，antialias参数可以使PIL图像和tensor图像的输出更接近。

```python
class RandomCrop(torch.nn.Module):
	def __init__(self, size, padding, pad_if_needed, fill, padding_mode) -> None
```

在随机位置裁剪给定的图像。如果图像是torch.Tensor，它应该具有[...,H,W]形状，其中...表示任意数量的前缀维度。但如果使用了非常量填充（non-constant padding），则输入应最多具有2个前缀维度。

其他更多内容，见TorchVision文档。

## 3. Color

```python
class ColorJitter(torch.nn.Module):
    def __init__(self, brightness, contrast, saturation, hue) -> None
```

随机改变图像的亮度、对比度、饱和度和色调。如果图像是torch.Tensor，它应该具有[...,1]形状或[...,3,H,W]，其中...表示任意数量的前缀维度。如果图像是PIL图像，则不支持“1”“I”“F”模式和透明度模式（alpha通道）。

其他更多内容，见TorchVision文档。

## 4. Composition

```python
class Compose:
    def __init__(self, transforms) -> None
```

将多个变换组合成一个变换。这个变换不支持torchscript。如下一个例子。

```python
transforms.Compose([
    transforms.CenterCrop(10),
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float),
])
```

其他更多内容，见TorchVision文档。

## 5. Miscellaneous

```python
class LinearTransformation(torch.nn.Module):
    def __init__(self, transformation_matrix, mean_vector) -> None
```

用离线（offline）计算的平方（square）变换矩阵transformation_matrix和mean_vector对tensor图像进行变换。此变换不支持PIL图像。给定transformation_matrix和mean_vector，将展平（flatten）torch.Tensor，并从中减去mean_vector，然后计算与变换矩阵的点积，然后将tensor重塑（rashape）为其原始形状。

```python
class Normalize(torch.nn.Module):
    def __init__(self, mean, std, inplace) -> None
```

用均值和标准差对tensor图像进行归一化。此变换不支持PIL图像。给定n个通道的均值和标准差为`mean[1],...,mean[n]`和`std[1],...,std[n]`，这个变换将对输入torch.Tensor的每个通道进行归一化，即输出`output[c]=(input[c]-mean[c])/std[c]`。该变换默认是非即位（out of place）的，即它不会改变输入张量。

其他更多内容，见TorchVision文档。

## 6. Conversion

注意，一些转换变换（conversion transforms）会在转换时缩放值，而有些可能不会进行任何缩放。例如，通过缩放，torch.uint8转换为torch.float32类型，会将值的范围从[0,255]映射到[0,1]，反之亦然。

```python
class ToPILImage:
    def __init__(self, mode) -> None
```

将tensor或ndarray转换为PIL图像，这不会缩放值。转换形状为[C,H,W]的torch.Tensor或形状为[H,W,C]的numpy.ndarray到PIL图像，同时保留值范围。这个转换不支持torchscript。

```python
class ToTensor:
    def __init__(self) -> None
```

将PIL图像或ndarray转换为tensor，并相应地缩放。如果PIL图像属于其中一种模式(L,LA,P,I,F,RGB,YCbCr,RGBA,CMYK,1)或如果numpy.ndarray的dtype为np.uint8，会转换形状为[H,W,C]的值范围为[0,255]的PIL图像或numpy.ndarray到torch.FloatTensor，转换后形状为[C,H,W]，值范围为[0.0,1,0]。在其他情况下，tensor将不缩放返回。这个转换不支持torchscript。

```python
class ConvertImageDtype(torch.nn.Module):
    def __init__(self, dtype) -> None
```

将tensor图像的值转换为给定的dtype类型，并相应地缩放值。此转换不支持PIL图像。

当从较小的整数dtype转换为较大的整数dtype时，最大值无法精确映射。但如果来回转换，这种不匹配不会产生影响。

其他更多内容，见TorchVision文档。

## 7. Auto-Augmentation

AutoAugment是一种常用的数据增强技术，可以提高图像分类模型的精度。尽管数据增强策略与其训练的数据集直接相关，但实证研究表明，当应用于其他数据集时，ImageNet策略提供了显著的改进。TorchVision在以下数据集上实现了三种策略，即ImageNet、CIFAR10和SVHN数据集。新的变换可以单独使用，也可以与现有的变换混合使用。

其他更多内容，见TorchVision文档。

## 8. Functional Transforms

此小节的内容均在`torchvision.transforms.functional`模块的命名空间中。

函数式变换能够细粒度地控制变换流水线（transformation pipeline）。与上面的变换不同，函数式变换的参数不包含随机数生成器，这意味着必须指定/生成所有的参数。不过，函数式变换能在不同的调用中提供可重现的结果。

一个示例，将具有相同参数的函数变换应用于多个图像，如下所示。

```python
import torchvision.transforms.functional as TF
import random

def my_segmentation_transforms(image, segmentation):
    if random.random() < 0.5:
        angle = random.randint(-30, 30)
        image = TF.rotate(image, angle)
        segmentation = TF.rotate(segmentation, angle)
    elif random.random() > 0.5:
    # more transforms ...
    return image, segmentation
```

一个示例，使用函数式变换来构建具有自定义行为的变换类，如下所示。

```python
import torchvision.transforms.functional as TF
import random

class MyRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

rotation_transform = MyRotationTransform(angles=[-30, -15, 0, 15, 30])
```

其他更多内容，见TorchVision文档。

# torchvision.datasets

本节内容若无特殊说明，均在`torchvision.datasets`模块的命名空间中。

TorchVision在torchvision.datasets模块中提供了许多内置数据集，以及用于构建自定义数据集的实用类。

## 1. Built-in datasets

TorchVision提供的所有内置数据集通常都是torchvision.datasets.VisionDataset的子类，而VisionDataset又是torch.utils.data.Dataset的子类，且实现了\_\_getitem\_\_()和\_\_len\_\_()方法。因此，它们都可以传递给torch.utils.data.Dataloader，并可以使用torch.multiprocessing并行地加载多个samples。例如：

```python
imagenet_data = torchvision.datasets.ImageNet('dataset/imagenet_root/')
dloader = torch.utils.data.DataLoader(imagenet_data, batch_size=4,
                                      shuffle=True, num_workers=4)
```

所有数据集都具有几乎相似的API，它们都有两个共同的参数，transform和target_transform，分别对输入和目标进行变换（torchvision.transforms）。此外，还可以使用torchvision.datasets提供的基类创建自定义数据集，见之后的内容。

内置的数据集有很多，可分为多种类型，如Image classification、Image detection or segmentation、Optical Flow、Stereo Matching、Image pairs、Image captioning、Video classification、Video prediction。这里仅列举前两类数据集中的几个，更多见TorchVision文档。

### 1.1 Image classification

```python
class MNIST(VisionDataset):
    def __init__(self, root, train, transform, target_transform, download) -> None
```

该数据集是[MNIST](http://yann.lecun.com/exdb/mnist/)数据集。

root参数，表示该数据集存放的根目录，类似于'./datasets/'形式的目录。对于MNIST数据集来说，其在根目录中的存放层级为'MNIST/raw/train-images-idx3-ubyte'和'MNIST/raw/t10k-images-idx3-ubyte'等形式。

train参数，表示是否用于训练，True则从train-images-idx3-ubyte创建Dataset，False则从t10k-images-idx3-ubyte创建Dataset。

transform参数和target_transform参数，表示分别对源样本和目标的变换，即torchvision.transforms模块中的变换类或变换函数，它接收一个PIL图像并返回转换后的图像，例如transforms.RandomCrop类。

download参数，True则从网络下载数据集并将其放在root目录下，若数据集已经下载，则不会再次下载。

**注意**，若download参数为True，则root表示下载数据集存放的根目录；若download为False，则root表示程序读取数据集的根目录，此时，若root指定的根目录中，没有该数据集给定层级结构一直的子目录，则会抛出错误。

```python
class ImageNet(ImageFolder):
    def __init__(self, root, split, **kwargs) -> None
```

该数据集是2012年[ImageNet](http://image-net.org/)分类数据集。

### 1.2 Image detection or segmentation

```python
class CocoDetection(VisionDataset):
    def __init__(self, root, annFile, transform, target_transform, transforms) -> None
```

该数据集是[MS Coco Detection](https://cocodataset.org/#detection-2016)数据集，它要求安装[COCO API](https://github.com/pdollar/coco/tree/master/PythonAPI)。

## 2. Base classes for custom datasets

### 2.1 DatasetFolder

```python
class DatasetFolder(VisionDataset):
    def __init__(
        self,
        root:              str,
        loader:            Callable[[str], Any],
        extensions:        Optional[Tuple[str, ...]]       = None,
        transform:         Optional[Callable]              = None,
        target_transform:  Optional[Callable]              = None,
        is_valid_file:     Optional[Callable[[str], bool]] = None
    ) -> None
```

该类表示一个通用的按文件夹形式组织的数据集，用于从数据集的目录结构中加载数据。这个默认的目录结构可以通过覆盖find_classes()方法来定制。

root参数，表示根目录路径。

loader参数，从给定路径path中加载一个sample的读取文件的函数。

extensions参数，所允许的文件扩展名列表。extensions参数和is_valid_file参数不应该同时使用。

transform参数和target_transform参数，表示分别对源样本和目标的变换。

is_valid_file参数，一个接收文件路径并检查文件是否有效的函数，用于检查损坏的文件。extensions参数和is_valid_file参数不应该同时使用。

```python
# class DatasetFolder(VisionDataset):
def find_classes(
    self, 
    directory:  str
) -> Tuple[List[str], Dict[str, int]]
```

该方法会在数据集中查找samples类别所对应的文件夹，并返回一个列表和一个字典，列表包含了samples所有类列名称，字典包含了所有“类别名称和表示该类别index”的键值对。默认情况下，列表包括directory目录下的所有子文件夹名称，且使用sorted()对这些子文件夹名称排序，并用从0开始的整数index表示。

数据集每个类别的目录结构如下所示

```
directory/
├── class_x
│   ├── xxx.ext
│   ├── xxy.ext
│   └── ...
│       └── xxz.ext
└── class_y
    ├── 123.ext
    ├── nsdf3.ext
    └── ...
        └── asd932_.ext
```

该方法可以被重写，用于只考虑samples类别的子集，或者用于适应不同的数据集目录结构。

```python
# class DatasetFolder(VisionDataset):
@staticmethod
def make_dataset(
    directory:      str,
    class_to_idx:   Dict[str, int],
    extensions:     Optional[Tuple[str, ...]]       = None,
    is_valid_file:  Optional[Callable[[str], bool]] = None
) -> List[Tuple[str, int]]:
```

生成一个samples表单(sample_path, class_index)的列表。该函数可以被覆盖，例如从压缩zip文件中读取文件，而不是从磁盘中读取文件。

### 2.2 ImageFolder

```python
class ImageFolder(DatasetFolder):
    def __init__(
        self,
        root:              str,
        transform:         Optional[Callable]              = None,
        target_transform:  Optional[Callable]              = None,
        loader:            Callable[[str], Any]            = default_loader,
        is_valid_file:     Optional[Callable[[str], bool]] = None,
    ) -> None
```

该类表示一个通用的按文件夹形式组织的图像数据集，这个类继承自DatasetFolder，因此可以重写相同的方法来定制数据集。

默认情况下图像数据集应以下示方式排列

```
root/dog/xxx.png
root/dog/xxy.png
root/dog/[...]/xxz.png

root/cat/123.png
root/cat/nsdf3.png
root/cat/[...]/asd932_.png
```

