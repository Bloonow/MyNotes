# scipy

scipy包含各种专用于科学计算中常见问题的工具箱。其不同的子模块对应不同的应用，如插值、积分、优化、图像处理、统计、特殊函数等。scipy是Python中科学计算的核心包，它旨在有效地在numpy数组上运行，以便numpy和scipy共同使用。可以使用pip按如下进行安装。

```
pip install scipy
```

scipy完成特定任务的子模块组成如下所示。

| 模块        | 功能               |
| ----------- | ------------------ |
| cluster     | 矢量量化/Kmeans    |
| constants   | 物理和数学常数     |
| fftpack     | 傅里叶变换         |
| integrate   | 积分               |
| interpolate | 插值               |
| io          | 数据输入输出       |
| linalg      | 线性代数           |
| ndimage     | n维图像包          |
| odr         | 正交距离回归       |
| optimize    | 优化               |
| signal      | 信号处理           |
| sparse      | 稀疏矩阵           |
| spatial     | 空间数据结构和算法 |
| special     | 任何特殊的数学函数 |
| stats       | 统计数据           |

需要注意的是，主scipy命名空间大部分包含了真正的numpy函数（scipy.cos，np.cos），这些都是由于历史原因造成的，不要在代码中使用`import scipy`，而应按如下方式导入。

```python
from scipy import stats
from scipy import integrate
import scipy.io as spio
```

# skimage

skimage全称是scikit-image（toolkit for SciPy），它是对scipy.ndimage的扩展，提供了更多的图片处理功能。可以使用pip按如下进行安装。

```
pip install scikit-image
```

skimage包由许多的子模块组成，各个子模块提供不同的功能，主要子模块组成如下。

| 模型         | 功能                                                        |
| ------------ | ----------------------------------------------------------- |
| color        | 颜色空间变换                                                |
| data         | 提供一些测试图片和样本数据                                  |
| draw         | 操作于numpy数组上的基本图形绘制，包括线条、矩形、圆和文本等 |
| exposure     | 图片强度调整，如亮度调整、直方图均衡等                      |
| feature      | 特征检测与提取等                                            |
| filters      | 图像增强、边缘检测、排序滤波器、自动阈值等                  |
| graph        | 理论图的操作，如最短距离                                    |
| io           | 读取、保存和显示图片或视频                                  |
| measure      | 图像属性的测量，如相似性或等高线等                          |
| morphology   | 形态学操作，如开闭运算、骨架提取等                          |
| restoration  | 图像恢复                                                    |
| segmentation | 图像分割                                                    |
| transform    | 几何变换或其它变换，如旋转、拉伸和拉东变换等                |
| util         | 通用函数                                                    |

# sklearn

sklearn全称是scikit-learn（toolkit for SciPy），它是基于python的机器学习工具包，集成了数据预处理、数据特征选择、数据特征降维，分类、回归、聚类模型，模型评估等非常全面算法。可以使用pip按如下进行安装。

```
pip install scikit-learn
```

sklearn也提供了K最近邻居算法，可按如下方式使用。

```python
from sklearn.neighbors import NearestNeighbors

nbrs = NearestNeighbors(n_neighbors=k).fit(raw_data)
dist, Idx = nbrs.kneighbors(raw_data)
```

- 数据raw_data和tar_data都是二维的，含义为：(\<numbers of data\>, \<a data vector\>)
- NearestNeighbors(n_neighbors=k).fit(raw_data)，是由raw_data构建K最近邻居的估算器estimator（NeighborsBase类型），用来基于raw_data对其他数据进行K最邻算法。
- 使用估算器的estimator.kneighbors(tar_data)方法，求raw_data与tar_data数据之间的K最近邻居；estimator.kneighbors()方法返回的结果是(distance, indices)，它是基于构建estimator时所使用数据raw_data的。
- distance是一个二维数据，distance[i, j]表示raw_data[i]数据与它的第j个最近邻居的距离，j从小到大，距离也从小到大。
- indices也是一个二维数组，indices[i, j]表示raw_data[i]数据的第j个最近邻居在目标数据tar_data中的索引。令idx=indices[i, j]，则tar_data[idx]就是raw_data[i]的第j个最近邻居。
- 需要注意的是，当raw_data与tar_data是同一数据集时，即在一个数据集上求数据点之间的K最近邻居，则distance与indices都包含数据点自身，即distance[:, 0]都是0，indices[i, 0]都是自身i，此时为了求数据集的K个最近邻居时，通常要传参数为n_neighbors=k+1。