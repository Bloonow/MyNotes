[toc]

# torch.utils.data

本节内容若无特殊说明，均在`torch.utils.data`模块的命名空间中。

PyTorch数据加载的核心组件是`torch.utils.data.DataLoader`类，它表示在一个数据集上的可迭代对象，能够支持：映射类型（map-style）数据集和可迭代类型（iterable-style）数据集，定制数据加载顺序，自动批量化（batching），单进程和多进程的数据加载，自动内存指定（memory pinning）。

这些选项由DataLoader类构造函数的参数配置，其语法格式如下

```python
class DataLoader(Generic[T_co]):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None, *, prefetch_factor=2,
                 persistent_workers=False) -> None
```

一个例子如下所示。

```python
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, length):
        self.length = length
        self.dataset = 'A dataset here.'

    def __getitem__(self, idx):
        data_1 = idx  # self.dataset[idx][0]
        data_2 = idx  # self.dataset[idx][1]
        label = idx   # self.dataset[idx][2]
        return data_1, data_2, label

    def __len__(self):
        return self.length

dloader = torch.utils.data.DataLoader(MyDataset(8), batch_size=4)
for batch_idx, batch in enumerate(dloader):
    print(batch_idx, batch)
"""
0, [tensor([0, 1, 2, 3]), tensor([0, 1, 2, 3]), tensor([0, 1, 2, 3])]
1, [tensor([4, 5, 6, 7]), tensor([4, 5, 6, 7]), tensor([4, 5, 6, 7])]
"""
```

下面的部分将详细描述DataLoader参数选项的效果和用法。

## 1. Dataset Types

DataLoader构造函数最重要的参数是dataset，它指示要从中加载数据的数据集对象。PyTorch支持两种不同类型的数据集：映射类型（map-style）数据集和可迭代类型（iterable-style）数据集。

### 1.1 Map-style datasets

map-style数据集是实现了\_\_getitem\_\_()和\_\_len\_\_()协议的数据集，并表示从索引或键（indices或keys）到数据样本（data samples）的映射，其中indices和keys可以是非整型的。

例如，对于一个dataset，当使用dataset[idx]访问时，能够从磁盘文件夹中读出第idx个图像和它对应的标签。

map-style数据集对应的是`torch.utils.data.Dataset`类，其语法格式如下

```python
class Dataset(Generic[T_co]):
    def __init__(self, *args, **kwds) -> None
```

它表示数据集的抽象类。所有表示从keys到data samples映射的数据集应该派生自Dataset。所有子类应该重写\_\_getitem\_\_()方法，以支持从给定keys获取相应data samples的操作。子类也可以选择性地覆盖\_\_len\_\_()方法，用来返回数据集的大小，许多Sampler的实现和DataLoader的默认参数选项会用该方法获得数据集大小。

默认情况下，DataLoader构造一个index sampler以生成整型indices，若需要使用非整型indices或keys的map-style数据集，需要提供一个自定义的sampler。

### 1.2 Iterable-style datsets

iterable-style数据集是`torch.utils.data.IterableDataset`子类的一个实例，其实现了\_\_iter\_\_()协议，表示在data samples上的可迭代对象。这种类型的数据集特别适用于“随机读取非常昂贵甚至不可能”的情况，以及“batch size取决于所取数据”的情况。

例如，对于一个dataset，当调用iter(dataset)时，可以返回从数据库（database）、远程服务器（remote serve）甚至实时生成的日志（logs）中读取数据的流（stream）。

iterable-style数据集对应的是`torch.utils.data.IterableDataset`类，其语法格式如下

```python
class IterableDataset(Dataset[T_co]):
    def __init__(self, *args, **kwds) -> None
```

它表示可迭代的数据集类。所有表示可迭代data samples的数据集应该派生自IterableDataset。当数据是从流中读取时，这种类型的数据集特别有用。所有子类应该重写\_\_iter\_\_()方法，以返回数据集中samples的迭代器（iterator）。

当IterableDataset子类与DataLoader一起使用时，数据集中的每一项都将由DataLoader迭代器生成。对于DataLoader来说，当num_workers>0时，每个worker进程都有不同的数据集对象副本，因此通常希望独立地配置每个副本，以避免从worker进程返回重复的数据。可在worker进程中调用get_worker_info()函数，返回有关worker进程的信息。它可以在数据集的\_\_iter\_\_()方法中使用，也可以在DataLoader的worker_init_fn参数选项中使用，以修改每个副本的行为。

```python
def get_worker_info()
```

该函数返回当前DataLoader迭代器worker进程的信息。当在worker中调用时，返回一个对象，该对象保证具有的属性包括，`id`，当前worker id；`num_workers`，worker总数；`seed`，当前worker的随机种子，这个值由主进程RNG和worker id决定；`dataset`，此进程中数据集对象的副本，它在不同进程中是不同的对象，且也不是主进程中的对象。

下面有两个例子，分别对应上述的两种方法。

例1，在\_\_iter\_\_()中划分所有workder的工作负载，如下所示。

```python
class MyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, start, end):
        super(MyIterableDataset).__init__()
        assert end > start, "this example code only works with end >= start"
        self.start = start
        self.end = end

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # single-process data loading, return the full iterator
            iter_start = self.start
            iter_end = self.end
        else:
            # in a worker process, split workload
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
        return iter(range(iter_start, iter_end))


if __name__ == '__main__':
    # should give same set of data as range(3, 7), i.e., [3, 4, 5, 6].
    ds = MyIterableDataset(start=3, end=7)

    # Single-process loading
    print(list(torch.utils.data.DataLoader(ds, num_workers=0)))
    """
    [tensor([3]), tensor([4]), tensor([5]), tensor([6])]
    """

    # Multi-process loading with two worker processes
    # Worker 0 fetched [3, 4].  Worker 1 fetched [5, 6].
    print(list(torch.utils.data.DataLoader(ds, num_workers=2)))
    """
    [tensor([3]), tensor([5]), tensor([4]), tensor([6])]
    """

    # With even more workers
    print(list(torch.utils.data.DataLoader(ds, num_workers=12)))
    """
    [tensor([3]), tensor([5]), tensor([4]), tensor([6])]
    """
```

例2，在worker_init_fn参数所指函数中划分所有workder的工作负载，如下所示。

```python
class MyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, start, end):
        super(MyIterableDataset).__init__()
        assert end > start, "this example code only works with end >= start"
        self.start = start
        self.end = end

    def __iter__(self):
        return iter(range(self.start, self.end))

# Define a `worker_init_fn` that configures each dataset copy differently
def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    overall_start = dataset.start
    overall_end = dataset.end
    # configure the dataset to only process the split workload
    per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
    worker_id = worker_info.id
    dataset.start = overall_start + worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, overall_end)


if __name__ == '__main__':
    # should give same set of data as range(3, 7), i.e., [3, 4, 5, 6].
    ds = MyIterableDataset(start=3, end=7)

    # Single-process loading
    print(list(torch.utils.data.DataLoader(ds, num_workers=0)))
    """
    [tensor([3]), tensor([4]), tensor([5]), tensor([6])]
    """

    # Directly doing multi-process loading yields duplicate data
    print(list(torch.utils.data.DataLoader(ds, num_workers=2)))
    """
    [tensor([3]), tensor([3]), tensor([4]), tensor([4]), tensor([5]), tensor([5]), tensor([6]), tensor([6])]
    """

    # Multi-process loading with the custom `worker_init_fn`
    # Worker 0 fetched [3, 4].  Worker 1 fetched [5, 6].
    print(list(torch.utils.data.DataLoader(ds, num_workers=2, worker_init_fn=worker_init_fn)))
    """
    [tensor([3]), tensor([5]), tensor([4]), tensor([6])]
    """

    # With even more workers
    print(list(torch.utils.data.DataLoader(ds, num_workers=12, worker_init_fn=worker_init_fn)))
    """
    [tensor([3]), tensor([5]), tensor([4]), tensor([6])]
    """
```

## 2. Data Loading Order and Sampler

对于iterable-style数据集来说，数据加载顺序完全由用户定义的可迭代对象控制，这使得块读取（chunk-reading）和动态batch size的实现更容易，例如，每次生成一个批量样本（batched samples）。

接下来考虑may-style数据集的情况。`torch.utils.data.Sampler`类用于指定数据加载中使用的索引序列（sequence of indices）。Sampler表示数据集indices上的可迭代对象。例如，在随机梯度下降（SGD）中，一个Sampler能对indices列表进行随机的重新排序，并每次生成一个index，或者在mini-batch时生成所需数量的indices。

根据DataLoader构造函数的shuffle参数，会自动构造顺序的或随机的Sampler，当然，用户也可使用sampler参数指定自定义的Sampler，它会在每次生成下个index或key时被使用。

一个每次生成批量indices列表的自定义Sampler，也可以作为DataLoader构造函数的batch_sampler参数传入。也可通过batch_size参数和drop_last参数启用自动批量化处理，详见下一节。

注意，无论sampler参数还是batch_sampler参数，都不支持iterable-style数据集，因为这种数据集没有indices或keys的概念。

Sampler对应的是`torch.utils.data.Sampler`类，其语法格式如下

```python
class Sampler(Generic[T_co]):
    def __init__(self, data_source) -> None
```

它是所有Sampler的基类。每个Sampler子类应该实现\_\_iter\_\_()方法，提供一种遍历数据集元素indices的方法，返回的是一个迭代器（可迭代对象）；还应该实现\_\_len\_\_()方法，以返回迭代器的长度。

还有一种分布式的Sampler，对应的是`torch.utils.data.distributed.DistributedSampler`类，其语法格式如下

```python
class DistributedSampler(Sampler[T_co]):
    def __init__(self, dataset, num_replicas=None, rank=None,
                 shuffle=True, seed=0, drop_last=False) -> None
```

它会将数据加载限制在数据集的一个子集中。

它在与`torch.nn.parallel.DistributedDatparallel`类结合使用时特别有用。在这种情况下，每个进程都可以将DistributedSampler实例作为DataLoader的sampler参数传递，并加载原始数据集的一个专有子集。注意，Dataset被假定为常量大小，并且它的任何实例总是以相同的顺序返回相同的元素。

在分布式模式下，在每个epoch的开始处，在创建DataLoader的iterator之前，必须调用DistributedSampler的set_epoch()方法，以使得乱序操作（shuffling）在多个epoch之间正确执行。否则，它会始终使用相同的顺序。一个例子如下所示。

```python
sampler = DistributedSampler(dataset) if is_distributed else None
loader = DataLoader(dataset, shuffle=(sampler is None), sampler=sampler)
for epoch in range(start_epoch, n_epochs):
    if is_distributed:
        sampler.set_epoch(epoch)
    train(loader)
```

## 3. Loading Batched and Non-Batched Data

DataLoader支持将单次访问的data samples自动整理为批量（batchs），可以通过batch_size参数，drop_last参数，batch_sampler参数，和collate_fn参数（存在默认函数）实现。

### 3.1 Automatic batching (default)

这是最常用的情况，对于获取的一个小批量（minibatch）数据，会将它们整理成批量样本（batched samples），也即整理成若干个Tensor，这些Tensor的某个维度会作为batch维度（通常是第一个维度）。

当batch_size参数（默认为1）非空时，DataLoader会生成批量samples而不是单独samples。batch_size参数和drop_last参数用于指定DataLoader如何获得数据集keys的批量。对于map-style数据集来说，用户也可用batch_sampler参数，来指定一次生成的keys的列表。

值得注意的是，batch_size参数和drop_last参数本质上是用于从sampler构造一个batch_sampler。对于map-style数据集来说，sampler要么由用户提供，要么基于shuffle参数构造。对于iterable-style数据集来说，它是一个虚拟的无限sampler。另外，当用多进程访问iterable-style数据集是，drop_last参数会抛弃每个worker中的最后非完整（last non-full）批量。

在使用来自sampler的indices获取到一个samples列表后，参数collate_fn所表示的函数，会将samples列表整理成批量。

在这种情况下，从map-style数据集中加载数据，大致相当于

```python
for indices in batch_sampler:
    yield collate_fn([dataset[i] for i in indices])
```

从iterable-style数据集中加载数据，大致相当于

```python
dataset_iter = iter(dataset)
for indices in batch_sampler:
    yield collate_fn([next(dataset_iter) for _ in indices])
```

可使用DataLoader构造函数的collate_fn参数，指定自定义的整理操作，例如，将连续数据填充到一个批量的最大长度。

### 3.2 Disable automatic batching

在某些情况下，用户可能希望在数据集代码中手动批量化处理，或者只是加载单独的samples。例如，直接加载批量数据可能成本更低（如数据库的批量读取或内存的连续块读取），或者batch size取决于数据，或者程序本就设计为处理单独的samples。在这些情况下，不使用自动批量化处理可能更好，而是让DataLoader直接返回数据集对象的每个成员。

当DataLoader构造函数的batch_size和batch_sampler参数都是None时，就不会使用自动批量化处理。从数据集中获得的每个sample，会传给由collate_fn函数来处理。当自动批量化处理禁用时，默认的collate_fn只是简单地将NumPy数组转换为PyTorch的Tensor，而其他部分都保持不变。

在这种情况下，从map-style数据集中加载数据，大致相当于

```python
for index in sampler:
    yield collate_fn(dataset[index])
```

从iterable-style数据集中加载数据，大致相当于

```python
for data in iter(dataset):
    yield collate_fn(data)
```

### 3.3 Working with collate_fn

根据自动批量化处理是否启用，collate_fn的使用有着些微不同。

当不启用自动批量化处理时，对每个单独的sample调用collate_fn函数，以对来自DataLoader迭代器的数据生成输出。在这种情况下，collate_fn只是简单地将NumPy数组转换为PyTorch的Tensor。

当启用自动批量化处理时，每次会对data samples的列表调用collate_fn函数。它会将输入samples整理为一个批量，以便从DataLoader迭代器中生成结果。接下来会介绍默认collate_fn函数的行为。

默认的collate_fn函数实现是`torch.utils.data.default_collate()`，它的语法格式如下

```python
def default_collate(batch)
```

该函数接受一批数据batch，并将批中的元素放入具有额外batch size外部维度的tensor中。实际的输出类型可能是一个torch.Tensor，一个torch.Tensor序列，一个torch.Tensor集合，或者保持不变，这取决于输入类型。当提供了DataLoader的batch_size参数或batch_sampler参数时，就会使用默认的整理函数default_collate()。

基于一批数据中每个数据，将其类型看作输入类型，则对应的输出类型如下

| Input Type                    | Output Type                                                  |
| ----------------------------- | ------------------------------------------------------------ |
| torch.Tensor                  | torch.Tensor (with an added outer dimension batch size)      |
| NumPy Arrays                  | torch.Tensor                                                 |
| int or float                  | torch.Tensor                                                 |
| Mapping[ K, V_i ]             | Mapping[ K, default_collate([V_1,V_2,...]) ]                 |
| NamedTuple[ V1_i, V2_i, ... ] | NamedTuple[ default_collate([V1_1,V1_2,...]), default_collate([V2_1,V2_2,...]), ... ] |
| Sequence[ V1_i, V2_i, ... ]   | Sequence[ default_collate([V1_1,V1_2,...]), default_collate([V2_1,V2_2,...]), ... ] |

例如，如果每个数据样本由一个三通道图像和一个整数标签组成，即数据集的每个元素返回一个元组(image, class_label)，则默认的collate_fn将这样的元组列表整理为单个元组，其包括一个批量的图像Tensor和一个批量的标签Tensor。

特别地，默认的collate_fn具有以下特性：

- 它总会在数据之前增加一个为新的维度，作为batch维度；
- 它自动将NumPy数组和Python的数值转换成PyTorch的Tensor；
- 它会保留数据结构，例如，若sample是一个字典，则输出也是一个字典，且输出字典的键与输入字典相同，不过字典的值会被批处理成Tensor（若值不支持构成Tensor，则会批处理成列表）；对于Python中的列表、元组，或命名元组也是如此。

用户可使用collate_fn参数来自定义批量化处理，例如，不沿着第一个维度批量化数据，填充变长序列，或者对自定义数据类型提供支持。

若发现DataLoader迭代器输出的数据维度或类型与预期不符，最好检查一下collate_fn函数的行为逻辑。

## 4. Single- and Multi-process Data Loading

DataLoader默认使用单进程数据加载。

在Python进程中，全局解释器锁（Global Interpreter Lock，GIL）阻止了跨线程真正地完全并行化Python代码。为了避免数据加载阻塞计算代码，PyTorch提供了一个简单的转换（switch）来执行多进程数据加载，只需简单地将DataLoader构造函数的num_workers参数设为一个正整数即可。

### 4.1 Single-process data loading (default)

在这种模式下，参数num_workers为0（默认），获取数据的进程与初始化DataLoader的进程是同一个。因此，数据加载可能会阻塞（block）计算。然而，当用于在进程之间共享数据的资源（例如，共享内存，文件描述符）有限时，或者当整个数据集很小并且可以完全加载在内存中时，这种模式可能是首选的。此外，单进程加载通常能打印更多可读的错误信息，对调试很有用。

### 4.2 Multi-process data loading

将DataLoader构造函数的num_workers参数设置为正整数，将使用指定数量的worker进程加，来开启多进程数据加载。需要注意的是，这会开启num_workers个新进程用于数据加载，此外还有一个主进程，共有num_workers+1个进程在运行。

在这种模式下，每当一个DataLoader迭代器被创建时，例如调用enumerate(dloader)时，都会创建num_workers个worker进程。此时，会把dataset、collate_fn、worker_init_fn传递给每个worker，用来初始化和获取数据。这意味着数据集的访问、内部IO、转换（包括collate_fn），都会在worker进程中运行。

`torch.utils.data.get_worker_info()`函数，在worker进程中会返回各种有用的信息（包括worker进程id、数据集副本、初始种子等），而在主进程中会返回None。可以在数据集代码中使用该函数，或在worker_init_fn所指定函数中使用该函数，以单独配置每个数据集副本，并确定代码是否运行在worker进程中。例如，这在对数据集进行分片（sharding）时特别有用。

当在传递给DataLoader的worker_init_fn中使用get_worker_info()函数时，可对每个工作进程进行不同的设置，例如，使用worker_id配置数据集对象，使其只读取分片数据集的特定部分，或使用seed为数据集代码中使用的其他库提供种子值。

对于map-style数据集，主进程使用sampler生成indices，并将它们发送给worker。因此，任何shuffle随机化都是在主进程中完成的，它通过分配indices来指导加载。

对于iterable-style数据集，由于每个worker进程都获得了数据集对象的副本，简单地多进程加载通常会导致重复的数据。使用get_worker_info()函数或worker_init_fn参数，用户可以独立地配置每个副本。出于类似的原因，在多进程加载中，drop_last参数会抛弃每个worker进程的iterable-style数据集副本的最后一批非完整批量。

一旦迭代结束，或者迭代器被垃圾回收机制收集时，worker线程将被关闭。

### 4.3 Platform-specific behaviors

由于worker依赖于Python的multiprocessing，所以worker的启动行为在不同操作系统（如Windows和Unix）上不同。

在Unix上，`fork()`是默认的multiprocessing启动方法。使用fork()，孩子worker进程通常可以通过克隆的地址空间，直接访问数据集和Python函数参数。

在Windows或MacOS上，`spawn()`是默认的multiprocessing启动方法。使用spawn()，启动另一个Python解释器（interpreter）来运行Python主脚本（main script），然后是其中的worker函数，它通过pickle序列化来接收数据集、collate_fn和其他参数。

这种单独的序列化意味着，用户应该采取两个步骤来确保，在使用多进程（multi-process）加载数据时与Windows或MacOS兼容：

- 将主脚本的大部分代码包装在`if __name__ == '__main__'`块中，以确保在每个worker进程启动时不会再次运行主脚本，否则很可能会产生错误。可以将数据集和DataLoader实例的创建逻辑放在\_\_main\_\_中，因为它不需要在worker中重新执行。
- 确保任何自定义collate_fn、worker_init_fn、dataset代码都声明为顶级定义（top level definitions），即放在\_\_main\_\_检查之外。这确保了它们在工作进程中可用，这是必需的，因为函数仅被pickle序列化为引用，而不是bytecode字节码。

### 4.4 Randomness in multi-process data loading

默认情况下，每个worker都将其PyTorch seed设置为base_seed+worker_id，其中base_seed是由主进程使用其RNG或指定生成器生成的长种子。但是，在初始化worker时，其他库的种子可能会重复，导致每个worker返回相同的随机数。

在worker_init_fn中，可以通过`torch.utils.data.get_worker_info().seed`或`torch.initial_seed()`，访问每个worker的PyTorch种子，并在数据加载之前使用它来种子化其他库。

## 5. Memory Pinning

当从主机到GPU的拷贝来自固定内存（锁定页面）时，它们要快得多。有关何时以及如何使用固定内存的详细信息，可参阅[Use pinned memory buffers](https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-pinning)。

对于数据加载，将DataLoader构造函数的pin_memory置为True，会让DataLoader自动将提取的数据Tensor放在固定内存中，从而使数据能更快地传输到支持CUDA的GPU上。

默认的内存固定逻辑（pinning logic）只能识别Tensor，以及包含Tensor的映射和可迭代对象。

默认情况下，如果固定逻辑看到一个自定义类型的批量（用户有一个返回自定义批量类型的collate_fn），或者批量中的每个元素都是自定义类型，那么固定逻辑不会识别它们，它返回该批量（或那些元素），而且不会使用固定内存。

要为自定义批量或数据类型启用内存固定，需要在自定义类型上定义pin_memory()方法。具体例子见PyTorch文档所示。

## 6. Else

```python
def random_split(dataset, generator)
```

随机将数据集分割为给定长度的不重叠的新数据集。如下一个例子所示。

```python
generator1 = torch.Generator().manual_seed(42)
random_split(range(10), [3, 7], generator=generator1)
```

如果给定lengths为一个和为1的分数frac列表，则每个新数据集的长度将自动计算为每个分数的`floor(frac*len(dataset))`。在计算完长度后，如果有余数，将以循环方式分配长度，直到没有余数为止。一个例子如下所示。

```python
generator2 = torch.Generator().manual_seed(42)
random_split(range(30), [0.3, 0.3, 0.4], generator=generator2)
```

# Saving and Loading a General Checkpoint in PyTorch

保存和加载一个通用的检查点模型（checkpoint model），以用于推理或恢复训练，可以帮助继续上次中断的工作。当保存一个通用checkpoint时，所要保存的不仅是模型的state_dict，还有在模型训练时会更新的优化器state_dict（包含参数状态等信息）。根据算法不同，可能还需要保存的其他项目，包括停止训练的时间、最新记录的训练损失、外部torch.nn.Embedding层等。

## 1. Introduction

要保存多个对象的checkpoint，必须将它们组织在一个字典中，并使用torch.save()方法将该字典序列化存储。PyTorch通常约定使用.tar文件扩展名保存这些checkpoint。要加载所保存的checkpoint条目，首先要使用torch.load()在加载字典，然后初始化模型和优化器对象，并使用模型或优化器对象的load_state_dict()方法，将checkpoint加载到相应对象中。

## 2. Save the general checkpoint

收集所有相关信息，构建要序列化存储的字典。

```python
net = MyNet()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
for epoch in epochs:
    # training an epoch here
    torch.save({
        'epoch': epoch,
        'loss': loss,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, PATH)
```

### 3. Load the general checkpoint

首先要初始化模型和优化器，然后加载checkpoint到模型或优化器对象。

```python
model = Net()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

checkpoint = torch.load(PATH)
epoch = checkpoint['epoch']
loss = checkpoint['loss']
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

model.eval()
# - or -
model.train()
```

需要注意，在运行推理之前，必须调用model.eval()将Dropout和Batch Normalization层设置为评估（evaluation）模式。否则将产生不一致的推理结果。如果想恢复训练，调用model.train()以确保这些层处于训练模式。

# Warmstarting model using parameters from a different model in PyTorch

在迁移学习或训练一个新的复杂模型时，加载某个模型的一部分，或加载一部分模型（共多个模型），是非常常见的场景。利用预训练好的参数，即使只有少数可用的参数，也将有助于启动训练过程，并比从头开始训练更快地收敛。

无论是从缺少一部分键的state_dict状态字典加载，还是从包含比正在加载的模型更多键的state_dict加载，都可以指定load_state_dict()方法中的strict参数为False以忽略不匹配的键。

```python
class NetA(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.f1 = torch.nn.Linear(784, 1024)

class NetB(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.f1 = torch.nn.Linear(784, 1024)
        self.f2 = torch.nn.Linear(1024, 10)

netA = NetA()
netB = NetB()

# training for NetA, and load its parameters to NetB
for para in netA.parameters():
    with torch.no_grad():
        para.zero_()

torch.save(netA.state_dict(), 'NetA.pt')
dump = torch.load('NetA.pt')
netB.load_state_dict(dump, strict=False)

for para in netB.parameters():
    print(para)
"""
Parameter containing:
Parameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.]], requires_grad=True)
Parameter containing:
tensor([0., 0., 0.,  ..., 0., 0., 0.], requires_grad=True)
Parameter containing:
tensor([[ 0.0192,  0.0251,  0.0142,  ...,  0.0043, -0.0100,  0.0124],
        ...,
        [ 0.0075, -0.0271,  0.0269,  ...,  0.0268,  0.0126, -0.0028]],
       requires_grad=True)
Parameter containing:
tensor([-0.0155, -0.0283, -0.0271, -0.0182, -0.0258], requires_grad=True)
"""
```

# torch.onnx

本节内容若无特殊说明，均在`torch.onnx`模块的命名空间中。

Open Neural Network eXchange（ONNX）是一个用于表示机器学习模型的开放标准格式。torch.onnx模块可以将PyTorch的Module模型转换为ONNX模型。该模型可用于任意支持ONNX格式的运行时（runtime）环境。

对于模型的推理优化来说，通常将模型导出成PB（Protocol Buffer）格式或ONNX（Open Neural Network eXchange）格式。

```python
# 加载其他类型的模型
model = torch.load(f='model.pt')

# 若模型接收多个输入，如model(X1,X2)，则需要将多个输入放到元组tuple中
# 其中X1与X2是与输入形状相同的随机张量，用来指定模型所接收输入的形状
# 可以在输入张量之前，前缀一个Batch的size维度，也可不加
# 但如果模型中有用到Batch维度，则需要加上
X1 = torch.randn([16, 3, 224, 224])
X2 = torch.randn([16, 3, 224, 224])
inputs = (X1, X2)
# Optional
input_names = ['X1', 'X2']
output_names = ['Y']
# 动态轴，在这里指定的维度，可接受动态维数的输入，通常用于指定动态Batch
dynamic_axes = {
    'X1': {0: 'B', 1: 'C', 2: 'H', 3: 'W'},
    'X2': {0: 'B', 1: 'C', 2: 'H', 3: 'W'},
}
training_mode = torch.onnx.TrainingMode.TRAINING if model.training else torch.onnx.TrainingMode.EVAL
constant_folding = False if model.training else True

# Export to onnx
torch.onnx.export(
    model=model, f='model.onnx', verbose=True, training=training_mode, do_constant_folding=constant_folding, 
    args=inputs, input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes,
)
```

