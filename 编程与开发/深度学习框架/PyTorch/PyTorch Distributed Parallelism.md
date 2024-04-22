[toc]

# PyTorch Distributed Overview

本节是对`torch.distributed`模块的概述，将参考文档分为不同的主题，并简要描述每个主题。

## 1. Introduction

自PyTorch v1.6.0以来，torch.distributed模块主要可分为三个主要组件。

[Distributed Data-Parallel Training](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)（DDP）是广泛应用的单程序多数据（single program multiple data）训练范式。在DDP模式，模型会复制到每个进程上，每个模型副本负责处理输入数据的不同子集。DDP进行梯度通信，以保持模型副本同步，并将通信和计算重叠以加速训练。

[RPC-Based Distributed Training](https://pytorch.org/docs/stable/rpc.html)（Remote Procedure Call，RPC）支持不适合用数据并行的通用训练架构，如分布式流水线并行（pipeline parallelism）、参数服务器模式（parameter server paradigm），以及组合使用DDP与其他并行训练模式。RPC会协助管理远程对象的生命周期，并在多机设备上扩展autograd引擎。

[Collective Communication](https://pytorch.org/docs/stable/distributed.html)（C10D）库支持在组内进程之间发送张量。C10D是性能驱动的，并对所有后端（Gloo、NCCL、MPI）完全异步运行。它提供集合（collective）通信API（如all_reduce、all_gather）与点对点（P2P）通信API（如send、isend）。DDP和RPC都基于C10D，其中前者使用集合通信，后者使用P2P通信。通常情况下，开发者无需使用这些原生通信API，而DDP和RPC能处理多数分布式训练场景。然而，在有些情况下会使用这些原生通信API。这可以将通信与计算解耦，并允许对通信内容进行更细粒度的控制，但另一方面，它也放弃了DDP提供的性能优化。

使用torch.nn.DistributedDataParallel分布式训练的示例代码如下所示。

```python
import sys
import time
import torch
import torch.utils.data
import torch.distributed as dist

if __name__ == '__main__':
    dist.init_process_group('nccl', init_method='env://')
    assert dist.is_initialized(), 'Error: dist.is_initialized == False'
    world_size = dist.get_world_size()
    my_rank = dist.get_rank()
    if my_rank != 0:
        # 将除主进程外的其他进程的标准输出置为None
        sys.stdout, sys.stderr = None, None
   	print(f'======== world_size: {world_size} ========')
    torch.cuda.set_device(my_rank)
    device = torch.cuda.current_device()

    train_ds = torch.utils.data.Dataset()
    train_sampler = torch.utils.data.DistributedSampler(train_ds)
    train_loader = torch.utils.data.DataLoader(train_ds, sampler=train_sampler, num_workers=8)

    # 从此处开始加载模型与训练配置，构建DDP模型
    torch.cuda.empty_cache()
    model = torch.nn.Module()
    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], output_device=[device])
    optimizer = torch.optim.Adam(model.parameters(), lr=1.e-3)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1.e-3, epochs=100, steps_per_epoch=len(train_loader),
    )
    train_loss_func = torch.nn.CrossEntropyLoss()

    time_start = time.time()
    _ = run_train_ddp(
        model, train_loader, train_sampler, train_loss_func, optimizer, lr_scheduler
    )
    time_end = time.time()
    print(f'========== train_time: {time_end - time_start:.6f} ==========')

    # 从DDP模型获得单进程模型
    model: torch.nn.Module = model.module
    # 主进程保存结果
    if my_rank == 0:
        torch.save({
            'Module': model.state_dict(),
        }, '/path/to/model.pt')
        print(f'========== Saving Results ==========')
```

使用torch.distributed.elastic启动训练的脚本如下所示。

```shell
#!/bin/bash
source ~/.bashrc
module purge
module load gcc/11.1.0
module load cuda/11.8
module load openmpi/4.1.1_gcc11.1.0_cuda11.8
module load anaconda/2020.11
module list
source activate py39
conda activate py39
export PYTHONUNBUFFERED=1

torchrun --standalone --nnodes=1 --nproc-per-node=8 TRAIN_SCRIPT.py
```

```shell
sbatch --gpus=8 run.sh
```

在run.sh提交脚本中，torchrun的--nproc-per-node参数指定分布式启动的进程数目，每个进程持有一块GPU加速卡，进程数目与系统GPU设备的数目一致。这些进程执行相同的TRAIN_SCRIPT.py脚本。在TRAIN_SCRIPT.py脚本中，torch.utils.data.DataLoader的num_workers指定读取数据集时使用的进程数目，它与上述--nproc-per-node参数无任何关系，这些进程是由上述执行进程派生出来的，它们仅负责加载数据集。

## 2. Data Parallel Training

PyTorch为数据并行训练提供了几种选项。对于从简单到复杂、从原型到生产环境的应用程序，常见的开发轨迹如下所示。

1. 使用单设备（single-device）训练，如果数据和模型能够放到单个GPU上，且训练速度并无紧要。
2. 使用单机多GPU（single-machine multi-GPU）数据并行[DataParallel](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html)训练，在单个机器上使用多个GPU加速训练，且代码改动最少。
3. 使用单机多GPU（single-machine multi-GPU）分布式数据并行[DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)进一步加速训练，并需要编写较多代码进行设置。
4. 使用多机（multi-machine）分布式数据并行[DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)和启动脚本[launching script](https://github.com/pytorch/examples/blob/master/distributed/ddp/README.md)，如果应用需要扩展到跨越多个机器进行训练。
5. 使用[torch.distributed.elastic](https://pytorch.org/docs/stable/distributed.elastic.html)启动分布式训练，如果一些错误是可以预期到的（例如内存不足），或者如果计算资源在训练期间动态挂载和分离。

另外，数据并行训练也适用于自动混合精度（[Automatic Mixed Precision](https://pytorch.org/docs/stable/notes/amp_examples.html#working-with-multiple-gpus)，AMP）。

### 2.1 torch.nn.DataParallel

torch.nn.DataParallel包能够以最低的编码门槛实现单机多GPU并行。它只需要对程序代码进行一行更改。尽管DataParallel非常容易使用，但它通常不能提供最好的性能，因为它在每个forward遍历中都会复制模型，而且它使用单进程多线程（single-process multi-thread）并行，自然会受到GIL竞争的影响。为了获得更好的性能，可以考虑使用DistributedDataParallel。

### 2.2 torch.nn.DistributedDataParallel

torch.nn.DistributedDataParallel需要更多的步骤来设置，即调用init_process_group()函数。DDP使用多进程并行，因此在模型副本之间不存在GIL竞争。此外，模型是在DDP构建时进行广播（broadcast），而不是在每次forward时广播，这有助于加快训练速度。

DDP使用多种性能优化技术，详情参阅论文[PyTorch Distributed: Experiences on Accelerating Data Parallel Training](http://www.vldb.org/pvldb/vol13/p3005-li.pdf)（VLDB'20）。

DDP的一些相关教程资料如下所列出。

1. [Distributed Data Parallel](https://pytorch.org/docs/stable/notes/ddp.html)提供了一个入门示例，并对其设计和实现进行了一些简要描述。
2. [Getting Started with Distributed Data Parallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)解释了DDP训练的一些常见问题，包括工作负载不平衡、检查点（checkpoint）和多设备模型。DDP可以很容易地与单机多设备模型并行相结合，这在[Single-Machine Model Parallel Best Practices](https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html)教程中进行了描述。
3. [Launching and configuring distributed data parallel applications](https://github.com/pytorch/examples/blob/main/distributed/ddp/README.md)文档展示了如何使用DDP启动脚本（DDP launching script）。
4. [Shard Optimizer States With ZeroRedundancyOptimizer](https://pytorch.org/tutorials/recipes/zero_redundancy_optimizer.html)展示了[ZeroRedundancyOptimizer](https://pytorch.org/docs/stable/distributed.optim.html)如何帮助减少优化器的内存占用。
5. [Distributed Training with Uneven Inputs Using the Join Context Manager](https://pytorch.org/tutorials/advanced/generic_oin.html)教程将介绍如何使用通用连接上下文（generic join context）进行非均匀输入的分布式训练。

### 2.3 torch.distributed.elastic

随着应用复杂性和规模的增长，故障恢复（failure recovery）成为一种需求。在使用DDP时，有时不可避免地会遇到内存不足（out-of-memory，OOM）之类的错误，但是DDP本身无法从这些错误中恢复，而且不可能使用标准的try-except结构来处理它们。这是因为DDP要求所有进程以紧密同步的方式运行，不同进程发起的所有AllReduce通信必须匹配。如果组中的某个进程抛出异常，很可能会导致去同步（不匹配的AllReduce操作），从而导致崩溃或挂起。[torch.distributed.elastic](https://pytorch.org/docs/stable/distributed.elastic.html)增加了容错性和使用动态机器池（dynamic pool of machines）的能力。

## 3. RPC-Based Distributed Training

许多训练范式不适合数据并行，例如，参数服务器、分布式流水线并行，多观察者或多代理的强化学习等。[torch.distributed.rpc](https://pytorch.org/docs/stable/rpc.html)旨在支持一般的分布式训练场景。

torch.distributed.rpc包含以下四点主要内容。

1. [RPC](https://pytorch.org/docs/stable/rpc.html#rpc)支持在远程进程上运行指定函数。
2. [RRef](https://pytorch.org/docs/stable/rpc.html#rref)协助管理远程对象的生命周期，引用计数协议等信息参考其文档。
3. [Distributed Autograd](https://pytorch.org/docs/stable/rpc.html#distributed-autograd-framework)将autograd引擎扩展超越单机边界。
4. [Distributed Optimizer](https://pytorch.org/docs/stable/rpc.html#module-torch.distributed.optim)自动联系所有参与worker进程，使用分布式autograd引擎计算的梯度来更新参数。

RPC的一些相关教程资料如下所列出。

1. [Getting Started with Distributed RPC Framework](https://pytorch.org/tutorials/intermediate/rpc_tutorial.html)教程首先使用一个简单的强化学习（Reinforcement Learning，RL）示例来演示RPC和RRef。然后，将一个基本的分布式模型并行应用于一个RNN示例，以展示如何使用分布式autograd和分布式optimizer。
2. [Implementing a Parameter Server Using Distributed RPC Framework](https://pytorch.org/tutorials/intermediate/rpc_param_server_tutorial.html)教程将其应用于异步参数服务器（Parameter Server，PS）训练。
3. [Distributed Pipeline Parallelism Using RPC](https://pytorch.org/tutorials/intermediate/dist_pipeline_parallel_tutorial.html)教程将[Single-Machine Model Parallel Best Practices](https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html)的单机流水线并行示例扩展到分布式环境，并展示如何使用RPC实现它。
4. [Implementing Batch RPC Processing Using Asynchronous Executions](https://pytorch.org/tutorials/intermediate/rpc_async_execution.html)教程演示了如何使用rpc.functions.async_execution装饰器实现批量RPC处理，它可以帮助加速推理和训练。它使用类似于上面教程1和2中的RL和PS示例。
5. [Combining Distributed DataParallel with Distributed RPC Framework](https://pytorch.org/tutorials/advanced/rpc_ddp_tutorial.html)教程演示了如何将DDP与RPC结合起来，以使用分布式数据并行与分布式模型并行来训练模型。

# Distributed Data Parallel

torch.nn.parallel.DistributedDataParallel（DDP）透明地执行分布式数据并行训练。本节描述了它的工作原理与实现细节。

[Distributed Data-Parallel Training](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)（DDP）是广泛应用的单程序多数据（single program multiple data）训练范式。在DDP模式，模型会复制到每个进程上，每个模型副本负责处理输入数据的不同子集。DDP进行梯度通信，以保持模型副本同步，并将通信和计算重叠以加速训练。

## 1. Example

```python
def example(rank, world_size):
    # Create default process group
    torch.distributed.init_process_group('gloo', rank=rank, world_size=world_size)

    model = torch.nn.Linear(10, 10).to(rank)
    model_ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model_ddp.parameters(), lr=0.001)

    samples = torch.randn(32, 10).to(rank)
    targets = torch.randn(32, 10).to(rank)
    outputs = model_ddp(samples)
    loss = loss_fn(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


if __name__ == '__main__':
    # Environment variables which need to be set when using c10d's default 'env' initialization mode.
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    world_size = 2
    torch.multiprocessing.spawn(example, args=(world_size,), nprocs=world_size, join=True)
```

DDP与TorchDynamo一起工作，当与TorchDynamo一起使用时，在编译模型之前应使用DDP包装模型，这样TorchDynamo就可以基于DDP的桶大小（bucket sizes）应用DDPOptimizer（graph-break optimizations）。详见本节之后的内容。

TorchDynamo对DDP的支持目前需要设置static_graph=False，这是由于图跟踪进程（graph tracing process）会与DDP中类似的机制产生影响，即DDP中用于观察其包装Module操作的机制，不过这个问题最终应该会被修复。

```python
model_ddp = DDP(model, device_ids=[rank])
model_ddp = torch.compile(model_ddp)
```

## 2. Internal Design

本节通过迭代中每个步骤的细节，揭示其在torch.nn.parallel.DistributedDataParallel框架下的工作原理。

### 2.1 Prerequisite

DDP依赖于c10d的ProcessGroup进行通信。因此，应用程序必须在构造DDP之前创建ProcessGroup实例。

### 2.2 Construction

DDP构造函数接受一个本地Module的引用，并将rank0进程的state_dict()广播给组中的所有其他进程，以确保所有模型副本都从完全相同的状态启动。

然后，每个DDP进程创建一个本地Reducer，它负责backward传递过程中的梯度同步。为了提高通信效率，Reducer将参数的梯度组织成桶（buckets），每次归约一个桶。桶的大小可以通过在DDP构造函数中设置bucket_cap_mb参数来配置。参数梯度到桶的映射在构建时，会根据桶的大小限制和参数大小确定。

模型参数被分配到桶中的顺序，（大致）与给定模型model.parameters()的参数顺序相反。使用相反顺序的原因是，DDP期望在进行反向传递时，参数梯度（大致）按照这个相反的顺序准备好（计算完成）。下图显示了一个示例。

<img src="PyTorch Distributed Parallelism.assets/DDP Construction.png" style="zoom:33%;" />

注意到，grad0和grad1保存在bucket1中，另外两个梯度grad2和grad3保存在bucket0中。当然，这个假设并不总是正确的。当不满足假设的这种情况时，可能会影响DDP的backward速度，因为Reducer无法尽可能早地开始通信。

除了分桶，Reducer还会在构造过程中注册autograd引擎的hook方法，每个hook对应一个参数。当梯度gradient就绪（计算完成）时，这些hook将在backward传递期间被触发。

### 2.3 Forward Pass

DDP接收输入（input）并将其传递给本地模型，然后在nn.parallel.DistributedDataParallel的find_unused_parameters参数置为True时分析本地模型的输出（output）。

这种模式允许在模型的子图上运行backward。DDP从模型输出的遍历autograd图，并将所有无用参数（unused parameters）标记为就绪归约（ready for reduction），来找出参与backward过程的参数。其中，无用参数指的是对backward中梯度计算无贡献的参数，即不参与梯度计算的参数。

在backward传递的过程中，Reducer只会等待未准备好的参数，但它仍然会归约所有桶。目前，将参数梯度标记为就绪不能帮助DDP跳过桶，但可以防止DDP在backward传递过程中一直等待无需使用的不存在梯度（absent gradients）。

注意，遍历autograd图会引入额外的开销，因此应该只在必要时将find_unused_parameters参数设置为True。

### 2.4 Backward Pass

直接在loss张量上调用backward()函数，这是DDP无法控制的。DDP使用在构造时注册的autograd引擎hook方法来触发梯度同步。

当一个梯度就绪时，它对应的梯度累加器（gradient accumulator）上的DDP的hook方法将被触发，然后DDP将该参数梯度标记为就绪归约。当一个桶中的梯度都就绪时，Reducer会在该桶上启动异步的allreduce，以计算所有进程上的梯度平均值。当所有的桶都就绪时，Reducer会阻塞地等待所有的allreduce操作完成。当所有allreduce操作完成后，平均梯度将被写入所有参数的grad字段。

因此，在backward传递之后，不同DDP进程上的相同参数上的grad字段应该是相同的。

> 实际上，DDP要求所有进程上的Reducer实例以完全相同的顺序调用allreduce，即始终按照桶索引的顺序调用allreduce，而不是实际的桶就绪顺序。跨进程的allreduce顺序不匹配可能会导致错误的结果或DDP挂起。

### 2.5 Optimizer Step

从优化器Optimizer的角度，它优化一个本地模型。所有DDP进程上的模型副本可以保持同步，因为它们都是从相同的状态开始，并且在每次迭代中具有相同的平均梯度。

## 3. Implementation

下面是DDP实现的关键组件，图片显示了代码的结构。

<img src="PyTorch Distributed Parallelism.assets/DDP Structure.png" style="zoom:33%;" />

### 3.1 ProcessGroup

ProcessGroup.hpp，包含所有进程组（process group）实现的抽象API。c10d库提供3种实现，即PorcessGroupNCCL、PorcessGroupGloo、PorcessGroupMPI。nn.parallel.DistributedDataParallel使用ProcessGroup::broadcast()，在初始化期间将rank0进程上的模型状态发送（send）到其他进程上，并使用ProcessGroup::allreduce()累积梯度。

Store.hpp，协助进程组实例的对接集合服务（rendezvous service）以查找彼此。

### 3.2 DistributedDataParallel

distributed.py，是Python实现DDP的入口点。它实现了nn.parallel.DistributedDataParallel模块的初始化步骤和forward()函数，该模块调用C++库。当一个DDP进程在多个设备上运行时，它的_sync_param()函数执行进程内（intra-process）参数同步，它还将rank0进程上的模型缓冲区（buffers）广播给其他所有进程。进程间（inter-process）参数同步发生在reducer.cpp中。

comm.h，实现了合并广播（coalesced broadcast）的辅助函数，该辅助函数用于在初始化期间广播模型状态，并在forward传递之前同步模型缓冲区。

reducer.h，提供用于backward过程中梯度同步的核心实现，它有三个关键函数。Reducer构造函数被distributed.py调用，并为梯度累加器（gradient accumulator）注册Reducer::autograd_hook()方法。autograd_hook()函数会在梯度就绪时被autogard引擎调用。prepare_for_backward()函数会在DDP的forward过程最后被调用，当DDP构造函数的find_unused_parameters参数置为True时，它遍历autograd图以查找无用参数。

### 3.3 TorchDynamo DDPOptimizer

DDP的性能优势来自于backward传递过程中，将allreduce集合通信与计算进行重叠。当与TorchDynamo一起编译整个forward和backward图时，AotAutograd会阻止这种重叠，因为allreduce操作是在整个优化backward计算完成之后，才由autograd的hook函数启动的。

TorchDynamo的DDPOptimizer通过在backward传递过程中，在DDP的allreduce桶的逻辑边界上分割forward图，会起到帮助。注意，目的是在backward时分割（break）图，最简单的实现是分割forward图，然后在每个部分上调用AotAutograd并进行编译（compilation）。这使得DDP的allreduce hook方法可以在backward过程的中间部分触发，并可以将通信安排在与compute重叠的地方。

要调试DDPOptimizer，设置torch.\_dynamo.config.log\_level为DEBUG（用于完整的图转储），或INFO（用于桶边界的基本信息）。

要禁用DDPOptimizer，设置torch.\_dynamo.config.optimize\_ddp为False。在没有DDPOptimizer的情况下，DDP和TorchDynamo仍然可以正常工作，但性能会下降。

# torch.nn.DataParallel

```python
class DataParallel(Module):
    def __init__(self, module, device_ids=None, output_device=None, dim=0)
    def forward(self, *inputs, **kwargs)
    def replicate(self, module, device_ids)
    def scatter(self, inputs, kwargs, device_ids)
    def gather(self, outputs, output_device)
    def parallel_apply(self, replicas, inputs, kwargs)
```

在Module级实现数据并行，单进程多线程。该容器对给定Module应用并行化，给定dim维度（默认0表示数据批量维度），通过对数据进行分块，将数据分割到指定的设备，而其他对象会被复制到每个设备上。给定dim维度的维数应该大于使用的GPU数量。

在forward过程中，Module被复制到每个设备上，每个副本处理输入的一部分。在backward过程中，每个副本的梯度汇总到原始Module中。

> 即使只有一个节点，也推荐使用torch.nn.parallel.DistributedDataParallel来进行多GPU训练，而不是这个类。通常来说，单节点多GPU的数据并行训练，torch.nn.parallel.DistributedDataParallel被证明比torch.nn.DataParallel快得多。

任意位置（positional）和关键字（keyword）输入都允许传给DataParallel，但有些类型会经过特殊处理。张量tensors将在指定dim维度上进行scatter。元组、列表和字典类型将被浅复制（shallow copy），其他类型将在不同的线程之间共享，如果在模型的forward中被写入，可能会损坏。

在运行DataParallel模型之前，并行化Module必须在device_ids[0]上持有其参数和缓冲区。

注意，在每次forward中，Module都被复制到每个设备上，因此对正在运行的Module的任何更新都会丢失。例如，如果Module有一个在每次forward过程中递增的counter属性，它将始终保持初始值，因为更新是在副本上完成的，而副本在每次backward后都会被销毁。但是，DataParallel保证device[0]上的副本，与被并行化的基本Module共享参数和缓冲区。因此，对device[0]上的副本的参数或缓冲区的就地更新会被记录。例如，BatchNorm2d()和spectral_norm()就依赖于这种行为来更新缓冲区。

在Module及其子代上定义的forward和backward过程中的hook函数将被调用len(device_ids)次，每个hook函数的输入都是位于特定设备上的。而且，只能保证对应设备上的hook按照该设备上的操作的正确顺序执行。例如，不能保证通过register_forward_pre_hook()设置的hook在所有的forward()调用之前执行，但每个这样的hook都在自己设备调用forward()之前执行。

当Module在forward()中返回一个标量（即0维张量）时，此DataParallel包装器将返回一个长度等于数据并行中使用的设备数量的向量，其中包含每个设备的结果（标量）。

```python
model_dp = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
optimizer = torch.optim.SGD(model_dp.parameters(), lr=cfg['lr'])
for inputs in dataloader:
    # inputs can be on any device, including CPU
	outputs = model_dp(inputs)
```

# torch.nn.parallel.DistributedDataParallel

```python
class DistributedDataParallel(Module, Joinable):
    def __init__(self, module, device_ids=None, output_device=None, dim=0, broadcast_buffers=True, 
                 process_group=None, bucket_cap_mb=25, find_unused_parameters=False,
                 check_reduction=False, gradient_as_bucket_view=False, static_graph=False,)
```

基于torch.distributed包，在Module级别实现分布式数据并行。要使用DistributedDataParallel，首先需要调用torch.distributed.init_process_group()初始化torch.distributed包。

该容器对给定Module应用并行化，在数据批量的维度，给定dim维度（默认0表示数据批量维度），通过对数据进行分块，将数据分割到指定的设备，而其他对象会被复制到每个设备上。给定dim维度的维数应该大于使用的GPU数量。

该容器通过同步每个模型副本的梯度来提供数据并行。要进行同步的设备由process_group参数指定，默认情况下它是整个进程组（entire world）。注意，DistributedDataParallel不会自动在GPU之间对输入进行分块或分片，而是由用户定义如何这样做，例如使用torch.utils.data.DistributedSampler类。

要在具有N个GPU的主机上使用DistributedDataParallel，应该执行N个进程，并确保每个进程只在从0到N-1的单个GPU上执行。这可通过为每个进程设置CUDA_VISIBLE_DEVICES环境变量来完成，也可通过调用torch.cuda.set_device(idx)函数来完成。在每个进程中，应按以下方式来构建该Module，如下。

```python
torch.distributed.init_process_group(backend='nccl', world_size=N, init_method='...')
model = DistributedDataParallel(model, device_ids=[idx], output_device=idx)
```

为在每个节点上生成多个进程，可以使用torch.distributed.launch模块或torch.multiprocessing.spawn模块。

## 1. Notes

DistributedDataParallel可以与torch.distributed.optim.ZeroRedundancyOptimizer一起使用，以减少每个进程（per-rank）的优化器状态的内存占用。参阅[ZeroRedundancyOptimizer recipe](https://pytorch.org/tutorials/recipes/zero_redundancy_optimizer.html)了解更多细节。

当使用GPU时，NCCL后端是目前最快的，也是强烈推荐的后端。这适用于单节点和多节点的分布式训练。

该模块还支持混合精度（mixed-precision）分布式训练。这意味着网络模型可以有不同精度数据类型的参数，比如FP16和FP32的混合类型，这些混合类型参数的归约仍可以很好地正常工作。

如果在某个进程中使用torch.save()保存模型checkpoint，然后在另一些进程种使用torch.load()加载模型checkpoint以恢复，应确保每个进程都正确指定torch.load()方法的map_location参数。否则，torch.load()会将模型恢复到保存模块的设备。

如果对一批样本的损失进行求和，而不是像往常那样平均，那么，当一个模型在M个节点上以batch_size=N训练时，与在单个节点上以batch_size=M*N训练相比，梯度将小M倍。因为不同节点之间的梯度是进行平均的。当希望获得与本地训练在数学意义上等价的训练过程时，应该考虑这一点。但在大多数情况下，可以将单个GPU上的DistributedDataParallel包装模型、DataParallel包装模型和普通模型视为相同的。例如，对于相同的批量大小使用相同的学习率。

DistributedDataParallel容器对梯度执行allreduce步骤，并假设优化器将在所有进程中以相同的方式更新模型参数。而模型参数永远不会在进程之间广播，而缓冲区（例如BatchNorm状态）会在每次迭代中从rank0的Module广播到系统中的所有其他副本上。

如果同时使用DistributedDataParallel和[Distributed RPC Framework](https://pytorch.org/docs/stable/rpc.html#distributed-rpc-framework)，应该始终使用torch.distributed.autograd.backward()来计算梯度，使用torch.distributed.optim.DistributedOptimizer来优化参数。这是beta版的，可能会发生变化。

```python
import torch.distributed.optim
import torch.distributed.autograd as dist_autograd
from torch.distributed import rpc

model = torch.nn.Linear(10, 10)
model_ddp = torch.nn.parallel.DistributedDataParallel(model)
loss_fn = torch.nn.MSELoss()

targets = torch.rand([16, 10])
t1 = torch.rand([16, 10], requires_grad=True)
t2 = torch.rand([16, 10], requires_grad=True)
t3 = rpc.remote('worker1', torch.add, args=(t1, t2))
# Setup optimizer
optim_params = [t3, ]
for param in model_ddp.parameters():
    optim_params.append(rpc.RRef(param))
dist_optim = torch.distributed.optim.DistributedOptimizer(torch.optim.SGD, optim_params, lr=0.05, )

with dist_autograd.context() as context_id:
    preds = model_ddp(t3.to_here())
    loss = loss_fn(preds, targets)
    dist_autograd.backward(context_id, [loss])
    dist_optim.step(context_id)
```

DistributedDataParallel目前通过torch.utils.checkpoint()对梯度checkpoint提供了有限的支持。当模型中没有无用参数，且每层最多只设置一次checkpoint时，DDP将按预期工作。请确保没有将find_unused_parameters=True传递给DistributedDataParallel。PyTorch目前不支持一个层多次checkpoint的情况，或者checkpoint模型中有无用参数。

为了让非DDP模型从DDP模型加载状态字典state_dict，在加载之前，要对DDP模型的状态字典调用consume_prefix_in_state_dict_if_present()方法，以剥离"module."前缀。

## 2. Warnings

构造函数、forward方法、输出的微分（或该Module的输出函数）是分布式的同步点（synchronization points）。如果不同的进程可能执行不同的代码，需要考虑到这一点。

该DDP容器在创建模型时，假设所有参数都已注册在模型中。且在这之后不应该添加或删除任何参数，对缓冲区来说也一样。而且，DDP假定每个分布式进程模型中注册的所有参数顺序相同。DDP本身将按照模型注册参数的逆序进行梯度allreduce。也即，用户需要确保每个分布式进程具有完全相同的模型，从而具有完全相同的参数注册顺序。

在使用DistributedDataParallel将模型封装起来之后，永远不要尝试更改模型的参数。因为，当使用DistributedDataParallel包装模型时，DistributedDataParallel的构造函数会在构造时，将额外的梯度归约函数（gradient reduction functions）注册到模型本身的所有参数上。如果之后改变了模型的参数，梯度归约函数将无法再匹配正确的参数。

该DDP容器允许参数具有非连续行主序（non-rowmajor-contiguous）的跨度（stride）。例如，模型可能包含一些参数，其torch.memory_format是torch.contiguous_format，而另一个参数的格式（format）为torch.channels_last。但是，不同进程中的相同参数必须具有相同的跨度。

该DDP容器不使用torch.autograd.grad()，torch.autograd只有在参数的grad属性中累积梯度时才会起作用。而应该使用分布式的torch.distributed.autograd。

如果打算将DDP容器与（使用InfiniBand的）GLOO或NCCL后端一起使用，以及与使用多个worker的DataLoader一起使用，需将多进程multiprocessing启动方法更改为forkserver或spawn。不幸的是，（使用Infiniband的）GLOO和NCCL2不是fork安全的，如果不更改此设置，可能会遇到死锁。

## 3. Methods

```python
class DistributedDataParallel(Module, Joinable):
    def __init__(self, module, device_ids=None, output_device=None, dim=0, broadcast_buffers=True, 
                 process_group=None, bucket_cap_mb=25, find_unused_parameters=False,
                 check_reduction=False, gradient_as_bucket_view=False, static_graph=False,)
```

- module参数(Module)，要被并行的模型。
- device_ids参数(list of int or torch.device)，CUDA设备。(1)对在单个CUDA设备上的Module来说，device_ids可以只包含一个设备id，表示该进程的Module所在的唯一CUDA设备。或者device_ids也可置为None。(2)对在多个CUDA设备上的Module，或在CPU上的Module来说，device_ids必须为None。当两种情况下的device_ids均为None时，forward输入数据和真正的Module都必须放置在正确的设备上。
- output_device参数(int or torch.device)，对在单个CUDA设备上的Module来说，该参数指定输出设备位置，默认为device_ids[0]。而对在多个CUDA设备上的Module，或在CPU上的Module来说，该参数必须为None，并且Module本身决定输出位置。
- broadcast_buffers参数(bool)，若为True，则在forward函数开始，同步（广播）模型的缓冲区。
- process_group参数，进行分布式数据allreduce的进程组。若为None，则使用由torch.distributed.init_process_group()创建的默认进程组。
- bucket_cap_mb参数，DistributedDataParallel将参数分到多个桶中，以便可能将每个桶的梯度归约与backward反向计算重叠。该参数bucket_cap_mb控制桶容量的大小，单位为MegaBytes（MB）。
- find_unused_parameters参数(bool)，从“被DDP包装的Module模型的forward()函数的返回值中包含的所有张量”开始遍历autograd图，而autograd图中不会接收到梯度的参数，会被提前标识为就绪归约（ready for reduction）。此外，对于在模型forward()函数中使用，但不参与loss计算的参数，自然不会接收到梯度，也会被提前标识为就绪归约。
- gradient_as_bucket_view参数(bool)，若为True，梯度gradient是指向allreduce通信桶（communication buckets）不同偏移位置（offset）的视图（view）。这可以减少内存的峰值使用，其节省的内存大小将等于总共gradient的大小。此外，它还避免了梯度gradient和allreduce通信桶之间进行复制的开销。当梯度gradient是视图时，不能在梯度上调用detach_()方法。注意，在第一次迭代后梯度才将会是视图，因此应该在第一次迭代后检查所节省的峰值内存。
- static_graph参数(bool)，若为True，通知DDP训练图是静态的。

静态图是指，(1)在整个训练循环中有用和无用的参数集不会发生变化，在这种情况下，用户是否设置find_unused_parameters=True并不重要。(2)图的训练方式在整个训练循环中不会改变，意味着没有基于迭代次数的控制流。

当static_graph置为True时，DDP将支持之前不支持的情况，如下所示，(1)可重入后向传递（Reentrant backwards）；(2)多次激活检查点（checkpoint）；(3)模型有无用参数时的激活检查点；(4)有些模型参数在forward()函数之外；(5)当存在无用的参数时，可能会提高性能，因为当static_graph设置为True时，DDP将不会在每次迭代中搜索图来检测无用参数。

要检查是否可以将static_graph设置为True，一种方法是在之前的模型训练结束时检查DDP日志数据，如果ddp_logging_data.get('can_set_static_graph')==True，大多数情况下可以将static_graph设置为True。如下所示。

```python
model_ddp = torch.nn.parallel.DistributedDataParallel(model)
# Training loop
ddp_logging_data = model_ddp._get_ddp_logging_data()
static_graph = ddp_logging_data.get('can_set_static_graph')
```

```python
class DistributedDataParallel(Module, Joinable):
    @contextmanager
    def no_sync(self)
```

一个上下文管理器，用于禁用DDP进程之间的梯度同步。在这个上下文中，梯度将在Module的模型参数（Parameter）上累积，这些参数的梯度将在退出上下文后的第一次forward-backward传递中同步。

```python
model_ddp = torch.nn.parallel.DistributedDataParallel(model, process_group=pg)
with model_ddp.no_sync():
    for inp in inputs:
        model_ddp(inp).backward()  # no synchronization, accumulate grads
model_ddp(another_inp).backward()  # synchronize grads
```

```python
class DistributedDataParallel(Module, Joinable):
    def join(self, divide_by_initial_world_size=True, enable=True, throw_on_early_termination=False,)
```

一个与torch.nn.parallel.DistributedDataParallel实例一起使用的上下文管理器，以便在参与进程上使用不均匀的输入（uneven inputs）进行训练。

- divide_by_initial_world_size参数(bool)，若为True，将梯度除以启动DDP训练时设置的初始world_size。若为False，将计算有效的world_size（尚未耗尽输入的rank进程数量），并在allreduce期间将梯度除以有效world_size。设置divide_by_initial_world_size=True，以确保每个输入样本（包括非均匀输入）在对全局梯度的贡献方面具有相同的权重。
- enable参数(bool)，是否开启非均匀输入检测（uneven input detection）。如果用户知道输入数量跨进程是均匀的（even），则可以传递enable=False来禁用此上下文管理器。
- throw_on_early_termination参数(bool)，当某个rank进程的输入耗尽时，是抛出错误还是继续训练。若为True，将抛出第一个耗尽数据的rank。若为False，将以更小的有效world_size继续训练，直到所有进程都并入（join）完成。注意，如果指定了该标志，则divide_by_initial_world_size标志将被忽略。

此上下文管理器将跟踪已并入（already-joined）的DDP进程，并通过插入集合通信操作（collective communication operations），来与未并入的DDP进程的通信操作相匹配，从而隐藏（shadow）前向forward传递和后向backward传递。这将确保对每个集合调用，已并入的DDP进程都会有相应的调用，以防止在跨进程输入不均匀的情况下进行训练时发生挂起或错误。

或者，如果将标志参数throw_on_early_termination指定为True，则一旦任一rank进程的输入耗尽，所有训练器进程（trainer）都会抛出一个错误，从而允许根据程序的逻辑捕获和处理这些错误。

如果这个上下文管理器所围绕的模型或训练循环具有额外的分布式集合操作，例如模型forward过程中的SyncBatchNorm，则必须启用throw_on_early_termination标志。这是因为此上下文管理器无法识别非DDP的集合通信。当任一rank进程的输入耗尽时，这个标志将导致所有rank进程抛出错误，这些错误允许被捕获并处理相应逻辑。

一旦所有的DDP进程都已并入（joined），此上下文管理器会将最后一个并入进程上的模型，广播给所有进程，以确保模型在所有进程中都是相同的。着由DistributedDataParallel提供保证。

要用它来支持跨进程不均匀输入的训练，只需将此上下文管理器用于（围绕住）一般的训练循环即可，而无需对模型或数据加载做进一步修改。如下一个例子所示。

```python
# On each spawned worker
def worker(rank):
    torch.distributed.init_process_group("nccl", rank=rank, world_size=2)
    torch.cuda.set_device(rank)
    model = nn.Linear(10, 10, bias=False).to(rank)
    model_ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)
    # Rank 1 gets one more input than rank 0.
    inputs = [torch.randn([10,]).float() for _ in range(10 + rank)]
    with model_ddp.join():
        for _ in range(5):
            for inp in inputs:
                loss = model_ddp(inp).sum()
                loss.backward()
    # Without the join() API, the below synchronization will hang blocking 
    # for rank 1's allreduce to complete. (2 ranks totally)
    torch.cuda.synchronize(device=rank)
```

```python
class DistributedDataParallel(Module, Joinable):
    def join_hook(self, **kwargs,)
```

返回DDP的join的hook方法，它通过对forward和backward中的集合通信创建影子（shadow），来提供对不均匀输入训练的支持。

```python
class DistributedDataParallel(Module, Joinable):
	def register_comm_hook(self, state: object, hook: callable)
"""
The callable hook function is defined like below:
def hook_fn(state: object, bucket: dist.GradBucked) -> torch.futures.Future[torch.Tensor]
"""
```

注册一个通信的hook方法，可以指定DDP如何在多个worker进程之间聚合梯度（aggregates gradients）。DDP通信hook方法只能注册一次，并且应该在backward之前注册。

这个hook对研究人员尝试新想法非常有用。例如，这个hook可以用于实现几种算法，如GossipGrad和梯度压缩（gradient compression），这些算法涉及分布式数据并行训练中用于参数梯度同步的不同通信策略。

- state参数(object)，传递给hook函数的，用以维护训练过程中的任何状态信息。包括错误反馈（error feedback），如梯度压缩中的，GossipGrad中与下一个同行peers通信中的，等等。每个worker将其存储到本地，并由worker上的所有梯度张量共享。
- hook参数(callable)，一个可调用的函数，其签名（signature）如上所示。前述，Reducer将参数的梯度组织成桶（buckets），每次归约一个桶。当某个桶就绪时，会调用该hook函数。该hook函数可以执行任何需要的处理，并返回一个表示任何异步操作（如allreduce）执行结果的torch.futures.Future对象。即使hook不执行任何通信，仍需返回一个Future对象。

Future应该拥有梯度桶中的梯度张量的Tensor新值。一旦某个桶就绪，c10d库的Reducer将调用这个hook，并使用Future返回的张量（新的梯度张量），并将梯度复制到各个参数的grad属性。注意，Future的返回类型必须是单个张量，且其形状应与梯度桶内的梯度张量形状相同。

PyTorch还提供了c10d::ProcessGroup::Work的一个get_future()函数，用于获取表示c10d.ProcessGroup.Work执行结果的Future对象。get_future()目前支持NCCL，也支持GLOO和MPI上的大多数操作，除了点对点操作（send/recv）。

在注册hook的情况下，梯度桶中的梯度张量不会默认除以world_size。对于allreduce这样的操作，需要用户负责除以world_size。

下面是一个返回相同张量的hook示例，不进行任何通信。

```python
def noop_fn(state, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
    fut = torch.futures.Future()
    fut.set_result(bucket.buffer())
    return fut
model_ddp.register_comm_hook(state=None, hook=noop_fn)
```

下面是一个并行SGD算法的示例，其中梯度在allreduce之前编码，然后在allreduce之后解码。

```python
def encode(tensor):
    # Some encode operation.
    return tensor
def decode(future):
    # Some decode operation.
    tensor = future.value()[0]
    return tensor

def encode_and_decode_hook(state, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
    encoded_tensor = encode(bucket.buffer())
    fut = torch.distributed.all_reduce(encoded_tensor).get_future()
    fut.then(decode)
    return fut

model_ddp.register_comm_hook(state=None, hook=encode_and_decode_hook)
```

# torch.distributed - Distributed Communication Packages

本节内容若无特殊说明，均在`torch.distributed`模块的命名空间中。

[Collective Communication](https://pytorch.org/docs/stable/distributed.html)（C10D）库支持在组内进程之间发送张量。C10D是性能驱动的，并对所有后端（Gloo、NCCL、MPI）完全异步运行。它提供集合（collective）通信API（如all_reduce、all_gather）与点对点（P2P）通信API（如send、isend）。DDP和RPC都基于C10D，其中前者使用集合通信，后者使用P2P通信。通常情况下，开发者无需使用这些原生通信API，而DDP和RPC能处理多数分布式训练场景。然而，在有些情况下会使用这些原生通信API。这可以将通信与计算解耦，并允许对通信内容进行更细粒度的控制，但另一方面，它也放弃了DDP提供的性能优化。

## 1. Backends

torch.distributed支持三种内嵌的后端（backend），每个都有不同的功能。

下表列出了哪些函数可用于CPU或CUDA张量。MPI仅在构建PyTorch的实现支持CUDA时才支持CUDA。

| BACKEND for DEVICE | Gloo-CPU | Gloo-GPU | MPI-CPU | MPI-GPU | NCCL-CPU | NCCL-GPU |
| ------------------ | -------- | -------- | ------- | ------- | -------- | -------- |
| send               | Yes      | -        | Yes     | ?       | -        | Yes      |
| recv               | Yes      | -        | Yes     | ?       | -        | Yes      |
| broadcast          | Yes      | Yes      | Yes     | ?       | -        | Yes      |
| all_reduce         | Yes      | Yes      | Yes     | ?       | -        | Yes      |
| reduce             | Yes      | -        | Yes     | ?       | -        | Yes      |
| all_gather         | Yes      | -        | Yes     | ?       | -        | Yes      |
| gather             | Yes      | -        | Yes     | ?       | -        | Yes      |
| reduce_scatter     | -        | -        | -       | -       | -        | Yes      |
| all_to_all         | -        | -        | Yes     | ?       | -        | Yes      |
| barrier            | Yes      | -        | Yes     | ?       | -        | Yes      |

### 1.1 Backends that come with PyTorch

PyTorch分布式模块支持Linux(stable)、MacOS(stable)、Windows(prototype)平台。默认情况下，Linux的Gloo和NCCL后端被构建并包含在PyTorch的torch.distributed模块中（NCCL后端仅在CUDA构建项目时使用）。MPI是一个可选的后端，只有在从源代码构建PyTorch时才能包含它。例如，在安装了MPI的主机上编译PyTorch。

> 从PyTorch v1.8开始，Windows支持除NCCL之外的所有集合通信后端，如果init_process_group()的init_method参数指向一个文件，则它必须遵循以下模式：
>
> 对本地文件系统，init_method='file:///d:/tmp/some_file'
>
> 对共享文件系统，init_method='file://////{machine_name}/{share_folder_name}/some_file'
>
> 与Linux平台一样，可以通过设置环境变量MASTER_ADDR和MASTER_PORT来开启TcpStore。

### 1.2 Which backend to use?

经验法则：(1)分布式CPU训练使用Gloo后端；(2)分布式GPU训练使用NCCL后端。

> InfiniBand（直译为“无限带宽”技术，IB）是一个用于高性能计算的计算机网络通信标准，它具有极高的吞吐量和极低的延迟，用于计算机与计算机之间的数据互连。InfiniBand也用作服务器与存储系统之间的直接或交换互连，以及存储系统之间的互连。

使用无限带宽（InfiniBand）互连的CPU主机。如果InfiniBand已启用IP over IB，使用Gloo，否则使用MPI。PyTorch计划在即将发布的版本中添加对Gloo的InfiniBand支持。

使用以太网（Ethernet）互连的CPU主机。使用Gloo，除非有特定的使用MPI的理由。

使用无限带宽（InfiniBand）互连的GPU主机。使用NCCL，因为它是目前唯一支持InfiniBand和GPUDirect的后端。

使用以太网（Ethernet）互连的GPU主机。使用NCCL，因为它目前提供了最好的分布式GPU训练性能，特别是对于单节点多进程或多节点分布式训练。如果在使用NCCL时遇到任何问题，可使用Gloo作为备用选项。但请注意，Gloo目前在GPU上的运行速度比NCCL慢。

### 1.3 Common environment variables


#### 1.3.1 Choosing the network interface to use

默认情况下，NCCL和Gloo后端都会尝试找到要使用的正确网络接口。如果自动检测到的接口不正确，可以使用以下环境变量（适用于各自的后端）覆盖它：

- NCCL_SOCKET_IFNAME，环境变量，覆盖的示例，export NCCL_SOCKET_IFNAME=eth0
- GLOO_SOCKET_IFNAME，环境变量，覆盖的示例，export GLOO_SOCKET_IFNAME=eth0

如果使用Gloo后端，可以通过使用逗号分隔多个接口，如export GLOO_SOCKET_IFNAME=eth0,eth1,eth2,eth3。Gloo后端将在这些接口之间以轮询方式分发操作（operations）。所有进程必须在该环境变量中指定相同数目的接口。

#### 1.3.2 Other NCCL environment variables

调试（Debugging）。在NCCL错误的情况下，可以设置环境变量NCCL_DEBUG=INFO，打印明确的警告消息以及基本的NCCL初始化信息。

还可以使用NCCL_DEBUG_SUBSYS来获得有关NCCL特定方面的更多细节。例如，NCCL_DEBUG_SUBSYS=COLL将打印集合调用的日志，这可能有助于调试挂起（debugging hangs），特别是那些由集合类型或消息大小不匹配导致的挂起。在拓扑检测（topology detection）错误的情况下，设置NCCL_DEBUG_SUBSYS=GRAPH有助于查看详细的检测结果，并可以在需要NCCL团队进一步帮助时，将检测结果保存为参考（reference）。

性能调优（Performance tuning）。NCCL根据其拓扑检测进行自动调优，以节省用户的调优工作。在一些基于套接字（socket）的系统上，用户仍然可以尝试调优NCCL_SOCKET_NTHREADS和NCCL_NSOCKS_PERTHREAD来增加套接字网络带宽。NCCL已经为一些云提供商（如AWS或GCP）预调优了这两个环境变量。

有关NCCL环境变量的完整列表，请参阅[NVIDIA NCCL’s official documentation](https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/env.html)文档。

## 2. Basics

torch.distributed模块为运行在一台或多台机器上的多个计算节点上的多进程并行，提供了PyTorch支持和通信原语（communication primitives）。torch.nn.parallel.DistributedDataParallel类构建于此功能之上，作为任何PyTorch模型nn.Module的包装器（wrapper）提供同步的分布式训练。这与torch.multiprocessing模块和torch.nn.DataParallel提供的并行不同，因为torch.distributed支持多个网络互连的机器，并且用户必须为每个进程显式启动主训练脚本（main training script）的单独副本。

在单机同步的情况下，torch.distributed模块或torch.nn.parallel.DistributedDataParallel包装器可能仍然比其他数据并行方法（如torch.nn.DataParallel）有优势，原因有以下两点。

- 每个进程都维护自己的优化器optimizer，并在每次迭代中执行完整的优化步骤。虽然这可能看起来是多余的，但因为梯度已经聚集（gather）在一起并在各个进程中平均，因此梯度对每个进程来说都是相同的，这意味着不需要对模型参数进行广播，减少了节点之间传输张量的时间。
- 每个进程都包含一个独立的Python解释器（interpreter），消除了由单个Python进程驱动多个“执行线程、模型副本、GPU”而产生的额外解释器开销和GIL抖动（GIL-thrashing）。这对于大量使用Python运行时（runtime）环境的模型尤其重要，包括具有循环层或许多小组件的模型。

## 3. Initialization

在调用分布式包（torch.distributed模块）的其他任何方法前，需要调用torch.distributed.init_process_group()方法初始化，这会阻塞直到所有进程都并入（join）。

```python
def is_available() -> bool
```

如果torch.distributed模块可用，则返回True；否则，torch.distributed不暴露任何其他API。目前，torch.distributed可在Linux、MacOS、Windows平台上使用。在从源代码构建PyTorch时设置USE_DISTRIBUTED=1以启用它。目前，默认值为Linux和Windows的USE_DISTRIBUTED=1，MacOS的USE_DISTRIBUTED=0。

```python
def is_mpi_available() -> bool
def is_nccl_available() -> bool
def is_gloo_available() -> bool
```

检查相应的mpi,nccl,gloo后端是否可用。

```python
def init_process_group(backend, init_method=None, timeout=default_pg_timeout,
    world_size=-1, rank=-1, store=None, group_name='', pg_options=None,)
```

初始化默认的分布式进程组（distributed process group），这也将初始化torch.distributed模块。

初始化进程组主要有两种方式：(1)显式指定world_size,rank,store参数；(2)指定init_method（URL字符串）参数，它指出如何定位/寻找另一个（peers），此时也可指定可选的world_size,rank参数，或直接在URL中编码所有必需的参数并忽略它们。如果上述两种方式都没有指定，则假定init_method默认为'env://'。

- backend参数(str or Backend, optional)，要使用的后端。根据构建时的配置，有效值包括mpi,gloo,nccl,ucc。如果未提供后端，则将同时创建gloo和nccl后端。这个参数可以是小写字符串（如'gloo'），也可以是Backend类的属性字段（如Backend.GLOO）。如果用nccl后端在每台机器上使用多个进程，每个进程必须独占访问它使用的每个GPU，因为在进程之间共享GPU可能会导致死锁。注意，若要启用mpi后端，需要在支持MPI的系统上源码构建PyTorch。

- init_method参数(str, optional)，指定如何初始化进程组的URL。如果未指定init_method或store，则init_method默认为'env://'。该参数与store参数互斥。
- world_size参数(int, optional)，参与作业（job）的进程数。如果指定了store参数，则必须要指定该参数。
- rank参数(int, optional)，当前进程的编号（rank），是一个介于0和world_size-1之间的整数。如果指定了store参数，则必须要指定该参数。
- store参数(Store, optional)，所有进程worker都可以访问的键值对存储（key-value store），用于交换连接/地址信息（connection/address information）。该参数与init_method参数互斥。
- group_name参数（str, optional, deprecated），进程组的名称。
- pg_options参数（ProcessGroupOptions, optional），进程组选项，指定构造进程组需要传递的附加选项。到目前为止，PyTorch唯一支持的选项是nccl后端的ProcessGroupNCCL.Options，可指定为is_high_priority_stream，以使得nccl后端在有计算内核等待时执行高优先级的cuda流。
- timeout参数(timedelta, optional)，对进程组执行操作的超时时间（timeout），默认值为30分钟。这适用于gloo后端。对于nccl，仅当环境变量NCCL_BLOCKING_WAIT或NCCL_ASYNC_ERROR_HANDLING设置为1时才适用。当设置了NCCL_BLOCKING_WAIT时，这是进程阻塞并等待集合（collectives）完成的持续时间，时间结束后抛出异常。当设置了NCCL_ASYNC_ERROR_HANDLING时，持续时间结束后，集合（collectives）将被异步中止，并且进程将崩溃（crash）。

```python
def is_initialized() -> bool
```

检查默认进程组是否已初始化。

```python
def is_torchelastic_launched() -> bool
```

检查进程是否通过torch.distributed.elastic（又名torchelastic）启动。可根据TORCHELASTIC_RUN_ID环境变量是否存在，确定当前进程是否用torchelastic启动。因为TORCHELASTIC_RUN_ID映射到集合id（rendezvous id），该id总是一个非空值，表示另一个（peer）用于发现的作业id。

目前支持三种初始化方法，如下所示。

### 3.1 TCP initialization

使用TCP进行初始化，需要一个所有进程都可访问的网络地址（network address），以及world_size参数。需要用init_method参数指定一个rank0进程的地址，该初始化方法要求所有进程都手动指定其rank编号。

```python
# Use address of one of the machines
dist.init_process_group(backend, init_method='tcp://10.0.0.1:23456', rank=myrank, world_size=4)
```

注意，最新的分布式包不再支持多播地址，group_name参数也被弃用。

### 3.2 Shared file-system initialization

该初始化方法利用了一个对进程组内所有机器都可见的共享文件系统，以及world_size参数。需要用init_method参数指定一个'file://'开头的URL，它包含一个路径，指向共享文件系统中不存在的文件（但应在一个已存在的目录中）。如果该文件不存在，文件系统将自动创建初始化该文件，但不会删除该文件。因此，若下次调用init_process_group()使用相同的文件路径/名称，应在调用之前清理上次自动创建的文件。

这种使用共享文件系统初始化的方式总是会创建文件，并尽可能在程序结束时清理和删除文件。换句话说，每次使用文件方式进行初始化，都需要一个全新的空文件，这样初始化才能成功。如果再次使用前一次初始化中使用的相同文件（碰巧没有被清理），这是意外的行为，通常会导致死锁和错误。

这种方式假设文件系统支持fcntl文件锁，大多数系统和NFS都支持。

注意，在最新的分布式模块中，不再支持自动分配rank（需手动指定），且group_name参数也已弃用。

```python
# rank should always be specified
dist.init_process_group(backend, init_method='file:///mnt/nfs/sharedfile', rank=myrank, world_size=4)
```

### 3.3 Environment variable initialization

在使用c10d库之前，需要设置环境变量MASTER_ADDR和MASTER_PORT，分别表示“主”进程地址及端口号。

```python
# Environment variables which need to be set when using c10d's default 'env' initialization mode.
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'
```

这个方法将从环境变量中读取配置，允许用户完全自定义获取信息的方式。需要设置的变量有：

- MASTER_ADDR，环境变量，应是rank0进程所在机器的URL地址，除rank0号进程外，其他进程都需要该环境变量。
- MASTER_PORT，环境变量，应是rank0进程所在机器上的自由网络端口。
- WORLD_SIZE，环境变量，可以在环境变量中设置，也可在init_process_group()初始化方法中设置。
- RANK，环境变量，可以在环境变量中设置，也可在init_process_group()初始化方法中设置。

rank0进程的机器将用于建立所有连接。

这是默认使用的方法，也就是说不必指定init_method（将采用默认的'env://'），当然也可显式指定init_method参数为'env://'。

## 4. Post-Initialization

在使用torch.distributed.init_process_group()初始化后，可以使用分布式包所提供的类与方法，在下面列出。要检查进程组是否已经初始化，可使用torch.distributed.is_initialized()函数。

```python
class Backend:
    UNDEFINED = 'undefined'
    GLOO = 'gloo'
    NCCL = 'nccl'
    MPI = 'mpi'
    TCP = 'tcp'
    def __new__(cls, name: str)
    @classmethod
    def register_backend(cls, name, func)
```

一个类似枚举的后端类，包含GLOO,NCCL,MPI,UCC和其他已注册的后端。该类的值为小写字符串，例如'gloo'，它们也可作为属性访问，例如Backend.GLOO。条目Backend.UNDEFINED只用作某些字段的初始值，用户不应该直接使用它，也不应该假设它的存在。

这个类可以直接调用来解析字符串，例如，Backend(backend_str)将检查backend_str是否有效，如果有效则返回解析后的小写字符串。它也接受大写字符串，例如，Backend('GLOO')返回'gloo'。

```python
def get_backend(group=None) -> str
```

返回给定进程组group的后端。

- group参数(ProcessGroup, optional)，指定某个进程组，默认是通用主进程组。若指定某个特定进程组，则调用该方法的进程必须属于该进程组。

```python
def get_world_size(group=None) -> int
```

返回给定进程组group中的进程数目。


```python
def get_rank(group=None) -> int
```

返回当前进程在给定进程组group中的rank编号，若未指定group则为默认进程组。

rank是分配给分布式进程组中的每个进程的唯一标识符，它们总是连续的整数，范围从0到world_size-1。如果调用该方法的进程不属于指定进程组，则返回-1。

## 5. Distributed key-value store

分布式包提供了分布式的键值对存储（key-value store）方式，可用于在组中的进程之间共享信息。此外，在torch.distributed.init_process_group()中初始化torch.distributed时也可指定store参数，即显式创建键值对存储，替代指定init_method方法。

```python
class Store(__pybind11_builtins.pybind11_object):
    def __init__(self)
    def set(self, arg0, arg1)
    def get(self, arg0)
    def compare_set(self, arg0, arg1, arg2)
    def add(self, arg0, arg1)
    def delete_key(self, arg0)
    def num_keys(self)
    def set_timeout(self, arg0)
    def wait(self, *args, **kwargs)
```

所有键值对存储store实现的基类，PyTorch分布式提供了三种实现子类，TCPStore、FileStore和HashStore。

```python
class TCPStore(Store):
    def __init__(self, host_name, port, world_size=-1, is_master=False, timeout=None, *args, **kwargs)
```

一种基于TCP的分布式键值对存储store的实现。服务器store保存数据，客户端store通过TCP连接到服务器，执行set()方法插入键值对、get()方法取得键值对等操作。该方式应始终要初始化一个服务器store，因为客户端store将等待与服务器store连接。

- host_name参数(str)，服务器store的主机名或IP地址。
- port参数(int)，服务器监听请求的端口。
- world_size参数(int, optional)，键值对存储的用户总数（使用该store的进程数），为客户端store数量+1，其中1表示服务器store。默认为-1，表示数量不固定。
- is_master参数(bool, optional)，初始化服务器store时为True，初始化客户端store时为False。默认为False。
- timeout参数(timedelta, optional)，store在初始化期间以及get()和wait()等方法的超时，默认值是300秒。
- wait_for_worker参数(bool, optional)，是否等待所有worker进程与服务器store连接，默认为True。这只适用于world_size为固定值的情况。

```python
# Run on process 1 (server)
server_store = dist.TCPStore('127.0.0.1', 1234, 2, True, timedelta(seconds=30))
# Run on process 2 (client)
client_store = dist.TCPStore('127.0.0.1', 1234, 2, False)
# Use any of the store methods from either the client or server after initialization
server_store.set('first_key', 'first_value')
client_store.get('first_key')
```

```python
class HashStore(Store):
    def __init__(self)
```

基于哈希映射的，线程安全的，键值对存储store的实现。此store可以在同一进程中使用，例如由其他线程使用，但不能跨进程使用。

```python
# store can be used from other threads
store = dist.HashStore()
# Use any of the store methods after initialization
store.set('first_key', 'first_value')
```

```python
class FileStore(Store):
    def __init__(self, file_name, world_size=-1)
```

基于文件的，键值对存储store的实现。

- file_name参数(str)，用于键值对store的文件路径。
- world_size参数(int, optional)，使用该store的进程数。默认为-1，表示数量不固定。

```python
store1 = dist.FileStore('/tmp/filestore', 2)
store2 = dist.FileStore('/tmp/filestore', 2)
# Use any of the store methods from either the client or server after initialization
store1.set('first_key', 'first_value')
store2.get('first_key')
```

```python
class PrefixStore(Store):
    def __init__(self, prefix, store)
```

对TCPStore、FileStore和HashStore三种键值对存储中的任何一个的包装器，它为插入到store中的每个键添加前缀prefix，参数store指定了其底层的键值对存储的实现。

## 6. Groups

默认进程组（名为world）上的默认集合，要求所有进程都进入（调用）分布式函数。然而，一些工作负载可以从更细粒度的通信中受益。这就是分布式组（distributed groups）发挥作用的地方。

```python
def new_group(ranks=None, timeout=default_pg_timeout, backend=None, pg_options=None)
```

创建一个新的分布式组，其中能包含所有进程的任意子集。它返回一个不透明的组句柄（opaque group handle），可以用作所有集合通信（collectives）的group参数。

此函数要求主进程组（main group）中的所有进程（即分布式作业的所有进程）进入（调用）此函数，即使它们不打算成为新进程组的成员。此外，在所有进程中应以相同的顺序创建组。

- ranks参数(list[int])，表示组成员rank进程的列表。若为None，则为所有rank进程。
- timeout参数(timedelta, optional)，对进程组执行操作的超时时间（timeout），默认值为30分钟。这适用于gloo后端。对于nccl，仅当环境变量NCCL_BLOCKING_WAIT或NCCL_ASYNC_ERROR_HANDLING设置为1时才适用。
- backend参数(str or Backend, optional)，新进程组要使用的后端。根据编译构建时的配置，有效值是gloo和nccl。默认情况下，使用与全局进程组相同的后端。
- pg_options参数(ProcessGroupOptions, optional)，进程组选项，指定构造进程组需要传递的附加选项。

> 并发使用多个具有NCCL后端的进程组是不安全的，用户应在程序中执行显式同步，以确保一次只使用一个进程组。这意味着，在另一个进程组的集合通信入队（enqueue）之前，上一个来自某进程组的集合通信应该已经在设备上完成执行，而不仅仅是入队，因为CUDA执行是异步的。有关详细信息，请参阅[Using multiple NCCL communicators concurrently](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html#using-multiple-nccl-communicators-concurrently)。

```python
def get_group_rank(group, global_rank) -> int
```

将某个进程在全局进程组中的编号global_rank，转换成它在进程组group中的编号group_rank。该进程必须是基础组group的一部分，否则会引发RuntimeError错误。在默认进程组上调用这个函数会返回其本身。

```python
def get_global_rank(group, group_rank) -> int
```

将某个进程在进程组group中的编号group_rank，转换成它在全局进程组中的编号global_rank。该进程必须是基础组group的一部分，否则会引发RuntimeError错误。在默认进程组上调用这个函数会返回其本身。

```python
def get_process_group_ranks(group)
```

返回进程组所有进程的全局rank编号列表，列表中的元素按照它在group组中的编号，进行排序。

## 7. Point-to-point communication

> 对于基于NCCL的进程组，在进行通信之前，必须将对象本身的张量表示移动到GPU设备。在这种情况下，使用的设备由torch.cuda.current_device()提供，且用户需要通过torch.cuda.set_device()设置该值，并使每个rank进程都拥有一个单独的GPU设备。
>
> 通常在每个进程开始处，在使用torch.distributed.init_process_group()初始化完后，调用torch.cuda.set_device()设置当前rank进程所使用的CUDA设备。

```python
def send(tensor, dst, group=None, tag=0) -> None
```

同步地发送一个tensor张量。

- tensor参数(torch.Tensor)，要发送的张量。
- dst参数(int)，目标接收进程的rank编号，不能为自己所在进程的编号。
- group参数(ProcessGroup, optional)，指定在哪个进程组group上进行。若为None，则使用默认的进程组。
- tag参数(int)，用以匹配send和recv的标识。

```python
def recv(tensor, src=None, group=None, tag=0) -> int
```

同步地接收一个tensor张量。

- tensor参数(torch.Tensor)，要接受的张量，使用接收到的数据填充该张量。
- src参数(int, optional)，发送源进程的rank编号。若为None，则可从进程组中的任何进程接收。

而非阻塞式的isend()和irecv()函数在使用时会返回分布请求对象（distributed request objects），它是torch.distributed.Work类型。一般来说，该对象的类型是未指定的，它不应该手动创建，但可以保证它支持is_completed()、wait()、get_future()方法。

```python
class Work(__pybind11_builtins.pybind11_object):
    def is_completed(self)
    def wait(self, timeout=None, *args, **kwargs)
    def get_future(self)
```

is_completed()函数，如果某个操作（operation）已经完成，则返回True。wait()函数，将阻塞进程，直到操作完成，此后再调用is_completed()会保证返回True。

```python
def isend(tensor, dst, group=None, tag=0) -> torch.distributed.Work
```

异步地发送一个tensor张量。返回一个分布式请求对象（distributed request objects），若进程不是group组的成员，则返回None。注意，在发送请求完成之前修改tensor会导致未定义的行为。

```python
def irecv(tensor, src, group=None, tag=0) -> torch.distributed.Work
```

异步地接收一个tensor张量，将接收到的数据填充到tensor张量。返回一个分布式请求对象（distributed request objects），若进程不是group组的成员，则返回None。

```python
def batch_isend_irecv(p2p_op_list)
```

异步地发送或接收一批张量，并返回一个请求的列表（list of requests）。调用p2p_op_list中的每个op操作，并返回相应的分布式请求对象的列表。目前支持NCCL、Gloo和UCC后端。

- p2p_op_list参数(list of torch.distributed.P2POp)，点对点操作的列表。列表中isend/irecv的顺序很重要，它需要与远端相应的isend/irecv匹配。

```python
# 2 ranks totally
send_tensor = torch.arange(2) + 2 * rank
recv_tensor = torch.randn(2)
send_op = dist.P2POp(dist.isend, send_tensor, (rank + 1) % world_size)
recv_op = dist.P2POp(dist.irecv, recv_tensor, (rank - 1 + world_size) % world_size)
reqs = batch_isend_irecv([send_op, recv_op])
for req in reqs:
    req.wait()
print(recv_tensor)
"""
tensor([2, 3])    # Rank 0
tensor([0, 1])    # Rank 1
"""
```

注意，当此API与NCCL PG后端一起使用时，用户必须使用torch.cuda.set_device()设置当前GPU设备，否则将导致意外的挂起问题。

此外，如果这个API是传递给torch.distributed.P2POp所涉及进程组的第一个集合调用，那么组中的所有成员都必须参与这个API调用，否则行为是未定义的。如果这个API调用不是组的第一个集合调用，则允许只涉及组中的一个子集的执行批量P2P操作。

```python
class P2POp:
    def __init__(self, op, tensor, peer, group=None, tag=0)
```

为batch_isend_irecv()构建点对点操作的类，用以构建P2P操作op、通信缓冲区tensor、peer进程的rank编号、进程组group和通信标识tag。该类的实例将被传递给batch_isend_irecv()以进行点对点通信（point-to-point communications）。

- op参数(Callable)，向对等进程发送或从对等进程接收数据的函数，其类型是torch.distributed.isend或torch.distributed.irecv。
- peer参数(int)，发送操作的接收进程，或接收操作的发送源进程。

## 8. Synchronous and asynchronous collective operations

每个集合操作函数都支持两种类型的操作（同步或异步），具体取决于传递给集合通信函数的async_op标志位。

同步操作（synchronous operation），async_op为False（默认模式）。当函数返回时，保证集合操作已被执行。而对CUDA集合操作来说，无法保证函数返回时CUDA集合操作已被执行，因为CUDA集合操作是异步的。对CPU上的集合操作，任何使用“集合操作的输出”的函数都能按照预期行为执行。对于CUDA集合操作，使用“同一CUDA流（CUDA stream）上的输出”的函数都能按预期行为执行。用户必须考虑不同CUDA流运行场景下的同步。关于CUDA语义的详细信息，如流同步（stream synchronization），参阅[CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html)。

异步操作（asynchronous operation），async_op为True。此时，集合操作函数返回异步进程句柄（async work handle），即分布式请求对象（distributed request objects）。若async_op为False，或当前进程不属于group组，则返回None。

一般来说，不要手动创建异步进程句柄，不过它会保证支持三个方法。

```python
class Work(__pybind11_builtins.pybind11_object):
    def is_completed(self)
    def wait(self, timeout=None, *args, **kwargs)
    def get_future(self)
```

is_completed()函数。对于CPU集合操作，如果执行完成则返回True。对于CUDA集合操作，如果操作已经成功入队（enqueue）到CUDA流中，并且输出可以无需进一步同步地在默认流上使用，则返回True。

wait()函数。对于CPU集合操作，将阻塞该进程，直到操作完成。对于CUDA集合操作，将阻塞直到操作成功入队到CUDA流中，并且输出可以无需进一步同步地在默认流上使用时。

get_future()函数，返回torch.futures.Future对象。支持NCCL，也支持GLOO和MPI上的大多数操作，除了点对点操作（如send和recv）。注意，PyTorch仍在继续维护修改Futures和并合并API，get_future()调用可能会变得多余。

下面的示例代码，展示了使用分布式集合通信时，CPU操作语义和CUDA操作语义的差异。在不同CUDA流上使用集合输出时需要同步。

```python
# Code runs on each rank.
torch.distributed.init_process_group('nccl', rank=rank, world_size=2)
output = torch.tensor([rank]).cuda(rank)
s = torch.cuda.Stream()
handle = torch.distributed.all_reduce(output, async_op=True)
# Wait ensures the operation is enqueued, but not necessarily complete.
handle.wait()
# Using result on non-default stream.
with torch.cuda.stream(s):
    s.wait_stream(torch.cuda.default_stream())
    output.add_(100)
# if the explicit call to wait_stream was omitted, the output below will be non-deterministically 
# 1 or 101, depending on whether the allreduce overwrote the value after the add completed.
if rank == 0:
    print(output)
"""
tensor([101])
"""
```

## 9. Collective functions

更详细的集合通信语义，见MPI文档。

> 对于基于NCCL的进程组，在进行通信之前，必须将对象本身的张量表示移动到GPU设备。在这种情况下，使用的设备由torch.cuda.current_device()提供，且用户需要通过torch.cuda.set_device()设置该值，并使每个rank进程都拥有一个单独的GPU设备。
>
> 通常在每个进程开始处，在使用torch.distributed.init_process_group()初始化完后，调用torch.cuda.set_device()设置当前rank进程所使用的CUDA设备。

### 9.1 broadcast

```python
def broadcast(tensor, src, group=None, async_op=False)
```

将tensor张量广播给整个group组。tensor在集合通信的所有参与进程中，必须具有相同的数据类型与形状。

- tensor参数(torch.Tensor)，若当前进程rank编号等于src，则为发送进程，将tensor张量广播到其他进程；若当前进程ranke编号不等于src，则为接收进程，将接收到的数据填充到tensor张量中。
- src参数(int)，表示发送进程的rank编号。
- group参数(ProcessGroup, optional)，执行集合通信的进程组，若为None则表示默认进程组。
- async_op参数(bool)，表示该集合通信操作是否异步执行。

若async_op为True，则返回异步进程句柄（async work handle），即分布式请求对象（distributed request objects）；若async_op为False，或当前进程不属于group组，则返回None。

```python
def broadcast_object_list(object_list, src=0, group=None, device=None)
```

将object_list列表中的对象广播给整个group组。

该方法与broadcast()类似，但可以传入Python对象，而且该函数不提供async_op参数，因此是一个阻塞式调用。应注意，所广播的对象都必须是picklable可打包的。

- object_list参数(List[Any])，要进行广播的对象列表，或接收广播数据的对象列表，对象都必须是picklable可打包的。
- device参数(torch.device, optional)，若不为None，则广播之前，先将广播对象序列化并转换成device设备上的tensor对象。

> broadcast_object_list()隐式地使用pickle模块，这是不安全的。有可能pickle打包恶意的数据，这些数据将在unpickle解包期间执行任意代码。故应该确保只有信任的数据才能调用这个函数。

```python
# Assumes world_size of 3.
if torch.distributed.get_rank() == 0:
    # any picklable object
    objects = ['foo', 12, {1: 2}]
else:
    objects = [None, None, None]
# Assumes backend is not NCCL
torch.distributed.broadcast_object_list(objects, src=0, device=torch.device('cpu'))
print(objects)
"""
['foo', 12, {1: 2}]    # rank 0, 1 and 2
"""
```

### 9.2 reduce

```python
class ReduceOp(__pybind11_builtins.pybind11_object)
```

一个枚举类，用于表示归约操作，包括：SUM、PRODUCT、MIN、MAX、BAND、BOR、BXOR。注意，PRODUCT、MAX、MIN不支持复数张量。

当使用NCCL后端时，BAND、BOR和BXOR归约不可用。此外，仅适用于NCCL后端2.10及以上版本的还有AVG归约，它先对输入除以world_size，再在各个rank进程上进行求和。此外，仅适用于NCCL后端2.11及以上版本的还有PREMUL_SUM归约，它将输入与给定标量本地相乘，再在各个rank进程上进行求和。用户应该使用torch.distributed._make_nccl_premul_sum()。

```python
def all_reduce(tensor, op=ReduceOp.SUM, group=None, async_op=False)
```

对组group所有进程上的tensor张量按op操作进行归约，结果存储到每个进程的tensor张量中，每个进程持有完全相同的结果。即调用之后，tensor张量在每个进程中按位相同（bitwise identical）。

```python
# 2 ranks totally
tensor = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank
print(tensor)
"""
tensor([1, 2])    # Rank 0
tensor([3, 4])    # Rank 1
"""
torch.distributed.all_reduce(tensor, op=ReduceOp.SUM)
print(tensor)
"""
tensor([4, 6])    # Rank 0 and 1
"""
```

```python
def reduce(tensor, dst, op=ReduceOp.SUM, group=None, async_op=False)
```

对组group所有进程上的tensor张量按op操作进行归约，结果存储到dst进程的tensor张量中，仅dst进程持有结果。

### 9.3 gather

```python
def all_gather(tensor_list, tensor, group=None, async_op=False)
```

将组group所有进程上的tensor张量进行聚集，结果存储到每个进程的tensor_list列表中，每个进程持有完全相同的结果。

注意，每个进程上的tensor张量可以是不同的形状。

```python
# 2 ranks totally
tensor_list = [torch.zeros([2,]) for _ in range(2)]
print(tensor_list)
"""
[tensor([0, 0]), tensor([0, 0])]    # Rank 0 and 1
"""
tensor = torch.arange(2) + 1 + 2 * rank
print(tensor)
"""
tensor([1, 2])     # Rank 0
tensor([3, 4])     # Rank 1
"""
torch.distributed.all_gather(tensor_list, tensor)
print(tensor_list)
"""
[tensor([1, 2]), tensor([3, 4])]    # Rank 0 and 1
"""
```

```python
def all_gather_into_tensor(output_tensor, input_tensor, group=None, async_op=False)
```

将组group所有进程上的tensor张量进行聚集，结果存储到每个进程的output_tensor张量中，每个进程持有完全相同的结果。

注意，每个进程上的tensor张量必须是相同的形状。聚集而成的output_tensor张量有两种构造方式，(1)由每个进程的tensor张量联接torch.concat()而成，(2)由每个进程的tensor张量堆叠torch.stack()而成，通过将output_tensor指定为相应形状来选择不同的构造方式。

注意，GLOO后端不支持此函数。

```python
# 2 ranks totally
device = torch.device(f'cuda:{rank}')
tensor = torch.arange(2, device=device) + 1 + 2 * rank
print(tensor)
"""
tensor([1, 2], device='cuda:0')    # Rank 0
tensor([3, 4], device='cuda:1')    # Rank 1
"""
# Output in concatenation form
out1 = torch.zeros([4,], device=device)
torch.distributed.all_gather_into_tensor(out1, tensor)
print(out1)
"""
tensor([1, 2, 3, 4], device='cuda:0')    # Rank 0
tensor([1, 2, 3, 4], device='cuda:1')    # Rank 1
"""
# Output in stack form
out2 = torch.zeros([2, 2], device=device)
torch.distributed.all_gather_into_tensor(out2, tensor)
print(out2)
"""
tensor([[1, 2], [3, 4]], device='cuda:0')    # Rank 0
tensor([[1, 2], [3, 4]], device='cuda:1')    # Rank 1
"""
```

```python
def all_gather_object(object_list, obj, group=None)
```

将组group所有进程上的obj对象进行聚集，结果存储到每个进程的object_list列表中，每个进程持有完全相同的结果。

该方法与all_gather()类似，但可以传入Python对象，而且该函数不提供async_op参数，因此是一个阻塞式调用。应注意，所聚集的对象都必须是picklable可打包的。

```python
# 3 ranks totally
gather_objects = ['foo', 12, {1: 2}]  # any picklable object
output = [None for _ in gather_objects]
torch.distributed.all_gather_object(output, gather_objects[dist.get_rank()])
print(output)
"""
['foo', 12, {1: 2}]    # rank 0, 1 and 2
"""
```

```python
def gather(tensor, tensor_gather_list=None, dst=0, group=None, async_op=False)
```

对组group所有进程上的tensor张量进行聚集，结果存储到dst进程的tensor_gather_list列表中，仅dst进程持有结果。

```python
def gather_object(obj, object_gather_list=None, dst=0, group=None)
```

对组group所有进程上的obj对象进行聚集，结果存储到dst进程的object_gather_list列表中，仅dst进程持有结果。

该方法与gather()类似，但可以传入Python对象，而且该函数不提供async_op参数，因此是一个阻塞式调用。应注意，所聚集的对象都必须是picklable可打包的。

```python
# 3 ranks totally
gather_objects = ['foo', 12, {1: 2}]    # any picklable object
output = [None for _ in gather_objects]
torch.distributed.gather_object(
    gather_objects[dist.get_rank()],
    output if dist.get_rank() == 0 else None,
    dst=0
)
print(output)
"""
['foo', 12, {1: 2}]    # rank 0
"""
```

### 9.4 scatter

```python
def scatter(tensor, scatter_list=None, src=0, group=None, async_op=False)
```

将scatter_list列表中的张量分配到group组所有进程上，数据源由src进程提供，组中每个进程收到一个tensor张量，结果存储到每个进程的tensor张量中，每个进程持有不同的结果。

```python
# 2 ranks totally
if dist.get_rank() == 0:
    scatter_list = [torch.ones([2,]) * 1, torch.ones([2,]) * 2]
else:
    scatter_list = None
output_tensor = torch.zeros([2,])
torch.distributed.scatter(output_tensor, scatter_list, src=0)
print(output_tensor)
"""
tensor([1, 1])    # rank 0
tensor([2, 2])    # rank 1
"""
```

```python
def scatter_object_list(scatter_object_output_list, scatter_object_input_list, src=0, group=None)
```

将scatter_object_input_list列表中的对象分配到group组所有进程上，数据源由src进程提供，组中每个进程收到一个object对象，结果存储到每个进程的scatter_object_output_list列表的第一个元素中，每个进程持有不同的结果。

该方法与scatter()类似，但可以传入Python对象，而且该函数不提供async_op参数，因此是一个阻塞式调用。应注意，所分配的对象都必须是picklable可打包的。

```python
# 3 ranks totally
if dist.get_rank() == 0:
    objects = ['foo', 12, {1: 2}]  # any picklable object
else:
    objects = [None, None, None]
output_list = [None]
torch.distributed.scatter_object_list(output_list, objects, src=0)
print(output_list)
"""
['foo']     # rank 0
[12]        # rank 1
[{1: 2}]    # rank 2
"""
```

```python
def reduce_scatter(output, input_list, op=ReduceOp.SUM, group=None, async_op=False)
```

在组group上，先对input_list列表进行逐元素归约，再将归约结果分配到每个进程的output上。

```python
def reduce_scatter_tensor(output, input, op=ReduceOp.SUM, group=None, async_op=False)
```

在组group上，先对input张量在相应的前缀维度，进行“逐子张量”归约，再将归约结果分配到每个进程的output上。

注意，每个进程上的output张量必须是相同的形状。而input张量有两种形状，(1)由每个进程的output张量联接torch.concat()而成，(2)由每个进程的output张量堆叠torch.stack()而成，通过将input指定为相应形状来选择不同的构造方式。

```python
# 3 ranks totally
device = torch.device(f'cuda:{rank}')
tensor_out = torch.zeros([2,], device=device)
# Input in concatenation form
tensor_in = torch.arange([4,], device=device)
print(tensor_in)
"""
tensor([0, 1, 2, 3], device='cuda:0')    # Rank 0
tensor([0, 1, 2, 3], device='cuda:1')    # Rank 1
"""
torch.distributed.reduce_scatter_tensor(tensor_out, tensor_in)
print(tensor_out)
"""
tensor([0, 2], device='cuda:0')    # Rank 0
tensor([4, 6], device='cuda:1')    # Rank 1
"""
# Input in stack form
tensor_in = tensor_in.reshape([2, 2])
print(tensor_in)
"""
tensor([[0, 1], [2, 3]], device='cuda:0')    # Rank 0
tensor([[0, 1], [2, 3]], device='cuda:1')    # Rank 1
"""
torch.distributed.reduce_scatter_tensor(tensor_out, tensor_in)
print(tensor_out)
"""
tensor([0, 2], device='cuda:0')    # Rank 0
tensor([4, 6], device='cuda:1')    # Rank 1
"""
```

注意，GLOO后端不支持此函数。

### 9.5 all-to-all

```python
def all_to_all(output_tensor_list, input_tensor_list, group=None, async_op=False)
```

在group组上，将所有进程input_tensor_list列表中的张量，分配到每个进程的output_tensor_list张量列表中。

对于编号为src的发送进程，它将自己的in_list[dst]元素（for dst in world_size），分配给rank编号为dst的进程，存储到接收进程的out_list[src]中。

对于编号为dst的接收进程，它自己的out_list[src]元素（for src in workd_size），接收rank编号为src进程的in_list[dst]张量数据。

```python
in_list = list((torch.arange(4) + rank * 4).chunk(4))
print(in_list)
"""
[tensor([0]), tensor([1]), tensor([2]), tensor([3])]        # Rank 0
[tensor([4]), tensor([5]), tensor([6]), tensor([7])]        # Rank 1
[tensor([8]), tensor([9]), tensor([10]), tensor([11])]      # Rank 2
[tensor([12]), tensor([13]), tensor([14]), tensor([15])]    # Rank 3
"""
out_list = list(torch.empty([4,]).chunk(4))
torch.distributed.all_to_all(out_list, in_list)
print(out_list)
"""
[tensor([0]), tensor([4]), tensor([8]), tensor([12])]     # Rank 0
[tensor([1]), tensor([5]), tensor([9]), tensor([13])]     # Rank 1
[tensor([2]), tensor([6]), tensor([10]), tensor([14])]    # Rank 2
[tensor([3]), tensor([7]), tensor([11]), tensor([15])]    # Rank 3
"""
```

### 9.6 barrier

```python
def barrier(group=GroupMember.WORLD, async_op=False, device_ids=None)
```

显式的栅障，用于同步所有进程。若async_op为False，或调用了进程句柄的wait()函数，则阻塞整个进程组group的所有进程，直到该进程组的所有进程都到达该栅障，即都进入（调用）该barrier()函数。

```python
def monitored_barrier(group=GroupMember.WORLD, timeout=None, wait_all_ranks=False)
```

与torch.distributed.barrier()类似，同步所有进程，但可以配置一个超时时间timeout参数，并且能够报告在该超时内没有通过该屏障的进程rank编号。

具体来说，该monitored_barrier()是通过判断send/recv是否按时完成来实现，且由rank为0的进程负责。rank编号非0的进程将阻塞，直到来自进程0的send/recv被处理。rank编号为0的进程将阻塞，直到所有来自其他进程的send/recv被处理，并将报告未能及时响应的失败进程的rank编号。注意，若一个rank进程未能达到monitored_barrier()栅障（例如由于挂起），那么所有其他rank进程都将在monitored_barrier()中失败。

该monitored_barrier()函数将阻塞组中的所有rank进程，直到整个进程组成功退出此函数，这对于调试和同步是有用的。但是它可能会影响性能，应只用于调试场景，或需要主机端完全同步点的场景。出于调试目的，可以在程序的集合通信调用之前插入该屏障，以检查是否有任何rank进程没有同步。

wait_all_ranks参数(bool, optional)，是否收集所有失败进程的rank编号。默认wait_all_ranks为False，此时进程0会在遇到第一个失败的进程rank时抛出异常。若将wait_all_ranks社为True，将收集所有失败进程的rank编号，并抛出一个包含所有失败rank信息的错误。

注意，仅有GLOO后端支持此函数。

```python
if torch.distributed.get_rank() != 1:
    # Raises exception indicating that rank 1 did not call into monitored_barrier.
    torch.distributed.monitored_barrier()
```

```python
# Example with wait_all_ranks=True
if torch.distributed.get_rank() == 0:
    # Raises exception indicating that ranks 1, 2, ... world_size - 1
    # did not call into monitored_barrier.
    torch.distributed.monitored_barrier(wait_all_ranks=True)
```

## 10. Profiling collective communication

可以使用torch.autograd.profiler或torch.profiler对上述的点对点通信API和集合通信API进行分析，推荐使用torch.profiler分析器。支持所有预定义的后端（gloo,nccl,mpi），在分析输出/跟踪（output/traces）时，可按预期呈现集合通信的使用情况。可以直接像分析任何标准torch算子那样，分析上述API，如下示例。

```python
with torch.profiler() as prof:
    tensor = torch.randn(20, 10)
    torch.distributed.all_reduce(tensor)
print(prof)
```

完整的分析器概述可参阅[profiler documentation](https://pytorch.org/docs/master/profiler.html)文档。

## 11. Multi-GPU collective functions

多GPU集合函数将被弃用。如果必须使用它们，请稍后重新访问PyTorch的文档。

如果每个节点上有多个GPU，当使用NCCL和GLOO后端时，broadcast_multigpu()、all_reduce_multigpu()、reduce_multigpu()、all_gather_multigpu()、reduce_scatter_multigpu()支持每个节点内多个GPU之间的分布式集合操作。这些函数可以潜在地提高整体分布式训练性能，并且可以通过传递张量列表来方便地使用。传递的张量列表中的每个张量都需要位于函数调用者主机的单独GPU设备上。注意，张量列表的长度需要在所有分布式进程中相同。还需注意，目前多GPU集合函数仅由NCCL后端支持。

```python
def broadcast_multigpu(tensor_list, src, group=None, async_op=False, src_tensor=0)
def all_reduce_multigpu(tensor_list, op=ReduceOp.SUM, group=None, async_op=False)
def reduce_multigpu(tensor_list, dst, op=ReduceOp.SUM, group=None, async_op=False, dst_tensor=0)
def all_gather_multigpu(output_tensor_lists, input_tensor_list, group=None, async_op=False)
def reduce_scatter_multigpu(output_tensor_list, input_tensor_lists, op=ReduceOp.SUM,
                            group=None, async_op=False)
```

例如，如果用于分布式训练的系统有2个节点，每个节点有8个GPU。在16个GPU中的每一个上，都有一个要allreduce的张量。可以参考下述代码示例。

```python
# 2 ranks totally
torch.distributed.init_process_group(backend='nccl', init_method='file:///distributed_test',
                                     world_size=2, rank=my_rank)
tensor_list = []
for dev_idx in range(torch.cuda.device_count()):
    tensor_list.append(torch.FloatTensor([1]).cuda(dev_idx))
torch.distributed.all_reduce_multigpu(tensor_list)
```

上述示例代码调用之后，2个节点上的所有16个张量的allreduced值都将为tensor([16.])。

## 12. Third-party backends

除内嵌的GLOO/MPI/NCCL后端外，PyTorch分布式包支持运行时的第三方后端注册机制。关于如何使用C++扩展开发第三方后端，可参考[Custom C++ and CUDA Extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html)教程和test/cpp_extensions/cpp_c10d_extension.cpp源码。第三发后端的功能由其实现决定。

新后端派生自c10d::ProcessGroup，并在导入时通过torch.distributed.Backend.register_backend()注册后端名称和实例化接口。当手动导入此后端并使用相应的后端名称调用torch.distributed.init_process_group()时，torch.distributed包将在新后端上运行。

## 13. Launch utility

分布式包的torch.distributed.launch提供了启动工具（launch utility），这个辅助工具可以用于在每个节点上启动多个进程进行分布式训练。torch.distributed.launch模块可用于CPU训练或GPU训练，在每个训练节点上，生成（spawn up）一个或多个分布式训练进程。

> 该模块将被弃用，使用Torch Distributed Elastic工具模块中的[torchrun](https://pytorch.org/docs/stable/elastic/run.html#launcher-api)替代。

该工具可用于单节点分布式训练（single-node distributed training），其中每个节点将生成一个或多个进程。若用于GPU训练，则每个进程将在单个GPU上操作，这可以很好地提高单节点训练性能。

该工具也可用于多节点分布式训练（multi-node distributed training），通过在每个节点上启动多个进程，也可以很好地提高多节点分布式训练的性能。这对于具有多个“能直接支持GPU的InfiniBand接口”的系统尤其有利，因为所有这些接口都可以用于聚合通信带宽。

在单节点分布式训练或多节点分布式训练的两种情况下，此工具将在每个节点上启动指定数量的进程（用命令行参数--nproc-per-node指定）。如果用于GPU训练，这个数字需要小于或等于一个节点上的GPU数量（nproc_per_node），且每个进程将在单个GPU上运行，其GPU编号从0到nproc_per_node-1。

### 13.1 How to use this module

以GPU集群系统为例。

单节点多进程分布式训练，假设一个节点有8张GPU卡，该节点的启动示例如下。

```shell
python -m torch.distributed.launch --nproc-per-node=8
                                   TRAIN_SCRIPT.py --script-args
```

多节点多进程分布式训练，假设有2个节点，一个节点有8张GPU卡，节点0为主节点（IP为192.168.1.1，端口为1234），两个节点的启动示例如下。

```shell
python -m torch.distributed.launch --nnodes=2 --master-addr="192.168.1.1" --master-port=1234
                                   --node-rank=0 --nproc-per-node=8
                                   TRAIN_SCRIPT.py --script-args
```

```shell
python -m torch.distributed.launch --nnodes=2 --master-addr="192.168.1.1" --master-port=1234
                                   --node-rank=1 --nproc-per-node=8
                                   TRAIN_SCRIPT.py --script-args
```

查看torch.distributed.launch模块提供的可选参数的方法如下。

```shell
python -m torch.distributed.launch --help
```

### 13.2 Important Notices

该工具和（单节点或多节点）多进程分布式GPU训练，目前仅能在NCCL分布式后端上获得最佳性能。因此NCCL后端是推荐用于GPU训练的后端。

在程序中，必须解析命令行参数--local-rank=LOCAL_PROCESS_RANK，它是由torch.distributed.launch模块提供的。如果训练程序使用GPU，应确保代码只能在LOCAL_PROCESS_RANK所指定的GPU设备上运行。

可以通过以下方式解析--local-rank命令行参数，并指定GPU设备编号。

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--local-rank", type=int)
args = parser.parse_args()

# Set device to local rank before code run
torch.cuda.set_device(args.local_rank)
# or
with torch.cuda.device(args.local_rank):
    # code to run here
    pass
```

在训练程序中，应首先调用torch.distributed.init_process_group()函数初始化分布式后端，对于init_method参数，强烈建议使用'env://'，其他的init_method（如'tcp://'）可能有效，但torch.distributed.launch正式支持'env://'方式。

```python
torch.distributed.init_process_group(backend='nccl', init_method='env://')
```

在训练程序中，可以使用常规的分布式通信函数，也可以使用torch.nn.parallel.DistributedDataParallel分布式数据并行。如果使用GPU进行训练，并且想结合DistributedDataParallel使用，下述示例展示了如何配置。

```python
model = torch.nn.parallel.DistributedDataParallel(
    model, device_ids=[args.local_rank], output_device=args.local_rank)
```

应确保device_ids参数设置为运行代码所在的唯一GPU设备ID，通常通过local_rank来设置。

另一种将local_rank参数传递给子进程（subprocess）的方法，是通过LOCAL_RANK环境变量。当指定命令行参数--use-env=True时，可以启用此行为，即python -m torch.distributed.launch --use-env TRAIN_SCRIPT.py。此时使用os.environ['LOCAL_RANK']环境变量即可作为local_rank，而无需再为启动器（launcher）指定--local-rank命令行参数。

注意，local_rank不是全局唯一的，它只对机器（节点）上的每个进程唯一。

## 14. Spawn utility

[Multiprocessing package - torch.multiprocessing](https://pytorch.org/docs/stable/multiprocessing.html#multiprocessing-doc)包在torch.multiprocessing.spawn()中提供了spawn（生成）函数。该辅助函数可用于产生多个进程。它的工作原理是，传入想要运行的函数，并生成N个进程来运行它。这也可以用于多进程分布式训练。

有关如何使用它的参考资料，可参阅[PyTorch example - ImageNet implementation](https://github.com/pytorch/examples/tree/master/imagenet)示例。

注意，此函数需要Python 3.4或更高版本。

## 15. Debugging torch.distributed applications

由于挂起、崩溃、跨rank行为不一致等情况难以分析，调试分布式应用程序具有挑战性。torch.distributed提供了一套工具，以帮助调试应用程序。

### 15.1 Monitored barrier

截至PyTorch v1.10，监控栅障torch.distributed.monitored_barrier()函数作为普通栅障torch.distributed.barrier()函数的替代方案。当程序崩溃（crash）时，监控栅障monitored_barrier()能提供哪个rank进程可能出错的有用信息，即不是所有rank进程都能在超时内调用torch.distributed.monitored_barrier()函数。

torch.distributed.monitored_barrier()在相应的进程中使用send/recv原语实现了一个主机侧（host-side）的barrier，类似于确认（acknowledgement）机制，它允许rank0进程报告哪个rank进程没有及时确认barrier。

一个例子如下，rank1进程没有调用monitored_barrier()，实际上，这可能是由于程序BUG或集合通信被挂起导致的。

```python
def worker(rank):
    torch.distributed.init_process_group('nccl', rank=rank, world_size=2)
    # monitored barrier requires gloo process group to perform host-side sync.
    group_gloo = torch.distributed.new_group(backend='gloo')
    if rank != 1:
        torch.distributed.monitored_barrier(group=group_gloo, timeout=datetime.timedelta(seconds=2))

if __name__ == '__main__':
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29501'
    mptorch.multiprocessing.spawn(worker, nprocs=2, args=())
```

然后rank0进程会产生如下的错误消息，可让用户确定哪个（哪些）rank进程可能是错误的，以用于进一步调查。

```
RuntimeError: Rank 1 failed to pass monitoredBarrier in 2000 ms
 Original exception:
[gloo/transport/tcp/pair.cc:598] Connection closed by peer [2401:db00:eef0:1100:3560:0:1c05:25d]:8594
```

### 15.2 TORCH_DISTRIBUTED_DEBUG

在环境变量TORCH_CPP_LOG_LEVEL=INFO的情况下，环境变量TORCH_DISTRIBUTED_DEBUG可用于日志记录和集合操作同步检查，以确保所有rank都正确同步。TORCH_DISTRIBUTED_DEBUG可设为OFF（默认值）、INFO或DETAIL，具体取决于所需的调试级别。请注意，DETAIL是最冗长详细（verbose）的选项，可能会影响应用程序的性能，因此应该只在调试问题时使用。

> 为了在运行时细粒度地控制调试级别，还可以使用torch.distributed模块中的set_debug_level()、set_debug_level_from_env()、get_debug_level()函数。

此外，可以同时使用TORCH_SHOW_CPP_STACKTRACES=1和TORCH_DISTRIBUTED_DEBUG=DETAIL环境变量，以便在检测到集合调用失去同步时记录整个调用堆栈。这些集合调用的未同步检查适用于所有使用c10d集合操作的程序，这些c10d集合操作，由init_process_group()和new_group()创建的进程组支持。

在使用torch.nn.parallel.DistributedDataParallel()训练模型进行初始化时，设置TORCH_DISTRIBUTED_DEBUG=INFO将导致额外的调试日志记录。而TORCH_DISTRIBUTED_DEBUG=DETAIL将根据选定的迭代次数，额外记录运行时性能统计数据。这些运行时统计数据包括forward时间、backward时间、梯度通信（gradient communication）时间等。

下面是一个应用示例。

```python
class TwoLinLayerNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(10, 10, bias=False)
        self.b = torch.nn.Linear(10, 1, bias=False)
    def forward(self, x):
        a = self.a(x)
        b = self.b(x)
        return a, b

def worker(rank):
    torch.distributed.init_process_group('nccl', rank=rank, world_size=2)
    torch.cuda.set_device(rank)
    print('init model')
    model = TwoLinLayerNet().cuda()
    print('init model_ddp')
    model_ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    inp = torch.randn(10, 10).cuda()
    print('start train')
    for _ in range(20):
        output = model_ddp(inp)
        loss = output[0] + output[1]
        loss.sum().backward()

if __name__ == '__main__':
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29501'
    os.environ['TORCH_CPP_LOG_LEVEL'] = 'INFO'
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'  # set to DETAIL for runtime logging.
    torch.multiprocessing.spawn(worker, nprocs=2, args=())
```

在初始化时生成的日志如下所示。

```
I0607 16:10:35.739390 515217 logger.cpp:173] [Rank 0]: DDP Initialized with:
broadcast_buffers: 1
bucket_cap_bytes: 26214400
find_unused_parameters: 0
gradient_as_bucket_view: 0
is_multi_device_module: 0
iteration: 0
num_parameter_tensors: 2
output_device: 0
rank: 0
total_parameter_size_bytes: 440
world_size: 2
backend_name: nccl
bucket_sizes: 440
cuda_visible_devices: N/A
device_ids: 0
dtypes: float
master_addr: localhost
master_port: 29501
module_name: TwoLinLayerNet
nccl_async_error_handling: N/A
nccl_blocking_wait: N/A
nccl_debug: WARN
nccl_ib_timeout: N/A
nccl_nthreads: N/A
nccl_socket_ifname: N/A
torch_distributed_debug: INFO
```

在运行时生成的日志如下所示。

```
I0607 16:18:58.085681 544067 logger.cpp:344] [Rank 1 / 2] Training TwoLinLayerNet
unused_parameter_size=0
Avg forward compute time: 40838608
Avg backward compute time: 5983335
Avg backward comm. time: 4326421
Avg backward comm/comp overlap time: 4207652
I0607 16:18:58.085693 544066 logger.cpp:344] [Rank 0 / 2] Training TwoLinLayerNet
unused_parameter_size=0
Avg forward compute time: 42850427
Avg backward compute time: 3885553
Avg backward comm. time: 2357981
Avg backward comm/comp overlap time: 2234674
```

此外，由于DDP模型中存在无用参数，因此在TORCH_DISTRIBUTED_DEBUG=INFO，记录DistributedDataParallel的日志时，会增加崩溃。目前，如果在forward传递中可能存在无用参数，则在初始化DDP时必须设置find_unused_parameters=True参数。并且从PyTorch v1.10开始，模型的所有输出都应对损失由贡献（即参与损失计算），因为DDP在backward传递中不支持无用参数。满足这些约束十分困难，特别是对于较大的模型而言，因此当出现错误时，DDP将记录所有无用参数的完全限定名称（fully qualified name）。

例如，在上面的程序示例中，如果将loss修改为loss=output[1]，则TwoLinLayerNet.a在backward传递中没有接收到梯度，因此导致DDP失败。在崩溃时，用户会收到关于无用参数的信息，这对于大型模型来说可能很难手动查找，如下所示。

```
RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one.
This error indicates that your module has parameters that were not used in producing loss.
You can enable unused parameter detection by passing the keyword argument `find_unused_parameters=True` to `torch.nn.parallel.DistributedDataParallel`, and by making sure all `forward` function outputs participate in calculating loss.
If you already have done the above, then the distributed data parallel module wasn't able to locate the output tensors in the return value of your module's `forward` function.
Please include the loss function and the structure of the return value of `forward` of your module when reporting this issue (e.g. list, dict, iterable).
Parameters which did not receive grad for rank 0: a.weight Parameter indices which did not receive grad for rank 0: 0
```

设置TORCH_DISTRIBUTED_DEBUG=DETAIL将对用户直接或间接发出的每个集合调用（如DDP的allreduce）触发额外的一致性和同步检查。这是通过创建一个包装进程组（wrapper process group）来完成的，它包装了由init_process_group()函数和new_group()函数创建的所有进程组。这些API将返回一个包装进程组，该进程组可以像普通进程组一样使用，但在将集合操作分派到底层进程组之前，会执行一致性检查。

目前，这些检查包括torch.distributed.monitored_barrier()，它确保所有rank进程完成它们的未完成集合调用，并报告卡住的进程rank编号。接下来检查集合调用本身的一致性，需要确保所有集合函数调用相匹配，并且张量的形状应一致。如果不一致，则在程序崩溃时包括详细的错误报告，而不是挂起或没有错误消息。

如下示例，考虑下面的函数，它在torch.distributed.all_reduce()中使用了不匹配的张量形状。

```python
def worker(rank):
    torch.distributed.init_process_group('nccl', rank=rank, world_size=2)
    torch.cuda.set_device(rank)
    tensor = torch.randn([10 if rank == 0 else 20, ]).cuda()
    torch.distributed.all_reduce(tensor)
    torch.cuda.synchronize(device=rank)

if __name__ == '__main__':
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29501'
    # os.environ['TORCH_CPP_LOG_LEVEL'] = 'INFO'
    # os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
    torch.multiprocessing.spawn(worker, nprocs=2, args=())
```

使用NCCL后端，上述程序可能会导致挂起，这可能很难找到根本原因。如果启用TORCH_DISTRIBUTED_DEBUG=DETAIL并重新运行应用程序，则会显示以下错误消息，显示根本原因。

```
work = default_pg.allreduce([tensor], opts)
RuntimeError: Error when verifying shape tensors for collective ALLREDUCE on rank 0. 
This likely indicates that input shapes into the collective are mismatched across ranks. Got shapes:  10
[ torch.LongTensor{1} ]
```

## 16. Logging

除了通过torch.distributed.monitored_barrier()和TORCH_DISTRIBUTED_DEBUG进行显式调试之外，torch.distributed的底层C++库也会输出不同级别的日志信息。这些消息有助于了解分布式训练作业的执行状态，排除（troubleshoot）网络连接故障等问题。

下面的表格显示了如何通过TORCH_CPP_LOG_LEVEL和TORCH_DISTRIBUTED_DEBUG环境变量的组合来调整日志级别。

| TORCH_CPP_LOG_LEVEL | TORCH_DISTRIBUTED_DEBUG | Effective Log Level |
| ------------------- | ----------------------- | ------------------- |
| ERROR               | -                       | Error               |
| WARNING             | -                       | Warning             |
| INFO                | -                       | Info                |
| INFO                | INFO                    | Debug               |
| INFO                | DETAIL                  | Trace (a.k.a. All)  |

torch.distributed有一个从RuntimeError派生的自定义异常类型，即torch.distributed.DistBackendError类。当特定于后端的错误发生时，此异常会被抛出。例如，当使用NCCL后端，用户试图使用NCCL库不可用的GPU时。

# Distributed Parallel Shard Strategy

在torch.distributed.fsdp模块中，类FullyShardedDataParallel是一个包装器，用于在数据并行的worker进程之上划分（shard）模型参数，通常简称为FSDP。该策略的实现是受*Automatic Cross-Replica Sharding of Weight Update in Data-Parallel Training*论文以及DeepSpeed的ZeRO Stage 3的启发。详细参考https://pytorch.org/docs/stable/fsdp.html文档。

```python
class FullyShardedDataParallel(torch.nn.Module, _FSDPState):
    def __init__(
        self, module: torch.nn.Module, process_group, sharding_strategy, cpu_offload, auto_wrap_policy,
        backward_prefetch, mixed_precision, ignored_modules, param_init_fn, device_id, sync_module_states,
        forward_prefetch, limit_all_gathers, use_orig_params, ignored_parameters,
    ) -> None
```

```python
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

torch.cuda.set_device(device_id)
sharded_module = FSDP(my_module)
optim = torch.optim.Adam(sharded_module.parameters(), lr=0.0001)
pred = sharded_module(x, y=3, z=torch.Tensor([1]))
loss = pred.sum()
loss.backward()
optim.step()
```

在torch.distributed.optim模块中，类DistributedOptimizer表示分布式优化器，它接受对分散在各个worker进程中的参数列表的远程引用（RRef），并在参数所在的worker进程本地为每个参数应用给定的优化器。分布式优化器可以使用任何本地优化器torch.optim.Optimizer来更新每个worker进程的梯度。详细参考https://pytorch.org/docs/stable/distributed.optim.html文档。

```python
class DistributedOptimizer:
    def __init__(self, optimizer_class, params_rref, *args, **kwargs) -> None
```

优化器ZeroRedundancyOptimizer可以对任意torch.optim.Optimizer优化器进行包装，并将优化器状态在组中的各个进程上划分（shard），划分是按照*ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*论文所描述的方式实现的。ZeroRedundancyOptimizer可以与torch.nn.parallel.DistributedDataParallel一起使用，以减少每个进程的峰值内存消耗。详细参考https://pytorch.org/docs/stable/distributed.optim.html文档。

```python
class ZeroRedundancyOptimizer(Optimizer, Joinable):
    def __init__(
        self, params, optimizer_class, process_group, parameters_as_bucket_view,
        overlap_with_ddp, **defaults: Any,
    ) -> None
```

在torch.distributed.tensor.parallel模块中，将提供张量并行（Tensor Parallelism）策略，包括逐行（row-wise）并行和逐列（col-wise）并行，该模块是基于PyTorch分布式张量（Distributed Tensor，DTensor）的基础之上的。该API正在开发中。详细参考https://pytorch.org/docs/stable/distributed.tensor.parallel.html文档与https://github.com/pytorch/pytorch/blob/main/torch/distributed/_tensor/README.md文档。

分布式检查点（Distributed Checkpoint，DCP）支持从多个并行进程中加载和保存模型。它在加载时会进行重划分（reshard），支持在一个集群拓扑中保存并加载到另一个集群拓扑中。详细参考https://pytorch.org/docs/stable/distributed.checkpoint.html文档。

# torch.distributed.elastic

本节内容若无特殊说明，工具torchrun在`torch.distributed`模块的命名空间中，而其他内容均在`torch.distributed.elastic`模块的命名空间中。

其中的torchrun模块可以使分布式PyTorch作业具有容错性和弹性，它是torch.distributed.launch启动工具的一个超集。

## 1. Quickstart

要启动容错（fault-tolerant）的作业，需在所有节点上执行程序，使用下述命令。

```shell
torchrun
    --nnodes=$NUM_NODES
    --nproc-per-node=$NUM_TRAINERS_PER_NODE
    --max-restarts=$NUM_ALLOWED_FAILURES
    --rdzv-id=$JOB_ID
    --rdzv-backend=c10d
    --rdzv-endpoint=$HOST_NODE_ADDR
    TRAIN_SCRIPT.py --script-args
```

要启动弹性（elastic）的作业，需在至少MIN_SIZE个节点且至多MAX_SIZE个节点上执行程序，使用下述命令。

```shell
torchrun
    --nnodes=$MIN_SIZE:$MAX_SIZE
    --nproc-per-node=$NUM_TRAINERS_PER_NODE
    --max-restarts=$NUM_ALLOWED_FAILURES_OR_MEMBERSHIP_CHANGES
    --rdzv-id=$JOB_ID
    --rdzv-backend=c10d
    --rdzv-endpoint=$HOST_NODE_ADDR
    TRAIN_SCRIPT.py --script-args
```

torchelastic模块会因其成员（参与节点）数量的变化而失败。当一个节点发生故障时，视为scale down事件，当故障节点被调度器（scheduler）替换时，视为scale up事件。因此，对于容错作业和弹性作业，--max-restarts命令行参数用于控制放弃继续执行前，可重启节点的总数量，而不管重启是由于故障还是scale事件引起的。

HOST_NODE_ADDR，指定c10d集合后端实例化和主机连接（hosted）所在的地址，形式为\<host\>\[:\<port\>\]，例如node1.example.com:29400。注意，若不指定端口号，则默认是29400。该主机地址host可以是训练集群中的任何节点，但理想情况下，应该选择一个具有高带宽的节点。

使用--standalone选项可启动一个单节点作业，其使用附接集合（sidecar rendezvous）后端。当使用--standalone时，无需传递--rdzv-id、--rdzv-backend和--rdzv-endpoint命令行参数。

若torchrun不能满足作业需求，可以直接使用[elastic agent](https://pytorch.org/docs/stable/elastic/agent.html)的API进行定制启动，详见之后的内容。

## 2. Train script

如果PyTorch的训练脚本train.py能用torch.distributed.launch启动，它也能用torchrun启动。只是在使用torchrun启动时，训练脚本有以下不同。

1. 无需手动传递RANK,WORLD_SIZE,MASTER_ADDR,MASTER_PORT参数。
1. 可以指定rdzv_backend和rdzv_endpoint参数，对大多数用户来说是使用c10d后端（详见[rendezvous](https://pytorch.org/docs/stable/elastic/rendezvous.html)）。默认情况下，rdzv_backend创建一个非弹性的集合（rendezvous），且rdzv_endpoint节点拥有主地址（master address）。
1. 确保训练脚本中有load_checkpoint(path)和save_checkpoint(path)逻辑。当任意worker进程失败时，则会对所有的worker进程进行重启，使用的程序参数与之前设置的相同，这会丢失所有未保存的检查点（checkpoint）。
1. use_env标志已不再使用。如果通过命令行的--local-rank参数解析本地进程rank编号，需要从环境变量LOCAL_RANK中获得该本地进程rank编号，例如int(os.environ['LOCAL_RANK'])。

如下是训练脚本的一个示例，它在每轮epoch训练时保存checkpoint，因此最坏的情况就是仅丢失一轮epoch完整的训练。

```python
def main():
    args = parse_args(sys.argv[1:])
    state = load_checkpoint(args.checkpoint_path)
    initialize(state)
    # torch.distributed.run ensures that this will work
    # by exporting all the environment variables needed to initialize the process group
    torch.distributed.init_process_group(backend=args.backend)
    
    for epoch in range(state.epoch, state.total_num_epochs):
        for batch in iter(state.dataset):
            train(batch, state.model)
        state.epoch += 1
        save_checkpoint(state)
```

对于具体的torchelastic训练脚本的例子，可以参看[elastic/examples README](https://github.com/pytorch/elastic/tree/master/examples)。

## 3. torchrun (Elastic Launch)

torchrun的功能是torch.distributed.launch的超集，它额外具有的功能如下：

- 对worker进程失败的情况，它能够优雅地重启所有worker进程。
- 可以自动指定WORLD_SIZE大小，和每个进程的RANK编号。
- 节点数量是可变的，即可在MIN_SIZE和MAX_SIZE之间改变（弹性）。

> torchrun是torch.distributed.run主模块的一个Python控制台脚本（[console script](https://packaging.python.org/en/latest/specifications/entry-points/#use-for-scripts)），其在PyTorch的setup.py中配置，在entry_points.txt中声明。它等价于`python -m torch.distributed.run`。

### 3.1 Transitioning from torch.distributed.launch to torchrun

torchrun支持与torch.distributed.launch相同的命令行参数，除了被弃用的--use-env命令行参数外。要将torch.distributed.launch启动迁移成torchrun启动，可按以下步骤进行。

若训练脚本已经从LOCAL_RANK环境变量中读取local_rank，则只需简单地移除--use-env命令行参数，如下所示。

```shell
python -m torch.distributed.launch --use-env TRAIN_SCRIPT.py
```

```shell
torchrun TRAIN_SCRIPT.py
```

若训练脚本从--local-rank命令行参数中读取local_rank，则需要将训练脚本改为从LOCAL_RANK环境变量中读取，如下所示。

```python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--local-rank', type=int)
args = parser.parse_args()
local_rank = args.local_rank
```

```python
import os
local_rank = int(os.environ['LOCAL_RANK'])
```

上面提到的修改足以将torch.distributed.launch迁移到torchrun。

#### 3.1.1 Uasge

单节点多进程（single-node multi-worker）。

```shell
torchrun
    --standalone
    --nnodes=1
    --nproc-per-node=$NUM_TRAINERS_PER_NODE
    TRAIN_SCRIPT.py --script-args
```

堆叠的单节点多进程（stacked single-node multi-worker）。

要在同一台主机上运行多个独立的“单节点多进程”实例（作业），需要确保每个实例作业设置在不同的端口上，以避免端口冲突（或者更糟糕的是，两个作业合并为一个作业）。为此，需要以--rdzv-backend=c10d命令行参数运行，并通过设置--rdzv-endpoint=localhost:\$PORT_k为每个作业指定不同的端口。对于--nodes=1的情况，可通过指定\$PORT=0，让torchrun自动选择一个空闲的随机端口，而不必手动分配不同的端口。

```shell
torchrun
    --nnodes=1
    --nproc-per-node=$NUM_TRAINERS_PER_NODE
    --rdzv-backend=c10d
    --rdzv-endpoint=localhost:0
    TRAIN_SCRIPT.py --script-args
```

容错作业（fault tolerant）。

```shell
torchrun
    --nnodes=$NUM_NODES
    --nproc-per-node=$NUM_TRAINERS_PER_NODE
    --max-restarts=$NUM_ALLOWED_FAILURES
    --rdzv-id=$JOB_ID
    --rdzv-backend=c10d
    --rdzv-endpoint=$HOST_NODE_ADDR
    TRAIN_SCRIPT.py --script-args
```

弹性作业（elastic）。

```shell
torchrun
    --nnodes=$MIN_SIZE:$MAX_SIZE
    --nproc-per-node=$NUM_TRAINERS_PER_NODE
    --max-restarts=$NUM_ALLOWED_FAILURES_OR_MEMBERSHIP_CHANGES
    --rdzv-id=$JOB_ID
    --rdzv-backend=c10d
    --rdzv-endpoint=$HOST_NODE_ADDR
    TRAIN_SCRIPT.py --script-args
```

当启用弹性作业时（min_size!=max_size），不要硬编码对WORLD_SIZE的假设，因为其大小可以随着节点的退出和加入而改变。

#### 3.1.2 Note on rendezvous backend

采用多节点训练，需要指定如下torchrun的命令行参数。

- --rdzv-id，唯一的作业编号（job id），由参与该作业的所有节点共享。
- --rdzv-backend，类torch.distributed.elastic.rendezvous.RendezvousHandler的实现，提供主要的集合接口。
- --rdzv-endpoint，集合后端运行所在的端点（endpoint），即端机，通常形式为\<host\>:\<port\>表示。

目前，PyTorch支持使用c10d（推荐）、etcd-v2、etcd集合后端，要使用etcd-v2或etcd，需要设置启用etcd服务器的v2 API，例如--enable-v2。

### 3.2 Definitions

这里列出PyTorch分布式所用的概念定义。

Node，是一个物理实例（physical instance）或容器（container），它映射到作业管理器（job manager）所运行的单元。

Worker，分布式训练环境中的worker，通常指训练进程。

WorkerGroup，执行相同函数的worker集合，即进程组，集合通信在进程组上进行，例如trainer的集合。

LocalWorkerGroup，进程组WorkerGroup中运行在同一个节点上的worker子集。

RANK，一个worker在进程组WorkerGroup中的编号。

WORLD_SIZE，进程组WorkerGroup中worker的总数。

LOCAL_RANK，当前worker在本地进程组LocalWorkerGroup中的编号。

LOCAL_WORLD_SIZE，本地进程组LocalWorkerGroup中worker的总数。

rdzv_id，用户定义的id，唯一标识一个作业的进程组WorkerGroup。每个节点使用该id加入特定WorkerGroup以成为其成员。

rdzv_backend，集合的后端，例如c10d。这通常是一个强一致性的Key-Value Store键值存储。

rdzv_endpoint，集合后端所允许的端点，通常形式为\<host\>:\<port\>。

一个Node运行LOCAL_WORLD_SIZE个worker，这些worker组成一个LocalWorkerGroup。一个作业中，所有Node上的LocalWorkerGroup组成总的WorkerGroup。

### 3.3 Environment Variables

可以在训练脚本中使用以下环境变量。

LOCAL_RANK，某个worker本地的rank编号。

RANK，某个worker全局的rank编号。

GROUP_RANK，进程组WorkerGroup的rank编号，取从0到max_nnodes之间的值。当一个节点上运行一个进程组时，该编号即是节点的rank编号。

ROLE_RANK，在所有扮演相同role角色的worker集合中，某个worker的rank编号。worker的角色在WorkerSpec中指定。

LOCAL_WORLD_SIZE，本地进程组中worker的总数，它等于torchrun所指定的--nproc-per-node命令行参数。

WORLD_SIZE，该job的所有参与worker的总数。

ROLE_WORLD_SIZE，在WorkerSpec中指定的以相同角色启动的所有worker的总数。

MASTER_ADDR，rank编号为0的worker进程所运行的主机的正式域名（Fully Qualified Domain Name，FQDN），用于初始化Torch Distributed后端。

MASTER_PORT，主机MASTER_ADDR上的一个端口，用于承载C10d TCP store存储的端口。

TORCHELASTIC_RESTART_COUNT，目前为止，进程组重新启动的次数。

TORCHELASTIC_MAX_RESTARTS，所配置的，重新启动的最大次数。

TORCHELASTIC_RUN_ID，等于集合rendezvous的run_id，例如唯一的作业id。

PYTHON_EXEC，系统可执行覆盖（system executable override），如果提供该参数，Python用户脚本会使用PYTHON_EXEC，默认使用sys.executable。

### 3.4 Deployment

#### 3.2.1 Steps

启动rendezvous后端服务器（C10d不需要），并获取端点endpoint的地址，即端机的标识符，将其传递给--rdzv-endpoint以启动脚本。

单节点多进程，启动主机上的启动器（launcher），以启动代理进程（agent process），用于创建和监控本地进程组。

多节点多进程，在参与训练的所有节点上，启动使用相同参数的启动器（launcher）。当使用作业/集群管理器（job/cluster manager）时，这个启动器应该是多节点作业的入口点命令（entry point command），如main()函数。

#### 3.2.2 Failure Modes

worker失败，对于有n个worker的训练任务，如果有k（小于等于n）个worker失败，则所有worker将停止并重新启动，直到达到max_restarts次数。

agent失败，一个agent失败将导致本地进程组失败。并由作业管理器决定是否让整个作业失败（gang semantics），还是尝试替换节点。这两种行为agent都支持。

node故障，与agent失败相同。

#### 3.2.3 Membership Changes

节点退出（scale-down），agent会收到退出（departure）通知，所有现有的worker都会停止，形成一个新的WorkerGroup组，所有的worker会以新的RANK和WORLD_SIZE开始运行。

节点加入（scale-up），新节点加入到作业中，所有现有的worker都会停止，形成一个新的WorkerGroup组，所有的worker会以新的RANK和WORLD_SIZE开始运行。

在失败或成员变更时，所有幸存的worker都会立即被杀死（kill），所以一定要及时保存checkpoint进度。在重新启动时，不保证RANK编号是稳定的，即一个节点上的本地worker可能分配到与之前不同的rank范围。不要硬编码任何关于RANK稳定性的假设，或者RANK和LOCAL_RANK之间相关性的假设。

### 3.5 Important Notices

该工具torchrun和多进程分布式（单节点或多节点）的GPU训练，目前只能在NCCL后端上实现最佳性能。因此推荐使用NCCL后端进行GPU训练。

此模块仅支持同构的LOCAL_WORLD_SIZE，也即，假设所有节点运行的本地worker的数量相同。

初始化Torch进程组的环境变量，由该模块提供，而无需手动传递RANK参数。要在训练脚本中初始化进程组，只需运行如下代码。

```python
import torch.distributed
torch.distributed.init_process_group(backend='nccl')
```

在训练程序中，可以使用常规的分布式函数（集合通信函数），也可以使用torch.nn.parallel.DistributedDataParallel模块。如果程序使用GPU进行训练，并且使用DistributedDataParallel模块，可按如下代码进行配置。

```python
local_rank = int(os.environ['LOCAL_RANK'])
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
```

确保device_ids被设置为，程序代码运行所在的唯一GPU设备id，这通常是进程的本地rank编号。也即，device_ids=[int(os.environ('LOCAL_RANK'))]，而output_device=int(os.environ('LOCAL_RANK'))。

建议训练脚本使用以下结构。

```python
def main():
  load_checkpoint(checkpoint_path)
  initialize()
  train()

def train():
  for batch in iter(dataset):
    train_step(batch)
    if should_checkpoint:
      save_checkpoint(checkpoint_path)
```

在worker的错误error上，此工具将总结错误的详细信息，例如time,rank,host,pid,traceback等。在每个节点上，第一个错误（按时间戳）被报告为“根本原因（Root Cause）”错误。为打印出错误摘要的一部分traceback信息，需在训练脚本中对主入口点函数（main entry point）进行注释，如下面的例子所示。

```python
from torch.distributed.elastic.multiprocessing.errors import record

@record
def main():
    pass  # do train

if __name__ == "__main__":
    main()
```

## 4. Elastic Agent

本节内容若无特殊说明，均在`torch.distributed.elastic.agent`模块的命名空间中。

### 4.1 Server

弹性代理（elastic agent）是torchelastic的控制平面（control plane），它是一个进程，用于启动和管理底层的worker进程。该代理负责：

1. 与torch分布式协同工作，准备好worker进程所有必要的信息，以便顺利地调用torch.distributed.init_process_group()初始化。
2. 容错（fault tolerance），监视worker进程，一旦检测到worker进程故障或不健康（unhealthiness），就杀死所有worker进程并重新启动所有进程。
3. 弹性（elasticity），对成员变更作出反应，并使用新成员重新启动worker。

> 控制平面（control plane），在计算机网络中，控制平面负责管理和控制数据平面的工作，例如路由协议、策略和配置。

最简单的代理部署在每个节点上，并与本地进程协同工作；更高级的代理可以远程启动和管理worker进程。代理agent可以是完全分散的（decentralized），根据其自己管理的worker做出决策；或者可以进行协调，与其他（管理同一作业worker的）代理进行通信，以做出集合决策（collective decision）。

下面是一个管理本地进程组的代理的示意图。

<img src="PyTorch Distributed Parallelism.assets/Agent diagram.jpg" style="zoom:50%;" />

### 4.2 Concepts

本节描述了高层的抽象类和概念，以更好地理解代理在torchelastic中的作用。

```python
# Abstract Base Classes (ABCs) according to PEP 3119.
class ElasticAgent(abc.ABC):
    @abc.abstractmethod
    def get_worker_group(self, role: str = DEFAULT_ROLE) -> WorkerGroup
    @abc.abstractmethod
    def run(self, role: str = DEFAULT_ROLE) -> RunResult
```

ElasticAgent代理进程负责管理一个或多个worker进程，worker进程被假定为常规的分布式PyTorch脚本。当代理创建worker进程时，代理为worker进程提供必要的信息，以正确地初始化一个torch进程组。

agent-to-worker的确切部署拓扑和比例，取决于agent的具体实现和用户作业的布局偏好（placement preference）。例如，要在具有8个训练器的GPU上运行分布式训练作业（每个GPU一个训练器），可以有如下划分：(1)使用8个“单GPU实例”，每个实例1个代理，每个代理1个worker进程；(2)使用4个“双GPU实例”，每个实例1个代理，每个代理2个worker进程；(3)使用2个“四GPU实例”，每个实例1个代理，每个代理4个worker进程；(4)使用1个“八GPU实例”，每个实例1个代理，每个代理8个worker进程。

代理agent的一个使用示例如下代码所示。

```python
group_result = agent.run()
if group_result.is_failed():
    # workers failed
    failure = group_result.failures[0]
    log.exception(f"worker 0 failed with exit code : {failure.exit_code}")
else:
    return group_result.return_values[0] # return rank 0's results
```

```python
@dataclass
class WorkerSpec:
    role: str
    local_world_size: int
    rdzv_handler: rdzv.RendezvousHandler
    entrypoint: Union[Callable, str, None] = None
    args: Tuple = ()
    max_restarts: int = 3
    monitor_interval: float = 30.0
    master_port: Optional[int] = None
    master_addr: Optional[str] = None
    redirects: Union[Std, Dict[int, Std]] = Std.NONE
    tee: Union[Std, Dict[int, Std]] = Std.NONE
    def __post_init__(self)
    def get_entrypoint_name(self)
```

WorkerSpec规格（specification）包含特定类型worker的构造（blueprint）信息。对于一个给定的角色role，必须只存在一个WorkerSpec规格。WorkerSpec规格在所有节点（机器）上应该是相同的，即每个节点所运行某个WorkerSpec规格的worker的数量，应该是相同的。

```python
class WorkerState(str, Enum):
    UNKNOWN = "UNKNOWN"      # agent lost track of worker group state, unrecoverable
    INIT = "INIT"            # worker group object created not yet started
    HEALTHY = "HEALTHY"      # workers running and healthy
    UNHEALTHY = "UNHEALTHY"  # workers running and unhealthy
    STOPPED = "STOPPED"      # workers stopped (interrupted) by the agent
    SUCCEEDED = "SUCCEEDED"  # workers finished running (exit == 0)
    FAILED = "FAILED"        # workers failed to successfully finish (exit != 0)
    @staticmethod
    def is_running(state: "WorkerState") -> bool
```

WorkerGroup的状态。worker进程组中所有worker的状态是一个整体（unit），是一起改变的。如果worker进程组中的一个worker失败了，那么整个worker组就被认为失败了。

worker组从初始的INIT状态开始，然后进展到HEALTHY或UNHEALTHY状态，最后到达终端SUCCEEDED或FAILED状态。

代理agent可以中断（interrupte）工作组，使其暂时进入STOPPED状态，并安排其在不久的将来重新启动。当代理检测到：(1)worker组失败/不健康，(2)或成员数量改变时，worker进程会被置为STOPPED状态。

当worker组上的操作（start,stop,rdzv,retry等）失败，并导致该操作只应用到worker组的一部分时，worker组的状态将是UNKNOWN。通常，这出现在“代理上发生状态更改事件（state change event）”的期间，由于未捕获/未处理异常所致。代理不并不会恢复UNKNOWN状态的worker组，最好自行终止并允许作业管理器重试该节点。

```python
class Worker:
    def __init__(self, local_rank, global_rank=-1, role_rank=-1, world_size=-1, role_world_size=-1,)
```

一个worker实例。Worker是根据某个WorkerSpec规格创建的，Worker之于WorkerSpec就像对象之于类。

每个worker都有一个唯一的id号，由ElasticAgent的特定实现负责解释。对于本地代理（local agent），它可以是worker的pid（int）；对于远程代理（remote agent），它可以编码为\<host\>:\<port\>（string）。

```python
class WorkerGroup:
    def __init__(self, spec: WorkerSpec)
```

表示由ElasticAgent管理的给定WorkerSpec的Worker实例集。worker组是否包含跨实例的worker，取决于代理的实现。

### 4.3 Implementations

```python
class SimpleElasticAgent(ElasticAgent):
    def __init__(self, spec: WorkerSpec, exit_barrier_timeout: float = 300)
    def get_worker_group(self, role: str = DEFAULT_ROLE) -> WorkerGroupp

    @abc.abstractmethod
    def _start_workers(self, worker_group: WorkerGroup) -> Dict[int, Any]
    @abc.abstractmethod
    def _stop_workers(self, worker_group: WorkerGroup) -> None
    @abc.abstractmethod
    def _monitor_workers(self, worker_group: WorkerGroup) -> RunResult
    @abc.abstractmethod
    def _shutdown(self, death_sig: signal.Signals = signal.SIGTERM) -> None
    @staticmethod
    def _set_master_addr_port(store: Store, master_addr: Optional[str], master_port: Optional[int])
    @staticmethod
    def _get_master_addr_port(store: Store) -> Tuple[str, int]

    def _assign_worker_ranks(self, store, group_rank: int, group_world_size: int, spec: WorkerSpec) -> List[Worker]
    def _initialize_workers(self, worker_group: WorkerGroup) -> None
    def _restart_workers(self, worker_group: WorkerGroup) -> None
    def run(self, role: str = DEFAULT_ROLE) -> RunResult
```

SimpleElasticAgent提供了实现的框架，实现了一些具体方法，并留声明了一些抽象方法以供特定实现。

```python
class LocalElasticAgent(SimpleElasticAgent):
    def __init__(self, spec: WorkerSpec, start_method="spawn", exit_barrier_timeout=300, log_dir=None)
    def _stop_workers(self, worker_group: WorkerGroup) -> None
    def _start_workers(self, worker_group: WorkerGroup) -> Dict[int, Any]
    def _shutdown(self, death_sig: signal.Signals = signal.SIGTERM) -> None
    def _monitor_workers(self, worker_group: WorkerGroup) -> RunResult
```

torch.distributed.elastic.agent.server.local_elastic_agent.LocalElasticAgent是由torchelastic模块提供的代理实现，用于处理主机host本地的worker进程。该代理部署在每个主机上，并被配置为生成N个worker进程。使用GPU时，N表示主机可用的GPU数量。

本地代理LocalElasticAgent不与部署在其他主机上的其他本地代理通信，即使worker进程可以与其他主机通信。

## 5. Multiprocessing

本节内容若无特殊说明，均在`torch.distributed.elastic.multiprocessing`模块的命名空间中。模块torch.distributed.elastic.multiprocessing是模块torch.multiprocessing的分布式版本，而torch.multiprocessing是Python模块multiprocessing的包装。

一个库，能够启动并管理N个工作子进程（worker subprocess）副本的库，子进程要执行的操作可由函数或二进制可执行文件指定。对于函数，它使用torch.multiprocessing（也即Python的multiprocessing模块）来spawn/fork工作进程；对于二进制文件，它使用Python的subprocessing.Popen模块来创建工作进程。该库由torchelastic代理使用。

要启动子进程执行函数，如下示例代码。

```python
from torch.distributed.elastic.multiprocessing import Std, start_processes

def trainer(a, b, c):
    pass  # train

# runs two trainers
# LOCAL_RANK=0 trainer(1,2,3)
# LOCAL_RANK=1 trainer(4,5,6)
ctx = start_processes(
    name='trainer',
    entrypoint=trainer,
    args={0: (1, 2, 3),         1: (4, 5, 6),         },
    envs={0: {'LOCAL_RANK': 0}, 1: {'LOCAL_RANK': 1}, },
    log_dir='/tmp/foobar',
    redirects=Std.ALL,  # write all worker stdout/stderr to a log file
    tee={0: Std.ERR},   # tee only local rank 0's stderr to console
)

# waits for all copies of trainer to finish
ctx.wait()
```

要启动子进程执行二进制文件，如下示例代码。

```python
from torch.distributed.elastic.multiprocessing import Std, start_processes

# same as invoking
# echo hello
# echo world > stdout.log
ctx = start_processes(
    name='echo',
    entrypoint='echo',
    log_dir='/tmp/foobar',
    args={0: 'hello', 1: 'world'},
    redirects={1: Std.OUT},
)

# waits for all copies of trainer to finish
ctx.wait()
```

与torch.multiprocessing一样，函数start_processes()的返回值是一个进程上下文torch.distributed.elastic.multiprocessing.api.PContext对象。如果从函数启动，返回的是api.SubprocessContext对象；如果从二进制文件启动，返回的是api.SubprocessContext对象；它们都是父类api.PContext的特定实现。

### 5.1 Starting Multiple Workers

函数torch.distributed.elastic.multiprocessing.start_processes的语法定义如下所示。

```python
def start_processes(
    name:          str,
    entrypoint:    Union[Callable, str],
    args:          Dict[int, Tuple],
    envs:          Dict[int, Dict[str, str]],
    log_dir:       str,
    start_method:  str = "spawn",
    redirects:     Union[Std, Dict[int, Std]] = Std.NONE,
    tee:           Union[Std, Dict[int, Std]] = Std.NONE,
) -> PContext
```

由所提供的一些options选项，启动n个由entrypotion参数指定进程副本，其中entrypoint可以是一个函数，或者是由字符串str指定的二进制可执行文件。进程副本的数量由args和envs参数的项数决定，它们需要相同的键集（key set）。

其中，args和envs是传递给entrypoint的参数和环境变量，由字典的键指定其相应的值映射到local rank进程，键集应该是{0,1,...,nprocs-1}。当entrypoint是二进制文件时，args的值只能是字符串，如果使用其他类型，则会将其强制转换为字符串表示。

```python
# caution. arguments casted to string, runs: echo "1" "2" "3" and echo "[1, 2, 3]"
start_processes(
   name = "trainer",
   entrypoint = "/usr/bin/echo",
   args = {0:(1,2,3), 1:([1,2,3],)},
   envs = {0:{}, 1:{}},
   log_dir=/temp/test/out
 )
```

> 如果main函数使用torch.distributed.elastic.multiprocessing.errors.record修饰，二进制文件执行失败时只会写一个error.json错误文件。当entrypoint是函数启动时，默认写错误文件，不需要手动使用@record注释。

redirects和tee参数是位掩码（bitmasks），指定要将哪些std标准流（stream）重定向到log_dir目录下的日志文件，有效的掩码值是在`Std`中定义的。要仅redirects/tee特定的local rank进程，将redirects作为一个map传递，它的键作为local rank来指定重定向行为。任何缺失（默认）的local rank进程默认使用Std.NONE。

> tee参数的作用类似于Unix的tee命令，它会重定向/打印到控制台，若要避免将worker的stdout/stderr打印到控制台，请使用redirects参数。
>
> Linux中的tee命令从标准输入读取数据后，将数据重定向到给定的文件和标准输出。

对于每个进程来说，目录log_dir下将包括：

- {local_rank}/error.json，如果进程失败，向该文件写入错误信息。
- {local_rank}/stdout.json，当redirect & STDOUT == STDOUT为真时。
- {local_rank}/stderr.json，当redirect & STDERR == STDERR为真时。

注意，log_dir应该是一个空的目录。

### 5.2 PContext

类torch.distributed.elastic.multiprocessing.api.PContext的语法定义如下所示。

```python
class PContext(abc.ABC):
    def __init__(self, name, entrypoint, args, envs, stdouts, stderrs, tee_stdouts, tee_stderrs, error_files, )
    def start(self) -> None
    def wait(self, timeout: float = -1, period: float = 1) -> Optional[RunProcsResult]
```

基类，是对不同机制启动的一组进程上的操作的标准化。其名为PContext是为了消除与torch.multiprocessing.ProcessContext之间的歧义。

注意，stdout和stderrs应该总是tee_stdout和tee_stderrs的超集。

```python
class SubprocessContext(PContext)
class MultiprocessContext(PContext)
```

函数start_processes()如果从函数启动，返回的是api.SubprocessContext对象；如果从二进制文件启动，返回的是api.SubprocessContext对象。

```python
class RunProcsResult:
    return_values:  Dict[int, Any] = field(default_factory=dict)
    failures:       Dict[int, ProcessFailure] = field(default_factory=dict)
    stdouts:        Dict[int, str] = field(default_factory=dict)
    stderrs:        Dict[int, str] = field(default_factory=dict)
    def is_failed(self) -> bool
```

类RunProcsResult是由start_processes()启动的进程完成运行的结果，它由PContext返回。

注意，RunProcsResult所有的字段都是由local rank指定的映射（map）。

## 6. Else

除上述的torch.distributed.elastic.agent和torch.distributed.elastic.multiprocessing之外，torch.distributed.elastic库中还包含一些其它内容，例如Error Propagation、Rendezvous、Expiration Timers、Metrics、Events等。

分布式PyTorch作业中的每个主机都运行一个torchelastic代理（agent）和多个worker进程（作为torchelastic代理的子进程）。由于worker进程是用户提供的PyTorch脚本，因此torchelastic可以通过agent代理在训练器（trainer）上传播错误，并向上传递到调度器，调度器最终通知用户有关作业的状态并应用任何重试策略。相关功能位于torch.distributed.elastic.multiprocessing.errors模块中。

在Torch Distributed Elastic的上下文中，使用术语集合（rendezvous）来指代将分布式同步原语（distributed synchronization primitive）与节点发现（peer discovery）相结合的特定功能。Torch Distributed Elastic使用rendezvous来收集训练作业的参与者（即节点），以便这些节点知道所有参与者和每个节点的定位，并能针对训练的开始/恢复做出一致的集体决定。相关功能位于torch.distributed.elastic.rendezvous模块中。

可以在与代理相同的进程上设置过期计时器（expiration timer），并在脚本中使用过期计时器以处理被卡住的工作进程。当用户执行有可能卡住的代码块时，可以获取一个过期计时器，它指示计时器服务器如果在设定的过期截止日期之前没有释放计时器，就终止进程。相关功能位于torch.distributed.elastic.timer模块中。

指标接口（metrics API）用于发布遥测指标（telemetry metrics），它被设计为由torchelastic的内部模块使用，为最终用户发布指标，以提高可见性和帮助调试。但是，用户仍可以在作业中使用相同的API将指标发布到相同的指标接收器。相关功能位于torch.distributed.elastic.metrics模块中。

此外，模块torch.distributed.elastic.events包含与标准Python日志集成的事件处理机制。

详细参考https://pytorch.org/docs/stable/distributed.elastic.html文档。

# Distributed RPC Framework

[RPC-Based Distributed Training](https://pytorch.org/docs/stable/rpc.html)（Remote Procedure Call，RPC）支持不适合用数据并行的通用训练架构，如分布式流水线并行（pipeline parallelism）、参数服务器模式（parameter server paradigm），以及组合使用DDP与其他并行训练模式。RPC会协助管理远程对象的生命周期，并在多机设备上扩展autograd引擎。

分布式RPC框架通过一组允许远程通信的原语提供了多机模型训练的机制，并提供高级API来自动划分跨多台机器的模型。
