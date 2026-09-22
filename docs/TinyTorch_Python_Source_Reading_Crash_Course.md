# TinyTorch 源码阅读专用 Python 速成教程

> 目标读者：有编程基础（尤其是 Java），但几乎没有 Python 基础，希望快速看懂 TinyTorch 源码。
>
> 学习目标：不是“系统学会 Python”，而是把 Python 语法障碍降到最低，把注意力放到 Tensor、Layer、DataLoader、Autograd、Optimizer 和 Training 的原理上。

---

# 0. 先建立正确目标

学习 TinyTorch 时，你真正要掌握的是两层东西：

```text
第一层：Python 只是代码表达工具
第二层：TinyTorch 才是你真正要学习的对象
```

例如：

```python
for layer in self.layers:
    x = layer(x)
```

如果卡在 Python，你会问：

- `for` 是什么？
- `self` 是什么？
- `layer(x)` 为什么对象还能加括号？

真正应该看到的是：

```text
输入 x
  ↓
第 1 层
  ↓
新的 x
  ↓
第 2 层
  ↓
新的 x
  ↓
...
```

所以本教程只讲 **TinyTorch 源码高频 Python**。

---

# 1. Python 与 Java 的最小差异

## 1.1 变量不需要声明类型

Java：

```java
int batchSize = 32;
boolean shuffle = true;
String name = "TinyTorch";
```

Python：

```python
batch_size = 32
shuffle = True
name = "TinyTorch"
```

你可以暂时理解为：

```text
变量名 = 对象
```

例如：

```python
x = Tensor([1, 2, 3])
```

就是：

```text
创建 Tensor 对象
    ↓
让变量 x 指向它
```

---

## 1.2 Python 用缩进代替 `{}`

Java：

```java
if (training) {
    doSomething();
}
```

Python：

```python
if training:
    do_something()
```

所以 Python 源码阅读时，**缩进就是代码结构**。

```python
if training:
    x = dropout(x)
    x = relu(x)

return x
```

表示：

```text
if training
├── dropout
└── relu

return x
```

`return x` 已经不属于 `if`。

---

## 1.3 `None` 基本等价于 Java `null`

```python
self.bias = None
```

理解为：

```java
this.bias = null;
```

常见判断：

```python
if self.bias is not None:
```

翻译：

```text
如果 bias 存在
```

---

# 2. TinyTorch 必会的数据结构

# 2.1 List：列表

```python
params = [weight, bias]
```

理解成 Java：

```java
List<Tensor> params = List.of(weight, bias);
```

访问：

```python
params[0]
params[1]
```

追加：

```python
params.append(x)
```

TinyTorch 高频模式：

```python
def parameters(self):
    params = [self.weight]

    if self.bias is not None:
        params.append(self.bias)

    return params
```

逐句翻译：

```text
创建参数列表
↓
先放 weight
↓
如果 bias 存在
    再放 bias
↓
返回所有参数
```

---

# 2.2 Tuple：元组

```python
shape = (32, 784)
```

非常常见，因为 Tensor shape 通常就是 tuple。

```python
x.shape
```

可能返回：

```python
(32, 784)
```

理解：

```text
32 个样本
每个样本 784 个特征
```

注意：

```python
(784,)
```

表示 **一维 tuple**。

---

# 2.3 Dict：字典

Python：

```python
state = {
    "weight": weight,
    "bias": bias
}
```

Java 可以类比：

```java
Map<String, Tensor>
```

访问：

```python
state["weight"]
```

以后保存模型、状态、配置时会经常遇到。

核心理解：

```text
key → value
```

---

# 3. 索引与切片：DataLoader 的基础

## 3.1 索引从 0 开始

```python
x[0]
```

第一项。

```python
x[1]
```

第二项。

---

## 3.2 切片

```python
x[2:5]
```

取：

```text
2
3
4
```

**左闭右开**，不包含 5。

---

## 3.3 DataLoader 为什么大量使用切片

```python
batch_indices = indices[i:i + batch_size]
```

假设：

```python
i = 32
batch_size = 32
```

那么：

```python
indices[32:64]
```

得到第 32~63 个索引。

这就是：

```text
一个 batch 的索引
```

---

# 4. if / for / range：读懂控制流程

# 4.1 if

常见形式：

```python
if x is None:
```

没有 x。

```python
if x is not None:
```

x 存在。

```python
if training:
```

训练模式。

```python
if not training:
```

非训练模式。

Dropout 常见逻辑：

```python
if not training:
    return x
```

翻译：

```text
推理阶段
↓
Dropout 不做任何事
↓
直接返回输入
```

---

# 4.2 for

```python
for layer in self.layers:
    x = layer(x)
```

不要逐字翻译。

整体理解：

```text
遍历每一层
↓
把当前 x 输入这一层
↓
输出重新保存到 x
↓
继续下一层
```

如果：

```text
self.layers = [
    Linear,
    ReLU,
    Linear
]
```

循环就是：

```text
x
↓
Linear
↓
x
↓
ReLU
↓
x
↓
Linear
↓
output
```

---

# 4.3 range

```python
range(5)
```

产生：

```text
0 1 2 3 4
```

```python
range(0, 100, 32)
```

产生：

```text
0 32 64 96
```

因此：

```python
for i in range(0, len(indices), batch_size):
```

意思：

```text
按照 batch_size 为步长处理数据
```

---

# 5. 函数：看懂输入、处理、输出

Python：

```python
def add(a, b):
    return a + b
```

源码阅读时不要纠结语法。

每个函数固定问四个问题：

```text
1. 函数叫什么？
2. 输入是什么？
3. 中间处理什么？
4. 返回什么？
```

例如：

```python
def forward(self, x):
    output = x.matmul(self.weight)

    if self.bias is not None:
        output = output + self.bias

    return output
```

阅读：

```text
名字：forward

输入：
    x

处理：
    x × weight
    如果有 bias，再加 bias

输出：
    output
```

---

# 6. 参数默认值

```python
def __init__(self, batch_size=32, shuffle=False):
```

表示：

```text
batch_size 默认 32
shuffle 默认 False
```

因此：

```python
DataLoader(dataset)
```

等价于：

```python
DataLoader(
    dataset,
    batch_size=32,
    shuffle=False
)
```

---

# 7. 解构：AI Python 高频语法

```python
x, y = batch
```

如果：

```python
batch = (features, labels)
```

那么相当于：

```python
x = batch[0]
y = batch[1]
```

再例如：

```python
batch_size, features = x.shape
```

如果：

```python
x.shape == (32, 784)
```

那么：

```text
batch_size = 32
features   = 784
```

---

# 8. class：TinyTorch 的真正核心

TinyTorch 大量概念都是 class：

```text
Tensor
Layer
Linear
Dropout
Sequential
Dataset
DataLoader
Optimizer
SGD
Adam
```

例如：

```python
class Linear:
    ...
```

意思不是“定义一个函数”。

而是：

```text
定义一种新类型：Linear
```

一个 Linear 对象里面既可以保存：

```text
weight
bias
```

也可以拥有行为：

```text
forward()
parameters()
```

因此：

```text
Linear
├── 数据
│   ├── weight
│   └── bias
│
└── 方法
    ├── forward()
    └── parameters()
```

---

# 9. self：直接当作 Java this

这是有 Java 基础的人最容易掌握的地方。

Python：

```python
class Linear:

    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
```

可以近似理解成 Java：

```java
class Linear {

    int inFeatures;
    int outFeatures;

    Linear(int inFeatures, int outFeatures) {
        this.inFeatures = inFeatures;
        this.outFeatures = outFeatures;
    }
}
```

所以以后看到：

```python
self.weight
```

脑内直接换成：

```java
this.weight
```

就行。

---

# 10. `__init__`：构造函数

```python
class Linear:

    def __init__(self, in_features, out_features):
        ...
```

调用：

```python
layer = Linear(10, 5)
```

Python 创建对象以后会调用：

```text
__init__
```

因此：

```python
__init__
```

脑内翻译成：

```text
构造函数
```

---

# 11. TinyTorch 最重要的一组：`__xxx__`

Python 允许 class 定义一些特殊方法。

TinyTorch 源码阅读只需要重点掌握下面这些。

| 特殊方法 | 你应该理解成 |
|---|---|
| `__init__` | 构造函数 |
| `__call__` | `obj(...)` |
| `__len__` | `len(obj)` |
| `__getitem__` | `obj[i]` |
| `__iter__` | `for x in obj` |
| `__add__` | `a + b` |
| `__sub__` | `a - b` |
| `__mul__` | `a * b` |
| `__truediv__` | `a / b` |
| `__matmul__` | `a @ b` |
| `__repr__` | 对象打印显示 |

---

# 12. 运算符重载：Tensor 为什么能 `a + b`

假设 Tensor：

```python
class Tensor:

    def __add__(self, other):
        ...
```

那么：

```python
a + b
```

Python 可以理解为：

```python
a.__add__(b)
```

所以：

```text
a + b
```

并不是 Tensor 天生就支持。

而是：

```text
Tensor 作者实现了 __add__
```

同理：

```python
a * b
```

↓

```python
a.__mul__(b)
```

---

# 13. `__call__`：为什么 Layer 可以写成 `layer(x)`

假设：

```python
class Layer:

    def __call__(self, x):
        return self.forward(x)
```

于是：

```python
layer(x)
```

实际会触发：

```python
layer.__call__(x)
```

而 `__call__` 又调用：

```python
layer.forward(x)
```

因此你可以建立这个固定模型：

```text
layer(x)
↓
__call__(x)
↓
forward(x)
↓
真正计算
```

这也是以后理解：

```python
model(x)
```

的关键。

---

# 14. 继承：Layer / Optimizer 体系核心

Python：

```python
class Linear(Layer):
    ...
```

直接类比 Java：

```java
class Linear extends Layer
```

关系：

```text
        Layer
       /     \
   Linear   Dropout
```

意思：

```text
Linear 是一种 Layer
Dropout 也是一种 Layer
```

同样：

```text
        Optimizer
        /      \
      SGD      Adam
```

---

# 15. `super()`：调用父类

你以后会看到：

```python
class Linear(Layer):

    def __init__(self, ...):
        super().__init__()
```

意思：

```text
先执行父类 Layer 的初始化
```

Java 类似：

```java
super();
```

源码看到 `super()` 不要慌。

---

# 16. `*args` 与 `**kwargs`

看到：

```python
def __call__(self, x, *args, **kwargs):
```

初学阶段只要记：

```text
*args
=
额外的位置参数

**kwargs
=
额外的“参数名=值”参数
```

例如：

```python
layer(x, True)
```

True 可以进入 `args`。

```python
layer(x, training=True)
```

`training=True` 可以进入 `kwargs`。

常见转发：

```python
return self.forward(x, *args, **kwargs)
```

翻译：

```text
把收到的额外参数继续交给 forward
```

---

# 17. `*tensors`：任意多个 Tensor

例如：

```python
def __init__(self, *tensors):
    self.tensors = tensors
```

调用：

```python
TensorDataset(x, y)
```

内部：

```python
tensors == (x, y)
```

调用：

```python
TensorDataset(x, y, mask)
```

内部：

```python
tensors == (x, y, mask)
```

所以 `*` 在这里不要理解成指针。

理解为：

```text
把任意多个位置参数收集起来
```

---

# 18. List Comprehension：列表推导式

这是 Python 源码压缩代码最常用的方式之一。

普通写法：

```python
result = []

for x in data:
    result.append(x * 2)
```

Python 常写：

```python
result = [x * 2 for x in data]
```

统一阅读规则：

```python
[表达式 for 元素 in 集合]
```

翻译：

```text
遍历集合
↓
对每个元素计算表达式
↓
把所有结果组成 List
```

DataLoader：

```python
batch = [self.dataset[idx] for idx in batch_indices]
```

翻译：

```text
遍历这一批索引
↓
依次 dataset[idx]
↓
拿到这一批样本
↓
组成 batch 列表
```

---

# 19. enumerate：同时获得编号和元素

```python
for i, param in enumerate(params):
```

假设：

```python
params = [W1, b1, W2]
```

得到：

```text
i=0, param=W1
i=1, param=b1
i=2, param=W2
```

Optimizer 很需要这种模式：

```text
第 i 个 parameter
对应
第 i 个 momentum buffer
```

---

# 20. zip：并行遍历

```python
for x, y in zip(features, labels):
```

如果：

```text
features = [x1, x2, x3]
labels   = [y1, y2, y3]
```

那么：

```text
x1 y1
x2 y2
x3 y3
```

阅读：

```text
把几个序列按相同位置配对
```

---

# 21. isinstance：判断对象类型

```python
if isinstance(other, Tensor):
```

意思：

```text
如果 other 是 Tensor
```

Java 类比：

```java
other instanceof Tensor
```

Tensor 运算经常需要区分：

```text
Tensor + Tensor

和

Tensor + 数字
```

例如：

```python
if isinstance(other, Tensor):
    other_data = other.data
else:
    other_data = other
```

---

# 22. property：为什么 `x.shape` 不写括号

Python 可能有：

```python
@property
def shape(self):
    return self.data.shape
```

调用时：

```python
x.shape
```

而不是：

```python
x.shape()
```

你可以理解成：

```text
把一个方法伪装成属性读取
```

现阶段知道怎么读即可，不需要研究装饰器原理。

---

# 23. 类型标注：阅读即可

```python
def forward(self, x: Tensor) -> Tensor:
```

解释：

```text
x 预期是 Tensor
返回值预期是 Tensor
```

```python
params: list[Tensor]
```

解释：

```text
params 是 Tensor 列表
```

重要：

> Python 的类型标注首先是给人和工具看的。

源码阅读时如果觉得复杂，可以暂时“擦掉”类型标注。

例如：

```python
def forward(self, x: Tensor) -> Tensor:
```

先看成：

```python
def forward(self, x):
```

---

# 24. NumPy：TinyTorch 必须掌握的最小集合

TinyTorch 的 Tensor 会使用 NumPy 做底层数值计算。

最重要的不是学完整 NumPy，而是认识下面这些。

## 创建数组

```python
np.array(...)
np.zeros(...)
np.ones(...)
np.random.randn(...)
```

## 数学

```python
np.exp(...)
np.log(...)
np.sqrt(...)
np.maximum(...)
```

## 聚合

```python
np.sum(...)
np.mean(...)
np.max(...)
```

## 形状

```python
x.shape
x.reshape(...)
x.transpose(...)
```

## 拼接

```python
np.stack(...)
np.concatenate(...)
```

---

# 25. `axis`：NumPy 最容易卡住的新手概念

假设：

```python
x.shape == (3, 4)
```

可以看成：

```text
3 行
4 列
```

```python
np.sum(x)
```

全部元素一起求和。

```python
np.sum(x, axis=0)
```

压掉第 0 维。

结果 shape：

```text
(4,)
```

可以先通俗理解：

```text
沿着“行方向”汇总
最后剩下每一列
```

```python
np.sum(x, axis=1)
```

结果：

```text
(3,)
```

每一行得到一个结果。

在 Softmax、Loss、归一化中会大量遇到。

---

# 26. Dataset 的 Python 本质

典型 Dataset：

```python
class Dataset:

    def __len__(self):
        ...

    def __getitem__(self, idx):
        ...
```

这两个方法非常关键。

有：

```python
__len__
```

就可以：

```python
len(dataset)
```

有：

```python
__getitem__
```

就可以：

```python
dataset[10]
```

所以 Dataset 的抽象本质只有：

```text
问题 1：
你一共有多少条数据？

问题 2：
给你一个编号，你能不能返回这一条数据？
```

---

# 27. DataLoader 的核心：迭代

你会看到：

```python
for batch in loader:
```

人类视角：

```text
不断从 loader 取下一批数据
直到没有数据
```

Python 底层与：

```text
iterator
__iter__
yield
```

有关。

不需要一开始研究完整 iterator protocol，但要真正理解 `yield`。

---

# 28. yield：DataLoader 最重要 Python 语法

普通函数：

```python
def f():
    return 1
```

调用一次：

```text
返回 1
函数结束
```

Generator：

```python
def f():
    yield 1
    yield 2
    yield 3
```

逻辑：

```text
第一次：
yield 1
↓
暂停

第二次：
从暂停的位置继续
↓
yield 2
↓
暂停

第三次：
yield 3
```

DataLoader：

```python
def __iter__(self):

    for i in range(...):
        ...
        yield batch
```

理解：

```text
构造 batch 1
↓
交给训练代码
↓
暂停

训练代码处理 batch 1

回来继续
↓
构造 batch 2
↓
交给训练代码
↓
暂停
```

因此 DataLoader 不需要先一次性制造所有 batch。

---

# 29. `__iter__`：为什么能 `for batch in loader`

当你写：

```python
for batch in loader:
```

Python 会尝试从 loader 获取迭代能力。

例如：

```python
class DataLoader:

    def __iter__(self):
        ...
```

于是：

```text
loader
↓
__iter__()
↓
不断 yield batch
```

阅读模型：

```text
for batch in loader

≈

不断向 loader 要“下一批”
```

---

# 30. ABC / abstractmethod：抽象类

```python
class Dataset(ABC):

    @abstractmethod
    def __len__(self):
        pass
```

如果你懂 Java，可以近似看成：

```java
abstract class Dataset {
    abstract int size();
}
```

含义：

```text
Dataset 只规定规则

具体子类负责真正实现
```

例如：

```python
class TensorDataset(Dataset):

    def __len__(self):
        ...
```

---

# 31. 装饰器 `@xxx`：现阶段只学“读法”

可能看到：

```python
@property
```

```python
@staticmethod
```

```python
@abstractmethod
```

不要现在研究装饰器底层实现。

暂时统一理解：

```text
给下面的函数添加某种特殊语义
```

分别大概理解为：

```text
@property
    方法可以像属性一样访问

@staticmethod
    方法不依赖具体 self

@abstractmethod
    子类必须实现
```

---

# 32. lambda：匿名小函数

可能看到：

```python
lambda x: x * 2
```

等价于：

```python
def f(x):
    return x * 2
```

看到 lambda 的固定翻译：

```text
临时定义一个很小的函数
```

Autograd / hook / 排序等代码里可能遇到。

---

# 33. 函数也是对象

Python 可以：

```python
fn = relu
```

然后：

```python
y = fn(x)
```

也可以把函数作为参数：

```python
def apply(x, fn):
    return fn(x)
```

调用：

```python
apply(x, relu)
```

这对后面理解 callback、hook、activation function 等设计有帮助。

---

# 34. Autograd 前必须补：对象引用

Python 变量通常可以理解成：

```text
变量
↓
指向对象
```

例如：

```python
a = Tensor([1, 2])
b = a
```

此时：

```text
a ─┐
   ├──> 同一个 Tensor
b ─┘
```

不是自动复制一份 Tensor。

所以：

```python
b.data[0] = 100
```

可能会影响 `a` 看到的数据。

这对理解：

```text
参数对象
计算图节点
梯度对象
```

非常重要。

---

# 35. Autograd 前必须补：递归

简单递归：

```python
def visit(node):

    for parent in node.parents:
        visit(parent)

    print(node)
```

核心：

```text
函数调用自己
```

为什么 Autograd 需要？

因为计算图：

```text
        loss
       /    \
      a      b
     / \
    x   y
```

需要沿着：

```text
当前节点
↓
父节点
↓
父节点的父节点
↓
...
```

不断遍历。

所以递归不是 Python 特有知识，真正是：

```text
图遍历
```

---

# 36. Autograd 前必须补：集合 set

```python
visited = set()
```

添加：

```python
visited.add(node)
```

判断：

```python
if node in visited:
```

为什么计算图经常需要 set？

为了避免：

```text
同一个节点被重复遍历
```

例如：

```text
    x
   / \
  a   b
   \ /
   loss
```

x 被多条路径引用。

需要：

```text
visited
```

记录：

```text
这个节点我处理过了
```

---

# 37. Autograd 前必须补：拓扑排序时怎么读代码

以后看到类似：

```python
visited = set()
topo = []

def build(node):

    if node not in visited:
        visited.add(node)

        for parent in node.parents:
            build(parent)

        topo.append(node)
```

不要先陷入 Python。

直接翻译：

```text
访问当前节点

如果从没访问过：
    标记已访问

    先递归访问它的所有父节点

    最后再把自己放进 topo
```

这会得到：

```text
输入节点在前
输出节点在后
```

反向传播再：

```python
for node in reversed(topo):
```

倒序处理。

---

# 38. reversed

```python
for node in reversed(topo):
```

解释：

```text
倒过来遍历 topo
```

Autograd 中通常意味着：

```text
loss
↓
中间节点
↓
输入/参数
```

这正是反向传播方向。

---

# 39. Optimizer 阅读需要掌握的 Python

Optimizer 常见结构：

```python
class SGD(Optimizer):

    def __init__(self, params, lr=0.01):
        self.params = list(params)
        self.lr = lr

    def step(self):
        for param in self.params:
            param.data -= self.lr * param.grad
```

源码阅读：

```text
SGD 保存所有参数

step()：
    遍历参数
    使用梯度更新参数
```

Python 其实不是难点。

真正难点是数学：

```text
new_parameter
=
old_parameter - learning_rate × gradient
```

---

# 40. Training Loop 阅读所需 Python

最典型训练循环：

```python
for epoch in range(num_epochs):

    for x, y in loader:

        pred = model(x)

        loss = loss_fn(pred, y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
```

把 Python 去掉以后就是：

```text
重复多个 epoch

    不断拿 batch

        forward
        ↓
        loss
        ↓
        清空旧梯度
        ↓
        backward
        ↓
        更新参数
```

这说明：

> 当你学到 Training 时，Python 应该已经退居幕后。

---

# 41. `import`：只要掌握基本读法

```python
import numpy as np
```

意思：

```text
导入 numpy
并给它一个简称 np
```

```python
from tinytorch import Tensor
```

意思：

```text
从 tinytorch 中取出 Tensor
```

```python
from abc import ABC, abstractmethod
```

意思：

```text
从 abc 模块导入两个名字
```

不用现在深入 Python package 机制。

---

# 42. 异常：只需先认识 raise

```python
if shape_invalid:
    raise ValueError("shape mismatch")
```

理解：

```text
发现非法情况
↓
主动报错
↓
停止正常执行
```

Tensor 的 shape 检查会大量使用。

---

# 43. assert：测试里非常常见

```python
assert x.shape == (2, 3)
```

翻译：

```text
我要求这个条件必须成立
```

如果不成立：

```text
测试失败
```

TinyTorch 学习时，tests 是非常好的“源码说明书”。

遇到不知道某函数应该干什么：

```text
先看测试
```

往往比先看实现更容易。

---

# 44. with：暂时只需认识

可能看到：

```python
with something:
    ...
```

现在只需要知道：

```text
进入某个临时上下文
↓
执行代码
↓
退出时自动清理
```

如果 Foundation 前期没遇到，可以先不学。

---

# 45. TinyTorch 01：Tensor 要会哪些 Python

优先掌握：

```text
class
self
__init__
property
__add__
__sub__
__mul__
__truediv__
isinstance
NumPy
shape
indexing
```

源码阅读模型：

```text
Tensor
=
Python Class
+
NumPy ndarray
+
一组数学操作
```

---

# 46. TinyTorch 02：Activations 要会哪些 Python

主要：

```text
def
return
np.exp
np.maximum
axis
shape
```

此时 Python 很简单。

重点应该转到：

```text
Sigmoid
ReLU
Tanh
Softmax
```

的数学含义。

---

# 47. TinyTorch 03：Layers 要会哪些 Python

最重要：

```text
class
继承
self
__init__
__call__
forward
list
for
parameters
```

阅读模型：

```text
Layer
=
保存状态/参数
+
定义 forward
```

Linear：

```text
参数：
    W
    b

计算：
    xW + b
```

Sequential：

```text
保存 layers
↓
循环执行 layers
```

---

# 48. TinyTorch 04：Losses 要会哪些 Python

主要：

```text
函数
shape
axis
NumPy
条件判断
数值处理
```

真正重点：

```text
预测值
↓
和目标比较
↓
得到一个标量 loss
```

---

# 49. TinyTorch 05：DataLoader 要会哪些 Python

这是 Python 语法突然增加的一章。

重点：

```text
Dataset
__len__
__getitem__
__iter__
yield
range
切片
List Comprehension
*args
shuffle
```

建议你单独花时间吃透：

```text
yield
```

---

# 50. TinyTorch 06：Autograd 要会哪些 Python

最重要：

```text
对象引用
list
tuple
set
递归
for
reversed
函数对象
closure（看到时再补）
```

这一章真正困难的是：

```text
计算图
链式法则
拓扑排序
梯度累积
```

不要误以为自己是 Python 不会。

---

# 51. TinyTorch 07：Optimizers 要会哪些 Python

主要：

```text
继承
list
enumerate
for
None
状态保存
```

真正重点：

```text
SGD
Momentum
Adam
```

的参数更新公式。

---

# 52. TinyTorch 08：Training 要会哪些 Python

主要：

```text
for
range
函数调用
对象组合
```

这章 Python 反而最简单。

重点：

```text
DataLoader
↓
Model
↓
Loss
↓
Backward
↓
Optimizer
```

如何组成一个完整系统。

---

# 53. 源码阅读五步法

以后打开任何 TinyTorch 文件，按照这个顺序。

## 第一步：先看有哪些 class / function

例如：

```text
class Layer
class Linear
class Dropout
class Sequential
```

先画结构，不看实现。

---

## 第二步：看 `__init__`

搞清楚：

```text
对象内部保存哪些数据？
```

例如 Linear：

```text
in_features
out_features
weight
bias
```

---

## 第三步：看 forward

回答：

```text
这个对象最核心的计算是什么？
```

---

## 第四步：看输入输出 shape

例如：

```text
x
(B, in_features)

W
(in_features, out_features)

↓

output
(B, out_features)
```

这一点通常比 Python 语法重要 10 倍。

---

## 第五步：最后才看辅助语法

例如：

```text
list comprehension
isinstance
property
*args
```

避免一开始陷入细节。

---

# 54. Python → 人话翻译训练

## 示例 1

```python
params = [self.weight]

if self.bias is not None:
    params.append(self.bias)

return params
```

不要翻译成：

```text
声明 params……
执行 if……
调用 append……
```

应该翻译：

```text
返回当前层所有可学习参数：
weight 必须有；
bias 如果存在，也返回。
```

---

## 示例 2

```python
for layer in self.layers:
    x = layer(x)
```

正确翻译：

```text
让输入依次通过每一层。
```

---

## 示例 3

```python
batch = [
    self.dataset[idx]
    for idx in batch_indices
]
```

正确翻译：

```text
根据当前 batch 的索引，从 Dataset 取出这一批样本。
```

---

## 示例 4

```python
for node in reversed(topo):
    node.backward()
```

正确翻译：

```text
按照计算图从输出到输入的顺序执行反向传播。
```

---

# 55. 看到源码卡住时的优先级

不要一看到不认识的 Python 就去系统学一小时。

按这个顺序判断：

```text
1. 不认识的语法会影响我理解算法吗？

不会
→ 暂时跳过

会
→ 查清它的输入输出

仍然不懂
→ 写一个 3~5 行最小 Python Demo
```

例如不理解：

```python
[x * 2 for x in data]
```

不要读“Python 列表推导式高级教程”。

自己执行：

```python
data = [1, 2, 3]

result = [x * 2 for x in data]

print(result)
```

输出：

```text
[2, 4, 6]
```

马上够用了。

---

# 56. 你的学习顺序

如果完全没有 Python 基础，建议按下面顺序补。

## 第一阶段：2~3 小时

```text
变量
list
tuple
dict
if
for
range
函数
索引
切片
```

目标：

```text
能读普通 Python 函数
```

---

## 第二阶段：3~4 小时

```text
class
self
__init__
继承
super
None
isinstance
```

目标：

```text
能看懂 Tensor / Layer 的对象结构
```

---

## 第三阶段：2~3 小时

```text
__add__
__mul__
__len__
__getitem__
__call__
__iter__
```

目标：

```text
理解 Python 语法糖背后的方法调用
```

---

## 第四阶段：3~4 小时

```text
NumPy
shape
axis
reshape
transpose
sum
mean
stack
```

目标：

```text
能读 Tensor / Activation / Loss
```

---

## 第五阶段：2~3 小时

```text
list comprehension
enumerate
zip
*args
**kwargs
yield
```

目标：

```text
能读 DataLoader
```

---

## 第六阶段：Autograd 前

补：

```text
set
递归
函数对象
对象引用
reversed
拓扑排序概念
```

目标：

```text
不再让 Python 语法干扰理解计算图
```

---

# 57. 最核心 20 个 Python 知识点速查

```text
01 变量              x = ...
02 None              类似 null
03 List              [...]
04 Tuple             (...)
05 Dict              {...}
06 if                条件判断
07 for               遍历
08 range             数字序列
09 def                定义函数
10 return            返回
11 class             定义类型
12 self              类似 Java this
13 __init__          构造函数
14 继承              class A(B)
15 __call__          obj(...)
16 __getitem__       obj[i]
17 __len__           len(obj)
18 运算符重载        __add__/__mul__
19 list comprehension [... for ...]
20 yield             生成一个结果后暂停
```

这 20 个吃透，Foundation 大部分 Python 障碍已经解决。

---

# 58. TinyTorch 阅读心智模型

以后永远先看：

```text
数据是什么？
↓
shape 是什么？
↓
这一步数学操作是什么？
↓
输出 shape 是什么？
↓
参数在哪里？
↓
梯度怎么回来？
```

最后才看：

```text
Python 为什么这么写？
```

例如 Linear：

```text
输入
(B, in)

↓

W
(in, out)

↓

矩阵乘法

↓

(B, out)

↓

+bias
(out,)

↓

广播

↓

(B, out)
```

这个理解比背 50 个 Python 语法规则重要。

---

# 59. 最终目标

当你看到：

```python
for epoch in range(epochs):

    for x, y in loader:

        output = model(x)
        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

你不应该再看到：

```text
for
range
函数
对象
method
```

而应该直接看到：

```text
             Dataset
                ↓
             DataLoader
                ↓
          x, y 一个 Batch
                ↓
             Model
                ↓
           Prediction
                ↓
              Loss
                ↓
            Autograd
                ↓
            Gradients
                ↓
            Optimizer
                ↓
       更新 Weight / Bias
                ↓
          下一轮训练
```

这就是你补 Python 的最终目的。

---

# 60. 一句话总结

> 学 TinyTorch 时，Python 是“阅读语言”，Tensor / Autograd / Neural Network 才是“学习内容”。

不要等 Python 学完再学 TinyTorch。

最有效的路线是：

```text
TinyTorch
↓
遇到 Python 障碍
↓
补一个最小知识点
↓
马上回源码
↓
继续 TinyTorch
```

这样最快。
