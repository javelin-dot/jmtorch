# TinyTorch 第 03 章：Layers 学习问答

> 面向有 Java 开发经验、没有 Python 语言基础、正在入门神经网络的学习者。
>
> 学习目标：能把 `Linear → ReLU → Dropout → Linear` 从中文含义、数字计算和 Python 代码三个角度讲清楚，并能独立检查形状与参数量。

本文以你上传的《粘贴的 markdown (1)。md》为主要教材，重新组织学习顺序，补充原创小例子、Python 语法解释和练习。官网用于核对本章的范围与接口：[TinyTorch Module 03: Layers](https://mlsysbook.ai/tinytorch/modules/03_layers.html)。

你上传的是教学说明，并未包含完整 `.py` 源码。因此，文中标注“教学片段”的代码用于解释实现思路，不冒充你本地文件的逐行还原；第七部分提供一份只依赖 NumPy 的完整练习程序。

## 怎么使用这份文档

**每次只完成一小段：先读问题，尝试回答，再看解释，最后动手算或运行。** 不用一次记住所有术语。

| 学习轮次 | 阅读范围 | 本轮完成的标志 |
| --- | --- | --- |
| 第一轮：先懂计算 | 第一、二部分，Q1—Q14 | 手算出一个 Linear 层的输出，能解释每个维度 |
| 第二轮：读懂语言 | 第三、四部分，Q15—Q31 | 看懂 `self`、构造方法、前向方法和参数列表 |
| 第三轮：组合网络 | 第五、六部分，Q32—Q44 | 解释 Dropout 缩放，并把多个层接起来 |
| 第四轮：自己验证 | 第七、八部分，Q45—Q50 | 运行完整程序，主动制造并修复形状错误 |
| 第二遍再学 | 第九部分，Q51—Q54 | 估算参数内存、运算量，识别实现细节 |

文中的代码分三种，请先看代码前的说明：

- **可独立运行**：该代码块自身包含必要导入和数据。
- **教学片段**：只展示当前知识点，需要已有的类、变量或 TinyTorch 环境。
- **完整实验**：第七部分整段复制后可以运行，不依赖你是否完成了 TinyTorch 前两章。

快速跳转：[概念](#concepts) · [手算 Linear](#linear-math) · [Python 语法](#python-basics) · [Linear 实现](#linear-code) · [Dropout](#dropout) · [Sequential](#sequential) · [完整实验](#experiment) · [错误排查](#debugging) · [进阶理解](#advanced) · [练习答案](#practice) · [检查表](#checklist)

---

<a id="concepts"></a>

## 一、先回答：这一章到底在解决什么问题？

### Q1：我已经学过 Tensor 和激活函数，为什么还需要 Layer？

**因为计算公式需要被组织成可以反复使用、保存参数、组合连接的对象。**

例如，你能写一次矩阵乘法，但一个网络可能包含多个矩阵乘法，每次使用不同的权重。你需要知道：每组权重属于谁、怎样计算、怎样把这些权重交给训练程序。

`Layer` 就是在组织这些事情。

结合 Java 来理解：你可以把一个具体的层看作一个对象。它有成员变量，也有方法：

| Java 中熟悉的概念 | 在 Linear 层中对应什么 |
| --- | --- |
| 成员变量 | 保存权重 `weight`、偏置 `bias` |
| 构造方法 | 创建对象时，确定输入输出宽度并初始化参数 |
| 业务方法 | `forward(x)` 根据输入计算输出 |
| 返回内部对象的集合 | `parameters()` 返回后续训练需要更新的参数 |

例如 `Linear(2, 3)` 创建后，它会一直保有自己的权重，而不是每次调用时重新抽一组随机权重。

### Q2：Tensor、Linear、激活函数、Dropout 和网络，分别是什么？

**Tensor 装数据；不同的层处理数据；网络把处理步骤组织起来。**

| 名称 | 本章中负责什么 | 是否有可学习参数 | 是否改变特征个数 |
| --- | --- | --- | --- |
| Tensor | 保存数值，并提供计算操作 | 取决于它装的是输入、参数还是其他数据 | 取决于执行的操作 |
| Linear | 对输入做加权组合并加偏置 | 有：权重、可选偏置 | 可以改变，也可以保持 |
| ReLU | 负数变为 0，非负数保持原值 | 没有 | 不改变 |
| Dropout | 训练时随机把部分元素置零并缩放 | 没有 | 不改变 |
| Sequential | 按顺序调用内部各层 | 汇总内部层的参数，本身不额外增加权重 | 由内部层决定 |

因此，**“是一个层”不等于“必定有权重”**。Dropout 就是一个没有可学习参数的层。

### Q3：`forward()` 运行一次，模型就学到东西了吗？

**没有。前向计算只是用当前参数算出结果。**

你可以类比一个已有规则的 Java 方法：运行它会返回一个结果，不代表方法中的系数会自己改变。

完整训练还要有以下分工：

| 环节 | 可以先怎样理解 |
| --- | --- |
| 前向计算 | 用当前权重算预测结果 |
| 损失函数 | 衡量预测与目标差多少 |
| 自动求导 | 计算损失对各个参数变化的敏感程度，即梯度 |
| 优化器 | 根据梯度调整参数 |
| 重复训练 | 再计算、再衡量、再调整 |

本章重点完成第一项，以及“怎样收集可调整的参数”。

**即使你调用一个新建的 Linear 一万次，只要没有更新参数，它也不会因此学会分类。** 使用 Dropout 时结果可能变化，但随机变化不等于学习。

### Q4：输入数据、模型参数、超参数，为什么都叫“参数”？

因为“参数”在编程和机器学习里有不同用法。读代码时要分清语境。

| 对象 | 例子 | 谁决定它 | 是否由本章所讨论的梯度训练更新 |
| --- | --- | --- | --- |
| 输入数据 | `x`，每条样本的特征 | 数据来源与预处理 | 通常不是模型要更新的对象 |
| 模型参数 | `weight`、`bias` | 先初始化，再通过训练调整 | 是 |
| 超参数或结构配置 | 输入宽度、输出宽度、Dropout 的 `p` | 开发者设置或通过实验选择 | 通常不是 |
| 函数调用参数 | `forward(x, training=False)` 中的实参 | 调用者 | 这是编程语法概念 |

例如 `Dropout(p=0.5)` 虽然在构造函数里传了一个参数，但这个 `p` 不是 `parameters()` 应返回的可学习权重。

### Q5：这一章学到什么程度，就可以继续下一章？

先达到以下四项即可：

1. 给出 `x`、`W`、`b`，你能手算 `xW+b`。
2. 给出 `Linear(2, 3)`，你能说出权重、偏置和输出的形状。
3. 看到 `layer(x)`，你知道它怎样调用到 `forward()`。
4. 你知道 Dropout 在训练和推理时的区别，以及怎样串联层。

初始化的严格概率推导、底层矩阵乘法优化、完整反向传播，都可以后续再深入。

---

<a id="linear-math"></a>

## 二、先不用 Python，把 Linear 算明白

### Q6：`Linear(2, 3)` 中的 2 和 3，究竟表示什么？

**每条样本输入 2 个特征，输出 3 个新特征。**

它不表示“输入 2 条数据，输出 3 条数据”，也不表示“有 2 层和 3 层”。

假设一次输入两条样本：

| 样本 | 输入特征 1 | 输入特征 2 |
| --- | ---: | ---: |
| A | 1 | 2 |
| B | 3 | 1 |

经过 `Linear(2, 3)` 后，仍然是 A、B 两条样本，只是每条都有 3 个输出数字。

这里用的是人为设置的小数字，仅用来理解计算，不代表一个已经训练好的业务模型。

### Q7：`shape=(2, 2)`、`(2, 3)` 这些形状应该怎样读？

本章主要讨论二维批量输入，约定：

**第一个数字是样本数，第二个数字是每条样本的特征数。**

| 对象 | 形状 | 中文含义 |
| --- | --- | --- |
| 输入 `x` | `(2, 2)` | 2 条样本，每条 2 个特征 |
| 权重 `W` | `(2, 3)` | 2 个输入位置，到 3 个输出位置的连接权重 |
| 偏置 `b` | `(3,)` | 3 个输出位置各有一个偏置 |
| 输出 `y` | `(2, 3)` | 2 条样本，每条 3 个输出 |

注意，**不是所有二维数组的第一维都叫样本数**。`W` 的两维表示输入和输出特征，含义与 `x` 不同。

Python 中 `(3,)` 表示只有一个轴、长度为 3。它不是 `(1, 3)`，也不是 `(3, 1)`。这里的逗号是元组语法，Q16 会解释。

### Q8：为什么权重有 `2×3=6` 个？一个“神经元”在哪里？

因为每个输出都要考虑全部 2 个输入，而现在有 3 个输出。

我们固定下面这组权重：

| 权重来自哪个输入 | 输出 1 | 输出 2 | 输出 3 |
| --- | ---: | ---: | ---: |
| 输入特征 1 | 1 | -1 | 2 |
| 输入特征 2 | 2 | 1 | 0 |

每一列对应一个输出的计算规则：

- 输出 1 使用权重 `[1, 2]`。
- 输出 2 使用权重 `[-1, 1]`。
- 输出 3 使用权重 `[2, 0]`。

本章可以把一个输出单元理解成一个“加权求和器”。有 3 个这样的求和器，每个需要 2 个权重，所以共有 6 个权重。

这里的“全连接”表示：结构上，每个输出都连接全部输入。某个权重恰好为 0，不会让这个层变成另一种层。

### Q9：能不能把 `y=xW+b` 的每个数都算出来？

可以。继续用同一组数据：

```text
x = [[1, 2],
     [3, 1]]

W = [[1, -1, 2],
     [2,  1, 0]]

b = [0, 1, -1]
```

这只是数字展示，不是要求你执行的 Python 程序。

矩阵乘法的规则是：**取输入的一行，与权重的一列对应相乘，再求和。最后加该输出位置的偏置。**

| 样本与输出 | 对应相乘并求和 | 加偏置 | 结果 |
| --- | --- | --- | ---: |
| A 的输出 1 | `1×1 + 2×2 = 5` | `5+0` | 5 |
| A 的输出 2 | `1×(-1) + 2×1 = 1` | `1+1` | 2 |
| A 的输出 3 | `1×2 + 2×0 = 2` | `2+(-1)` | 1 |
| B 的输出 1 | `3×1 + 1×2 = 5` | `5+0` | 5 |
| B 的输出 2 | `3×(-1) + 1×1 = -2` | `-2+1` | -1 |
| B 的输出 3 | `3×2 + 1×0 = 6` | `6+(-1)` | 5 |

所以：

```text
y = [[5,  2, 1],
     [5, -1, 5]]
```

建议你暂时合上表格，只重新算“B 的输出 2”。如果能独立得到 `-1`，说明你已经掌握了一个输出的完整计算过程。

### Q10：偏置为什么只有 3 个？不是每条样本都需要一份吗？

**每个输出位置一个偏置，所有样本共享它。**

`xW` 的形状是 `(2, 3)`，加上 `(3,)` 的 `b` 时，效果相当于每行都加 `[0, 1, -1]`。

| 样本 | 矩阵乘法结果 | 加上的偏置 | 最终结果 |
| --- | --- | --- | --- |
| A | `[5, 1, 2]` | `[0, 1, -1]` | `[5, 2, 1]` |
| B | `[5, -2, 6]` | `[0, 1, -1]` | `[5, -1, 5]` |

这叫**广播**。它描述运算怎样对齐，不要求程序先真正复制出两份偏置数组。NumPy 的规则是从最右边的轴开始比较，长度相同或其中一个为 1 时可以匹配；缺少的前置轴按长度 1 理解。[NumPy 广播规则](https://numpy.org/doc/stable/user/basics.broadcasting.html)

共享参数意味着：不管输入 2 条还是 200 条样本，这个层都只有同一组 6 个权重、3 个偏置。

### Q11：偏置有什么用？为什么 `bias=False` 可以关闭它？

看最简单的单输入公式：

```text
y = 2x + 3
```

权重 `2` 决定输入变化怎样影响输出，偏置 `3` 提供额外的平移量。

如果没有偏置，`x=0` 时输出只能是 0；有偏置时，`x=0` 也可以输出 3。

对于 `Linear(2, 3, bias=False)`：

- 仍然有 6 个权重。
- 不创建那 3 个可学习偏置。
- 计算变成 `y=xW`。
- 在教材描述的实现中，`self.bias` 保存为 `None`。

严格的数学术语里，`xW+b` 在包含非零偏置时是仿射变换。框架仍习惯把这个组件叫作 Linear。

### Q12：Linear 输出的 3 个数，一定是 3 个概率吗？

**不一定。Linear 只负责数值变换，数值的含义由任务和所处位置决定。**

刚才的输出 `[5, 2, 1]` 不是概率分布：它们的和不是 1，其他输入还可能产生负数。

| Linear 所处位置 | 输出可能表示什么 |
| --- | --- |
| 网络中间 | 供后续层使用的隐藏特征 |
| 分类网络最后一层 | 类别分数，常称 logits |
| 回归网络最后一层 | 一个或多个连续数值的预测 |

隐藏特征通常不是开发者预先命名的字段。输出 1 不自动代表“收入”，输出 2 也不自动代表“风险”。训练会决定这些组合怎样帮助完成任务。

如果任务需要类别概率，才考虑匹配的概率变换。不要在还没有理解任务和损失函数时，给所有 Linear 后面都加 Softmax。

### Q13：为什么 Linear 后面经常接 ReLU？多叠几层 Linear 不行吗？

**中间加入非线性，才能表达单纯线性组合做不到的关系。**

先看单个数，不用矩阵：

```text
第一层：h = 2x + 1
第二层：y = 3h + 4

代入以后：y = 3(2x+1)+4 = 6x+7
```

两层合起来，仍然只是一个“乘系数、加常数”的公式。

如果中间加入 ReLU：

```text
h = max(0, 2x+1)
y = 3h+4
```

此时输入落在不同区间时，计算行为会不同，不能统一写成一个 `ax+b`。

在我们的二维例子里：

| 样本 | Linear 输出 | 经过 ReLU |
| --- | --- | --- |
| A | `[5, 2, 1]` | `[5, 2, 1]` |
| B | `[5, -1, 5]` | `[5, 0, 5]` |

ReLU 不改变特征数量，只改变对应位置的数值。

### Q14：batch 变了，要不要重新创建 Linear？特征数变了呢？

**只改变样本数，通常不用重建；改变每条样本的输入特征数，就必须重新检查层的结构。**

对于已经创建好的 `Linear(2, 3)`：

| 输入形状 | 能否按本章的二维输入约定使用 | 输出形状 |
| --- | --- | --- |
| `(1, 2)` | 可以：1 条样本 | `(1, 3)` |
| `(8, 2)` | 可以：8 条样本 | `(8, 3)` |
| `(100, 2)` | 可以：100 条样本 | `(100, 3)` |
| `(8, 4)` | 不可以：每条变成 4 个特征了 | 需要匹配的权重形状 |

即使一个样本，也建议初学时写成 `[[1, 2]]`，保留 `(1, 2)` 的二维形状。

`[1, 2]` 的形状是 `(2,)`。有些底层矩阵接口支持这种一维输入，但输出轴的行为会不同。本章先统一使用二维输入，减少歧义。

---

<a id="python-basics"></a>

## 三、只补阅读本章需要的 Python

### Q15：`import numpy as np`、`from ... import ...`，怎样对应 Java？

先把 `import` 理解成“让当前文件能使用其他模块提供的名字”。

| Python 写法 | 中文读法 |
| --- | --- |
| `import numpy as np` | 导入 NumPy，并在本文件中把它简称为 `np` |
| `np.array(...)` | 调用 NumPy 提供的数组构造功能 |
| `from tinytorch.core.tensor import Tensor` | 从指定模块中导入 `Tensor` 这个名字 |
| `from tinytorch.core.layers import Linear` | 从层模块中导入 `Linear` |

NumPy 是数值数组计算库。TinyTorch 在它之上封装自己的 Tensor 和网络组件。

学习文件和导出包也要分清：

| 路径或名称 | 作用 |
| --- | --- |
| `src/03_layers/03_layers.py` | 教学源码的位置 |
| `03_layers_zh.py` | 你的教材提到的中文副本 |
| `tinytorch.core.layers` | 示例代码实际导入的软件包模块 |

修改一个文件，不代表另一个文件自动改变。尤其不能把“看到了中文源码”当成“导入时一定使用中文源码”。

### Q16：`[]`、`()`、列表、数组、Tensor，分别是什么？

**列表是 Python 自带容器；NumPy 数组专门处理数值；TinyTorch Tensor 是项目自己的数据对象。**

| 写法 | 类型或含义 |
| --- | --- |
| `[1, 2, 3]` | Python 列表，`list` |
| `(2, 3)` | Python 元组，`tuple`，这里常用来表示形状 |
| `(3,)` | 只有一个元素的元组 |
| `(3)` | 只是整数 3 外面加括号，不是元组 |
| `np.array([1, 2, 3])` | NumPy 数组，类型是 `ndarray` |
| `Tensor([1, 2, 3])` | TinyTorch 的 Tensor 对象，前提是已导入该类 |

下面是**可独立运行**的小例子：

```python
import numpy as np

items = [1, 2, 3]
values = np.array([1, 2, 3], dtype=np.float32)

print(items * 2)   # [1, 2, 3, 1, 2, 3]：重复列表
print(values * 2)  # [2. 4. 6.]：每个数乘以 2
print(values.shape)  # (3,)
```

**同一个 `*`，在不同对象上可以有不同含义。** 所以读代码时，不能只看符号，还要知道左右两边是什么类型。

列表与元组的语法和基本行为，可查 [Python 数据结构教程](https://docs.python.org/3/tutorial/datastructures.html)。

### Q17：Python 没有 Java 那样的类型声明和大括号，我怎样读一个方法？

看下面这个**可独立运行**的例子：

```python
def score(x, weight, bias):
    result = x * weight + bias
    return result

answer = score(2, 3, 1)
print(answer)  # 7
```

逐行读：

1. `def`：定义一个函数，函数名是 `score`。
2. 括号内是函数接收的三个参数。
3. 行末 `:` 表示下面开始一个代码块。
4. 缩进的两行属于这个函数；通常使用 4 个空格。
5. `return` 把结果交给调用者。
6. `#` 后面的内容是注释，不参与运算。

`answer = ...` 不用提前写 `int` 或 `float`，但对象仍然有类型。Python 不是“没有类型”，而是变量名不需要这样预先声明类型。

源码中的 `# %%`、`# %% [markdown]` 是工具使用的单元标记，对普通 Python 执行而言属于注释。三引号可以包围多行字符串；它们经常用于文档说明，不会神奇地变成训练逻辑。

### Q18：`class Linear(Layer)`、`self`、`__init__` 分别是什么？

先看结构，不考虑矩阵运算。这是一个**可独立运行**的语法例子：

```python
class FeatureMapper:
    def __init__(self, input_width, output_width):
        self.input_width = input_width
        self.output_width = output_width

mapper = FeatureMapper(2, 3)
print(mapper.input_width)   # 2
print(mapper.output_width)  # 3
```

| Python | 与 Java 对照着理解 |
| --- | --- |
| `class FeatureMapper:` | 定义一个类 |
| `FeatureMapper(2, 3)` | 创建对象，Python 这里不写 `new` |
| `__init__` | 对新对象做初始化，入门时可类比构造方法 |
| `self` | 当前对象，可类比 `this` |
| `self.input_width` | 当前对象的一个成员属性 |
| `class Linear(Layer):` | 定义 Linear，它继承 Layer |

`self` 是常用命名约定，不是需要你额外传入的业务参数。调用 `FeatureMapper(2, 3)` 时，Python 会把新对象交给初始化方法的第一个参数。

严格说，`__init__` 负责初始化，创建实例还涉及其他机制；目前不需要展开。[Python 类与实例](https://docs.python.org/3/tutorial/classes.html)

### Q19：为什么 `self.in_features = in_features` 两边名字一样？

它们处于不同位置，作用也不同。

```python
# 教学片段：位于某个类的方法内部。
self.in_features = in_features
```

| 位置 | 含义 |
| --- | --- |
| 右边 `in_features` | 本次方法调用传进来的值 |
| 左边 `self.in_features` | 保存到当前对象上的属性 |

对应的 Java 思路就是：

```java
this.inFeatures = inFeatures;
```

如果只写 `in_features = 2`，通常只是给当前作用域的变量赋值，并没有把这个值保存成对象属性。

再看 `bias=True`：它是默认参数值。

| 调用 | 含义 |
| --- | --- |
| `Linear(2, 3)` | 没有传 `bias`，使用默认值 `True` |
| `Linear(2, 3, bias=False)` | 按名字指定：关闭偏置 |
| `dropout(x, training=False)` | 本次调用按推理模式执行 |

`True`、`False` 分别对应布尔真、假，首字母大写；`None` 表示没有相应值，可类比 Java 的 `null`。默认参数和关键字参数参见 [Python 函数定义](https://docs.python.org/3/tutorial/controlflow.html#defining-functions)。

### Q20：Layer 基类是不是 Java 的抽象类？

**它的用途可以类比统一接口或父类，但教材描述的实现不等同于 Java 的编译期抽象类约束。**

下面是用于理解约定的**教学片段**：

```python
class Layer:
    def forward(self, x):
        raise NotImplementedError("子类需要提供前向计算")

    def parameters(self):
        return []
```

`raise` 类似 Java 的 `throw`。

这个写法表达：如果子类没有实现 `forward()`，实际调用它时就报错。它没有使用 Python 的抽象基类机制来阻止实例化。

`parameters()` 默认返回空列表，适合没有可学习参数的组件。有参数的 Linear 要提供自己的实现。

**把 Tensor 保存到 `self.weight`，不会自动让这个简单基类发现它。** 参数收集需要具体实现。

如果后来看到 `super().__init__()`，可先理解为“调用父类初始化”；是否需要这样写，取决于父类是否有需要完成的初始化工作。

### Q21：`layer(x)` 看起来是在调用对象，为什么合法？

因为 Python 允许类定义 `__call__()`，让它的实例可以用函数调用语法执行。[Python `__call__` 说明](https://docs.python.org/3/reference/datamodel.html#object.__call__)

下面是**可独立运行**的例子：

```python
class AddOne:
    def forward(self, x):
        return x + 1

    def __call__(self, x):
        return self.forward(x)

layer = AddOne()
print(layer.forward(3))  # 4
print(layer(3))          # 4
```

第二个调用会进入 `__call__()`，再由它调用 `forward()`。

注意区分：

- `AddOne()`：调用类，创建一个对象。
- `layer(3)`：调用已有对象，执行一次计算。

在这个极简实现里，两种前向写法的结果一样。成熟框架还可能在对象调用入口加入其他处理，因此不要把所有框架的 `layer(x)` 都理解成永远只有这一行转发。

### Q22：`*layers`、`*args`、`**kwargs` 是乘法吗？

**在函数参数的位置，它们用于收集或展开参数，不是乘法。**

| 写法 | 在本章中的作用 |
| --- | --- |
| `def __init__(self, *layers):` | 把多个位置参数收集成元组 |
| `*args` | 常用名字，收集额外的位置参数 |
| `**kwargs` | 常用名字，收集额外的关键字参数，得到字典 |
| `self.forward(x, *args, **kwargs)` | 把收集到的参数再次展开，继续传给另一个方法 |

这是**可独立运行**的例子：

```python
def show(*args, **kwargs):
    print(args)
    print(kwargs)

show(10, 20, training=False)
```

输出：

```text
(10, 20)
{'training': False}
```

所以 `Sequential(layer1, relu, layer2)` 可以通过 `*layers` 一次收下三个对象；而 `training=False` 可以通过 `**kwargs` 转交给具体的层。

### Q23：`x.data`、`x.shape`、`.astype(np.float32)`，各自在做什么？

要先看 `x` 的类型。

| 表达式 | 本文中的解释 |
| --- | --- |
| `tensor.data` | 对 TinyTorch Tensor，取出它保存的底层 NumPy 数值数组 |
| `tensor.shape` | 读取 Tensor 暴露的形状 |
| `array.shape` | 对 NumPy 数组，直接读取形状 |
| `array.size` | NumPy 数组中标量元素的总数 |
| `array.dtype` | 数组元素的数据类型 |
| `array.astype(np.float32)` | 得到元素类型转换为 float32 的数组 |

`float32` 表示每个数的数据部分占 32 位，也就是 4 字节。

不要混淆两种 `.data`：NumPy 数组也有这个属性，但不是这里“取出 TinyTorch 底层数组”的意思。**本章练习中，对 NumPy 数组直接操作它本身；对 TinyTorch Tensor 才按教材使用 `.data` 查看数值。**

还有两个常见细节：

- `x.shape` 没有括号，因为是在读取属性。
- `layer.parameters()` 有括号，因为是在调用方法。

### Q24：`@`、`*`、`+` 在这些代码中怎么分辨？

对本章的 NumPy 数值数组：

| 运算 | 含义 | 本章用途 |
| --- | --- | --- |
| `x @ W` | 矩阵乘法 | 把输入特征组合成输出特征 |
| `x * mask` | 逐元素乘法，可按规则广播 | 应用 Dropout 掩码 |
| `y + b` | 逐元素加法，可按规则广播 | 给每条样本加偏置 |

TinyTorch 的 Linear 教材使用 `x.matmul(weight)` 表达矩阵乘法。不要据此假定你的 Tensor 版本必定实现了 `@` 运算符。

**一个结果取决于一行和一列，是矩阵乘法；一个位置只跟对应位置相乘，是逐元素乘法。**

如果变量实际上是 Python 列表，`+` 会拼接列表，`*` 可以重复列表。Q16 的例子就是为了防止把这两套规则混在一起。

### Q25：剩下这些“小符号”，现在需要懂到什么程度？

先能读懂下表即可。这里是语法速查，不要求一次背会。

| 写法 | 中文理解 |
| --- | --- |
| `if self.bias is not None:` | 如果这个对象确实保存了偏置 |
| `if not training:` | 如果不是训练模式 |
| `for layer in self.layers:` | 逐个取出容器里的层 |
| `len(params)` | 参数列表里有几个对象 |
| `params.append(item)` | 把一个对象作为一个新元素加入列表 |
| `params.extend(items)` | 把另一组对象逐个加入列表 |
| `return []` | 返回空列表 |
| `x is y` | 判断是不是同一个对象，而不是仅比较数值 |
| `assert condition` | 检查条件；不成立时抛出断言错误 |
| `f"shape={x.shape}"` | 把表达式结果填进字符串 |
| `def forward(self, x: Tensor) -> Tensor:` | 类型提示：期望输入和返回值都是 Tensor |
| `def __repr__(self):` | 自定义对象的调试文字表示 |
| `if __name__ == "__main__":` | 文件被直接运行时，执行下面的入口代码 |

类型提示通常帮助阅读和静态检查，并不自动完成数值转换或运行时类型校验。

`assert` 很适合本章的学习测试；正式运行时必须执行的输入校验，应使用明确的条件判断和异常，而不是只依赖断言。

---

<a id="linear-code"></a>

## 四、回到 Linear：把代码与数学一一对上

### Q26：第一次打开 Linear 源码，最适合按什么顺序读？

**先看 `forward()`，再看 `__init__()`，最后看 `parameters()` 和调用入口。**

原因是你已经手算过 `xW+b`，先找到这条主线，最容易建立联系。

| 阅读位置 | 只回答一个问题 |
| --- | --- |
| `forward()` | 输入怎样变成输出？ |
| `__init__()` | 本次计算用的权重和偏置从哪里来？ |
| `parameters()` | 哪些对象需要交给后续训练程序？ |
| `__call__()` | `layer(x)` 怎样进入前向方法？ |
| 测试函数 | 什么结果算正确？ |

暂时跳过打印格式、计时表和大型网络示例。等小例子跑通后再回头看。

### Q27：能给我一个贴近 TinyTorch、但更容易读的 Linear 实现吗？

下面是**教学片段**。假设已经有 `Layer`、`Tensor`、`np` 和随机数生成器 `rng`。它展示教材中的计算约定，未加入完整参数校验，不是可单独运行的文件。

```python
class ReadableLinear(Layer):
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features

        std = 1.0 / np.sqrt(in_features)
        numbers = rng.normal(
            loc=0.0,
            scale=std,
            size=(in_features, out_features),
        )
        self.weight = Tensor(numbers)

        self.bias = None
        if bias:
            self.bias = Tensor(np.zeros(out_features))

    def forward(self, x):
        result = x.matmul(self.weight)
        if self.bias is not None:
            result = result + self.bias
        return result

    def parameters(self):
        items = [self.weight]
        if self.bias is not None:
            items.append(self.bias)
        return items
```

逐段翻译：

| 代码 | 数学或程序含义 |
| --- | --- |
| 保存 `in_features`、`out_features` | 对象记住自己接收几个特征、输出几个特征 |
| `std = 1.0 / np.sqrt(in_features)` | 计算初始化权重的标准差 |
| `rng.normal(...)` | 按指定均值、标准差和形状生成随机数 |
| `loc=0.0` | 这组正态分布的均值是 0 |
| `scale=std` | 这组正态分布的标准差是 `std` |
| `size=(in_features, out_features)` | 生成与权重矩阵形状一致的数组 |
| `Tensor(numbers)` | 把数值包装成 TinyTorch Tensor |
| `np.zeros(out_features)` | 创建一维全零偏置数组 |
| `x.matmul(self.weight)` | 计算 `xW` |
| `result + self.bias` | 广播加上 `b` |
| `items.append(self.bias)` | 把偏置对象也加入待训练参数列表 |

括号内部可以分成多行，所以上面的 `rng.normal(...)` 仍然是一次函数调用。

**实现思路用一句话说：构造时创建并保存 W、b；前向时计算 xW+b；参数接口把 W、b 的原对象交给训练程序。**

### Q28：为什么权重不直接随便随机一下，还要除以输入数的平方根？

因为一个输出是很多项相加。如果输入项越来越多，单个权重仍保持相同波动大小，求和结果就可能越来越不稳定。

本章教材采用的初始化标准差是：

```text
标准差 = sqrt(1 / in_features)
       = 1 / sqrt(in_features)
```

| 输入特征数 | 初始化标准差 |
| ---: | ---: |
| 4 | 0.5 |
| 100 | 0.1 |
| 1000 | 约 0.03162 |

这不表示每个权重都等于这个数字，也不表示权重一定落在“正负这个数字”之间。**标准差描述随机值的波动尺度。**

先记住直觉：求和的输入越多，每项的初始权重通常需要适当缩小。

这是 LeCun 风格的缩放。它有助于控制数值传播，但不保证任意深度、任意激活函数的网络都不会梯度消失或爆炸。

### Q29：为什么不能把权重都设成 0？偏置设成 0 却可以？

对有多个隐藏单元的常见网络，如果同一层各单元的参数从完全相同的状态出发，并受到相同的更新，它们就容易一直做相同的事。

你原本希望 3 个输出单元学习不同组合，结果它们可能变成 3 份重复计算。这叫**对称性问题**。

随机初始化权重可以让各单元从不同的组合出发。权重已经提供差异时，偏置从 0 开始通常没有这个问题。

需要保留一个边界：这不是“所有机器学习模型都绝对禁止零初始化”。单独的线性回归等模型可以采用零初始化。这里讨论的是常见多层网络中隐藏单元的对称性。

把所有权重设成 1，也没有解决“各单元完全相同”的问题，并且可能放大输出。

### Q30：为什么 `len(layer.parameters())` 是 2，参数量却是 9？

**一个是在数装参数的对象，另一个是在数对象里有多少个标量。**

对于 `Linear(2, 3)`：

| 参数对象 | 形状 | 包含多少个数 |
| --- | --- | ---: |
| `weight` | `(2, 3)` | 6 |
| `bias` | `(3,)` | 3 |
| 合计 | 2 个 Tensor 对象 | 9 个标量参数 |

类比 Java：一个 `List<float[]>` 有两个元素，不代表里面合计只有两个浮点数。

对 TinyTorch 参数列表，可以这样统计。下面是**教学片段**，假设 `layer` 已经存在：

```python
params = layer.parameters()
total = 0

for parameter in params:
    total = total + parameter.data.size

print(len(params))  # 参数 Tensor 对象的数量
print(total)        # 所有参数中的标量总数
```

这里 `parameter` 是 TinyTorch Tensor，所以通过 `.data.size` 数底层数组元素。

公式也可以直接算：

```text
带偏置：in_features × out_features + out_features
不带偏置：in_features × out_features
```

后面看到 `sum(p.data.size for p in params)` 时，可以把它展开成上面的循环。`p` 只是循环变量名，没有特殊魔法。

### Q31：`parameters()` 为什么必须返回原对象？前向时为什么不能重建 Linear？

因为训练程序必须更新**实际参与计算的那组参数**。

假设层使用对象 A 保存权重，你却复制一份对象 B 返回给优化器。即使 B 被更新，下一次前向仍然使用 A，就不能得到预期的训练效果。

因此，参数列表装的是对象引用。Python 中 `params[0] is layer.weight` 可以检查是否为同一个对象。

返回参数列表本身不等于完成梯度计算。以后看到 `requires_grad`，可以先理解为“是否需要跟踪梯度”的标志；它何时被启用、怎样产生梯度，属于后续模块的集成工作。

还有一个常见错误：

```python
# 反例：教学片段，不要这样组织持续训练的模型。
def predict(x):
    layer = Linear(2, 3)
    return layer(x)
```

每次进入 `predict()` 都创建新层，就会重新初始化参数。通常应先创建并保存模型，后续用它处理一批又一批数据。

这一点与 Java 中“长期持有一个有状态对象”和“每次请求都 new 一个对象”很相似。

---

<a id="dropout"></a>

## 五、Dropout：随机置零为什么有帮助？

### Q32：Dropout 解决什么问题？它会删除神经元吗？

它是一种缓解过拟合的手段。

**过拟合**可以先理解为：对见过的训练数据表现很好，对新数据表现却不好。

Dropout 在训练时让部分中间数值暂时不能使用，促使模型不要过度依赖少数信号。这个作用可能改善泛化，但并不保证一定提高效果。

本章是逐元素 Dropout：

- 改变的是这一次前向计算中的部分数值。
- 不会永久删除权重。
- 不会让数组少几列。
- 下一次调用会重新抽取掩码。

对于 `(2, 3)` 的输入，Dropout 的输出仍然是 `(2, 3)`。

### Q33：`p=0.5` 是丢掉一半的样本，还是一半的特征？

**这里表示每个元素独立地以 50% 的概率被置零。**

它不是按样本整行删除，也不要求整批共享同一列的开关。其他类型的 Dropout 可以有不同规则，但不是本章的重点。

用 ReLU 后的结果做一次示意：

```text
输入 h = [[5, 2, 1],
          [5, 0, 5]]

某次二值掩码 m = [[1, 0, 1],
                 [0, 1, 1]]
```

`1` 表示保留，`0` 表示丢弃。先逐元素相乘，再除以保留概率 `0.5`：

```text
h * m / 0.5 = [[10, 0,  2],
               [0, 0, 10]]
```

这个掩码是人为给定的演示，不是在承诺随机运行会得到它。

注意第二行第二个位置：掩码虽然保留它，但输入本来就是 0，输出仍然是 0。所以**不能对任意输入只数输出的零，就断言 Dropout 实际丢弃了多少项**。

### Q34：为什么保留下来的数还要乘 2？这样不会变大吗？

**单次被保留时会变大；目的是让反复随机实验的平均值保持不变。**

先只看一个输入 `x=6`，丢弃概率 `p=0.5`：

| 发生的情况 | 概率 | 不缩放时输出 | 缩放后输出 |
| --- | ---: | ---: | ---: |
| 被丢弃 | 0.5 | 0 | 0 |
| 被保留 | 0.5 | 6 | 12 |

不缩放时，期望输出是：`0×0.5 + 6×0.5 = 3`。

缩放后，期望输出是：`0×0.5 + 12×0.5 = 6`。

“期望”就是按每种结果的概率做加权平均，并不是每一次运行的实际结果。

一般情况下，保留概率是 `1-p`，所以：

```text
训练输出 = 输入 × 二值掩码 / (1-p)
```

再换一个数检验：`x=6`、`p=0.25` 时，保留概率为 `0.75`，保留后输出 `6/0.75=8`。于是 `0×0.25+8×0.75=6`。

这叫 inverted dropout，即把补偿缩放放到训练阶段。它保持的是**固定输入下，本层输出的期望**，不保证整个后续非线性网络的平均预测也完全相同。

### Q35：为什么推理时关闭 Dropout？还需要乘回 0.5 吗？

普通推理时，我们希望使用完整特征，并避免这一步额外随机改变结果。

本章已经在训练时做了 `1/(1-p)` 的缩放，因此推理时直接返回输入，**不用再乘 `1-p`，也不用继续放大**。

| 情况 | 本章行为 |
| --- | --- |
| `training=True`，且 `0<p<1` | 随机置零，保留项除以 `1-p` |
| `training=False` | 原样返回输入 |

TinyTorch 本章通过调用参数显式选择模式，例如下面的**教学片段**：

```python
train_result = dropout(x, training=True)
eval_result = dropout(x, training=False)
```

不要仅因为熟悉其他框架，就直接假定本章有 `model.eval()`。

另外，`training=True` 只选择层的行为，不会替你计算损失、求梯度或更新参数。

### Q36：随机掩码的代码怎么读？

下面是**可独立运行**的 NumPy 演示，用固定“随机数”帮助你对照中间结果：

```python
import numpy as np

p = 0.5
keep_prob = 1.0 - p

numbers = np.array([0.2, 0.8, 0.4, 0.9], dtype=np.float32)
keep = numbers < keep_prob
binary_mask = keep.astype(np.float32)
scaled_mask = binary_mask / keep_prob

print(keep)         # [ True False  True False]
print(binary_mask)  # [1. 0. 1. 0.]
print(scaled_mask)  # [2. 0. 2. 0.]
```

理解过程：

1. 每个位置拿到一个 `[0,1)` 区间的数。
2. 小于 `keep_prob`，这个位置就保留。
3. 比较结果是布尔数组，含 `True`、`False`。
4. 转成浮点数后，变成 `1.0`、`0.0`。
5. 除以 `keep_prob` 后，得到可以直接与输入相乘的缩放掩码。

真实随机抽样时，可用 `rng.random(x.shape)` 生成 `numbers`。当保留概率为 0.5 时，均匀随机数落在 `[0,0.5)` 的概率就是 0.5。

所以教材里可能出现两种掩码：**二值掩码是 0/1，缩放掩码是 0/补偿系数。** 看到 `mask` 这个变量名时，要看它是否已经包含缩放。

### Q37：`p=0`、`p=1`、`p=1.5` 应该怎么办？

| `p` 与模式 | 行为 | 原因 |
| --- | --- | --- |
| `p=0`，训练 | 原样返回 | 没有元素需要丢弃 |
| `0<p<1`，训练 | 随机置零并缩放 | 正常 Dropout |
| `p=1`，训练 | 返回同形状全零 | 所有元素都丢弃，需单独处理以免除以 0 |
| 合法 `p`，推理 | 原样返回 | 推理关闭 Dropout，包括 `p=1` 的情况 |
| `p<0` 或 `p>1` | 报输入错误 | 不是合法概率 |

Q34 的保持期望推导要求 `p<1`。`p=1` 是全零边界，不能把 `1/(1-p)` 硬套进去。

教材描述的推理与 `p=0` 分支会直接返回输入对象。因此“没有改数值”不代表“创建了一份数据副本”。

### Q38：为什么每次结果不同？设了随机种子就永远一样了吗？

**同一个随机数生成器在每次抽样后会推进内部状态，所以连续调用通常得到不同掩码。**

下面是**可独立运行**的实验：

```python
import numpy as np

first_rng = np.random.default_rng(7)
a = first_rng.random(4)
b = first_rng.random(4)

second_rng = np.random.default_rng(7)
c = second_rng.random(4)

print(np.allclose(a, c))  # True：相同起点、相同首次调用
print(np.allclose(a, b))  # False：同一个生成器的两次连续抽样
```

`7` 是种子，你可以把它理解为某条伪随机序列的起点。在相同实现和调用条件下，用相同种子重新创建生成器，便于重现实验。[NumPy 随机数生成器](https://numpy.org/doc/stable/reference/random/generator.html)

两个常见坑：

- 不要在每次 `forward()` 里重新用同一个种子创建生成器，否则可能反复使用同一个掩码。
- `np.random.seed(...)` 管理的是另一套旧式随机接口，不会重置已经创建的 `default_rng(...)` 对象。

小数组连续两次抽到相同掩码也完全可能。不能把“每次输出都必须不一样”写成必定通过的测试。

### Q39：Dropout 没有参数，那它怎样帮助模型“学”？

**它通过改变训练时流过网络的数据，影响其他层参数的更新过程。**

它本身不需要学习一组权重，也不会在 `parameters()` 中返回 `p` 或随机掩码。

可以类比练习时随机遮住部分提示：提示的遮挡规则不是你要学的答案，但会改变你练习的方式。

Dropout 的输入和输出仍然参与网络计算。以后接入自动求导时，常规掩码分支应通过 Tensor 运算保持计算关系，而不是随意取出 `.data` 算完后重新包装，就认为梯度关系仍然自动存在。

当前先理解前向行为即可；第七部分的 NumPy 实验没有自动求导，也不会执行参数学习。

---

<a id="sequential"></a>

## 六、Sequential：把层连起来，而不是再发明一种计算

### Q40：不用 Sequential，我能先把网络写出来吗？

能，而且这是初学时最值得先写的方式。

下面是**TinyTorch 教学片段**，假设类已正确导入，`x` 是形状 `(2,2)` 的 Tensor：

```python
first = Linear(2, 3)
activation = ReLU()
dropout = Dropout(0.5)
last = Linear(3, 1)

h1 = first(x)
h2 = activation(h1)
h3 = dropout(h2, training=False)
result = last(h3)
```

| 步骤 | 输入形状 | 输出形状 |
| --- | --- | --- |
| 第一个 Linear | `(2,2)` | `(2,3)` |
| ReLU | `(2,3)` | `(2,3)` |
| Dropout | `(2,3)` | `(2,3)` |
| 第二个 Linear | `(2,3)` | `(2,1)` |

这里新建的权重是随机的；想得到 Q44 中的确定结果，需要先手动设置我们约定的权重，完整操作在第七部分。

### Q41：Sequential 里面究竟做了什么？

**保存层的顺序，再把上一步的输出交给下一步。**

核心是这样的循环。这是**教学片段**，暂时省略训练模式传递：

```python
for layer in self.layers:
    x = layer.forward(x)
return x
```

其中 `x = ...` 表示变量名现在指向这一层的结果，不代表一定在原输入数组上修改。

把循环展开，就是 Q40 那种手动调用。Sequential 不会自动训练、不自动选择层的宽度，也不会替你修好不匹配的形状。

对于 `Linear(2,3) → ReLU → Dropout → Linear(3,1)`，它收集的标量参数总数为：

```text
第一个 Linear：2×3 + 3 = 9
第二个 Linear：3×1 + 1 = 4
合计：13
```

ReLU、Dropout 和容器本身都没有额外增加可学习标量。

### Q42：为什么收集参数时用 `extend()`，不能直接用 `append()`？

因为一层返回的是一个列表，整个网络通常需要把每层列表“摊平”成一个列表。

用名字字符串模拟参数引用，这是**可独立运行**的语法实验：

```python
first_params = ["W1", "b1"]
last_params = ["W2", "b2"]

nested = []
nested.append(first_params)
nested.append(last_params)

flat = []
flat.extend(first_params)
flat.extend(last_params)

print(nested)
print(flat)
```

输出：

```text
[['W1', 'b1'], ['W2', 'b2']]
['W1', 'b1', 'W2', 'b2']
```

真正的参数收集用 Tensor 对象替代这些字符串。

教材描述的简单容器不自动去重。如果你把同一个有参数的层对象加入两次，列表可能重复包含同一参数对象。初学时先使用各自独立的层对象。

### Q43：有些层没有 `training` 参数，Sequential 怎么统一调用？

你上传的教材描述了一种兼容方式：先尝试传入 `training`，遇到 `TypeError` 后再尝试不传。查阅时的项目源码也包含这一逻辑，并提供容器的 `training` 参数；官网展示的某些简化片段没有展开它。[TinyTorch 第 03 章源码](https://github.com/harvard-edge/cs249r_book/blob/main/tinytorch/src/03_layers/03_layers.py)

因此，阅读项目时要检查实际方法签名，不能只看网页上的简化示例。

这个兼容方式还有一个调试细节：层内部其他原因产生的 `TypeError` 也可能被捕获，容易让错误定位不直观。

**为了让完整实验容易读懂，本文的 Demo 类统一接受 `training` 参数。** Linear 和 ReLU 收到后不改变行为；Dropout 根据它切换行为。这样容器就能直接逐层传递，不需要异常重试。

这是一项明确的教学简化，不是在声称你的 TinyTorch 版本已经这样实现。

### Q44：能把整个小网络最后一个输出也手算出来吗？

继续使用前面的确定数值，当前在推理模式，Dropout 原样传递。

第一个 Linear 与 ReLU 之后：

```text
h = [[5, 2, 1],
     [5, 0, 5]]
```

第二个 Linear 为 `Linear(3,1)`，我们指定：

```text
W2 = [[ 1.0],
      [-1.0],
      [ 0.5]]

b2 = [0.5]
```

| 样本 | 最后一层计算 | 输出 |
| --- | --- | ---: |
| A | `5×1 + 2×(-1) + 1×0.5 + 0.5` | 4 |
| B | `5×1 + 0×(-1) + 5×0.5 + 0.5` | 8 |

最终结果：

```text
[[4.0],
 [8.0]]
```

形状是 `(2,1)`：两条样本，每条得到一个输出。

这个例子只证明我们的计算链路正确。权重是人为指定的，所以不能把这些数字当成真实的业务预测效果。

---

<a id="experiment"></a>

## 七、完整实验：不依赖 TinyTorch 导出状态，也能运行

### Q45：我没有 Python 基础，怎样运行下面的完整代码？

先新建文件 `layers_learning_demo.py`，把 Q46 中的**一个完整 Python 代码块**复制进去保存。

在这个文件所在目录打开终端，执行：

```bash
python3 layers_learning_demo.py
```

如果你的环境使用的是 `python` 命令，就把上述 `python3` 换成 `python`。已经启用了 TinyTorch 虚拟环境时，继续使用该环境即可。

如果明确提示缺少 NumPy，再在同一个环境执行：

```bash
python3 -m pip install numpy
```

这份程序只做前向计算，使用 NumPy 数组直接保存权重。类名带 `Demo`，方便与你项目里的正式类区分。

| 这个实验 | TinyTorch 教材 |
| --- | --- |
| `DemoLinear` | `Linear` |
| `DemoDropout` | `Dropout` |
| `DemoSequential` | `Sequential` |
| 参数是 NumPy 数组 | 参数是 Tensor，底层保存数值数组 |
| 参数元素数写 `parameter.size` | 按教材写 `parameter.data.size` |
| Linear 用 `x @ weight` | 教材用 `x.matmul(weight)` |
| 全部 Demo 层接受 `training` | 项目中需要适配不同层的接口 |
| Demo 容器继承 `DemoLayer`，复用调用入口 | 教材中的容器自行实现相同接口 |

这个实验的容器只演示 `DemoSequential(layer1, layer2, ...)`，不额外兼容传入单个列表。它不包含自动求导和优化器，也不是要覆盖你的项目源码。

### Q46：完整代码是什么？运行后应该核对哪些结果？

下面这段代码可以完整复制运行。

```python
# 完整实验：保存为 layers_learning_demo.py
import numpy as np


# 只在这里创建一次生成器，后续抽样会推进它的状态。
rng = np.random.default_rng(7)


class DemoLayer:
    def forward(self, x, training=True):
        raise NotImplementedError("子类需要实现 forward")

    def __call__(self, x, training=True):
        return self.forward(x, training=training)

    def parameters(self):
        return []


class DemoLinear(DemoLayer):
    def __init__(self, in_features, out_features, bias=True):
        if in_features <= 0 or out_features <= 0:
            raise ValueError("输入、输出特征数必须为正整数")

        self.in_features = in_features
        self.out_features = out_features

        std = 1.0 / np.sqrt(in_features)
        values = rng.standard_normal((in_features, out_features))
        self.weight = (values * std).astype(np.float32)

        self.bias = None
        if bias:
            self.bias = np.zeros(out_features, dtype=np.float32)

    def forward(self, x, training=True):
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError(
                f"需要形状 (样本数, {self.in_features})，收到 {x.shape}"
            )

        result = x @ self.weight
        if self.bias is not None:
            result = result + self.bias
        return result

    def parameters(self):
        items = [self.weight]
        if self.bias is not None:
            items.append(self.bias)
        return items


class DemoReLU(DemoLayer):
    def forward(self, x, training=True):
        # 对每个位置，取输入与 0 中较大的那个。
        return np.maximum(x, 0.0)


class DemoDropout(DemoLayer):
    def __init__(self, p=0.5):
        if not 0.0 <= p <= 1.0:
            raise ValueError("p 必须在 0 到 1 之间")
        self.p = p

    def forward(self, x, training=True):
        if not training or self.p == 0.0:
            return x

        if self.p == 1.0:
            return np.zeros_like(x)

        keep_prob = 1.0 - self.p
        numbers = rng.random(x.shape)
        binary_mask = (numbers < keep_prob).astype(np.float32)
        scaled_mask = binary_mask / keep_prob
        return x * scaled_mask


class DemoSequential(DemoLayer):
    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x, training=True):
        for layer in self.layers:
            x = layer(x, training=training)
        return x

    def parameters(self):
        all_parameters = []
        for layer in self.layers:
            all_parameters.extend(layer.parameters())
        return all_parameters


def main():
    x = np.array([[1, 2], [3, 1]], dtype=np.float32)

    first = DemoLinear(2, 3)
    activation = DemoReLU()
    dropout = DemoDropout(0.5)
    last = DemoLinear(3, 1)

    # 为了核对手算，先用固定数值替换初始随机参数。
    # 正常训练由优化器更新参数；本实验没有执行训练。
    first.weight = np.array([[1, -1, 2], [2, 1, 0]], dtype=np.float32)
    first.bias = np.array([0, 1, -1], dtype=np.float32)
    last.weight = np.array([[1], [-1], [0.5]], dtype=np.float32)
    last.bias = np.array([0.5], dtype=np.float32)

    model = DemoSequential(first, activation, dropout, last)

    h1 = first(x)
    h2 = activation(h1)
    h3 = dropout(h2, training=False)
    result = last(h3)
    container_result = model(x, training=False)

    print("Linear1:")
    print(h1)
    print("ReLU:")
    print(h2)
    print("Output:")
    print(result)

    parameters = model.parameters()
    total = 0
    for parameter in parameters:
        total = total + parameter.size

    print("Parameter objects:", len(parameters))
    print("Parameter scalars:", total)

    # 核对人工推导、形状、组合行为和参数引用。
    expected = np.array([[4], [8]], dtype=np.float32)
    assert result.shape == (2, 1)
    assert np.allclose(result, expected)
    assert np.allclose(container_result, expected)
    assert len(parameters) == 4
    assert total == 13
    assert parameters[0] is first.weight

    # 核对 Dropout 的几个确定性边界。
    assert dropout(h2, training=False) is h2
    assert DemoDropout(0.0)(h2, training=True) is h2
    assert np.allclose(DemoDropout(1.0)(h2, training=True), 0.0)
    assert DemoDropout(1.0)(h2, training=False) is h2

    print("Checks passed.")
    print("One training-mode output (random dropout):")
    print(model(x, training=True))


if __name__ == "__main__":
    main()
```

重点核对下面这一段输出。不同显示环境中的空格可能不同，数值应该一致：

```text
Linear1:
[[ 5.  2.  1.]
 [ 5. -1.  5.]]
ReLU:
[[5. 2. 1.]
 [5. 0. 5.]]
Output:
[[4.]
 [8.]]
Parameter objects: 4
Parameter scalars: 13
Checks passed.
```

最后还会打印一次训练模式的输出。它受到随机掩码影响，不要拿它与上面固定的推理输出强行比较是否相等。

这里新出现的几个函数：

| 函数 | 用途 |
| --- | --- |
| `np.maximum(x, 0.0)` | 逐元素实现 ReLU |
| `np.zeros_like(x)` | 创建与 `x` 形状、类型一致的全零数组 |
| `np.allclose(a, b)` | 允许很小的浮点误差，比较数值是否接近 |
| `list(layers)` | 把收到的多个层对象整理成列表 |
| `print(...)` | 打印中间结果，方便观察 |

### Q47：跑通以后，我应该改哪些地方，才能确认自己真的理解了？

每次只改一处，先预测现象，再运行。修改实验时，原有断言可能不再适用；不要为了消除报错而无条件删除它们，要先解释预期值为什么变化。

| 小实验 | 先预测什么 | 可以核对的结果 |
| --- | --- | --- |
| 给 `x` 增加一行 `[1,1]` | 样本数变了，参数量变不变？ | 参数仍是 13；新增样本推理输出为 3 |
| 把第二行输入改为 `[0,0]` | 第一层是不是输出全零？ | 第一层输出偏置 `[0,1,-1]`；最终输出为 `-0.5` |
| 把 `DemoDropout(0.5)` 改为 `DemoDropout(0.0)` | 训练与推理路径是否还受 Dropout 随机性影响？ | 同一参数、同一输入下，两条路径输出相同 |
| 从容器中移除 ReLU | 哪条样本会发生变化？ | A 仍为 4，B 从 8 变成 9 |
| 把第一层的输入宽度改为 4，输入仍是两列 | 是否能继续计算？ | 触发输入形状检查；别忽略手动参数覆盖步骤 |

最后一个实验中，程序手动设置的权重仍是 `(2,3)`。这提醒你：**层的配置、参数形状和输入形状必须保持一致**，不能只改一个数字就认为网络已经正确扩展。

如果要认真练习关闭偏置，建议单独新建 `DemoLinear(2,3,bias=False)`，先只检查参数量为 6。不要保留实验程序中随后给 `first.bias` 重新赋数组的语句，否则又人为加回了偏置。

---

<a id="debugging"></a>

## 八、遇到错误，先分清是语言、形状还是项目环境

### Q48：遇到这些 Python 错误，我先检查哪里？

| 报错或现象 | 常见原因 | 第一检查点 |
| --- | --- | --- |
| `IndentationError` | 缩进层级不正确 | `class`、`def`、`if`、`for` 后面的代码是否正确缩进 |
| `NameError: name 'np' is not defined` | 只复制了后半段代码 | 是否先执行 `import numpy as np` |
| `NameError: name 'Tensor' is not defined` | TinyTorch 片段缺少导入或环境 | 当前复制的是教学片段，还是完整 NumPy 实验？ |
| `ModuleNotFoundError: No module named 'numpy'` | 当前解释器没有 NumPy | 安装和运行是否使用同一个 Python 环境 |
| `ModuleNotFoundError: No module named 'tinytorch'` | 项目包没有正确安装或解释器不对 | 是否进入项目所用的虚拟环境 |
| `AttributeError: ... has no attribute 'shape'` | 传入了普通列表等对象 | 完整实验需要 `np.array(...)`，TinyTorch 示例需要 Tensor |
| 对象不能调用，提示 `not callable` | 没有 `__call__`，或变量被别的值覆盖 | 该对象的类和调用入口是什么？ |
| `unexpected keyword argument 'training'` | 当前方法不接受该关键字参数 | 检查实际版本的 `forward()` 或 `__call__()` 签名 |

排查顺序：**先看报错的最后一行，再找自己代码对应的位置，最后确认变量类型和形状。**

“程序报错”不一定意味着你没懂神经网络，有时只是执行了不完整的代码片段。

### Q49：形状不匹配时，我应该怎样定位？

把每一步输出形状写出来，重点看相邻 Linear 的宽度是否接得上。

例如：

```python
# 反例：第一层输出 3 个特征，第二层却要求 4 个。
first = Linear(2, 3)
last = Linear(4, 1)
```

如果输入是 `(8,2)`：

| 步骤 | 形状 |
| --- | --- |
| 第一层输入 | `(8,2)` |
| 第一层输出 | `(8,3)` |
| 第二层权重 | `(4,1)` |
| 失败的计算 | `(8,3) @ (4,1)` |

问题在于中间两个维度 `3` 和 `4` 不相等。

如果你希望第一层输出 3 个特征，那么后一层应匹配 `Linear(3,1)`。

一般二维规则：

```text
(样本数, 输入宽度) @ (输入宽度, 输出宽度)
得到 (样本数, 输出宽度)
```

不要为了让报错消失就随意转置输入。转置有时能“凑对形状”，却可能把样本轴和特征轴的业务含义颠倒。

### Q50：教材、网页、本地 `.py` 文件不完全一样，我应该以谁为准？

**理解概念可以相互参考；判断运行行为，要以当前解释器实际导入的代码为准。**

你可以在已配置好的 TinyTorch 环境中运行下面的**排查片段**：

```python
import inspect
import tinytorch.core.layers as layers_module

print(layers_module.__file__)
print(inspect.signature(layers_module.Sequential.forward))
```

第一行输出模块实际所在的文件，第二行查看当前方法支持哪些参数。只有模块成功导入后，这段检查才能继续执行。

目前已核对的几处差别，阅读时这样处理：

| 容易混淆的地方 | 本文处理方式 |
| --- | --- |
| 网页图示笼统说容器继承 Layer | 教材与查阅到的源码中 `Sequential` 按相同接口工作，并未继承 Layer |
| 网页的容器简化片段未传递 `training` | 实际查阅到的源码含模式传递；本地是否如此仍应检查版本 |
| 网页片段使用旧式 NumPy 随机接口 | 教材描述和查阅到的源码使用独立 `rng`；本文练习也使用 `default_rng` |
| 中文文件已经包含实现 | 不等于它已导出到 `tinytorch.core.layers` |

这些项目实现差别可对照 [当前源码](https://github.com/harvard-edge/cs249r_book/blob/main/tinytorch/src/03_layers/03_layers.py)；`main` 分支后续可能变化。

你提供的讲义提到 `tito module complete 03`。它具体读取哪份源码，需以你安装版本的工具配置为准；不要假定它会自动选择 `_zh.py`。

---

<a id="advanced"></a>

## 九、第二遍再看：初始化、内存与运算量

### Q51：LeCun、Xavier、He 三种初始化，现在怎么区分？

先记住“它们在控制随机权重的规模”，再看采用什么标准差。

下面只比较常见的**正态分布版本**，不能直接拿这些标准差当成均匀分布的上下界。

| 名称 | 标准差形式 | 本阶段的记法 |
| --- | --- | --- |
| LeCun 风格 | `sqrt(1/fan_in)` | 根据输入数量缩放；本章教材采用这一形式 |
| Xavier / Glorot | `sqrt(2/(fan_in+fan_out))` | 同时考虑输入和输出数量 |
| He | `sqrt(2/fan_in)` | 常与 ReLU 一类激活配合 |

在本章 Linear 中，`fan_in` 就是输入特征数，`fan_out` 就是输出特征数。

进阶直觉：设每个输出是多个 `输入×权重` 之和。在输入和权重近似独立、均值与方差满足相应假设时，求和项增多会增加输出方差，缩小权重方差可以进行补偿。

第一遍不需要证明这套概率计算，只要别把本章的 `sqrt(1/in_features)` 误称为 He 初始化即可。

### Q52：参数内存怎么估算？batch 增大后，什么会变大？

如果参数全部是 float32，先用：

```text
参数数据字节数 = 标量参数总数 × 4
```

我们的 `2→3→1` 网络有 13 个标量参数，参数数据为 `13×4=52` 字节。

这不是整个 Python 程序只用 52 字节。对象、数组头部、运行环境和中间结果都需要额外内存。

换成你教材中的网络 `784→256→128→10`：

| 层 | 权重数 | 偏置数 | 合计 |
| --- | ---: | ---: | ---: |
| `Linear(784,256)` | 200,704 | 256 | 200,960 |
| `Linear(256,128)` | 32,768 | 128 | 32,896 |
| `Linear(128,10)` | 1,280 | 10 | 1,290 |
| 合计 | 234,752 | 394 | 235,146 |

参数数据大小：`235146×4=940584` 字节，约 `918.54 KiB`，也就是约 `0.941 MB`。

这里 `1 KiB=1024` 字节，`1 MB=1,000,000` 字节，不能混用换算方式。

batch 从 1 增加到 32 时：

- 权重和偏置共享，参数数量不变。
- 输入、输出、中间激活、Dropout 掩码的元素数通常随 batch 增长。
- 训练还可能保存梯度、计算图与优化器状态。

教材中的 Dropout 可能先产生布尔掩码，再转成 float32 缩放掩码。因此不能看到“布尔值”就断言所有掩码内存都只有每元素 1 字节，也不能用一个固定倍率代表真实峰值。

### Q53：为什么矩阵乘法常估算为 `2×batch×in×out` 次运算？

**因为每个输出要做多次乘法，以及把结果相加。**

先数一个输出。如果它接收 `in` 个输入：

- 有 `in` 次乘法。
- 把这 `in` 项求和，需要 `in-1` 次加法。

所以不含偏置时，按这种直接计数方式，一个输出需要 `2×in-1` 次标量运算。

一共有 `batch×out` 个输出，因此矩阵乘法是：

```text
batch × out × (2×in - 1)
```

工程估算常忽略小的差值，记成：

```text
约 2 × batch × in × out FLOPs
```

FLOP 表示一次浮点运算；一次乘法加一次加法，常按 2 FLOPs 统计。MAC 是乘加计数，通常按 1 MAC 对应约 2 FLOPs 理解。两者不是同一个计数单位。

现在算 `100` 条样本、特征宽度 `50→20→5`：

| 层 | 输出有几个数 | 每个输出约多少次运算 | 矩阵乘法近似运算量 |
| --- | ---: | ---: | ---: |
| `50→20` | `100×20=2000` | `2×50=100` | 200,000 |
| `20→5` | `100×5=500` | `2×20=40` | 20,000 |
| 合计 | — | — | 220,000 |

也就是约 22 万次，并不是有 22 万次矩阵乘法。

若坚持按“每次点积 `in-1` 次加法”的算术方式精确计数，两次矩阵乘法合计为 `198000+19500=217500`；若两层都带偏置，再加 `2000+500=2500` 次加法，总计恰好是 `220000`。

这些都是数学工作量的统计方式，不是 CPU/GPU 实际指令数，也不能直接换算成固定毫秒数。

### Q54：`Linear→ReLU→Dropout` 的顺序，换了就一定不同吗？

不能笼统地说“一定”。要看具体操作。

对同一个非负缩放掩码 `m`，ReLU 有：

```text
ReLU(m×x) = m×ReLU(x)
```

例如 `x=-3`、`m=2`：先缩放再 ReLU 是 0，先 ReLU 再缩放也是 0。`x=3` 时两边都得到 6。

因此，在这个局部计算里，如果用的是同一个掩码，ReLU 和逐元素 Dropout 可以得到相同结果。两次独立抽样的掩码不同，则输出可能不同。

换成 Sigmoid 就不能直接套用：

- 先 Dropout 把某项置零，再 Sigmoid：`sigmoid(0)=0.5`。
- 先 Sigmoid，再用零掩码丢弃：输出是 0。

这题的学习价值是：**判断代码行为时，先看具体公式和条件，不要只记一个绝对化的层顺序口号。**

---

<a id="practice"></a>

## 十、闭卷练习：先写答案，再向下核对

这些都是开放式练习，目的不是背定义，而是检查你能否自己解释和计算。

### 练习 A：只看形状

输入 `x` 的形状为 `(4,3)`，层是带偏置的 `Linear(3,2)`。

请写出：权重形状、偏置形状、输出形状、参数 Tensor 个数、标量参数总数。

### 练习 B：只算一条样本

给定：

```text
x = [[2, -1]]
W = [[1, 2],
     [3, 4]]
b = [0.5, -0.5]
```

先算 Linear 输出，再算 ReLU 输出。

### 练习 C：改变 batch

把练习 A 的输入从 `(4,3)` 改成 `(40,3)`。

权重形状、标量参数量、输出形状分别怎样变化？

### 练习 D：解释 Python 对象调用

别人问你：“`layer` 明明是对象，为什么可以写 `layer(x)`？”

请用 `__call__`、`forward` 两个词，完整解释调用过程。

### 练习 E：算 Dropout 的两个可能结果

固定输入 `x=6`，`p=0.25`。训练时可能输出什么？各自概率是多少？期望是多少？推理时输出多少？

### 练习 F：找连接错误

网络是：`Linear(6,4) → ReLU → Linear(5,2)`。

错误在哪里？如果保留第一层结构不变，怎样调整最后一层？

### 练习 G：参数列表和标量参数

`Linear(4,3) → ReLU → Dropout(0.2) → Linear(3,2,bias=False)`。

假设两层独立、没有共享参数，整个网络有几个参数 Tensor？共多少个可学习标量？

### 练习 H：判断模型是否已经学习

你把 `model(x, training=True)` 放进循环，执行了 1000 次。结果会随机变化，但没有损失函数、求梯度和优化器更新。

能否因此断言模型学到了知识？为什么？

### 参考答案

| 练习 | 答案与原因 |
| --- | --- |
| A | `W=(3,2)`，`b=(2,)`，输出 `(4,2)`；2 个参数 Tensor，`3×2+2=8` 个标量 |
| B | 第一项 `2×1+(-1)×3+0.5=-0.5`；第二项 `2×2+(-1)×4-0.5=-0.5`。Linear 输出 `[[-0.5,-0.5]]`，ReLU 输出 `[[0,0]]` |
| C | 权重仍是 `(3,2)`，参数仍是 8；输出变为 `(40,2)` |
| D | 实例调用语法进入类提供的 `__call__()`，本章在其中把输入继续交给 `forward()` |
| E | 25% 概率输出 0，75% 概率输出 8；期望为 6；推理直接输出 6 |
| F | 第一层输出 4 个特征，最后一层却要求 5 个；调整为 `Linear(4,2)` |
| G | 第一层 2 个参数 Tensor，第二层仅权重 1 个，总计 3 个；标量数 `4×3+3+3×2=21` |
| H | 不能。Dropout 的随机性会改变前向结果，但参数没有因学习目标被更新 |

---

<a id="checklist"></a>

## 十一、返回源码时，带着这张检查表

遇到任意一个层，都按相同顺序询问：

1. **输入是什么？** 类型、形状、每个轴的含义是什么？
2. **保存了什么？** 哪些是模型参数，哪些只是配置？
3. **怎样计算？** 对应什么数学公式？
4. **输出是什么？** 形状、数值含义是否符合预期？
5. **怎样组合？** 下一个层能否接住它的输出？
6. **怎样训练？** 参数是否被正确收集，是否需要区分训练和推理？

学习完成后，你应该能不看原文回答：

- [ ] `Linear(8,4)` 中，8 和 4 各表示什么？
- [ ] 为什么权重是 `(8,4)`，偏置是 `(4,)`？
- [ ] 为什么不同 batch 共享同一套权重？
- [ ] 为什么多层 Linear 中间通常需要非线性？
- [ ] `self`、`__init__`、`__call__`、`forward` 分别做什么？
- [ ] 为什么两个参数 Tensor 可能包含几千个标量？
- [ ] 为什么 Dropout 不减少形状，也不减少参数量？
- [ ] 为什么训练时要除以 `1-p`，推理时不用再缩放？
- [ ] 为什么 Sequential 不会自动让模型学会任务？
- [ ] 为什么改中文教学文件，不一定改变实际导入的软件包？

如果这些问题基本能解释，下一章学习损失函数时，就可以围绕一个新问题继续：**“已经算出了预测，怎样用一个数衡量它与正确答案之间的差距？”**

## 常用词汇速查

| 英文或代码名 | 中文 | 在本章中的具体含义 |
| --- | --- | --- |
| layer | 层 | 一个可组合的数据处理组件 |
| linear / dense / fully connected | 线性层 / 稠密层 / 全连接层 | 本章实现 `xW+b` 的组件 |
| feature | 特征 | 每条样本中的一个数值维度 |
| batch | 批次 | 一起处理的一组样本 |
| weight | 权重 | 输入参与输出组合时的系数 |
| bias | 偏置 | 每个输出位置的额外平移量 |
| forward pass | 前向计算 | 使用当前参数计算结果 |
| activation | 激活；也可指中间输出 | 结合上下文分清激活函数和激活数值 |
| parameter | 模型参数 | 后续训练需要调整的数值 |
| hyperparameter | 超参数 | 人为设置或通过实验选择的配置 |
| dropout | 随机丢弃 | 本章训练时对部分元素置零的操作 |
| mask | 掩码 | 指定哪些位置保留，以及可能包含怎样的缩放 |
| inference | 推理 | 使用模型进行预测 |
| regularization | 正则化 | 用来约束学习、帮助改善泛化的一类方法 |
| overfitting | 过拟合 | 训练数据上好，新数据上差 |
| generalization | 泛化 | 模型处理未见数据的能力 |
| initialization | 初始化 | 在学习开始前设置参数起点 |
| logits | 类别原始分数 | 分类模型中尚未归一化为概率的分数 |
| sequential | 按顺序组合 | 将多个层依次连接的容器 |

## 资料范围与核对说明

- **主要学习材料**：你上传的《粘贴的 markdown (1)。md》，包括 Linear、Dropout、Layer、Sequential、参数统计和系统分析的中文说明。
- **项目核对**：[TinyTorch Layers 教程](https://mlsysbook.ai/tinytorch/modules/03_layers.html) 与 [第 03 章源码](https://github.com/harvard-edge/cs249r_book/blob/main/tinytorch/src/03_layers/03_layers.py)。网页和远程源码是查阅时的内容，不代表你本地检出版本完全相同。
- **语言与数组语法**：相关位置已链接 Python 与 NumPy 官方文档。文中的 Java 类比只用于理解，不表示两种语言的机制完全一致。
- **验证范围**：已在 Python 3.12.14、NumPy 2.3.5 环境运行完整实验与 8 段可独立运行的小示例，并核对手算结果、练习关键数值、参数统计及 Dropout 边界；没有在你的本地 TinyTorch 项目上执行导出、训练或官方集成测试。
