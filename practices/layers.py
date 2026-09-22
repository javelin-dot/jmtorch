# 阅读路线：先看 Layer 规定“层”怎样接收输入，再看 Linear 和 Dropout 怎样改变数字，最后看 Sequential 怎样串联各层。
# Python 速查：= 是把右边的值交给左边的名字；== 才是比较。冒号后的缩进行属于同一个代码块，缩进退回即结束代码块。
# class 定义一类对象；Linear(10, 5) 会创建一个实例。方法中的 self 指这个实例，self.weight 是它持有的权重，不是所有实例共享的全局变量。
# NumPy 的 ndarray 是存数字的数组；本项目 Tensor 把数组包起来，提供 shape、matmul 等接口，和 PyTorch 的 torch.Tensor 不是同一个类。
# shape=(32, 784) 表示“一批 32 条样本，每条 784 个数字”；一维 shape=(784,) 的末尾逗号表示只有一个维度。
# forward 只是让输入经过现有权重算出输出；学习还需要损失、梯度、优化器更新权重，这个文件没有实现完整训练。
"""03 网络层练习；复用张量与激活函数模块。"""

# time 中的计时器在后面的性能实验里使用。
import time

# as 给 NumPy 起短名 np；后面 np.sqrt、np.zeros 等都从这个库调用。
import numpy as np

# 固定随机数生成器的种子为 7：同样的调用顺序会产生可重现的随机序列，而非每次抽样都得到相同数字。
rng = np.random.default_rng(7)

# 从各自的定义模块导入，依赖关系直接对应前面两章。
# Python 缓存正常导入的模块，使层参数与外部输入使用同一个 Tensor 类。
from .tensor import Tensor
# ReLU 给网络加入非线性；Sigmoid 留给后续组合练习使用。
from .activations import ReLU, Sigmoid

# 大写通常用作常量名；当前用 1.0 / 输入宽度计算权重初始方差。
INIT_SCALE_FACTOR = 1.0
# 此常量虽然叫 HE_SCALE_FACTOR，却没有在本文件的权重初始化里使用，不能说当前 Linear 用了 He 初始化。
HE_SCALE_FACTOR = 2.0
# Dropout 的概率下界为 0：训练时任何位置都不丢弃。
DROPOUT_MIN_PROB = 0.0
# Dropout 的概率上界为 1：训练时所有位置都变成 0。
DROPOUT_MAX_PROB = 1.0


# def 定义函数，不会立刻运行函数体；* 之后的 data 只能按名字传入，如 _show("输入", x, data=False)。
def _show(label, tensor, *, data=True):
    # [] 创建列表；f"...{值}..." 会把花括号里的值转换为文字，四项分别显示形状、元素数、维度数、数据类型。
    parts = [f"shape={tensor.shape}", f"size={tensor.size}", f"ndim={tensor.ndim}", f"dtype={tensor.dtype}"]
    # if 只在条件为真时执行下面缩进的代码；data 默认 True，表示还要显示实际数字。
    if data:
        # insert(0, 值) 把元素插在列表最前面；.data 是 Tensor 包装的 NumPy 数组。
        parts.insert(0, f"data={tensor.data}")
    # join 用逗号连接多段字符串；print 只负责显示，不参与神经网络计算。
    print(f"   {label}：{', '.join(parts)}")


# 类是创建对象的说明书；Layer 给各类层规定共同接口，本身并未指定数学计算。
class Layer:
    # self 是调用它的层对象，x 是交给层的输入张量；具体层应自己写 forward。
    def forward(self, x):
        # raise 表示主动抛出错误；若直接使用基础 Layer，说明没有人实现“输入如何变输出”。
        raise NotImplementedError(
            # __class__.__name__ 取实际对象的类名；\n 是字符串里的换行符，f 前缀允许插入变量。
            f"forward() 未在 {self.__class__.__name__} 中实现\n"
            # 括号内相邻的两个字符串会自动拼接；这里继续写给开发者看的报错提示。
            f"  子类需要实现 forward()，定义输入如何被变换"
        )

    # __call__ 是 Python 特殊方法：对象写成 layer(x) 时会调用它。*args 收额外位置参数，**kwargs 收具名参数。
    def __call__(self, x, *args, **kwargs):
        # 星号在调用处会展开参数；return 把 forward 的结果交还调用者，于是 layer(x) 等价于走进该层的 forward。
        return self.forward(x, *args, **kwargs)

    # parameters 约定各层如何交出可学习的权重、偏置，方便以后交给优化器。
    def parameters(self):
        # 基础 Layer 没有可学习参数；[] 是空列表，不是一个数值为零的参数。
        return []

    # __repr__ 规定对象在调试打印时呈现的文字，不改变前向计算。
    def __repr__(self):
        # 返回如 Layer() 的字符串；此处括号属于字符串内容，不是在创建新对象。
        return f"{self.__class__.__name__}()"


# class Linear(Layer) 中的括号表示继承，Linear 可复用 Layer.__call__，再实现自己的前向计算。
class Linear(Layer):
    # __init__ 在 Linear(784, 256) 创建实例时执行；bias=True 是默认启用偏置，可传 bias=False 关闭。
    def __init__(self, in_features, out_features, bias=True):
        # self.in_features 是对象属性，保存每条样本有多少个输入数字；右边是传进来的形参。
        self.in_features = in_features
        # 输出宽度决定每条样本最终得到多少个数字，并决定权重矩阵的列数。
        self.out_features = out_features

        # sqrt(1 / 输入宽度) 是当前采用的 LeCun 风格标准差；让随机权重随输入增多而适当缩小。
        scale = np.sqrt(INIT_SCALE_FACTOR / in_features)
        # 创建形状 (输入宽度, 输出宽度) 的正态随机数，并逐元素乘 scale；例如 (784, 256)。
        weight_data = rng.standard_normal((in_features, out_features)) * scale
        # 用项目的 Tensor 包装 NumPy 数组。其构造器会转为 float32；这个权重由当前 Linear 实例持有。
        self.weight = Tensor(weight_data)

        # if bias 判断开关是否为真；不同 Linear 实例可以选择是否有偏置。
        if bias:
            # 每个输出位置各有一个偏置，开始时全是 0；长度等于输出宽度，不依赖一次处理多少条样本。
            self.bias = Tensor(np.zeros(out_features))
        # else 表示 bias 为假时走另一条分支。
        else:
            # None 是“没有这个偏置对象”，不同于有一个数值全为 0 的偏置张量。
            self.bias = None

    # 前向公式 y = xW + b；例如 x 为 (32, 784)，W 为 (784, 256)，输出是 (32, 256)。
    # 手算一项：输入 [1, 2]、权重一列 [3, 4]、偏置 5，会得到 1*3 + 2*4 + 5 = 16。
    def forward(self, x):
        # matmul 是矩阵乘法；前一个矩阵的列数必须等于后一个的行数。batch 大小 32 不改变共享的权重。
        output = x.matmul(self.weight)
        # is not None 检查是否存在偏置对象；它不是“偏置数字是否非零”的比较。
        if self.bias is not None:
            # 广播让形状 (256,) 的 b 加到 (32, 256) 的每一行：32 条样本共用同一组偏置。
            output = output + self.bias
        # 把新计算出的 Tensor 交给下一层；没有调用反向传播，也没有更新权重。
        return output

    # 返回本层可以训练的参数对象；带偏置时是两个 Tensor，而非两个标量数字。
    def parameters(self):
        # 先以权重张量作为列表第一项；一个 (784, 256) 的 Tensor 内有 784*256 个数字。
        params = [self.weight]
        # 只有对象确实有偏置时才把它作为第二项交给后续训练代码。
        if self.bias is not None:
            # append 是向列表末尾加入“一项”；这里加入的是装有 out_features 个数字的整个 Tensor。
            params.append(self.bias)
        # return 返回对原参数对象的引用，并不会复制权重数字。
        return params

    # __repr__ 返回便于看清尺寸、是否启用偏置的文字。
    def __repr__(self):
        # 条件表达式 self.bias is not None 的结果为布尔值 True 或 False，f-string 把它显示成文本。
        bias_str = f", bias={self.bias is not None}"
        # 两个数是每条样本的输入/输出宽度，不是 batch 里有多少条样本。
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}{bias_str})"

# Dropout 用于训练时随机关闭一部分位置，减少模型对特定位置的依赖；推理时要关闭随机丢弃。
class Dropout(Layer):
    # p 是“丢弃概率”，默认 0.5；p 是手动设定的超参数，不是通过训练学习的权重。
    def __init__(self, p=0.5):
        # 链式比较 a <= p <= b 要求两边都成立；not 取反，只对超出 [0, 1] 的 p 报错。
        if not DROPOUT_MIN_PROB <= p <= DROPOUT_MAX_PROB:
            # ValueError 表示参数的值不合法；raise 会立即中断这次实例创建。
            raise ValueError(
                # f-string 中 {p} 插入本次实际传入的值，\n 表示显示时换行。
                f"非法 dropout 概率：{p}\n"
                # 括号中相邻的 f-string 会连成一段完整的报错消息。
                f"  p 必须在 {DROPOUT_MIN_PROB} 和 {DROPOUT_MAX_PROB} 之间\n"
                # p=0.5 指每个位置各有 50% 概率被丢弃，不保证一次调用恰好丢弃一半。
                f"  p=0.0 表示全部保留；p=0.5 随机丢掉一半；p=1.0 全部置零"
            )
        # 把合法概率存到当前实例，以供每次前向计算使用。
        self.p = p

    # 名称前缀 _ 表示“主要供类内部使用”，不是 Python 强制的访问限制。
    def _should_apply_dropout(self, training):
        # and 要求“处于训练模式”且“p 大于 0”；推理模式或者 p=0 都直接原样通过。
        return training and self.p > DROPOUT_MIN_PROB

    # 掩码是与输入逐元素相乘的系数，shape 必须与输入形状相同。
    # 每个输入位置独立抽签：丢弃填 0；保留填 1/(1-p)，用于维持单个位置输出的期望值。
    def _generate_dropout_mask(self, shape):
        # p 指丢弃的概率，1-p 才是留下的概率；例如 p=0.3 时 keep_prob=0.7。
        keep_prob = 1.0 - self.p
        # rng.random(shape) 产生 [0, 1) 的数；< 得到真假数组，astype 把 True/False 变为 float32 的 1/0。
        binary_mask = (rng.random(shape) < keep_prob).astype(np.float32)
        # 保留下来的数放大：p=0.5 时系数为 2；期望值不变不代表某次输出一定不变。
        scale = 1.0 / keep_prob
        # 数组乘标量是逐元素乘法，再包装成 Tensor；p=1 会除以零，forward 先单独处理它。
        return Tensor(binary_mask * scale)

    # 默认 training=True，调用 dropout(x) 会执行训练式随机丢弃；推理需要显式传 training=False。
    def forward(self, x, training=True):
        # 如果处于推理模式或 p=0，则下面缩进的 return 会提前结束函数。
        if not self._should_apply_dropout(training):
            # 直接返回原输入对象，不重新生成张量，也不额外缩放。
            return x
        # p=1 时 keep_prob=0，若按普通公式求 1/keep_prob 会发生除零。
        if self.p == DROPOUT_MAX_PROB:
            # zeros_like 建立与 x.data 同形状的零数组；返回新的 Tensor，不修改原输入。
            return Tensor(np.zeros_like(x.data))
        # 输入数组的 .shape 决定掩码形状，例如 (32, 128) 会抽签 32*128 次。
        mask = self._generate_dropout_mask(x.data.shape)
        # * 对这两个 Tensor 逐元素相乘；形状不变，留下的数变大，被丢弃的位置变 0。
        return x * mask

    # 显式提供 training 参数，让 dropout(x, training=False) 容易看出当前模式。
    def __call__(self, x, training=True):
        # 这里的 training 作为第二个位置参数交给 forward；也可以写成 training=training。
        return self.forward(x, training)

    # 虽然 Dropout 会改变训练时的数据，它没有可学习的权重或偏置。
    def parameters(self):
        # 返回空列表；self.p 是配置，不是优化器要更新的参数。
        return []

    # 返回便于查看的设置说明，如 Dropout(p=0.5)。
    def __repr__(self):
        # 这里的 p 只决定前向丢弃概率，字符串显示不参与计算。
        return f"Dropout(p={self.p})"


# Sequential 是“顺序容器”，负责把多个层接起来；它提供相似方法，但没有继承 Layer。
class Sequential:
    # *layers 把不定数量的位置参数收集为元组，如 Sequential(a, b) 收到 (a, b)。
    def __init__(self, *layers):
        # len 求元素数量，[0] 取第一个；isinstance 判断它是否为 list 或 tuple，and 要求两个条件都成立。
        if len(layers) == 1 and isinstance(layers[0], (list, tuple)):
            # 也接受 Sequential([a, b])；list(...) 建立新列表保存传来的两个层。
            self.layers = list(layers[0])
        # 如果传的是 Sequential(a, b)，走下面的分支。
        else:
            # 这里只复制容器，不复制里面的层对象，因此仍是调用者传进来的那些层。
            self.layers = list(layers)

    # 前向传播像流水线：每一层接收上一层的输出。training 要交给需要区分训练/推理的 Dropout。
    def forward(self, x, training=True):
        # for 每次从列表按顺序取一个层；例如 Linear -> ReLU -> Dropout -> Linear。
        for layer in self.layers:
            # try 尝试下面缩进的调用；失败时由 except 处理指定类型的异常。
            try:
                # 先按支持 training 参数的接口调用，结果重新赋给 x，让下一层接着使用它。
                x = layer.forward(x, training=training)
            # Linear/ReLU 的 forward 只接收 x，传 training 通常会引发 TypeError。
            # 注意：如果 forward 内部真的出了其他 TypeError，这个宽泛的 except 也会捕获并可能掩盖原错误。
            except TypeError:
                # 第二次调用只传 x，以兼容不认识 training 参数的层。
                x = layer.forward(x)
        # 所有层算完才返回；这依然只是前向计算，没有自动进行参数更新。
        return x

    # __call__ 让容器也可以像函数那样写 model(x, training=False)。
    def __call__(self, x, training=True):
        # training=training 右边是当前变量，左边是传给 forward 的参数名。
        return self.forward(x, training=training)

    # 将各层持有的可学习参数集中起来；没有参数的 ReLU/Dropout 各返回空列表。
    def parameters(self):
        # 准备一个空列表，逐层把权重和偏置对象加入其中。
        params = []
        # 容器的每个子层都需要提供 parameters() 方法。
        for layer in self.layers:
            # extend 逐项加入子层返回的列表；和 append 把整个列表作为“一项”加入不同。
            params.extend(layer.parameters())
        # 返回所有子层的参数 Tensor；列表长度不等于网络中可学习数字的总数。
        return params

    # 用字符串显示内部层的顺序与配置，方便核对网络结构。
    def __repr__(self):
        # repr(layer) 逐层取显示文字；括号中的 for 是生成器表达式，join 用逗号连接它们。
        layer_reprs = ", ".join(repr(layer) for layer in self.layers)
        # 返回例如 Sequential(Linear(...), ReLU(), Dropout(p=0.5)) 的说明文字。
        return f"Sequential({layer_reprs})"


def test_unit_linear_layer():
    # def 定义函数；空括号表示调用时不用传参数，冒号后的缩进行组成函数体。
    # 定义函数不会立刻运行这些检查；后面调用 test_unit_linear_layer() 才会执行。
    # print 把引号里的字符串显示在终端，中文和表情都只是提示文字。
    print("🧪 单元测试：Linear 层...")

    # Linear(...) 创建一个层对象；两个位置参数分别指定输入 784 个特征、输出 256 个特征。
    # = 是赋值，让变量 layer 指向新对象；没有写 bias 时使用默认值 True，即带偏置。
    layer = Linear(784, 256)
    # 点号读取对象的属性；== 比较数值是否相等，区别于赋值使用的单个 =。
    # assert 条件：成立就继续，否则抛出 AssertionError；这里检查输入特征数是否保存正确。
    # assert 适合调试和测试；python -O 会移除断言，正式输入校验应使用 if 和 raise。
    assert layer.in_features == 784
    # 再检查输出特征数；属性保存正确，并不能单独证明矩阵计算也正确。
    assert layer.out_features == 256
    # (784, 256) 是含两个整数的元组，表示权重表有 784 行、256 列。
    # 每个输出都连接 784 个输入，所以一共需要 784 × 256 = 200704 个权重数值。
    assert layer.weight.shape == (784, 256)
    # (256,) 是单元素元组，逗号不能省略；(256) 只是整数外面套括号。
    # 偏置是一维的 256 个数，每个输出特征配一个；它与二维形状 (1, 256) 不同。
    assert layer.bias.shape == (256,)
    # f 字符串把 {表达式} 的结果插进文字；这里会显示权重和偏置的实际形状。
    print(f"   Linear(784, 256)：weight.shape={layer.weight.shape}，bias.shape={layer.bias.shape}")

    # .data 取出 Tensor 内部的 NumPy 数组；np.std 计算全部权重的标准差。
    # 标准差衡量数值围绕平均值分散得多大；这里检查随机初始权重的整体尺度。
    weight_std = np.std(layer.weight.data)
    # / 先计算除法，np.sqrt 再开平方；常量为 1.0，所以目标是 √(1/784) = 1/28。
    # 目标标准差约为 0.035714；它描述随机分布，不要求每个权重等于这个数。
    expected_std = np.sqrt(INIT_SCALE_FACTOR / 784)
    # a < b < c 表示同时满足两个比较；这里要求实测标准差在目标的一半到两倍之间。
    # 随机样本不会恰好符合理论值，所以留出容忍范围；逗号后的括号允许错误提示换行写。
    assert 0.5 * expected_std < weight_std < 2.0 * expected_std, (
        # 这是断言失败时显示的消息，不是 print；f 字符串报告实际值和理论值。
        f"权重标准差 {weight_std} 与期望 {expected_std} 相差过大"
    )
    # :.4f 表示显示小数点后 4 位；显示时四舍五入不会改动变量里的原值。
    # 此处 LeCun 初始化让权重方差约为 1 / 输入特征数，减少输入维度带来的尺度变化。
    print(f"   LeCun 初始化：weight_std={weight_std:.4f}，期望约 {expected_std:.4f}")
    # np.allclose 检查全部元素是否在允许的浮点误差内接近目标；0 会与每个偏置比较。
    # 它允许很小的数值差异；逗号后是失败提示，这里检验偏置从零开始。
    assert np.allclose(layer.bias.data, 0), "偏置应初始化为 0"
    # 花括号里也能调用函数；先计算 allclose，再把 True 或 False 插进显示文字。
    print(f"   bias 是否全零：{np.allclose(layer.bias.data, 0)}")

    # 从内到外读：(32, 784) 给出形状，standard_normal 生成均值 0、标准差 1 的随机数。
    # Tensor 再包装数组，并依照 tensor 模块的实现转成 float32；每行一个样本，共 32 个样本。
    x = Tensor(rng.standard_normal((32, 784)))
    # 调用前向方法：输入乘权重，再加偏置，即 y = x @ W + b；返回的 Tensor 交给 y。
    y = layer.forward(x)
    # 样本数仍为 32，每个样本从 784 个特征变成 256 个输出；失败时显示实际形状。
    assert y.shape == (32, 256), f"期望形状 (32, 256)，实际 {y.shape}"
    # _show 是自定义显示函数；前两个是位置参数，data=False 是按名字传入的关键字参数。
    # False 关闭具体数字显示，只看形状等信息，避免打印 32 × 784 个输入数。
    _show("输入 x (32, 784)", x, data=False)
    # 查看输出的形状等信息；标题里的 xW + b 只是文字，运算已在 forward 中完成。
    _show("输出 y = xW + b", y, data=False)

    # 新建另一层，输入特征数 10、输出 5；bias=False 关闭默认启用的偏置。
    layer_no_bias = Linear(10, 5, bias=False)
    # is 判断是否为同一个对象，== 通常比较值；检查缺失值推荐写 is None。
    # None 表示没有偏置对象，不是数字 0，也不是一个装着全零数值的偏置 Tensor。
    assert layer_no_bias.bias is None
    # parameters() 调用方法，返回装着可学习参数 Tensor 的列表；只写 .parameters 不会调用。
    params = layer_no_bias.parameters()
    # len 统计列表元素数；只有一个权重 Tensor，但其中包含 10 × 5 = 50 个可调数字。
    assert len(params) == 1
    # 这里输出的“参数个数”指 Tensor 容器数量，不是模型中单个权重数字的总数。
    print(f"   Linear(10, 5, bias=False) 参数个数：{len(params)}")

    # 重新赋值：params 改为指向最初那个有偏置层的参数列表，不会改变无偏置层本身。
    params = layer.parameters()
    # 两项是权重 Tensor 和偏置 Tensor；按单个数字统计，则共有 200704 + 256 = 200960 个。
    assert len(params) == 2
    # 列表下标从 0 开始，params[0] 取第一项；is 要求它就是层持有的同一个权重对象。
    # 这样以后通过参数列表更新该对象时，层使用的也是这份权重，而不是另一份同值副本。
    assert params[0] is layer.weight
    # params[1] 取第二项；这里再次检查对象身份，数值相同但另行复制的 Tensor 也不能通过。
    assert params[1] is layer.bias
    # 显示列表里的两项；字符串中的 weight + bias 只是说明名称，不在这里执行加法。
    print(f"   Linear(784, 256) 参数个数：{len(params)}（weight + bias）")

    # 只有前面检查都未失败，才会运行到这里；函数结束时没有 return，会默认返回 None。
    print("✅ Linear 层通过！")


def test_unit_edge_cases_linear():
    # 定义无参数测试函数；“边界情况”包括单样本、没有样本、较大权重以及关闭偏置。
    # print 只显示测试标题；函数内同名变量与其他函数中的局部变量相互独立。
    print("🧪 边界测试：Linear 层...")

    # 创建带偏置的 Linear 层；后面的单样本和空批次检查使用同一组权重。
    layer = Linear(10, 5)

    # (1, 10) 表示一个样本、10 个特征，仍是二维表，区别于一维形状 (10,)。
    # 先生成随机数组再用 Tensor 包装；变量名中的 2d 提醒我们它有两个维度。
    x_2d = Tensor(rng.standard_normal((1, 10)))
    # 同一套矩阵乘法也能处理只有一行的输入，返回结果保存为 y。
    y = layer.forward(x_2d)
    # 要求一个输入样本得到一个输出样本，每个输出样本包含 5 个数。
    assert y.shape == (1, 5), "应能处理单样本"
    # 没有写 data=False，就采用 _show 的默认值 data=True，显示这 10 个输入数字。
    _show("单样本输入 (1, 10)", x_2d)
    # 显示对应的 5 个输出数字与形状，便于观察这次小规模计算。
    _show("输出", y)

    # (0, 10) 是零行十列：样本数为零，但仍保留每个样本应有 10 个特征的约定。
    # 总元素数是 0 × 10 = 0；这不是“一个数值为 0 的样本”，而是没有任何样本。
    x_empty = Tensor(rng.standard_normal((0, 10)))
    # 本项目的二维矩阵乘法会创建 (0, 5) 输出，遍历行的循环执行零次，加偏置也兼容此形状。
    y_empty = layer.forward(x_empty)
    # 空输入仍应得到空输出，输出特征数为 5；检查的是形状处理，不是模型预测准确率。
    assert y_empty.shape == (0, 5), "应能处理空 batch"
    # batch 是一次送入模型的一批样本；f 字符串显示这一批为空时输入输出的形状。
    print(f"   空 batch：输入 {x_empty.shape} → 输出 {y_empty.shape}")

    # 创建独立的层，后面修改它的权重不会影响前面变量 layer 指向的对象。
    layer_large = Linear(10, 5)
    # np.ones((10, 5)) 创建全 1 数组，* 100 逐个变成 100，再直接替换权重的 .data。
    # 注意：np.ones 默认 float64，而 Tensor 只在构造时缓存 .dtype；此后 .data.dtype 与 .dtype 会失步。
    # 本例形状没变，所以形状缓存恰好没问题；一般更新可用 .data[...] = 100 保持原数组类型。
    layer_large.weight.data = np.ones((10, 5)) * 100
    # 输入是一行 10 个 1；每个输出可手算为 10 个“1 × 100”相加，再加初始为 0 的偏置。
    # Tensor 构造会把这里 np.ones 生成的数组转成 float32。
    x = Tensor(np.ones((1, 10)))
    # 前向计算应得到一行 5 个 1000；100 相对初始小权重较大，但远未达到浮点数最大范围。
    y = layer_large.forward(x)
    # np.isnan 逐个检查是否为 NaN，再由 np.any 判断是否至少存在一个，not 将真假反转。
    # NaN 表示无效数值，例如某些未定义计算的结果；整个断言要求输出里一个 NaN 都没有。
    assert not np.any(np.isnan(y.data)), "大权重不应产生 NaN"
    # np.isinf 逐个检查正负无穷；同样用 not any 要求一个都没有。
    # 过大数值的溢出可能产生 Inf；本例通过只说明这组数值没有溢出，不保证任意大权重都安全。
    assert not np.any(np.isinf(y.data)), "大权重不应产生 Inf"
    # 显示结果便于手算核对；前两条断言只检查 NaN/Inf，并没有断言具体值一定等于 1000。
    _show("大权重前向", y)

    # 再创建无偏置层，专门检查只执行 x @ W 的计算分支；布尔值 False 的首字母须大写。
    layer_no_bias = Linear(10, 5, bias=False)
    # 准备 4 个样本，每个 10 个随机特征；x 重新赋值后代表这批新数据。
    x = Tensor(rng.standard_normal((4, 10)))
    # 运行无偏置前向，只做矩阵乘法，并让 y 指向新的输出 Tensor。
    y = layer_no_bias.forward(x)
    # 关闭偏置也应保持样本数 4，并将每个样本映射到 5 个输出特征。
    assert y.shape == (4, 5), "无 bias 时应能前向"
    # 仅查看输入的形状等信息；关键字参数 data=False 让辅助函数不显示具体数值。
    _show("无 bias 输入 (4, 10)", x, data=False)
    # 查看对应输出，计算已经完成，显示函数不会修改这个 Tensor。
    _show("无 bias 输出", y, data=False)

    # 所有边界断言都通过，才会执行到这条成功提示。
    print("✅ 边界情况通过！")


def test_unit_parameter_collection_linear():
    # 定义参数收集检查函数；训练代码需要找到权重和偏置，才能在后续学习过程中调整它们。
    # 本函数只检验返回的列表，没有真的执行梯度计算或参数更新。
    print("🧪 参数收集测试：Linear 层...")

    # 创建有偏置的层：权重有 10 × 5 个数，偏置有 5 个数。
    layer = Linear(10, 5)
    # 方法调用返回列表；按本实现约定，第一项是权重，第二项是偏置。
    params = layer.parameters()
    # len(params) 是参数 Tensor 的数量 2；单个可学习数字共有 10 × 5 + 5 = 55 个。
    assert len(params) == 2, "应返回 2 个参数（weight 和 bias）"
    # 先用 [0] 取列表第一项，再读取 .shape；要求它符合权重的二维排列。
    assert params[0].shape == (10, 5), "第一个应是 weight"
    # [1] 取第二项；(5,) 是一维形状元组，不是整数 5，也不是二维的 (1, 5)。
    assert params[1].shape == (5,), "第二个应是 bias"
    # [p.shape for p in params] 是列表推导式：依次取出每个参数叫 p，读取形状，组成新列表。
    # 可以读成“收集 params 中每个 p 的形状”，显示 [(10, 5), (5,)]；此处 p 不是丢弃概率。
    print(f"   有 bias：{len(params)} 个参数，shapes={[p.shape for p in params]}")

    # bias=False 是按参数名称关闭偏置，不会改变输入维度 10 和输出维度 5。
    layer_no_bias = Linear(10, 5, bias=False)
    # 无偏置层的 .bias 为 None，不会加入参数列表，所以此列表只有权重 Tensor。
    params_no_bias = layer_no_bias.parameters()
    # 只返回 1 个 Tensor，但该 Tensor 内仍含 50 个可调权重数字。
    assert len(params_no_bias) == 1, "无 bias 时应只返回 weight"
    # 再次用列表推导式收集形状，得到 [(10, 5)]：外层方括号是列表，内层圆括号是形状元组。
    print(f"   无 bias：{len(params_no_bias)} 个参数，shapes={[p.shape for p in params_no_bias]}")

    # 这组检查通过说明参数列表符合预期，不能据此认为模型已经训练完成。
    print("✅ 参数收集通过！")


def test_unit_should_apply_dropout():
    # 定义“是否启用 Dropout”的测试；Dropout 在训练中随机把一些位置临时变为 0。
    # 它能减少网络对固定特征组合的过度依赖；推理是用模型产生结果的阶段，通常关闭这种随机性。
    print("🧪 单元测试：Dropout 是否生效...")

    # p=0.5 表示每个位置有 50% 的机会被丢弃，不保证某一批恰好丢掉一半。
    d = Dropout(0.5)
    # training=True 按名字传入训练标志；返回的真假结果保存为 train_on。
    # 方法名前的下划线表示内部辅助使用的约定，并不禁止测试从外部调用它。
    train_on = d._should_apply_dropout(training=True)
    # 同一对象改用推理标志 False 再判断；本行只检查决定，还没有生成随机掩码。
    infer_off = d._should_apply_dropout(training=False)
    # is True 要求返回的正是 Python 的 True 对象；训练且概率大于零，所以应当启用。
    assert train_on is True, "Dropout(0.5) 训练时应生效"
    # 推理时应当关闭；is False 检查布尔对象身份，区别于一般的数值相等比较 ==。
    assert infer_off is False, "推理时不应 dropout"
    # f 字符串显示两次判断；字符串里面的等号只是标签，不能与语句中的赋值号混淆。
    print(f"   Dropout(0.5) 训练={train_on}，推理={infer_off}")

    # 创建丢弃概率为零的对象；0.0 是浮点数，表示不丢弃任何位置。
    d_zero = Dropout(0.0)
    # 虽然是训练模式，但 p > 0 不成立，因此辅助方法应返回 False。
    zero_train = d_zero._should_apply_dropout(training=True)
    # 要求 p=0 时跳过 Dropout；对这里正常的布尔 training 标志，训练或推理都原样通过。
    assert zero_train is False, "Dropout(0.0) 永远不应生效"
    # 显示 False 意味着不启动随机丢弃，不意味着输入全部变成零。
    print(f"   Dropout(0.0) 训练={zero_train}")

    # p=1 是允许的边界，训练时丢弃全部位置；构造函数不会把 1.0 当作非法值。
    d_full = Dropout(1.0)
    # 训练时应决定启用；实际前向还会用专门的全零分支，避免计算 1 / (1-p) 产生除零。
    full_train = d_full._should_apply_dropout(training=True)
    # 即使 p=1，推理时依然应关闭 Dropout，保持输入原样。
    full_infer = d_full._should_apply_dropout(training=False)
    # True 只表示要执行丢弃逻辑，不表示输出中还有存活位置。
    assert full_train is True, "Dropout(1.0) 训练时应生效"
    # 推理标志优先让前向跳过随机处理，所以不应误走全零输出分支。
    assert full_infer is False, "即使 p=1.0，推理时也不应 dropout"
    # 显示 p=1 在两种模式下的决定，正常分别为 True 和 False。
    print(f"   Dropout(1.0) 训练={full_train}，推理={full_infer}")

    # 这组只检查决策结果；掩码取值和真实输出会由后面的测试继续验证。
    print("✅ Dropout 决策逻辑通过！")


def test_unit_generate_dropout_mask():
    # 定义掩码测试；掩码是一组与输入逐位置相乘的数，0 表示丢弃，非零系数表示保留并缩放。
    # 本组直接调用辅助方法检查掩码，暂时不需要提供真正的输入张量。
    print("🧪 单元测试：Dropout 掩码...")

    # 丢弃概率为 0.5，对应保留概率 0.5；存活位置要乘的系数为 1 / 0.5 = 2。
    d = Dropout(0.5)
    # 外层括号是方法调用，内层 (1000,) 是传入的单元素形状元组，表示一维 1000 个位置。
    mask = d._generate_dropout_mask((1000,))
    # 掩码必须具有预期形状，才能与计划中的输入逐位置相乘；失败消息报告实际形状。
    assert mask.shape == (1000,), f"期望形状 (1000,)，实际 {mask.shape}"

    # np.unique 提取出现过的不同数值，再用 set 转成不重复、无固定显示顺序的集合。
    # 集合只回答“有哪些取值”，不记录每种取值出现了多少次。
    unique_vals = set(np.unique(mask.data))
    # 对集合，<= 表示“左边是右边的子集”：所有实际值都必须属于 {0.0, 2.0}，不是比大小。
    # 子集检查允许只出现一种值，后面的数量检查再排除极端情况；f 字符串的 {{ 和 }} 显示普通花括号。
    assert unique_vals <= {0.0, 2.0}, f"掩码取值应为 {{0.0, 2.0}}，实际 {unique_vals}"

    # np.count_nonzero 统计不等于零的位置数量；对本掩码来说，它就是实际存活位置数。
    non_zero = np.count_nonzero(mask.data)
    # 1000 个位置各以 0.5 概率独立保留，存活总数的标准差是 √(1000 × 0.5 × 0.5) ≈ 15.81。
    # std_err 这个名字不够准确：此处是“计数的标准差”；存活比例的标准误才是再除以 1000，约 0.01581。
    std_err = np.sqrt(1000 * 0.5 * 0.5)
    # 期望存活数为 500，允许上下波动三个标准差；整数计数允许 453 到 547，并非必须正好 500。
    # 这是容忍随机波动的测试范围；正确随机过程理论上也有小概率超出范围，不能当作数学保证。
    assert 500 - 3 * std_err < non_zero < 500 + 3 * std_err, f"期望约 500 个存活，实际 {non_zero}"
    # 同时显示形状、不同取值和实际存活数；集合的显示顺序不影响其含义。
    print(f"   Dropout(0.5) 掩码 shape={mask.shape}，取值={unique_vals}，非零个数={non_zero}")

    # 创建丢弃概率为 0.3 的另一对象，意味着期望保留 70%，不是保留 30%。
    d2 = Dropout(0.3)
    # 生成一维 2000 个位置的掩码；数量较大时比例通常更接近期望，但单次结果仍然随机。
    mask2 = d2._generate_dropout_mask((2000,))
    # / 是普通除法，得到存活位置的理论缩放 1 / 0.7 ≈ 1.428571。
    # 每个位置约有 70% 机会保留，保留时放大 1 / 0.7，可让固定输入位置的输出期望保持原值。
    expected_scale = 1.0 / 0.7
    # 先算方括号内的 !=：逐个判断是否“不等于 0”，得到一组 True/False。
    # 再用这组真假值选择原数组中 True 对应的位置，这叫布尔索引；结果是筛出的 NumPy 数组。
    non_zero_vals = mask2.data[mask2.data != 0.0]
    # allclose 检查每个非零系数都接近理论值；float32 不能精确存下所有小数，所以不要求严格相等。
    # 单个 expected_scale 会与数组各项分别比较；逗号后的括号让失败提示可以分行书写。
    assert np.allclose(non_zero_vals, expected_scale), (
        # :.4f 显示四位小数；np.unique 列出实际出现的非零系数，供断言失败时排查。
        f"存活值应为 {expected_scale:.4f}，实际 {np.unique(non_zero_vals)}"
    )
    # 非零位置数除以总数就是实际存活比例，例如 1400 / 2000 = 0.7。
    survival_rate = np.count_nonzero(mask2.data) / 2000
    # 连写比较要求实际比例在 60% 与 80% 之间，为理论 70% 留随机余量。
    # :.1% 将 0.7 显示为 70.0%；这个格式只影响显示，不改变原比例。
    assert 0.60 < survival_rate < 0.80, f"p=0.3 时期望约 70% 存活，实际 {survival_rate:.1%}"
    # 显示理论缩放和实际存活率；缩放来自理论概率 0.7，没有根据本次抽样比例临时重算。
    print(f"   Dropout(0.3) 缩放={expected_scale:.4f}，存活率={survival_rate:.1%}")

    # 形状、取值和存活数量的检查都成功后，才会执行这条提示。
    print("✅ Dropout 掩码生成通过！")


def test_unit_dropout_layer():
    # 定义 Dropout 完整前向测试：检查推理、p=0、p=1、普通训练以及非法概率。
    # 与上一组掩码测试不同，本组会真正传入数据并检查输出。
    print("🧪 单元测试：Dropout 层...")

    # 创建 p=0.5 的对象；构造只保存概率并检查范围，还没有对任何输入执行丢弃。
    dropout = Dropout(0.5)
    # 读取保存的 .p，用 == 检查值等于 0.5；0.5 可被二进制浮点数精确表示。
    # 不能据此推断任意经过运算的小数都适合用 ==，含舍入误差时应考虑近似比较。
    assert dropout.p == 0.5
    # 显示配置概率；p 是人工设定的选项，不是像 Linear 权重那样由训练学习的参数。
    print(f"   Dropout(0.5)：p={dropout.p}")

    # [1, 2, 3, 4] 是 Python 列表，按顺序装着四个整数；Tensor 再把它转换成 float32 数组。
    # 结果形状是 (4,)，表示一维四个数，区别于一行四列的二维形状 (1, 4)。
    x = Tensor([1, 2, 3, 4])
    # training=False 指定推理模式；本实现直接 return x，所以这里得到的还是同一个 Tensor 对象。
    y_inference = dropout.forward(x, training=False)
    # np.array_equal 要求形状相同且对应数值完全相等，区别于允许数值误差的 np.allclose。
    # 它检查内容，没有检查对象身份，也不要求 dtype 必须相同；本处验证推理时数据原样通过。
    assert np.array_equal(x.data, y_inference.data), "推理应原样通过"
    # 显示输入四个数；不传 data=False 就使用 _show 默认的数据展示行为。
    _show("输入", x)
    # 显示推理输出，应仍为 1、2、3、4，既不随机清零，也不乘训练时的放大系数。
    _show("推理模式输出", y_inference)

    # 创建不丢弃任何位置的对象；p=0 时也无需放大存活值。
    dropout_zero = Dropout(0.0)
    # 即使 training=True，p=0 的启用判断仍为 False，于是直接返回原来的 x。
    y_zero = dropout_zero.forward(x, training=True)
    # 检查内容完全相同；变量名 zero 指概率为零，不是输出应全零。
    assert np.array_equal(x.data, y_zero.data), "p=0 应原样通过"
    # 显示 p=0 时的训练输出，应该依旧是原先四个数。
    _show("p=0 训练输出", y_zero)

    # 创建训练时全部丢弃的对象；p=1 属于合法边界，p>1 才会被拒绝。
    dropout_full = Dropout(1.0)
    # 前向走专门分支，生成与输入形状相同的全零 Tensor，避免 1/(1-p) 除零。
    y_full = dropout_full.forward(x, training=True)
    # allclose 要求每个输出都接近 0；实现实际生成精确零，但本断言只验证数值，不另查形状。
    assert np.allclose(y_full.data, 0), "p=1 应全部置零"
    # 输出应为四个零；这是新输出的内容，原输入 x 没被修改。
    _show("p=1 训练输出", y_full)

    # np.ones((1000,)) 生成一维 1000 个 1，再包装为 Tensor；使用全 1 输入便于检查结果。
    # p=0.5 时，被丢弃的位置应为 0，存活位置应为 1 × 2 = 2。
    x_large = Tensor(np.ones((1000,)))
    # 使用最初那个 p=0.5 对象并启用训练；生成同形状随机掩码后，与输入逐元素相乘。
    y_train = dropout.forward(x_large, training=True)
    # 统计非零输出数；本例输入全为 1，所以它等于存活数。
    # 若原输入就有零，只看非零输出数就无法区分“原本为零”和“被 Dropout 丢弃”。
    non_zero_count = np.count_nonzero(y_train.data)
    # 期望存活总数是 1000 × (1 - 0.5) = 500；这是平均目标，不要求每次都恰好达到。
    expected = 500
    # √(1000 × 0.5 × 0.5) ≈ 15.81 是存活计数的标准差，衡量计数通常波动多大。
    # std_error 名称不准确：若要算存活比例的标准误，应再除以 1000，约为 0.01581。
    std_error = np.sqrt(1000 * 0.5 * 0.5)
    # 下界是平均目标减三个标准差，约 452.57；命名中间结果让后面的判断更容易读。
    lower_bound = expected - 3 * std_error
    # 上界是平均目标加三个标准差，约 547.43；这些边界不会改变实际随机抽样结果。
    upper_bound = expected + 3 * std_error
    # 要求存活数严格位于上下界之间，即整数 453 到 547；正确随机过程也有小概率超出范围。
    # 逗号后的括号是跨行错误消息，不是开启另一个需要执行的代码块。
    assert lower_bound < non_zero_count < upper_bound, (
        # ± 是普通文本；{3*std_error:.0f} 先计算三倍标准差，再显示为不带小数位的约 47。
        # 显示会四舍五入，但断言使用的仍是完整精度的上下界。
        f"期望 {expected}±{3*std_error:.0f} 个存活，实际 {non_zero_count}"
    )
    # 先逐个比较 != 0，再用真假数组筛选原数组，得到全部非零输出；这仍是布尔索引。
    surviving_values = y_train.data[y_train.data != 0]
    # 全 1 输入乘保留系数 1 / (1 - 0.5)，所以每个存活输出应等于 2.0。
    expected_value = 2.0
    # 检查所有存活值都接近 2；前面的数量断言已保证有正常数量的存活值。
    # 这很关键，因为 allclose 对空数组比较也可能返回 True，仅靠它无法证明真的有值被检查。
    assert np.allclose(surviving_values, expected_value), f"存活值应为 {expected_value}"
    # 显示实际非零数量与预期存活值；引号中的 /1000 是总数量说明，不是此处执行除法。
    print(f"   p=0.5 训练：非零 {non_zero_count}/1000，存活值={expected_value}")

    # 获取 Dropout 的可学习参数列表；虽然它有 .p 配置，但此实现不会自动学习 p。
    params = dropout.parameters()
    # 空列表长度为 0，说明 Dropout 不贡献可学习参数 Tensor；它仍然会改变训练中的数据。
    assert len(params) == 0, "Dropout 不应有可学习参数"
    # 显示可学习参数列表的长度，普通配置属性 p 不计算在内。
    print(f"   Dropout 参数个数：{len(params)}")

    # try: 表示尝试执行缩进内语句；出现异常时，会寻找后面能匹配异常类型的 except 分支。
    # 这里故意使用错误输入，验证程序能否拒绝非法值，因此预期报错是测试成功的一部分。
    try:
        # -0.1 是负数，不是合法概率；构造函数应抛出 ValueError，然后跳过 try 中剩余语句。
        Dropout(-0.1)
        # assert False 永远失败：只有上一行没按预期报错，才会来到这里触发 AssertionError。
        # 它是测试保险；AssertionError 与 ValueError 类型不同，不会被下面的 except 吞掉。
        assert False, "负概率应抛出 ValueError"
    # except 与 try 同级缩进，只捕获预期的 ValueError，其他类型的错误仍会暴露出来。
    except ValueError:
        # 只有捕获到预期异常才显示本句；处理完后会继续下一组测试，不会因此结束整个程序。
        print("   p=-0.1：捕获 ValueError")

    # 第二组独立的 try 检查另一个非法边界：概率大于 1。
    try:
        # 1.1 相当于 110% 丢弃概率，超出 0 到 1 的允许范围，应在构造时抛出 ValueError。
        Dropout(1.1)
        # 若错误输入被悄悄接受，就强制产生 AssertionError，防止测试误报通过。
        assert False, "p>1 应抛出 ValueError"
    # 只处理本组预期的 ValueError，不会捕获上面用于宣告测试失败的 AssertionError。
    except ValueError:
        # 这句表示非法概率被正确拒绝，不表示创建那个 p=1.1 的对象成功了。
        print("   p=1.1：捕获 ValueError")

    # 正常输出和非法输入检查全部成功后显示通过；这些测试没有进行损失计算、反向传播或权重更新。
    print("✅ Dropout 层通过！")


# def 定义函数；analyze_layer_memory 是函数名，空括号表示调用时不用传参数，冒号后缩进的代码属于函数体。
# 定义函数时先记住这段做法，后面调用 analyze_layer_memory() 才执行。这里估算参数数组的内存，并非测量整个程序占用。
def analyze_layer_memory():
    # print(...) 把双引号中的字符串打印到终端；中文和表情也是字符串的内容。
    print("📊 分析层的内存占用...")

    # = 是赋值：把右边的列表交给变量 layer_configs。方括号 [] 表示按顺序存放元素的列表。
    # 因为左方括号还没闭合，列表内容可以自然换行；每一项都是一组“输入维度、输出维度”。
    layer_configs = [
        # (784, 256) 是包含两个整数的元组，代表输入 784 维、输出 256 维；末尾逗号分隔列表中的不同元素。
        (784, 256),
        # 第二个元组代表 256 → 256；维度相同也会通过权重和偏置重新组合特征，并非原样返回。
        (256, 256),
        # 第三个元组代表 256 → 10；此处仅保存配置数字，尚未创建 Linear 对象。
        (256, 10),
        # 第四个元组代表 2048 → 2048，权重将含 2048×2048 个数。列表最后一项后面保留逗号是合法写法。
        (2048, 2048),
    # 右方括号结束上面的列表；四个元组一起成为 layer_configs 的值。
    ]

    # 字符串中的 \n 是换行转义符：先换行，再显示小标题。它不会把反斜杠和字母 n 原样打印出来。
    print("\nLinear 层内存：")
    # 打印表格列名；引号内的 in、out、箭头只是说明文字，不是正在执行的 Python 语法。
    print("配置 (in, out) → 权重内存 → 偏置内存 → 合计")

    # for ... in ...: 依次遍历列表；in_feat, out_feat 把每个二元组分别拆给两个变量，这叫元组解包。
    # 第一轮 in_feat=784、out_feat=256，下一轮为 256、256；每轮都会执行下方缩进的内存计算。
    for in_feat, out_feat in layer_configs:
        # * 在这里是数值乘法：权重数量为输入维度×输出维度，再乘每个数的字节数。
        # 本项目 Tensor 构造器使用 float32，即每个数占 32 位=4 字节，因此 weight_memory 的单位是字节。
        weight_memory = in_feat * out_feat * 4
        # 每个输出特征有一个偏置，所以偏置数组有 out_feat 个数，占 out_feat×4 字节。
        # 同一层的所有样本共用权重和偏置；这些参数的内存不会随着 batch 大小增加。
        bias_memory = out_feat * 4
        # + 把权重和偏置的数据字节数相加。这里只算参数数组数据区，不含 Python 对象、输入输出、临时数组、梯度或优化器状态。
        total_memory = weight_memory + bias_memory
        # print( 开始一次跨行函数调用；右括号闭合之前，后面的内容仍属于这次调用。
        # 下面两个相邻的 f-string 会自动拼成一个字符串，不必另写 +。
        print(
            # f 开头的字符串能把 {...} 内的表达式结果插入文字。冒号后的内容规定显示方式，不改变变量保存的值。
            # :4d 表示十进制整数最少占 4 个字符，不足时左边补空格；/ 是普通除法。
            # :7.1f 表示最少宽 7、保留 1 位小数。字节数除以 1024 的严格单位是 KiB，原输出写 KB 不够准确。
            f"({in_feat:4d}, {out_feat:4d}) → {weight_memory/1024:7.1f} KB → "
            # 这一段继续拼接偏置和合计内存；:6.1f 与 :7.1f 只是列宽不同，都显示 1 位小数。
            # 宽度是最小宽度，数字太长会完整显示，不会被截断。这里除以 1024 后的 KB 也应理解为 KiB。
            f"{bias_memory/1024:6.1f} KB → {total_memory/1024:7.1f} KB"
        # 右括号结束 print 调用，将前面拼好的整行文字打印出来。
        )

    # 打印下一组比较的小标题。隐藏层是输入与最终输出之间的中间层，宽度是每个样本在该层的特征数。
    print("\n💡 多层模型内存随隐层变宽：")
    # 创建含五个整数的列表，作为接下来逐一尝试的第一隐藏层宽度。
    hidden_sizes = [128, 256, 512, 1024, 2048]

    # 依次把列表中的一个整数赋给 hidden_size，再执行缩进代码；循环变量不是整个列表。
    for hidden_size in hidden_sizes:
        # 第一层结构为 784 → hidden_size；权重有 784×hidden_size 个数，再加 hidden_size 个偏置。
        # 这里的参数量统计可训练的单个数，也叫标量数量，不是参数 Tensor 对象的数量。
        layer1_params = 784 * hidden_size + hidden_size
        # 第二层结构为 hidden_size → hidden_size//2，公式仍是“权重数+偏置数”。
        # // 是向下取整除法，例如 5//2 得到 2，而 5/2 得到 2.5；负数时 -5//2 得到 -3。
        # 本例宽度全是正偶数，因此 //2 恰好减半。括号让先计算出的第二层宽度参与后续乘法。
        layer2_params = hidden_size * (hidden_size // 2) + (hidden_size // 2)
        # 第三层结构为 hidden_size//2 → 10；第二层的输出宽度乘 10 得到权重数，再加 10 个偏置。
        layer3_params = (hidden_size // 2) * 10 + 10
        # 把三层可训练标量数量相加。此处没有给 ReLU、Dropout 加参数量，因为本项目里它们没有可训练权重。
        total_params = layer1_params + layer2_params + layer3_params
        # 先乘每个 float32 数的 4 字节，再除以 1024×1024，把参数数据量换算为 MiB。
        # 变量名 memory_mb 和输出中的 MB 是简写，严格说 1 MiB=1,048,576 字节，而 1 MB 通常是 1,000,000 字节。
        memory_mb = total_params * 4 / (1024 * 1024)
        # f-string 中 :4d 是整数最少宽 4；:7, 是最少宽 7 并用逗号分组，如 1234567 显示成 1,234,567。
        # :5.1f 表示最少宽 5、显示 1 位小数；这里打印的 MB 实际按 MiB 计算，且仍只是参数数据内存。
        print(f"隐层={hidden_size:4d}：{total_params:7,} 个参数 = {memory_mb:5.1f} MB")


# 定义无参数函数 analyze_layer_performance，用来估算计算量并实际测量前向传播耗时。
# 下面的重复计算只有函数被调用后才运行；定义本身不会立即开始耗时测试。
def analyze_layer_performance():
    # 打印开始分析计算量的提示；打印文字本身不会测量运行速度。
    print("📊 分析层的计算量...")

    # 列表中每个整数是一种 batch 大小，表示一次送入网络的样本数，分别测试 1、32、128、512 个样本。
    batch_sizes = [1, 32, 128, 512]
    # 调用 Linear 类创建 784 → 256 的层对象，保存到变量 layer。
    # 后面不同 batch 都使用这个对象；权重始终是 (784,256)，偏置始终是 (256,)，不会因 batch 变化而增加。
    layer = Linear(784, 256)

    # 打印计算量标题。MAC 是一次乘法并加进累计和，常用于估算矩阵乘法的工作量。
    print("\nLinear 层 MAC 分析：")
    # 注意表头的术语简化：加偏置只有加法，不是乘加，因此不能直接称为同样数量的 MAC。
    print("batch → 矩阵乘 MAC → 加偏置 MAC → 合计 MAC")
    # FLOP 指一次浮点运算；常见估算把一次 MAC 的乘法与加法分别计数，因此约为 2 FLOPs。
    # 此换算针对乘加工作量；偏置只有一次加法，应单独计数，不能直接把它也按一个 MAC 再乘 2。
    print("说明：FLOPs = 2 × MACs（一次乘加算一次 MAC）")

    # 依次遍历四种 batch 大小，用当前的 batch_size 计算这一批数据前向传播的大致工作量。
    for batch_size in batch_sizes:
        # 输入 (batch_size,784) 乘权重 (784,256)，得到 batch_size×256 个输出，每个输出合并 784 项乘积。
        # 因此常按 batch_size×784×256 个 MAC 估算；变量虽然叫 matmul_flops，实际这里存的是 MAC 数。
        matmul_flops = batch_size * 784 * 256
        # 每个输出元素再加一个偏置，共有 batch_size×256 次加法，可按同样数量的 FLOP 计数。
        # 这一项只有加法，没有对应的乘法，因此与上一行的 MAC 数不是同一种计数单位。
        bias_flops = batch_size * 256
        # Python 可以把这两个整数相加，但数学口径不一致：这里把矩阵 MAC 数与偏置加法数混在了一起。
        # 所以 total_flops 不能视为统一口径的 FLOPs，也不能直接视为总 MAC；保留原公式，仅在此说明。
        # 常见估算口径可用 2×batch×784×256+batch×256 FLOPs；若严格数每个点积的 784 次乘法、
        # 783 次累加，再加一次偏置，则恰好为 2×batch×784×256。差别来自是否按每项都计一次乘加的约定。
        total_flops = matmul_flops + bias_flops
        # 打印本轮结果；:10d 表示整数最少宽 10，:15,、:13,、:11, 表示设置最小宽度并加千位逗号。
        # 格式化让各列对齐，但不会修正上面混用计数单位的问题。
        print(f"{batch_size:10d} → {matmul_flops:15,} → {bias_flops:13,} → {total_flops:11,}")

    # 打印新的小标题，下面开始真正执行计算并测时间；上面只是通过公式估算运算数量。
    print("\nLinear 层耗时：")
    # 打印表头。ms 是毫秒，1 秒=1000 毫秒；吞吐量表示每秒能处理多少个样本。
    print("batch → 耗时 (ms) → 吞吐 (样本/秒)")

    # 这是一个新的 for 循环，会再次从 batch_sizes 的第一个元素开始，为每种 batch 分别计时。
    for batch_size in batch_sizes:
        # 先执行内部 rng.standard_normal((batch_size,784))，生成指定形状的标准正态随机数组，再用 Tensor(...) 包装。
        # (batch_size,784) 是二维形状元组，表示每行一个样本、每个样本 784 个特征；这里是随机假数据。
        x = Tensor(rng.standard_normal((batch_size, 784)))
        # range(10) 依次给出 0 到 9，因此循环 10 次，用于正式计时前的预热。
        # _ 是一个合法的普通变量名，惯例表示“这个值我不打算使用”，并非自动丢弃值的特殊语法。
        for _ in range(10):
            # 调用 layer.forward(x) 完成一次前向传播，把返回结果绑定给 _，表示不关心输出内容。
            # 此处复用了循环变量 _；下一轮 for 仍会把它重新赋成下一个轮次整数，因此不影响循环次数。
            _ = layer.forward(x)

        # 把正式计时的重复次数设为 100；多次运行再取平均通常能减小单次测量的偶然波动。
        iterations = 100
        # time.perf_counter() 返回适合测短时间间隔的高精度计时器读数，单位是秒。
        # start 不是当前日期或钟表时间；把它与后面的计时器读数相减，才得到经历了多久。
        start = time.perf_counter()
        # range(iterations) 此处等同于 range(100)，把缩进的前向传播正式重复 100 次。
        # 当前 tensor 模块的二维 matmul 使用两层 Python for 循环并逐项调用 np.dot，batch=512 时可能明显较慢。
        for _ in range(iterations):
            # 这次前向传播处在计时区间内；_ 接收返回结果但不用于后续业务计算。
            # 这里只执行 xW+b，没有损失计算、反向传播、参数更新，测得的不是完整训练一步的耗时。
            _ = layer.forward(x)
        # 再读取计时器，用减号 - 减去 start，得到这 100 次前向传播合计花费的秒数。
        elapsed = time.perf_counter() - start

        # 总秒数除以重复次数，得到一次前向传播的平均秒数，再乘 1000 换算为毫秒。
        # 例如 100 次共花 2 秒，平均每次就是 2/100×1000=20 毫秒。
        time_per_forward = (elapsed / iterations) * 1000
        # batch_size×iterations 是总共处理的样本数量，除以总秒数便得到吞吐量，单位是样本/秒。
        # 吞吐量高表示单位时间处理更多样本，但不直接等于单个样本等待时间短。
        throughput = (batch_size * iterations) / elapsed
        # f-string 中 :8.3f 表示最少宽 8、保留 3 位小数；:12,.0f 表示最少宽 12、千位分组、显示 0 位小数。
        # .0f 只对显示值做四舍五入，不会把 throughput 本身改成整数。
        print(f"{batch_size:10d} → {time_per_forward:8.3f} ms → {throughput:12,.0f} 样本/秒")

    # 打印经验说明的标题；后面几句话是预先写好的字符串，并非程序根据本次耗时结果自动推导出来的结论。
    print("\n💡 要点：")
    # O(...) 是大 O 表示法，描述数据规模增加时工作量的大致增长趋势，不是某台机器上的精确耗时公式。
    # 固定输入输出维度时，batch 翻倍，Linear 的核心算术工作量通常也约翻倍。
    print("   Linear 复杂度：O(batch × in_features × out_features)")
    # 这句要分开理解：固定网络时，输入和中间激活的内存通常随 batch 线性增加，共享参数的内存不随 batch 增加。
    # 当相邻两层宽度都从 h 增至 2h 时，中间权重从 h×h 变为 2h×2h，才出现约 4 倍的平方增长。
    # 只扩大某一维不一定是平方增长；这句也不能理解成程序的所有内存都服从同一个公式。
    print("   内存随 batch 线性增长，随层宽近似平方增长")
    # Dropout 对每个元素生成随机数、比较、乘掩码，工作量通常随元素总数线性增加，且没有可训练参数。
    # 它仍会产生随机掩码、分配结果数组，实际额外开销取决于实现和规模，不能保证任何情况下都很小。
    print("   Dropout 几乎只是逐元素运算，额外开销很小")
    # 更大 batch 有时能摊薄固定开销、提高硬件利用率，但吞吐量更高并非保证。
    # 本项目手写矩阵乘法有 Python 循环开销，缓存和内存也会影响结果；应以实际测量为准，过大 batch 还可能耗尽内存。
    print("   更大 batch 能摊掉固定开销，吞吐更高")


# def 定义无参数函数 test_module，把多个单元测试和网络集成演示组织在一起。
# 函数名字含 test 不会让 Python 自动运行它；需要后面的主入口主动调用。
def test_module():
    # 打印执行阶段提示。集成测试关注多个部件连接后能否协同工作。
    print("🧪 正在运行模块集成测试")
    # 字符串乘整数会重复字符串："="*50 生成 50 个等号，再由 print 打印为分隔线。
    # 同一个 * 运算符对数字表示乘法，对这里的字符串表示重复。
    print("=" * 50)

    # 打印即将执行单元测试的提示。单元测试通常一次关注一个较小功能。
    print("运行单元测试...")
    # 调用前面定义的 Linear 单元测试，检查属性、形状、初始化和前向结果等。
    # 空括号表示不传参数；若函数内有未捕获的 AssertionError，程序会在失败处停止，后面测试不会继续执行。
    test_unit_linear_layer()
    # 调用 Linear 边界测试，检查单样本、空 batch、较大数值、无偏置等情况。
    # 这些测试通过仅说明已覆盖的案例通过，不等于所有可能输入都不会出错。
    test_unit_edge_cases_linear()
    # 调用参数收集测试，确认 Linear 能提供权重和可选偏置，方便未来训练程序统一访问并更新它们。
    test_unit_parameter_collection_linear()
    # 调用 Dropout 开关判断的测试，检查 training 与丢弃概率如何共同决定是否启用丢弃。
    test_unit_should_apply_dropout()
    # 调用 Dropout 掩码测试，检查掩码的形状、取值与随机统计性质。
    # 掩码用对应位置的数决定哪些输入清零、哪些输入保留并缩放。
    test_unit_generate_dropout_mask()
    # 调用 Dropout 层的综合单元测试，检查训练、推理和极端概率等行为；正常返回后才继续集成演示。
    test_unit_dropout_layer()

    # 字符串开头的 \n 先换行，让单元测试和集成场景的输出分开。
    print("\n运行集成场景...")
    # 打印多层网络演示的提示。下面把 Linear、ReLU、Dropout 按顺序手工连接起来。
    print("🧪 集成测试：多层网络...")

    # 创建第一层 784 → 128：weight.shape=(784,128)，bias.shape=(128,)。
    # 调用类名 Linear(...) 会创建实例并执行初始化；每个样本的 784 个特征会被组合成 128 个中间特征。
    layer1 = Linear(784, 128)
    # ReLU() 创建激活层，逐元素计算 max(0,x)，把负数变成 0，引入非线性。
    # ReLU 不改变形状，也没有可训练参数；只有多个线性层而无非线性时，整体仍可合并成一次仿射变换。
    activation1 = ReLU()
    # 创建丢弃概率 p=0.5 的 Dropout。训练时各元素独立地有 50% 概率清零，保留项乘 1/(1-0.5)=2。
    # “概率 50%”不保证每个 batch 恰好清零一半。
    dropout1 = Dropout(0.5)
    # 创建第二层 128 → 64；它的输入宽度 128 必须接得上前一阶段的输出宽度 128。
    # 这一层把每个样本的 128 个中间特征重新组合成 64 个特征。
    layer2 = Linear(128, 64)
    # 再创建一个 ReLU 对象，用于第二次线性变换后的非线性处理；两个变量名帮助我们辨认网络中的不同位置。
    activation2 = ReLU()
    # 创建第二个 Dropout，丢弃概率 0.3，即平均保留约 70% 元素；保留项乘 1/0.7。
    # 下面直接调用 forward 时会使用默认 training=True，启用随机丢弃。
    dropout2 = Dropout(0.3)
    # 创建最后一层 64 → 10，准备为每个样本输出 10 个原始分数。
    # 后面没有接 Sigmoid 或 Softmax，所以分数不是概率，不保证在 0 到 1 之间，也不保证总和为 1。
    layer3 = Linear(64, 10)

    # 把 batch_size 设为 16，表示这一批有 16 个样本；它决定输入与输出的第一维，不改变已有权重的形状。
    batch_size = 16
    # 先生成形状 (16,784) 的标准正态随机数组，再包装成 Tensor；每行是一个样本。
    # 784=28×28，可以模拟灰度图片展平后的长度，但这里没有读取真实图片，也没有真实标签。
    x = Tensor(rng.standard_normal((batch_size, 784)))
    # 调用辅助函数 _show：前两个参数按位置传入，data=False 则按参数名称传入。
    # False 是布尔值“假”；此处隐藏完整数组，只显示 shape、size、ndim、dtype 等元信息。
    _show("输入", x, data=False)

    # 先算右侧 layer1.forward(x)，再把结果绑定给 h；形状由 (16,784) 变为 (16,128)。
    # h 常用作隐藏层表示的名字，本质上仍是普通变量；不是 Python 自带的神经网络符号。
    h = layer1.forward(x)
    # 显示第一层输出的元信息，此时 shape=(16,128)；字符串标签仅用于解释，不会再计算一次 Linear。
    _show("Linear(784, 128)", h, data=False)
    # 把当前 h 传入 ReLU，再让 h 指向返回结果；负数清零、非负数保留，shape 仍为 (16,128)。
    # h=...h... 是合法赋值：Python 先用旧 h 算右边，再更新左边绑定；它不是数学等式，也不意味着原地修改旧 Tensor。
    h = activation1.forward(h)
    # 显示 ReLU 输出的元信息。它改变数值，但保留 16 个样本和每样本 128 个特征的形状。
    _show("ReLU", h, data=False)
    # 未显式传 training，所以使用默认 True：第一个 Dropout 随机清零一部分元素，其余乘 2。
    # 输出 shape 仍为 (16,128)；训练模式只控制丢弃开关，这里没有学习或更新参数。
    h = dropout1.forward(h)
    # 打印第一个 Dropout 的元信息；只看 shape 无法知道哪些位置清零，因为 Dropout 不删除数组位置。
    _show("Dropout(0.5)", h, data=False)
    # 把 shape=(16,128) 的 h 送入第二层，得到 shape=(16,64)；第一维样本数 16 保持不变。
    h = layer2.forward(h)
    # 显示第二个 Linear 输出的形状和数据类型等信息；运算在上一行已经完成。
    _show("Linear(128, 64)", h, data=False)
    # 第二个 ReLU 逐元素把负数变成 0，再更新变量 h；shape 仍为 (16,64)。
    h = activation2.forward(h)
    # 打印第二个 ReLU 的输出信息；data=False 避免把完整数组打印出来。
    _show("ReLU", h, data=False)
    # 第二个 Dropout 以 0.3 的概率丢弃元素，保留项乘 1/0.7，shape 仍为 (16,64)。
    # 缩放保持的是各元素在随机掩码下的期望值，不保证这一次具体输出的总和与输入相同。
    h = dropout2.forward(h)
    # 打印第二个 Dropout 的元信息；标签中的 0.3 是丢弃概率，而不是保留概率。
    _show("Dropout(0.3)", h, data=False)
    # 最后一层把 (16,64) 变成 (16,10)，保存为 output；每个样本有 10 个输出分数。
    # 用于十分类时这些原始分数可称 logits。权重尚未训练，不能因形状正确就认为模型已经学会分类。
    output = layer3.forward(h)
    # 打印最终输出的元信息；共有 16×10=160 个数，但 data=False 让它们的具体值不显示。
    _show("Linear(64, 10) 输出", output, data=False)

    # assert 条件, 错误信息：条件为真则继续，为假则抛出 AssertionError；== 比较的是值，而 = 用于赋值。
    # 这里比较两个形状元组是否相等，预期 (16,10)；f-string 在失败时说明期望与实际。
    # 这个检查只验证形状，不能证明预测准确。python -O 可禁用 assert，因此断言不适合替代必须执行的输入校验。
    assert output.shape == (batch_size, 10), f"期望输出形状 ({batch_size}, 10)，实际 {output.shape}"

    # 三次 parameters() 各返回一个列表；列表之间的 + 表示拼接，不是把权重数值相加。
    # 得到的 all_params 包含三层的权重与偏置共 6 个 Tensor 引用；拼接列表不会复制各 Tensor 内部的参数数组。
    all_params = layer1.parameters() + layer2.parameters() + layer3.parameters()
    # 预期 6 指 3 层×每层 2 个参数 Tensor，不是只含 6 个可训练数。
    # 三层实际的标量总数为 784×128+128+128×64+64+64×10+10=109386。
    expected_params = 6
    # len(all_params) 返回列表元素数量，应该是 6；不符就抛出带中文信息的 AssertionError。
    # len 在这里不会钻进每个 Tensor 里累计浮点数个数。
    assert len(all_params) == expected_params, f"期望 {expected_params} 个参数，实际 {len(all_params)}"
    # f-string 打印列表长度 6；原提示“参数个数”在这里准确地说是“参数 Tensor 数量”。
    print(f"   三层 Linear 参数个数：{len(all_params)}")

    # 创建新的 shape=(4,784) 随机输入，专门对比 Dropout 的训练与推理行为；它不是前面 16 个样本的切片。
    test_x = Tensor(rng.standard_normal((4, 784)))
    # 创建一个丢弃概率为 0.5 的新 Dropout 对象；它保存配置 p，不含可训练权重。
    dropout_test = Dropout(0.5)
    # training=True 是关键字实参，显式打开随机清零和保留项缩放，结果保存为 train_output。
    # True 是布尔真值，不加引号；执行训练模式的前向传播并不等于执行参数训练。
    train_output = dropout_test.forward(test_x, training=True)
    # training=False 关闭 Dropout，结果存入 infer_output。本实现直接 return x，返回的就是原输入 Tensor 对象。
    infer_output = dropout_test.forward(test_x, training=False)
    # np.array_equal(a,b) 检查两数组形状和对应元素是否完全相等，assert 要求此处返回 True。
    # 它与 np.allclose 不同：array_equal 做完全相等检查，allclose 则允许小量数值误差。
    assert np.array_equal(test_x.data, infer_output.data), "推理模式应原样通过"
    # f-string 的花括号里可以调用函数；np.count_nonzero 统计训练结果中非零元素数，array_equal 检查推理是否原样通过。
    # 随机丢弃不保证恰好清零一半；若原输入自身有零，也会影响非零数量，所以非零统计不总等于保留掩码数量。
    print(f"   Dropout 训练非零：{np.count_nonzero(train_output.data)}，推理与输入相同：{np.array_equal(test_x.data, infer_output.data)}")

    # 先计算括号内四个构造调用，创建 Linear、ReLU、Dropout、Linear，再作为四个位置实参交给 Sequential。
    # 顺序网络结构是 10 → 8 → 3，ReLU 与 Dropout 保持中间的 8 维，不新增可训练参数。
    seq = Sequential(Linear(10, 8), ReLU(), Dropout(0.2), Linear(8, 3))
    # 创建 shape=(2,10) 的随机输入：2 个样本，每个 10 个特征，正好对应第一层要求的输入宽度 10。
    seq_x = Tensor(rng.standard_normal((2, 10)))
    # seq(...) 能像函数一样调用，是因为 Sequential 定义了 __call__，它会转去执行 forward。
    # training=False 传给容器，使其中的 Dropout 关闭；形状依次是 (2,10)→(2,8)→(2,8)→(2,8)→(2,3)。
    seq_y = seq(seq_x, training=False)
    # 显示 Sequential 输入的元信息，shape=(2,10)；data=False 隐藏数组具体数值。
    _show("Sequential 输入", seq_x, data=False)
    # 这次没有传 data，_show 使用默认 data=True，因此也打印 shape=(2,3) 的具体输出数值。
    # 每行 3 个值仍是未经训练网络的原始分数，不是概率。
    _show("Sequential 推理输出", seq_y)
    # seq.parameters() 收集各层参数列表；len 得到两个 Linear 各 2 个、合计 4 个 Tensor。
    # ReLU、Dropout 返回空参数列表；如果统计其中可训练的标量，则有 10×8+8+8×3+3=115 个数。
    print(f"   Sequential 参数个数：{len(seq.parameters())}")

    # 打印集成测试走到末尾的提示；它表示前面的代码未触发失败，不代表已经验证所有数值和所有可能的输入。
    print("✅ 多层网络集成通过！")

    # 先用字符串乘法重复 50 个等号，再用字符串 + 把换行接在前面，最后打印；此处 + 是连接文字。
    print("\n" + "=" * 50)
    # 打印作者预设的流程完成提示。“可以迁入”不是 Python 自动审核结论，仍需考虑测试未覆盖的场景。
    print("🎉 全部测试通过！模块可以迁入。")


# 定义无参数函数 test_layers，演示 Linear 如何改变特征维度。
# 名字包含 test 不会自动执行；这个函数主要打印说明，没有 assert 自动检查。
def test_layers():
    # 打印演示主题；Linear 改变每个样本的特征数，但保留 batch 中的样本数。
    print("🎯 层在变换形状")
    # 把等号重复 45 次并打印；数字 45 只影响终端里的分隔线长度。
    print("=" * 45)

    # 创建 784 → 10 的 Linear：权重有 784×10=7840 个数，偏置有 10 个数；刚创建时还没从数据中学习。
    layer = Linear(784, 10)
    # 生成 shape=(32,784) 的随机数组并包装成 Tensor，表示 32 个模拟样本，每个样本 784 个特征。
    # 784 可以对应 28×28 像素展平后的长度，但这里并没有加载或展平真实图片。
    batch = Tensor(rng.standard_normal((32, 784)))
    # layer(batch) 会触发 Linear 从 Layer 继承的 __call__，再转到 forward，等价于这里调用 layer.forward(batch)。
    # 计算 xW+b，把 (32,784) 变为 (32,10)，并把结果赋给 output。
    output = layer(batch)

    # f-string 读取 batch.shape 并显示 (32,784)；箭头和中文是说明文字。
    # “32 张图”是这个例子的形象解释，实际输入仍是随机特征数组。
    print(f"   输入形状：{batch.shape}  ← 32 张图，每张 784 像素")
    # 输出 shape=(32,10)，表示每个样本有 10 个类别分数，不是已经为每张图确定了类别。
    # 分数不是概率；通常可取每行最大分数的位置作为预测类别，但可靠预测仍需要先训练模型。
    print(f"   输出形状：{output.shape}  ← 32 张图，每张 10 类")
    # 花括号里先计算 784×10+10=7850，再用冒号后的逗号做千位分组，显示为 7,850。
    # 这一行数的是可训练标量，和 len(layer.parameters()) 得到的 2 个参数 Tensor 是不同口径。
    print(f"   参数量：  {784 * 10 + 10:,}（权重 + 偏置）")
    # 打印演示结论：Linear 可以把输入特征线性组合成类别分数。
    # 要让分数有可靠的分类意义，还需要真实数据、标签、损失、反向传播与参数更新，这些训练步骤不在本文件中。
    print("\n✨ Linear 层把图像特征变成类别分数。")


# __name__ 是 Python 给模块设置的名字变量；直接运行本文件时，它的值为字符串 "__main__"。
# if ...: 在 == 比较为真时执行下面缩进的入口代码；注意两边都是双下划线，== 是比较而不是赋值。
# 当别的文件导入它时，__name__ 通常是模块名，下面演示不会自动运行；但顶层 import、随机数生成器创建、
# 依赖导入及类/函数定义仍会执行。因此“导入不执行主入口”不等于“导入时这个文件什么都不执行”。
if __name__ == "__main__":
    # 直接运行时先调用 test_module，执行单元测试与集成演示；有未捕获异常就停止，后面的分析不会继续。
    test_module()
    # 字符串 \n 输出一个换行，print 默认末尾再补一个换行，所以这里合计输出两个换行字符，仅用于排版。
    print("\n")
    # 调用内存估算函数，按数字配置计算参数数组大小；它没有实际创建那些大型层，也没有测整个程序的内存。
    analyze_layer_memory()
    # 打印空白分隔；不会清空变量、释放全部内存或重置随机数生成器。
    print("\n")
    # 调用性能分析，完成四种 batch 的公式估算、每种 10 次预热和 100 次正式前向计时。
    # 这可能是直接运行最慢的一段，尤其 batch=512 配合当前 Python 双循环矩阵乘法；暂时没新输出不一定是报错。
    analyze_layer_performance()
    # 打印两个换行字符，把性能分析与最后的形状演示隔开；前一函数正常返回后才会执行到这里。
    print("\n")
    # 调用最后的 Linear 形状演示，正常返回后本文件的主入口流程结束。
    # 全程只有前向、检查和演示，没有训练循环；前面导入的 Sigmoid 与定义的 HE_SCALE_FACTOR 也没有参与本文件的计算。
    test_layers()
