# 阅读路线：先看如何沿用 01 阶段的 Tensor，再看五种激活函数，最后用后面的测试观察数值和形状。
# 把张量想成“带形状的一组数字”：例如 (2, 3) 表示两条数据、每条三个数字；激活函数通常不改变形状。
# 神经网络的线性层先做加权求和，激活函数再逐个处理结果，提供非线性；没有它，多层线性变换仍可合并为一层。
# 本文件只计算前向输出，不计算梯度、损失，也不会自动训练或更新模型参数。
# Python 入门：= 把右侧结果交给左侧名字；== 比较是否相等；点号如 x.data 表示访问对象的属性。
# def 定义可调用的方法；冒号后的缩进代码属于该方法，return 把结果交回调用者。
# class 定义一种对象的规则；Sigmoid() 创建实例；方法里的 self 指正在调用方法的那个实例。
# 函数后面的括号表示调用，例如 np.exp(x)；括号里可放参数，也可用来把多行表达式写清楚。
"""02 激活函数练习；通过普通包导入复用张量。"""

# time 是计时工具，供文件后面的性能练习使用；激活函数本身不依赖它。
import time
# Path 用对象表示文件路径，后面的绘图函数用它确定图片保存位置。
from pathlib import Path

# as np 给第三方库 NumPy 起别名；它负责对一整组数字进行数组计算。
import numpy as np

# 随机种子 7 使相同调用顺序下的测试数据可重现；连续两次取随机数仍通常不同。
rng = np.random.default_rng(7)

# 点号表示当前 practices 包；普通导入会复用已经加载的模块。
# 各章节共享同一个 Tensor 类，跨模块运算时 isinstance 才能正确识别张量。
from .tensor import Tensor

# 大写名字是常量命名习惯；1e-10 等于 0.0000000001，后面的误差测试会用它。
TOLERANCE = 1e-10
# __all__ 是字符串列表，声明用 from 模块 import * 时准备导出的类名；不会自动调用这些类。
__all__ = ["Sigmoid", "ReLU", "Tanh", "GELU", "Softmax"]


# 这个辅助函数只打印张量信息，便于检查；label 是标题，tensor 是要观察的对象。
# 单独的 * 使 data 成为“仅限关键字”的参数，调用时要写 data=False；默认 True 会显示具体数字。
def _show(label, tensor, *, data=True):
    # [] 是列表；f"...{表达式}..." 把属性值插入文字，shape/size/ndim/dtype 分别是形状/总数/轴数/类型。
    parts = [f"shape={tensor.shape}", f"size={tensor.size}", f"ndim={tensor.ndim}", f"dtype={tensor.dtype}"]
    # if 后是条件：data 为 True 时执行缩进的下一行；为 False 时跳过，避免打印大量数字。
    if data:
        # insert(0, 内容) 把文字插到列表的第一项；Python 下标从 0 开始，.data 是底层 NumPy 数组。
        parts.insert(0, f"data={tensor.data}")
    # join 用逗号和空格连接列表中的文字，print 输出到屏幕；只观察，不修改张量。
    print(f"   {label}：{', '.join(parts)}")


# Sigmoid 将每个有限输入映射到 0 和 1 之间，例如 0 -> 0.5，常用于表示二分类的概率倾向。
# 这里没有可学习权重；函数本身也不会把任意一组输出自动变成“总和等于 1”的类别概率。
class Sigmoid:
    # parameters 是给上层网络收集可学习参数的接口；这个方法中的 self 指当前 Sigmoid 对象。
    def parameters(self):
        # [] 是空列表，表示此层没有权重或偏置；这与输出里没有数字是两回事。
        return []

    # forward 定义从输入 x 到输出的计算；x: Tensor 和 -> Tensor 是类型提示，Python 默认不强制检查。
    def forward(self, x: Tensor) -> Tensor:
        # 读出 Tensor 内的 NumPy 数组；把输入值存在局部变量 x_data 中，不修改 x 本身。
        x_data = x.data
        # 大正数/负数可能使指数计算溢出；errstate 只不显示指定警告，并不改变结果或修复 NaN/无穷。
        # with 的缩进范围内临时启用该警告设置，离开代码块后恢复原设置。
        with np.errstate(over="ignore", invalid="ignore"):
            # np.where(条件, 真时取的数组, 假时取的数组) 会逐个位置选择结果，输出形状与输入相同。
            # NumPy 会先计算“真”和“假”两个候选表达式，where 只负责随后挑选；被舍弃分支仍可能溢出。
            result = np.where(
                # >= 比较是否大于或等于零；数组比较得到每个位置各自的 True/False 掩码。
                x_data >= 0,
                # 非负输入用 1/(1+exp(-x))；exp 是以自然常数 e 为底的指数函数。
                1.0 / (1.0 + np.exp(-x_data)), # x >= 0 时，sigmoid(x) = 1 / (1 + exp(-x))
                # 负输入改写为 exp(x)/(1+exp(x))，数学上相同，却能避免计算 exp(-很大负数) 的溢出。
                np.exp(x_data) / (1.0 + np.exp(x_data)),
            # 在括号未合上前允许换行；这一行只结束 np.where 调用，不额外执行一次计算。
            )
        # 把算好的数组重新包装成项目自己的 Tensor；极大的有限输入会因浮点精度接近 0 或 1。
        return Tensor(result)

    # Python 特殊方法 __call__ 使 sigmoid(x) 成立；它和显式调用 sigmoid.forward(x) 用的是同一计算。
    def __call__(self, x: Tensor) -> Tensor:
        # self.forward(x) 先运行前向计算，return 再把新 Tensor 交回调用者。
        return self.forward(x)


# ReLU 是整流线性激活：负数变 0，零与正数保持原值；例如 [-2, 0, 3] -> [0, 0, 3]。
# 它让网络能表达非线性关系，但负数一侧输出恒为 0，反向传播时那一侧通常没有梯度。
class ReLU:
    # 此层没有自己要学习的数字；约定用 parameters() 告诉后续组网代码。
    def parameters(self):
        # 空列表里没有参数 Tensor；函数计算仍会对输入的每个位置产生输出。
        return []

    # 类型提示说明预期输入和输出是 Tensor；方法内部实际依赖 NumPy 的数组运算。
    def forward(self, x: Tensor) -> Tensor:
        # np.maximum(0, x.data) 对每个位置取 0 与原值中较大的一个，形状不改变。
        result = np.maximum(0, x.data) # x >= 0 时，ReLU(x) = x，否则 ReLU(x) = 0
        # 包装为新 Tensor；x.data 没有被此操作改写。
        return Tensor(result)

    # 创建 relu = ReLU() 之后，relu(x) 就会进入这个特殊方法。
    def __call__(self, x: Tensor) -> Tensor:
        # 调用本类写好的 forward 并返回结果，避免调用者必须总写 .forward。
        return self.forward(x)


# Tanh 将有限输入压到 -1 与 1 之间，0 -> 0；与只输出非负数的 ReLU 不同。
# 输入绝对值很大时接近 -1 或 1，称为“饱和”；本文件不实现训练所需的反向传播。
class Tanh:
    # 没有权重、偏置可供优化器调整。
    def parameters(self):
        # 返回一个新空列表；没有参数不影响本层处理输入。
        return []

    # forward 是前向计算的约定名称，x: Tensor 是供读者和工具看的类型提示。
    def forward(self, x: Tensor) -> Tensor:
        # np.tanh 对数组逐个元素计算双曲正切；不用手写指数公式，较易处理极大的输入。
        result = np.tanh(x.data) # tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        # 将 NumPy 的数组结果交给 Tensor 构造方法，得到新对象。
        return Tensor(result)

    # 让对象像函数一样使用，例如 Tanh()(x)；前一个 () 创建对象，后一个 () 把 x 传给对象。
    def __call__(self, x: Tensor) -> Tensor:
        # 方法内的 self 是当前创建的 Tanh 实例，不需要调用者自己传入。
        return self.forward(x)


# GELU 是神经网络中常见的平滑激活；此练习使用 x * sigmoid(1.702*x) 的近似版，而非精确 GELU。
# 与 ReLU 直接把负数截为 0 不同，这个近似式对部分负输入会保留小的负输出。
class GELU:
    # 这里的 1.702 是固定系数，不是训练参数。
    def parameters(self):
        # 返回空列表；即使调用时创建了临时 Sigmoid 对象，也没有可学习的参数。
        return []

    # 先做逐元素数值变换，再交给 Sigmoid；同形状张量之间的 * 也是逐元素乘法。
    def forward(self, x: Tensor) -> Tensor:
        # x * 1.702 产生新 Tensor；Sigmoid() 创建临时对象，其后的 (x * 1.702) 调用它的 __call__。
        # 最后再 * x 得到近似 GELU；这里 Tensor.__mul__ 不是线性层的矩阵乘法。
        return Sigmoid()(x * 1.702) * x # GELU(x) = x * sigmoid(1.702 * x)

    # gelu(x) 转调 forward；上面 return 的结果会被传回 gelu(x) 的调用处。
    def __call__(self, x: Tensor) -> Tensor:
        # self.forward(x) 内部再调用 Sigmoid，保持本类的对外调用形式与其他激活一致。
        return self.forward(x)


# Softmax 通常把同一条样本的多个类别原始分数转为非负概率，沿指定轴的概率和为 1。
# 例如一批形状 (2, 3) 的分数，dim=-1 时分别对两行的三个类别归一化，而不是混合两条样本。
class Softmax:
    # 把分数变成概率不需要本层自有的权重；训练权重一般属于前面的 Linear 等层。
    def parameters(self):
        # [] 表示参数 Tensor 的数量是零，不代表 Softmax 计算时返回空张量。
        return []

    # dim: int 是类型提示；默认 -1 表示最后一个轴，2 维数组 (batch, classes) 中即类别轴。
    # 用户可以传其他 dim，例如 dim=0 表示按批次方向归一化；轴越界或被归一化的轴为空会报错。
    def forward(self, x: Tensor, dim: int = -1) -> Tensor:
        # np.max 沿 axis=dim 找每组最大值；keepdims=True 保留此轴为长度 1，例如 (2,3)->(2,1)。
        # 保留轴使下一行 (2,3)-(2,1) 可用广播：每行的三个数字都减去该行最大值。
        x_max = np.max(x.data, axis=dim, keepdims=True)
        # 同组分数减同一个数不改变 Softmax 概率；减最大值使最大的新分数为 0，避免 exp(很大正数) 溢出。
        # 对包含 +inf 或全为 -inf 的异常输入，inf-inf 仍可能得到 NaN；这一步不保证所有输入有效。
        x_shifted = x.data - x_max # 将 x 减去 x 的最大值，使得 x 的值范围在 [-∞, 0] 之间
        # 对有限分数逐个取 exp：数学上大于 0 且不超过 1，浮点下溢时实际值也可能变成 0。
        # 最大的新分数 0 对应 exp(0)=1，因此正常有限输入的同组指数总和至少是 1。
        exp_values = np.exp(x_shifted) # 将 x 的值范围在 [-∞, 0] 之间转换为 [0, 1] 之间
        # 沿类别轴求指数值的“总和”，例如每行三个数 [1,0.5,0.2] 总和为 1.7，可以大于 1。
        # 原行尾把“求和”说成“把值变到 [0,1]”并不准确；keepdims=True 同样用于下一行广播除法。
        exp_sum = np.sum(exp_values, axis=dim, keepdims=True) # 将 exp_values 的值范围在 [0, 1] 之间转换为 [0, 1] 之间
        # 每个 exp 值除以自己那一组的总和；对正常有限输入，每组结果位于 [0,1] 且总和为 1。
        # / 是逐元素除法；例如 [1,0.5,0.2]/1.7 得到同一组的三个概率，不修改 x。
        result = exp_values / exp_sum # 将 exp_values 的值范围在 [0, 1] 之间转换为 [0, 1] 之间
        # 返回项目自己的 Tensor；这里只得到概率，还没有计算损失、预测类别或训练模型。
        return Tensor(result) # 将 result 的值范围在 [0, 1] 之间转换为 [0, 1] 之间

    # softmax(x, dim=1) 能像函数一样调用；默认使用最后一轴，dim= 也属于按名字传参数。
    def __call__(self, x: Tensor, dim: int = -1) -> Tensor:
        # 把 x 与用户指定的 dim 交给 forward；return 返回前向输出，输入张量仍保持原状。
        return self.forward(x, dim)


# def 定义函数，缩进的语句只有在调用 test_unit_sigmoid() 时才执行；这个函数检查 Sigmoid 的典型输入与极值。
# 测试函数不用 return 结果；它靠 assert 报错或顺利运行到最后表示检查结果。
def test_unit_sigmoid():
    # print 把文字显示在终端；这里只标记测试开始，不参与神经网络运算。
    print("🧪 单元测试：Sigmoid...")

    # 类名后的 () 创建一个 Sigmoid 对象；激活函数本身没有通过训练更新的权重。
    sigmoid = Sigmoid()

    # [0.0] 是只含一个数的 Python 列表，Tensor(...) 把它包装成形状为 (1,) 的一维张量。
    x = Tensor([0.0])
    # 点号表示调用对象的方法；forward 对张量内每个数应用 sigmoid(z)=1/(1+exp(-z))。
    result = sigmoid.forward(x)
    # exp(0)=1，因此 sigmoid(0)=1/2；.data 是 Tensor 内保存的 NumPy 数组。
    # 例如线性层算出的原始分数 z=0，Sigmoid 会将其映射为 0.5；具体是不是分类概率还取决于模型用途。
    # assert 条件, 提示：条件为 False 才抛 AssertionError；allclose 允许浮点运算产生微小误差。
    # f"...{result.data}..." 是格式化字符串，花括号内的表达式会换成实际结果。
    assert np.allclose(result.data, [0.5]), f"sigmoid(0) 应为 0.5，实际 {result.data}"
    # _show 是文件上方定义的展示函数；先显示输入，再显示计算结果。
    _show("输入 [0.0]", x)
    _show("sigmoid(0)", result)

    # 五个数放进一个张量，便于比较负数、零、正数各自被映射到哪里。
    x = Tensor([-10, -1, 0, 1, 10])
    # 变量 x 和 result 重新赋值；先前测试用过的对象不会因此变成新对象。
    result = sigmoid.forward(x)
    # > 和 < 对数组逐元素比较，产生布尔数组；np.all 要求其中每一项都是 True。
    # and 连接两个整体真假值：这些有限输入的结果应严格大于 0 且小于 1。
    # 每个位置各自进行变换，不像 Softmax 那样要求五个结果加起来等于 1。
    assert np.all(result.data > 0) and np.all(result.data < 1), "Sigmoid 输出应落在 (0, 1)"
    # 输出形状仍是 (5,)：激活函数改变每个元素的值，不改变元素个数。
    _show("输入 [-10, -1, 0, 1, 10]", x)
    _show("sigmoid 输出（应在 0～1）", result)

    # -1000 和 1000 是极大的有限数字，不是 Python 中的正负无穷。
    x = Tensor([-1000, 1000])
    # 此类的 forward 使用不同公式处理正数和负数，以免直接计算指数时造成不必要的溢出。
    result = sigmoid.forward(x)
    # [0] 取得第一个输出；极大负数的 sigmoid 趋近 0，atol 指允许的绝对误差。
    assert np.allclose(result.data[0], 0, atol=TOLERANCE), "sigmoid(-∞) 应接近 0"
    # [1] 取得第二个输出；极大正数趋近 1，float32 的舍入可能使结果正好显示成 1。
    # 两侧接近 0/1 称为饱和；若将来做反向传播，这两端的斜率很小，学习可能变慢。
    assert np.allclose(result.data[1], 1, atol=TOLERANCE), "sigmoid(+∞) 应接近 1"
    # 查看实际输出，确认输入的两个元素分别落在两个饱和端附近。
    _show("极值输入 [-1000, 1000]", x)
    _show("sigmoid 极值输出", result)

    # 运行到此处说明前面的断言都未失败；它不表示网络已经训练。
    print("✅ Sigmoid 通过！")


# 用多个边界样本验证 ReLU：负数归零，零仍为零，正数原样保留。
# ReLU 在负输入一侧的斜率为 0，正输入一侧的斜率为 1；本文件仅计算输出，没有求导。
def test_unit_relu():
    # 函数调用需要括号；显示标题只是方便读终端输出。
    print("🧪 单元测试：ReLU...")

    # 新建一个 ReLU 对象；它的规则是 relu(z)=max(0,z)。
    relu = ReLU()

    # 一维列表有五个元素：两个负数、零、两个正数。
    # 正负号对应线性层给出的中间分数，不要求原始数据本身非负。
    x = Tensor([-2, -1, 0, 1, 2])
    # forward 返回新的 Tensor；不会让 x.data 自动变成计算后的结果。
    result = relu.forward(x)
    # expected 是普通 Python 列表，表示人为手算的期望输出。
    expected = [0, 0, 0, 1, 2]
    # allclose 逐个比较数值，全部足够接近才返回 True；报错时 f-string 会同时显示两组值。
    assert np.allclose(result.data, expected), f"ReLU 失败，期望 {expected}，实际 {result.data}"
    # 查看输入和输出，有助于把“逐个取 0 与原值的较大者”与结果对应起来。
    _show("输入 [-2, -1, 0, 1, 2]", x)
    _show("ReLU 输出", result)

    # 重新绑定变量 x；此测试专门覆盖“所有元素均为负数”的情况。
    x = Tensor([-5, -3, -1])
    # 对三个元素分别做 max(0,z)，预期全为零。
    # 若神经元一直处在负输入区域，训练时该 ReLU 对输入的梯度长期为 0，可能难以恢复。
    result = relu.forward(x)
    # 列表 [0,0,0] 是期望值，np.allclose 检查三个位置，而非只比较其中一个位置。
    assert np.allclose(result.data, [0, 0, 0]), "ReLU 应将全部负数置零"
    # _show 只展示数值、形状等信息，不会修改张量。
    _show("全负输入", x)
    _show("ReLU 输出", result)

    # 全正样本检验 ReLU 的另一半定义，正数直接保留。
    x = Tensor([1, 3, 5])
    # 每个元素经过同一个激活规则；输出形状仍为 (3,)。
    result = relu.forward(x)
    # 数字仍是 1、3、5，但一般不应假设返回对象和输入对象具有相同身份。
    assert np.allclose(result.data, [1, 3, 5]), "ReLU 应保持正数不变"
    # 接连调用两次 _show，分别观察激活前与激活后。
    _show("全正输入", x)
    _show("ReLU 输出", result)

    # 四个输入中恰有三个负数，因此激活后应该恰有三个零。
    x = Tensor([-1, -2, -3, 1])
    # 这只是刻意构造的示例，并不能推出任意网络或任意数据都一定有 75% 的零。
    result = relu.forward(x)
    # == 对数组每一项判断是否等于 0，np.sum 把 True 当 1、False 当 0，得到零的个数。
    zeros = np.sum(result.data == 0)
    # == 在此比较两个整数；只有零的个数恰好是 3，断言才通过。
    assert zeros == 3, f"ReLU 应产生稀疏，4 个元素里有 {zeros} 个零"
    # “稀疏”在这里是指结果中零元素较多，不代表网络已经学到了稀疏表示。
    # 零值可以减少随后某些位置的贡献，但不能仅据此推出运行速度一定变快。
    _show("稀疏输入", x)
    _show("ReLU 输出", result)
    # int(...) 把 NumPy 算出的个数转成 Python 整数，再放进 f-string 显示。
    print(f"   零的个数：{int(zeros)} / 4")

    # 前面的断言没有中断执行，才显示通过。
    print("✅ ReLU 通过！")


# Tanh 与 Sigmoid 都改变每个元素的值；Tanh 输出含负数，且满足 tanh(-z)=-tanh(z)。
# Tanh 的输出以 0 为中心，例如同样大小的正负输入对应相反数。
def test_unit_tanh():
    # 输出本组测试标题；稍后需显式调用这个函数才会真正执行。
    print("🧪 单元测试：Tanh...")

    # 创建 Tanh 对象，其 forward 内部调用 NumPy 的 np.tanh。
    tanh = Tanh()

    # [0.0] 表示一维、单元素的张量，不等同于 Tensor(0.0) 的零维标量。
    x = Tensor([0.0])
    # Tanh 将零映射到零，并保留输入形状 (1,)。
    result = tanh.forward(x)
    # np.allclose 容忍浮点误差；后面的 f-string 在断言失败时显示实际数组。
    assert np.allclose(result.data, [0.0]), f"tanh(0) 应为 0，实际 {result.data}"
    # 打印两个张量，方便核对该点的输入与输出。
    _show("输入 [0.0]", x)
    _show("tanh(0)", result)

    # 负、零、正输入一起测试，观察 Tanh 从接近 -1 逐渐变到接近 1。
    # 这五个数字同时处理，得到的仍是五个数字；它们之间没有做 Softmax 式的归一化。
    x = Tensor([-10, -1, 0, 1, 10])
    # 逐元素运算得到五个输出，不会将五个输入加总。
    result = tanh.forward(x)
    # >=、<= 在数组上逐项产生 True/False；np.all 和 and 要求每项都处于 [-1,1]。
    assert np.all(result.data >= -1) and np.all(result.data <= 1), "Tanh 输出应落在 [-1, 1]"
    # 显示范围；它不证明不同输入的输出一定都不同，极端值可能饱和到边界。
    _show("输入 [-10, -1, 0, 1, 10]", x)
    _show("tanh 输出（应在 -1～1）", result)

    # 选择一正一负、绝对值相同的输入，验证 Tanh 的奇函数对称性。
    x = Tensor([2.0])
    # 正数 2 的计算结果暂存为 pos_result。
    pos_result = tanh.forward(x)
    # x_neg 是另一个张量对象，负号写在数字前面，得到 -2.0。
    x_neg = Tensor([-2.0])
    # 再计算负数一侧；两个结果都是形状 (1,) 的数组。
    neg_result = tanh.forward(x_neg)
    # -neg_result.data 对整个 NumPy 数组逐元素取相反数，不会永久修改 neg_result。
    assert np.allclose(pos_result.data, -neg_result.data), "tanh 应对称：tanh(-x) = -tanh(x)"
    # 查看这两个输出的正负号、大小是否近似对称。
    _show("tanh(2)", pos_result)
    _show("tanh(-2)", neg_result)

    # 再测试两侧非常大的有限输入；这里只检查数值趋向，不是在传数学上的无穷。
    x = Tensor([-1000, 1000])
    # 由于浮点数表示精度有限，输出可能直接显示为 -1 和 1。
    # Tanh 饱和端的斜率也趋近零；这里只验证值，不测训练时的梯度大小。
    result = tanh.forward(x)
    # [0] 为第一项；atol=TOLERANCE 指绝对容忍误差，避免强求浮点计算精确相等。
    assert np.allclose(result.data[0], -1, atol=TOLERANCE), "tanh(-∞) 应接近 -1"
    # [1] 为第二项，极大正数的输出接近 1。
    assert np.allclose(result.data[1], 1, atol=TOLERANCE), "tanh(+∞) 应接近 1"
    # 打印原值和结果，将“趋近于边界”与实际数字联系起来。
    _show("极值输入 [-1000, 1000]", x)
    _show("tanh 极值输出", result)

    # 该测试函数执行到此处说明上述几组输入均通过检查。
    print("✅ Tanh 通过！")


# 此项目的 GELU 使用 x*sigmoid(1.702*x) 近似形式；不是精确积分定义的 GELU。
# 乘法中的 sigmoid 输出在 0 到 1 附近，可把它想成按输入大小调整 x 的保留比例。
def test_unit_gelu():
    # print 只用于标识当前检查项目。
    print("🧪 单元测试：GELU...")

    # 创建 GELU 对象；激活函数不负责修改前后线性层的权重。
    gelu = GELU()

    # 输入零时，近似公式中的末尾乘数 x 为 0，因此输出是零。
    x = Tensor([0.0])
    # forward 返回封装好的 Tensor；其 .data 可供后续断言读取。
    result = gelu.forward(x)
    # atol 是绝对误差上限；即使数学上为零，测试浮点结果时也用近似比较。
    assert np.allclose(result.data, [0.0], atol=TOLERANCE), f"GELU(0) 应约等于 0，实际 {result.data}"
    # 展示该边界点，输出形状依旧为单元素 (1,)。
    _show("输入 [0.0]", x)
    _show("GELU(0)", result)

    # 再选正输入 1；这个近似形式的输出约为 0.846，不会像 ReLU 那样恰好等于 1。
    x = Tensor([1.0])
    # 调用同一个 gelu 对象处理另一份输入。
    result = gelu.forward(x)
    # [0] 读单元素数组第一项；这里只规定必须大于 0.8，并未精确验证 0.846。
    assert result.data[0] > 0.8, f"GELU(1) 应约等于 0.84，实际 {result.data[0]}"
    # 检查正数在这里被保留了大部分，却没有直接原样通过。
    _show("输入 [1.0]", x)
    _show("GELU(1)", result)

    # 输入 -1 用于对比 ReLU：近似 GELU 的负侧输出通常不会直接截成零。
    x = Tensor([-1.0])
    # 结果在这个点约为 -0.154，仍保留一点负值。
    # 与 ReLU(-1)=0 对照，就能看到两种非线性激活即使形状相同，数字行为仍不同。
    result = gelu.forward(x)
    # and 要两次标量比较都成立，即输出落在 (-0.2, 0) 内；并非测试所有负输入。
    assert result.data[0] < 0 and result.data[0] > -0.2, f"GELU(-1) 应约等于 -0.16，实际 {result.data[0]}"
    # 对照打印近似 GELU 的负侧行为。
    _show("输入 [-1.0]", x)
    _show("GELU(-1)", result)

    # 选择 0 附近的三个点，检查输出没有突然跳变。
    x = Tensor([-0.001, 0.0, 0.001])
    # 一个 Tensor 同时存三个点，激活函数会分别计算三个输出。
    result = gelu.forward(x)
    # abs 是取绝对值；下标 [1] 为零处，[0] 为左侧，求两个相邻输出的差距。
    diff1 = abs(result.data[1] - result.data[0])
    # [2] 为右侧；再次求相邻点的输出差距。
    diff2 = abs(result.data[2] - result.data[1])
    # 小于 0.01 表明这几个选定点的输出相近；有限采样并不能严格证明函数处处平滑。
    # 这个断言也没有求导，因此不能代替梯度连续性或精确 GELU 公式的数学证明。
    assert diff1 < 0.01 and diff2 < 0.01, "GELU 在 0 附近应平滑"
    # 展示三个邻近位置的结果，观察随输入小幅变化时输出如何改变。
    _show("0 附近输入", x)
    _show("GELU 输出（应平滑）", result)

    # 运行到此处说明这组示例检查通过，没有进行求导或反向传播。
    print("✅ GELU 通过！")


# Softmax 把同一组原始分数转成非负且总和约为 1 的数；默认沿最后一维各组分别计算。
# 其核心是 exp(每个分数) / 同组 exp 的总和；不同于逐个独立计算的 Sigmoid。
def test_unit_softmax():
    # 测试标题，不属于计算逻辑。
    print("🧪 单元测试：Softmax...")

    # 创建 Softmax 对象；其 forward(x, dim=-1) 可指定沿哪条轴进行归一化。
    softmax = Softmax()

    # [1,2,3] 表示同一组的三个原始分数（logits），并不是三个已经训练好的概率。
    x = Tensor([1, 2, 3])
    # 一维输入只有一条轴；默认 dim=-1 指这条最后的轴，对三个数一起计算。
    result = softmax.forward(x)
    # np.sum 汇总所有输出；allclose 允许有限浮点误差，理论上这一组的和为 1。
    # 例如三个原始分数虽是 1、2、3，归一化后仍保留大小顺序，但和不再是 6。
    assert np.allclose(np.sum(result.data), 1.0), f"Softmax 应归一化为 1，实际和为 {np.sum(result.data)}"
    # 对这里有限且不太悬殊的三个输入，每个概率大于 0；极端数值下可能下溢显示为 0。
    assert np.all(result.data > 0), "Softmax 各分量应为正"
    # 同组有多个有限值，这个具体样本的每一项都小于 1；不表示所有实现都绝不会舍入成 1。
    assert np.all(result.data < 1), "Softmax 各分量应小于 1"
    # np.argmax 找出最大值的下标；Python 下标从 0 开始，所以此处最大原始分数位于下标 2。
    max_input_idx = np.argmax(x.data)
    # 再找归一化后最大的输出在哪个下标。
    max_output_idx = np.argmax(result.data)
    # Softmax 不会颠倒同组分数的大小顺序，所以两个最大值的位置应一致。
    assert max_input_idx == max_output_idx, "最大输入应对应最大输出"
    # 这里展示的是“原始分数”和“归一化结果”；没有标签和损失函数，仍未计算分类是否正确。
    _show("输入 [1, 2, 3]", x)
    _show("softmax 输出", result)
    # f-string 的 :.6f 把显示文字保留六位小数，内部的实际数值没有被四舍五入覆盖。
    print(f"   各分量之和：{np.sum(result.data):.6f}")

    # 数值很大时直接计算 exp(1002) 可能溢出；同减组内最大值可防止这一问题。
    x = Tensor([1000, 1001, 1002])
    # forward 内部先减去最大分数，差值成为 [-2,-1,0]，所得概率应与 [1,2,3] 接近。
    # 同组所有输入都加或减同一常数，Softmax 的比例不变，这是数值稳定处理的数学依据。
    result = softmax.forward(x)
    # 确认仍满足这一组的总和约等于 1。
    assert np.allclose(np.sum(result.data), 1.0), "大数输入时 Softmax 仍应归一化"
    # np.isnan 逐个找 NaN（无效数），np.any 检查有无命中，not 表示“不能出现”。
    assert not np.any(np.isnan(result.data)), "Softmax 不应出现 NaN"
    # np.isinf 同样逐个查正无穷或负无穷；这里检查没有 Inf。
    assert not np.any(np.isinf(result.data)), "Softmax 不应出现 Inf"
    # 展示结果，检查三个大数并未使 Softmax 输出溢出。
    _show("大数输入 [1000, 1001, 1002]", x)
    _show("softmax 输出（数值稳定）", result)

    # 嵌套列表 [[1,2],[3,4]] 是一个两行两列的矩阵，形状 (2,2)。
    x = Tensor([[1, 2], [3, 4]])
    # dim=-1 是按最后一维归一化，即每行的两个数各自成为一组，不跨行混合。
    # 对常见 (批次数, 类别数) 形状，行就是一条样本，列就是各个类别原始分数。
    result = softmax.forward(x, dim=-1)
    # shape 是各维长度构成的元组；(2,2) 不代表四个输出加在一起等于 1。
    assert result.shape == (2, 2), "Softmax 应保持输入形状"
    # np.sum 的 axis=-1 表示沿每行的最后一维求和，得到含两个数的一维数组。
    # 如果错误地沿 axis=0 求和，检查到的是“每列跨两个样本的和”，意义便不同。
    row_sums = np.sum(result.data, axis=-1)
    # 第一行的和约为 1，第二行的和也约为 1；两个条件都满足，断言才通过。
    assert np.allclose(row_sums, [1.0, 1.0]), "每一行应归一化为 1"
    # 将输入和两组概率分开显示，便于对照分组方式。
    _show("二维输入", x)
    _show("按最后一维 softmax", result)
    # f-string 将 NumPy 的 row_sums 数组嵌入输出文字。
    print(f"   各行之和：{row_sums}")

    # 测试通过只表示这些示例的前向数值符合预期，不表示模型已完成训练。
    print("✅ Softmax 通过！")


# 集成测试将前面的单元测试串在一起，再检查多种激活在实际张量上的配合。
# 函数以 def 定义时不会自动运行；文件末尾的主入口会调用它。
# “通过”仅表示下列示例检查正确，不代表已经训练好神经网络。
def test_module():
    # print 输出文字；这是本组测试标题。
    print("🧪 正在运行模块集成测试")
    # 字符串乘整数会重复，例如 "=" * 50 会产生一条五十字符的分隔线。
    print("=" * 50)

    # 以下五个 test_unit_*() 是函数调用；Python 顺序执行，某项 assert 失败就会中断。
    print("运行单元测试...")
    # Sigmoid 检查输出落在 0 到 1 的区间，并关注极大正负数的情况。
    test_unit_sigmoid()
    # ReLU 检查负数归零、非负数不变。
    test_unit_relu()
    # Tanh 检查输出在 -1 到 1 之间以及正负输入的对称性。
    test_unit_tanh()
    # 此处是近似 GELU，检查部分正负输入及零附近的平滑程度。
    test_unit_gelu()
    # Softmax 检查类别概率的总和及其对大数输入的数值稳定性。
    test_unit_softmax()

    # \n 表示换行；标题中的“集成”指将多种已实现的层放在一个场景验证。
    print("\n运行集成场景...")

    # 第一组集成场景确认各逐元素激活仍输出本项目的 Tensor，并保持形状。
    print("🧪 集成测试：保持张量属性...")
    # 两层嵌套列表是二维数据，shape 为 (2,2)；可看作两条样本、每条两个值。
    test_data = Tensor([[1, -1], [2, -2]])
    # 显示输入的数字和 shape，便于与各激活输出对照。
    _show("测试输入", test_data)

    # 方括号创建列表；四个 ClassName() 分别构造一个激活对象。
    # 没有 Softmax：它会在一整组分数之间做归一化，虽也保持 shape，却不是逐元素独立处理。
    activations = [Sigmoid(), ReLU(), Tanh(), GELU()]
    # for 依次取出列表中的一个对象，每次执行下方缩进的四行。
    for activation in activations:
        # 前向传播：使用本轮激活把输入每个数变为输出，结果交给局部变量 result。
        result = activation.forward(test_data)
        # 双下划线属性 __class__.__name__ 取得当前类名，供提示和展示使用。
        name = activation.__class__.__name__
        # == 比较形状元组；形状不相等就触发 AssertionError，逗号后是失败说明。
        assert result.shape == test_data.shape, f"{name} 未保持形状"
        # isinstance(对象, 类型) 检查输出是 Tensor，而不是直接返回一个 NumPy 数组。
        assert isinstance(result, Tensor), f"{name} 输出不是 Tensor"
        # 分别显示 Sigmoid、ReLU、Tanh、GELU 的具体数字。
        _show(name, result)

    # 所有四轮都走完才打印通过。
    print("✅ 各激活均保持张量属性！")

    # 第二组检查三维输入按某个轴归一化的行为。
    print("🧪 集成测试：Softmax 维处理...")
    # 三层嵌套列表得到 shape (2,2,3)：2 组，每组 2 行，每行 3 个类别分数。
    data_3d = Tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    # 创建 Softmax 对象；它本身没有需要训练的权重。
    softmax = Softmax()
    # 对象可像函数调用，因为实现了 __call__；dim=-1 指最后一轴的三个类别。
    # 每一行的 [1,2,3] 等三个分数分别转为概率；不会把两组数据混在一起。
    result_last = softmax(data_3d, dim=-1)
    # 归一化改变数值，但仍保留三个轴及其原先的长度。
    assert result_last.shape == (2, 2, 3), "Softmax 应保持形状"
    # np.sum(..., axis=-1) 沿最后一轴求和，输出 shape (2,2) 的四个行和。
    last_dim_sums = np.sum(result_last.data, axis=-1)
    # allclose 对数组的每个值做容许浮点误差的比较；标量 1.0 会广播到四个和。
    assert np.allclose(last_dim_sums, 1.0), "最后一维应归一化为 1"
    # 显示输入分数和归一化后的概率；每组内部较大的原分数对应较大概率。
    _show("三维输入", data_3d)
    _show("softmax(dim=-1)", result_last)
    # 打印四组结果的和，预期都是约 1.0；概率计算有浮点误差。
    print(f"   最后一维之和：{last_dim_sums}")

    # 前述两项断言均成功时显示通过。
    print("✅ Softmax 维处理通过！")

    # 第三组把两种激活前后连接：上一层输出就是下一层输入。
    print("🧪 集成测试：激活串联...")
    # 两层方括号使 x 为 shape (1,4)，即一条样本、四个原始数字。
    x = Tensor([[-1, 0, 1, 2]])
    # 创建非线性激活 ReLU。
    relu = ReLU()
    # 输入 [-1,0,1,2] 经过 ReLU 后变 [0,0,1,2]，shape 仍是 (1,4)。
    hidden = relu.forward(x)
    # 创建 Softmax，将这一行四个分数归一化成概率。
    softmax = Softmax()
    # 输出是 shape (1,4) 的新 Tensor；四个概率的和应约为 1。
    output = softmax.forward(hidden)
    # [0,0] 是第 0 行第 0 列；断言原来的负数 -1 已被 ReLU 截成零。
    assert hidden.data[0, 0] == 0, "ReLU 应将负数置零"
    # 对所有四个输出求和；这里仅一条样本，故整体求和与该行求和一致。
    assert np.allclose(np.sum(output.data), 1.0), "最终输出应为概率分布"
    # 按计算顺序打印：原始数据、ReLU 中间结果、Softmax 概率。
    _show("输入", x)
    _show("ReLU 后", hidden)
    _show("Softmax 后", output)
    # :.6f 只控制显示六位小数，不改变概率真实存储的数值。
    print(f"   Softmax 各分量之和：{np.sum(output.data):.6f}")

    # 以上连接过程的断言都没有失败。
    print("✅ 激活串联通过！")

    # 字符串用 + 拼接、* 重复，组合为空行与分隔线。
    print("\n" + "=" * 50)
    # 这一行是测试结果的固定文案；并没有真正执行“迁入”源代码的操作。
    print("🎉 全部测试通过！模块可以迁入。")


# 这只是本机上对四种实现的一次小型前向计时，耗时不是算法的固定属性。
# 它会统计创建新 Tensor 等运行开销，而且只有 ReLU/Sigmoid 提前预热。
def analyze_activation_performance():
    # 打印分析标题，未开始计时。
    print("📊 分析激活函数计算开销...")
    # 重复等号作视觉分隔。
    print("=" * 60)

    # Python 中 1000000 是整数一百万，代表这次创建一百万个输入数字。
    size = 1000000
    # rng.standard_normal 产生正态分布随机数，astype 转为 float32；Tensor 包装为张量。
    # 固定随机种子令相同执行顺序较容易复现数据，不会保证计时结果一致。
    test_data = Tensor(rng.standard_normal(size).astype(np.float32))
    # {size:,} 按千位分隔显示 1,000,000；这只是打印用的格式。
    print(f"\n测试规模：{size:,} 个元素（模拟较大隐层）")
    # 一维输入 shape 为 (1000000,)；dtype 为 float32。
    print(f"   shape={test_data.shape}，dtype={test_data.dtype}")
    # 打印横线，以下开始比较各激活。
    print("-" * 60)

    # 每个类名后跟 () 会创建一个可以执行前向计算的对象。
    relu = ReLU()
    # Sigmoid 把各输入压到 0 与 1 附近。
    sigmoid = Sigmoid()
    # Tanh 把各输入压到 -1 与 1 附近。
    tanh = Tanh()
    # 这里的 GELU 是 x*sigmoid(1.702*x) 的近似实现。
    gelu = GELU()

    # 预先运行 ReLU 与 Sigmoid；下划线 _ 是“结果不需要再用”的普通变量名。
    _ = relu(test_data)
    # Tanh 和 GELU 没有同等预热，因此四组计时条件并不完全一致。
    _ = sigmoid(test_data)

    # 每种函数各运行十次，取平均毫秒数；次数越少，短时系统波动越明显。
    n_runs = 10

    # time.time() 读取当前时间（秒）；减去开始时间得到这一段运行秒数。
    start = time.time()
    # range(10) 会生成 0 至 9；本循环执行十次，_ 的值没有被用来作索引。
    for _ in range(n_runs):
        # 每轮都处理一百万个值，并新建输出 Tensor。
        _ = relu(test_data)
    # 结束时间减开始时间，除以 10 得平均秒数，再乘 1000 转成毫秒。
    relu_time = (time.time() - start) / n_runs * 1000

    # 重新记录时间，避免把 ReLU 的耗时混到 Sigmoid 中。
    start = time.time()
    # 执行相同次数以便大致比较。
    for _ in range(n_runs):
        # Sigmoid 的具体计算包含 exp 等运算；仍对每个输入分别处理。
        _ = sigmoid(test_data)
    # 得到 Sigmoid 单次平均毫秒数。
    sigmoid_time = (time.time() - start) / n_runs * 1000

    # 第三次独立开始计时。
    start = time.time()
    # 重复运行 Tanh 十次。
    for _ in range(n_runs):
        # 各次输出被交给 _，下一轮会覆盖这个名字。
        _ = tanh(test_data)
    # 计算 Tanh 的平均毫秒数。
    tanh_time = (time.time() - start) / n_runs * 1000

    # 第四次独立开始计时。
    start = time.time()
    # 重复运行近似 GELU 十次。
    for _ in range(n_runs):
        # 此实现内部还会创建并调用一个临时 Sigmoid 对象。
        _ = gelu(test_data)
    # 计算这一实现的平均毫秒数，包含对象创建和新 Tensor 包装时间。
    gelu_time = (time.time() - start) / n_runs * 1000

    # 再准备一组易于人工查看的五个数字；这部分输出不计入上述耗时。
    sample = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    # 标题前的 \n 表示换行。
    print("\n小样本对照：")
    # 展示相同输入经过各函数后数字如何变化。
    _show("输入", sample)
    # ReLU 输出 [0,0,0,1,2]。
    _show("ReLU", relu(sample))
    # Sigmoid 每个位置落在 0 到 1，0 对应 0.5。
    _show("Sigmoid", sigmoid(sample))
    # Tanh 的 0 对应 0，负输入映射为负输出。
    _show("Tanh", tanh(sample))
    # 近似 GELU 在部分负输入会保留负输出。
    _show("GELU", gelu(sample))

    # 以下只展示上面得到的平均时间；屏幕显示会因设备和系统负载波动。
    print("\n🧪 激活耗时：")
    # :.2f 显示两位小数，不把原本浮点数据截断保存。
    print(f"   ReLU：   {relu_time:.2f}ms（基准）")
    # 用 Sigmoid 时间除以 ReLU 时间得本次相对倍数，不是普适速度常数。
    print(f"   Sigmoid：{sigmoid_time:.2f}ms（慢 {sigmoid_time/relu_time:.1f} 倍）")
    # :.1f 让倍数显示一位小数；若本次更快，固定的“慢”字也不会自动改。
    print(f"   Tanh：   {tanh_time:.2f}ms（慢 {tanh_time/relu_time:.1f} 倍）")
    # GELU 耗时依赖本文件使用的近似公式和 Python/NumPy 环境。
    print(f"   GELU：   {gelu_time:.2f}ms（慢 {gelu_time/relu_time:.1f} 倍）")

    # 以下四点都是原文件预写的经验说明，输出文案不会依据本次结果自动更新。
    print("\n" + "=" * 60)
    # 文字标题，不对应一个新的测试或断言。
    print("要点：")
    # ReLU 这里用逐元素 maximum，通常计算简单；实际总耗时还包括内存访问。
    print("   1. ReLU 最快：只需 max(0, x)，没有指数运算")
    # Sigmoid 用指数；Tanh 交给 NumPy 的 np.tanh，不能由这行断言其底层实现方式。
    print("   2. Sigmoid / Tanh 需要 exp()，更贵")
    # GELU 在本练习里调用 Sigmoid；别推广为所有 GELU 实现都这样计算。
    print("   3. GELU 内部用 Sigmoid，开销跟着它走")
    # 多层网络会多次调用激活；具体瓶颈仍取决于线性层、设备和 batch 等因素。
    print("   4. 隐层数量一大，ReLU 的速度优势会累加")

    # 下方是框架/模型举例的固定文字，不是在此创建 ResNet 或 GPT。
    print("\n实际影响：")
    # 这里没有加载 ResNet，也没有测其前向性能。
    print("   - ResNet 用 ReLU：一次前向有海量激活")
    # 这里没有训练 GPT、计算梯度或证明平滑激活在任何任务上都更好。
    print("   - GPT 用 GELU：多花算力换更平滑的梯度")
    # 具体模型也会使用其他激活；用途取决于任务和结构。
    print("   - Sigmoid / Tanh：多用于输出层或门控")
    # 结束打印分隔线。
    print("=" * 60)


# 用同一组五个数字直观对比 ReLU、Sigmoid 和 Softmax；这里没有权重更新。
def test_activations():
    # 标题文字说明激活的作用；非线性让多层网络有能力表达更复杂的关系。
    print("🎯 激活函数给网络加上非线性")
    # 用 * 重复分隔符，不是张量乘法。
    print("=" * 45)

    # np.array 构造五个输入数字，Tensor 再把底层数字统一为 float32。
    # 这一维 shape (5,) 只是一个向量；没有批次轴，不能直接说它有五条样本。
    x = Tensor(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
    # _show 打印原始数据和形状供对照。
    _show("输入", x)

    # ReLU() 创建激活对象。
    relu = ReLU()
    # 写 relu(x) 会进入类的 __call__，再执行 forward；得到新 Tensor。
    relu_out = relu(x)
    # 输出应对应 [0,0,0,1,2]，仍有五个元素。
    _show("ReLU", relu_out)
    # 固定说明文字，不是新的条件检查。
    print("   负数变 0，正数不变")

    # 创建 Sigmoid 对象。
    sigmoid = Sigmoid()
    # 对每个位置各自计算，0 对应 0.5；五个输出通常不会加起来等于 1。
    sigmoid_out = sigmoid(x)
    # 显示逐元素压缩到 0 与 1 之间的结果。
    _show("Sigmoid", sigmoid_out)
    # Sigmoid 适合解释单个二分类输出的倾向，不等于多类别整体归一化。
    print("   全部压到 (0, 1)")

    # Softmax() 创建无可学习参数的归一化对象。
    softmax = Softmax()
    # 默认 dim=-1，对向量的全部五个分数一起做指数归一化。
    softmax_out = softmax(x)
    # 五个输出仍是 shape (5,)，可解释为五个互斥类别的概率。
    _show("Softmax", softmax_out)
    # .data.sum() 是 NumPy 数组的方法；输出的五个概率之和应约为 1。
    # :.1f 显示一位小数，打印 1.0 不表示存储的浮点值绝对精确等于 1.0。
    print(f"   各分量之和 = {softmax_out.data.sum():.1f}（合法概率分布）")

    # 结语是概念说明，这个函数没有计算损失、梯度，也没有训练网络。
    print("\n✨ 激活带来非线性，这是深度网络能拟合复杂函数的关键。")

# 绘制两类别 Softmax 的示意图；此函数只在被主动调用时运行。
# 主入口末尾把 plot_activations() 注释掉了，正常执行此脚本不会生成图片。
def plot_activations():
    # 在函数内部导入 Matplotlib；只有真的调用此函数才需要安装该库。
    # 环境若没有 matplotlib，这里会抛 ModuleNotFoundError，前面的激活测试仍可独立运行。
    import matplotlib.pyplot as plt

    # rcParams 是绘图库的全局显示设置；列表按顺序列出供中文字符使用的候选字体。
    # 不同电脑不一定安装这些字体，缺字时图上的中文可能无法正常显示。
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "SimHei"]
    # False 让负数刻度尽量使用普通减号；只影响图的显示，不改变数据。
    plt.rcParams["axes.unicode_minus"] = False

    # linspace(-5,5,400) 在 -5 到 5（包括两端）均匀取 400 个 x 值。
    xs = np.linspace(-5, 5, 400)
    # zeros_like(xs) 生成 400 个零；stack(...,axis=1) 将两列分数合成 (400,2)。
    # 每行 [x,0] 表示当前 x 下类别 0 与类别 1 的两个原始分数（logits）。
    logits = np.stack([xs, np.zeros_like(xs)], axis=1)
    # Softmax() 先构造对象，后面的 (...) 调用对象；默认沿末轴归一化。
    # probs 为 (400,2) 的 NumPy 数组，每行两个概率相加约为 1。
    probs = Softmax()(Tensor(logits)).data

    # 建立宽 8 英寸、高 4 英寸的图；figsize 不是图片的像素尺寸。
    plt.figure(figsize=(8, 4))
    # probs[:,0]：冒号选所有 400 行，0 选每行第一列，得到类别 0 的概率曲线。
    plt.plot(xs, probs[:, 0], label="P(类0)，分数=x")
    # probs[:,1] 取第二列；类别 1 的 logit 恒为 0，但其概率随 x 变化。
    plt.plot(xs, probs[:, 1], label="P(类1)，分数=0")
    # 在概率 0.5 的高度画灰色虚线；lw 是线宽，ls="--" 表示虚线。
    plt.axhline(0.5, color="gray", lw=0.6, ls="--")
    # 在 x=0 处画竖线；此时两个原始分数一样，概率各为 0.5。
    plt.axvline(0, color="gray", lw=0.6, ls="--")
    # 添加图标题；logits=[x,0] 中的两个数字尚未归一化。
    plt.title("两类 Softmax：logits=[x, 0]")
    # 横轴表示第 0 类的原始分数 x。
    plt.xlabel("x")
    # 纵轴表示输出概率，正常有限输入应落在 0 与 1 之间。
    plt.ylabel("概率")
    # 将画图范围稍微扩到 -0.05 与 1.05，留一些上下边距，不改变概率。
    plt.ylim(-0.05, 1.05)
    # 根据前两条 plot 的 label 显示图例。
    plt.legend()
    # 显示浅色网格；alpha=0.3 控制透明度。
    plt.grid(True, alpha=0.3)

    # __file__ 是当前 activations.py 路径；resolve() 转绝对路径，parent 是所在目录。
    # / "softmax.png" 拼出同目录下的输出文件路径，若文件存在 savefig 会覆盖它。
    out = Path(__file__).resolve().parent / "softmax.png"
    # 只有主动调用此函数时，才以 120 DPI 把当前图保存到 softmax.png；没有 plt.show()。
    plt.savefig(out, dpi=120)
    # 打印保存路径；仅定义函数、导入模块或运行当前的主入口均不生成图片。
    print(f"已保存 {out}")


# Python 把直接运行的文件的 __name__ 设成 "__main__"；被其他文件加载时通常不是。
if __name__ == "__main__":
    # 先调用单元与集成测试；若 assert 失败，程序通常不会继续执行到下一行。
    test_module()
    # 打印一个空行，便于把两个部分的终端输出分开。
    print("\n")
    # 再运行四种激活的一百万数字计时；其结论只反映本次环境。
    analyze_activation_performance()
    # 再隔开终端输出。
    print("\n")
    # 展示五个数字经 ReLU、Sigmoid、Softmax 的结果。
    test_activations()
    # 最后仍只打印空行，不调用画图。
    print("\n")
    # 井号 # 开始注释，下面一行不是调用语句；正常运行脚本不需要 matplotlib 或写图。
    # plot_activations()
