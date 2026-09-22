"""验证练习模块可以按任意顺序导入，并复用同一套张量与前向计算。"""

# 子进程提供全新的模块缓存，避免当前测试进程掩盖重复加载问题。
import subprocess
# 复用运行 pytest 的解释器，保证子进程具有相同的依赖环境。
import sys
# 路径由测试文件定位，不依赖调用者启动命令时所在的目录。
from pathlib import Path
# 多行脚本统一去掉外层缩进，交给 Python 的 -c 参数执行。
from textwrap import dedent

# NumPy 直接计算预期值，避免用待测张量运算生成自己的断言答案。
import numpy as np
# 参数化测试分别覆盖从前置模块开始与从后续模块开始的导入顺序。
import pytest

# 包括尚未实现的练习：占位阶段也必须能够正常复用已有模块。
# 模块顺序仅用于测试输入，不约束实际程序的导入顺序。
MODULES = ("tensor", "activations", "layers", "losses", "autograd", "nn", "optim", "data", "training", "export")


# 正序从张量开始，逆序从导出开始，会触发不同的依赖加载路径。
# 两种顺序分别启动子进程，不共享 sys.modules 中的导入结果。
@pytest.mark.parametrize("module_names", [MODULES, MODULES[::-1]], ids=["forward", "reverse"])
def test_import_order_preserves_tensor_identity_without_demo_output(module_names):
    """任意学习阶段先导入，都应得到相同的 Tensor，且不会运行演示。"""
    # 脚本只执行导入与断言；因此捕获到的输出只能来自导入副作用或错误。
    # 导入顺序通过独立命令行参数传入，避免拼接可执行的模块名字符串。
    script = dedent("""\
        # importlib 遵循普通包导入规则和 Python 的模块缓存。
        import importlib
        # argv 从测试进程接收待导入的模块名。
        import sys
        # 一次导入全部十个练习，包括暂时只有依赖声明的模块。
        modules = {name: importlib.import_module('practices.' + name) for name in sys.argv[1:]}
        # tensor 模块是统一的类定义来源。
        tensor_type = modules['tensor'].Tensor
        # 这三个前向计算模块必须复用同一个类，而非重新执行张量文件。
        for name in ('activations', 'layers', 'losses'):
            # 类名相同仍可能是不同对象，is 检查能识别重复加载。
            assert modules[name].Tensor is tensor_type, name
        """)
    # -c 让解释器直接执行脚本，并在仓库根目录查找 practices 包。
    # 捕获所有输出，既便于展示失败原因，也能检查导入是否误跑演示。
    result = subprocess.run(
        [sys.executable, "-c", script, *module_names],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, timeout=30, check=False,
    )
    # 正常退出意味着全部模块可导入，并通过类身份断言。
    # 失败时保留子进程回溯，直接定位具体模块或依赖问题。
    assert result.returncode == 0, result.stdout + result.stderr
    # 导入应只注册定义；单元演示和性能测试应留在主入口。
    assert result.stdout == ""
    # 同时拒绝重复预加载等导入警告，确保模块启动路径清晰。
    assert result.stderr == ""


# 张量构造与算术依赖 isinstance，因此能直接暴露跨模块重复类的问题。
def test_tensors_from_other_modules_can_be_stacked_and_combined():
    """不同模块返回的 Tensor 可组成批次，并继续进行广播算术。"""
    # 此处使用日常代码中的标准包导入方式。
    from practices import activations, layers, losses, tensor

    # 第一行通过激活模块产生，模拟上一阶段的计算结果。
    positive = activations.ReLU()(tensor.Tensor([-1.0, 2.0]))
    # 第二行通过损失模块持有的 Tensor 引用创建。
    other = losses.Tensor([3.0, 4.0])
    # Tensor 列表构造器应能识别两个来源不同但类型相同的对象。
    batch = tensor.Tensor([positive, other])
    # 固定预期批次，覆盖 ReLU 负数归零与列表按行堆叠。
    expected_batch = np.array([[0.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    # 比较实际数值，避免只有形状正确却未正确合并数据。
    np.testing.assert_array_equal(batch.data, expected_batch)
    # 用另外两个模块的类引用创建操作数，覆盖加法与乘法中的类型判断。
    # 长度为二的向量按列广播到整个批次。
    result = (batch + layers.Tensor([1.0, -1.0])) * activations.Tensor([2.0, 0.5])
    # NumPy 独立给出广播运算预期，未复用 Tensor 的实现。
    expected = (expected_batch + [1.0, -1.0]) * [2.0, 0.5]
    # 后续步骤仍需同一个 Tensor 类，不能只返回底层 ndarray。
    assert type(result) is tensor.Tensor
    # 确认两个算术步骤都保留了预期的批次与数值。
    np.testing.assert_allclose(result.data, expected)


# 用已知参数串起前置练习，验证后续损失函数可以直接接收网络输出。
def test_linear_relu_and_mse_share_tensor_values():
    """线性层、激活函数与损失函数组成的前向链应与 NumPy 一致。"""
    # 每个能力从其语义模块导入，不需要文件路径或动态加载工具。
    from practices.activations import ReLU
    # Linear 内部参数与输入必须使用同一个 Tensor 类。
    from practices.layers import Linear
    # 损失模块直接消费之前阶段的计算结果。
    from practices.losses import MSELoss
    # 所有输入统一使用公开的张量模块创建。
    from practices.tensor import Tensor

    # 两条样本各含两个特征；有正有负，确保 ReLU 的两种分支均参与计算。
    inputs = Tensor([[2.0, -1.0], [-1.0, 3.0]])
    # 两维输出对应每条样本的两个目标值。
    targets = Tensor([[1.0, 0.0], [0.0, 4.0]])
    # 使用小矩阵即可覆盖前向接口，避免运行课程中的大型性能实验。
    layer = Linear(2, 2)
    # 覆盖随机初始化参数，让测试结果不依赖其他测试消耗随机数的顺序。
    layer.weight.data[...] = [[1.0, -2.0], [0.5, 1.0]]
    # 采用非零偏置，同时验证线性层的广播加法。
    layer.bias.data[...] = [0.25, -0.5]
    # 按真实使用顺序连接三个模块，不分别手动提取中间 NumPy 数组。
    predictions = ReLU()(layer(inputs))
    # MSELoss 接收前向链返回的张量，并给出标量张量。
    loss = MSELoss()(predictions, targets)
    # NumPy 的矩阵乘法、广播和最大值构成独立的前向参考。
    expected_predictions = np.maximum(inputs.data @ layer.weight.data + layer.bias.data, 0)
    # 参考损失按所有样本和输出位置取均值。
    expected_loss = np.mean((expected_predictions - targets.data) ** 2)
    # 两个模块返回的对象都应继续支持统一的 Tensor 接口。
    assert type(predictions) is Tensor and type(loss) is Tensor
    # 损失必须归约成标量，不能误保留批次维度。
    assert loss.shape == ()
    # 中间值与最终损失分别断言，出错时能够区分前向链与归约问题。
    np.testing.assert_allclose(predictions.data, expected_predictions)
    # float32 运算允许合理舍入误差，不依赖平台逐位一致。
    np.testing.assert_allclose(loss.data, expected_loss, rtol=1e-6)
