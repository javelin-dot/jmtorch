"""阶段草稿。实现写在这里，稳定后再迁入 src/jmtorch/。"""

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

rng = np.random.default_rng(7)

_tensor_path = Path(__file__).resolve().parent.parent / "01_tensor" / "practice.py"
_spec = importlib.util.spec_from_file_location("tensor_practice", _tensor_path)
_tensor_mod = importlib.util.module_from_spec(_spec)
sys.modules["tensor_practice"] = _tensor_mod
_spec.loader.exec_module(_tensor_mod)
Tensor = _tensor_mod.Tensor

TOLERANCE = 1e-10
__all__ = ["Sigmoid", "ReLU", "Tanh", "GELU", "Softmax"]


def _show(label, tensor, *, data=True):
    parts = [f"shape={tensor.shape}", f"size={tensor.size}", f"ndim={tensor.ndim}", f"dtype={tensor.dtype}"]
    if data:
        parts.insert(0, f"data={tensor.data}")
    print(f"   {label}：{', '.join(parts)}")


class Sigmoid:
    def parameters(self):
        return []

    def forward(self, x: Tensor) -> Tensor:
        x_data = x.data
        with np.errstate(over="ignore", invalid="ignore"):
            result = np.where(
                x_data >= 0,
                1.0 / (1.0 + np.exp(-x_data)), # x >= 0 时，sigmoid(x) = 1 / (1 + exp(-x))
                np.exp(x_data) / (1.0 + np.exp(x_data)),
            )
        return Tensor(result)

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)


class ReLU:
    def parameters(self):
        return []

    def forward(self, x: Tensor) -> Tensor:
        result = np.maximum(0, x.data) # x >= 0 时，ReLU(x) = x，否则 ReLU(x) = 0
        return Tensor(result)

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)


class Tanh:
    def parameters(self):
        return []

    def forward(self, x: Tensor) -> Tensor:
        result = np.tanh(x.data) # tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        return Tensor(result)

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)


class GELU:
    def parameters(self):
        return []

    def forward(self, x: Tensor) -> Tensor:
        return Sigmoid()(x * 1.702) * x # GELU(x) = x * sigmoid(1.702 * x)

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)


class Softmax:
    def parameters(self):
        return []

    def forward(self, x: Tensor, dim: int = -1) -> Tensor:
        x_max = np.max(x.data, axis=dim, keepdims=True)
        x_shifted = x.data - x_max # 将 x 减去 x 的最大值，使得 x 的值范围在 [-∞, 0] 之间
        exp_values = np.exp(x_shifted) # 将 x 的值范围在 [-∞, 0] 之间转换为 [0, 1] 之间
        exp_sum = np.sum(exp_values, axis=dim, keepdims=True) # 将 exp_values 的值范围在 [0, 1] 之间转换为 [0, 1] 之间
        result = exp_values / exp_sum # 将 exp_values 的值范围在 [0, 1] 之间转换为 [0, 1] 之间
        return Tensor(result) # 将 result 的值范围在 [0, 1] 之间转换为 [0, 1] 之间

    def __call__(self, x: Tensor, dim: int = -1) -> Tensor:
        return self.forward(x, dim)


def test_unit_sigmoid():
    print("🧪 单元测试：Sigmoid...")

    sigmoid = Sigmoid()

    x = Tensor([0.0])
    result = sigmoid.forward(x)
    assert np.allclose(result.data, [0.5]), f"sigmoid(0) 应为 0.5，实际 {result.data}"
    _show("输入 [0.0]", x)
    _show("sigmoid(0)", result)

    x = Tensor([-10, -1, 0, 1, 10])
    result = sigmoid.forward(x)
    assert np.all(result.data > 0) and np.all(result.data < 1), "Sigmoid 输出应落在 (0, 1)"
    _show("输入 [-10, -1, 0, 1, 10]", x)
    _show("sigmoid 输出（应在 0～1）", result)

    x = Tensor([-1000, 1000])
    result = sigmoid.forward(x)
    assert np.allclose(result.data[0], 0, atol=TOLERANCE), "sigmoid(-∞) 应接近 0"
    assert np.allclose(result.data[1], 1, atol=TOLERANCE), "sigmoid(+∞) 应接近 1"
    _show("极值输入 [-1000, 1000]", x)
    _show("sigmoid 极值输出", result)

    print("✅ Sigmoid 通过！")


def test_unit_relu():
    print("🧪 单元测试：ReLU...")

    relu = ReLU()

    x = Tensor([-2, -1, 0, 1, 2])
    result = relu.forward(x)
    expected = [0, 0, 0, 1, 2]
    assert np.allclose(result.data, expected), f"ReLU 失败，期望 {expected}，实际 {result.data}"
    _show("输入 [-2, -1, 0, 1, 2]", x)
    _show("ReLU 输出", result)

    x = Tensor([-5, -3, -1])
    result = relu.forward(x)
    assert np.allclose(result.data, [0, 0, 0]), "ReLU 应将全部负数置零"
    _show("全负输入", x)
    _show("ReLU 输出", result)

    x = Tensor([1, 3, 5])
    result = relu.forward(x)
    assert np.allclose(result.data, [1, 3, 5]), "ReLU 应保持正数不变"
    _show("全正输入", x)
    _show("ReLU 输出", result)

    x = Tensor([-1, -2, -3, 1])
    result = relu.forward(x)
    zeros = np.sum(result.data == 0)
    assert zeros == 3, f"ReLU 应产生稀疏，4 个元素里有 {zeros} 个零"
    _show("稀疏输入", x)
    _show("ReLU 输出", result)
    print(f"   零的个数：{int(zeros)} / 4")

    print("✅ ReLU 通过！")


def test_unit_tanh():
    print("🧪 单元测试：Tanh...")

    tanh = Tanh()

    x = Tensor([0.0])
    result = tanh.forward(x)
    assert np.allclose(result.data, [0.0]), f"tanh(0) 应为 0，实际 {result.data}"
    _show("输入 [0.0]", x)
    _show("tanh(0)", result)

    x = Tensor([-10, -1, 0, 1, 10])
    result = tanh.forward(x)
    assert np.all(result.data >= -1) and np.all(result.data <= 1), "Tanh 输出应落在 [-1, 1]"
    _show("输入 [-10, -1, 0, 1, 10]", x)
    _show("tanh 输出（应在 -1～1）", result)

    x = Tensor([2.0])
    pos_result = tanh.forward(x)
    x_neg = Tensor([-2.0])
    neg_result = tanh.forward(x_neg)
    assert np.allclose(pos_result.data, -neg_result.data), "tanh 应对称：tanh(-x) = -tanh(x)"
    _show("tanh(2)", pos_result)
    _show("tanh(-2)", neg_result)

    x = Tensor([-1000, 1000])
    result = tanh.forward(x)
    assert np.allclose(result.data[0], -1, atol=TOLERANCE), "tanh(-∞) 应接近 -1"
    assert np.allclose(result.data[1], 1, atol=TOLERANCE), "tanh(+∞) 应接近 1"
    _show("极值输入 [-1000, 1000]", x)
    _show("tanh 极值输出", result)

    print("✅ Tanh 通过！")


def test_unit_gelu():
    print("🧪 单元测试：GELU...")

    gelu = GELU()

    x = Tensor([0.0])
    result = gelu.forward(x)
    assert np.allclose(result.data, [0.0], atol=TOLERANCE), f"GELU(0) 应约等于 0，实际 {result.data}"
    _show("输入 [0.0]", x)
    _show("GELU(0)", result)

    x = Tensor([1.0])
    result = gelu.forward(x)
    assert result.data[0] > 0.8, f"GELU(1) 应约等于 0.84，实际 {result.data[0]}"
    _show("输入 [1.0]", x)
    _show("GELU(1)", result)

    x = Tensor([-1.0])
    result = gelu.forward(x)
    assert result.data[0] < 0 and result.data[0] > -0.2, f"GELU(-1) 应约等于 -0.16，实际 {result.data[0]}"
    _show("输入 [-1.0]", x)
    _show("GELU(-1)", result)

    x = Tensor([-0.001, 0.0, 0.001])
    result = gelu.forward(x)
    diff1 = abs(result.data[1] - result.data[0])
    diff2 = abs(result.data[2] - result.data[1])
    assert diff1 < 0.01 and diff2 < 0.01, "GELU 在 0 附近应平滑"
    _show("0 附近输入", x)
    _show("GELU 输出（应平滑）", result)

    print("✅ GELU 通过！")


def test_unit_softmax():
    print("🧪 单元测试：Softmax...")

    softmax = Softmax()

    x = Tensor([1, 2, 3])
    result = softmax.forward(x)
    assert np.allclose(np.sum(result.data), 1.0), f"Softmax 应归一化为 1，实际和为 {np.sum(result.data)}"
    assert np.all(result.data > 0), "Softmax 各分量应为正"
    assert np.all(result.data < 1), "Softmax 各分量应小于 1"
    max_input_idx = np.argmax(x.data)
    max_output_idx = np.argmax(result.data)
    assert max_input_idx == max_output_idx, "最大输入应对应最大输出"
    _show("输入 [1, 2, 3]", x)
    _show("softmax 输出", result)
    print(f"   各分量之和：{np.sum(result.data):.6f}")

    x = Tensor([1000, 1001, 1002])
    result = softmax.forward(x)
    assert np.allclose(np.sum(result.data), 1.0), "大数输入时 Softmax 仍应归一化"
    assert not np.any(np.isnan(result.data)), "Softmax 不应出现 NaN"
    assert not np.any(np.isinf(result.data)), "Softmax 不应出现 Inf"
    _show("大数输入 [1000, 1001, 1002]", x)
    _show("softmax 输出（数值稳定）", result)

    x = Tensor([[1, 2], [3, 4]])
    result = softmax.forward(x, dim=-1)
    assert result.shape == (2, 2), "Softmax 应保持输入形状"
    row_sums = np.sum(result.data, axis=-1)
    assert np.allclose(row_sums, [1.0, 1.0]), "每一行应归一化为 1"
    _show("二维输入", x)
    _show("按最后一维 softmax", result)
    print(f"   各行之和：{row_sums}")

    print("✅ Softmax 通过！")


def test_module():
    print("🧪 正在运行模块集成测试")
    print("=" * 50)

    print("运行单元测试...")
    test_unit_sigmoid()
    test_unit_relu()
    test_unit_tanh()
    test_unit_gelu()
    test_unit_softmax()

    print("\n运行集成场景...")

    print("🧪 集成测试：保持张量属性...")
    test_data = Tensor([[1, -1], [2, -2]])
    _show("测试输入", test_data)

    activations = [Sigmoid(), ReLU(), Tanh(), GELU()]
    for activation in activations:
        result = activation.forward(test_data)
        name = activation.__class__.__name__
        assert result.shape == test_data.shape, f"{name} 未保持形状"
        assert isinstance(result, Tensor), f"{name} 输出不是 Tensor"
        _show(name, result)

    print("✅ 各激活均保持张量属性！")

    print("🧪 集成测试：Softmax 维处理...")
    data_3d = Tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    softmax = Softmax()
    result_last = softmax(data_3d, dim=-1)
    assert result_last.shape == (2, 2, 3), "Softmax 应保持形状"
    last_dim_sums = np.sum(result_last.data, axis=-1)
    assert np.allclose(last_dim_sums, 1.0), "最后一维应归一化为 1"
    _show("三维输入", data_3d)
    _show("softmax(dim=-1)", result_last)
    print(f"   最后一维之和：{last_dim_sums}")

    print("✅ Softmax 维处理通过！")

    print("🧪 集成测试：激活串联...")
    x = Tensor([[-1, 0, 1, 2]])
    relu = ReLU()
    hidden = relu.forward(x)
    softmax = Softmax()
    output = softmax.forward(hidden)
    assert hidden.data[0, 0] == 0, "ReLU 应将负数置零"
    assert np.allclose(np.sum(output.data), 1.0), "最终输出应为概率分布"
    _show("输入", x)
    _show("ReLU 后", hidden)
    _show("Softmax 后", output)
    print(f"   Softmax 各分量之和：{np.sum(output.data):.6f}")

    print("✅ 激活串联通过！")

    print("\n" + "=" * 50)
    print("🎉 全部测试通过！模块可以迁入。")


def analyze_activation_performance():
    print("📊 分析激活函数计算开销...")
    print("=" * 60)

    size = 1000000
    test_data = Tensor(rng.standard_normal(size).astype(np.float32))
    print(f"\n测试规模：{size:,} 个元素（模拟较大隐层）")
    print(f"   shape={test_data.shape}，dtype={test_data.dtype}")
    print("-" * 60)

    relu = ReLU()
    sigmoid = Sigmoid()
    tanh = Tanh()
    gelu = GELU()

    _ = relu(test_data)
    _ = sigmoid(test_data)

    n_runs = 10

    start = time.time()
    for _ in range(n_runs):
        _ = relu(test_data)
    relu_time = (time.time() - start) / n_runs * 1000

    start = time.time()
    for _ in range(n_runs):
        _ = sigmoid(test_data)
    sigmoid_time = (time.time() - start) / n_runs * 1000

    start = time.time()
    for _ in range(n_runs):
        _ = tanh(test_data)
    tanh_time = (time.time() - start) / n_runs * 1000

    start = time.time()
    for _ in range(n_runs):
        _ = gelu(test_data)
    gelu_time = (time.time() - start) / n_runs * 1000

    sample = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    print("\n小样本对照：")
    _show("输入", sample)
    _show("ReLU", relu(sample))
    _show("Sigmoid", sigmoid(sample))
    _show("Tanh", tanh(sample))
    _show("GELU", gelu(sample))

    print("\n🧪 激活耗时：")
    print(f"   ReLU：   {relu_time:.2f}ms（基准）")
    print(f"   Sigmoid：{sigmoid_time:.2f}ms（慢 {sigmoid_time/relu_time:.1f} 倍）")
    print(f"   Tanh：   {tanh_time:.2f}ms（慢 {tanh_time/relu_time:.1f} 倍）")
    print(f"   GELU：   {gelu_time:.2f}ms（慢 {gelu_time/relu_time:.1f} 倍）")

    print("\n" + "=" * 60)
    print("要点：")
    print("   1. ReLU 最快：只需 max(0, x)，没有指数运算")
    print("   2. Sigmoid / Tanh 需要 exp()，更贵")
    print("   3. GELU 内部用 Sigmoid，开销跟着它走")
    print("   4. 隐层数量一大，ReLU 的速度优势会累加")

    print("\n实际影响：")
    print("   - ResNet 用 ReLU：一次前向有海量激活")
    print("   - GPT 用 GELU：多花算力换更平滑的梯度")
    print("   - Sigmoid / Tanh：多用于输出层或门控")
    print("=" * 60)


def test_activations():
    print("🎯 激活函数给网络加上非线性")
    print("=" * 45)

    x = Tensor(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
    _show("输入", x)

    relu = ReLU()
    relu_out = relu(x)
    _show("ReLU", relu_out)
    print("   负数变 0，正数不变")

    sigmoid = Sigmoid()
    sigmoid_out = sigmoid(x)
    _show("Sigmoid", sigmoid_out)
    print("   全部压到 (0, 1)")

    softmax = Softmax()
    softmax_out = softmax(x)
    _show("Softmax", softmax_out)
    print(f"   各分量之和 = {softmax_out.data.sum():.1f}（合法概率分布）")

    print("\n✨ 激活带来非线性，这是深度网络能拟合复杂函数的关键。")

def plot_activations():
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    xs = np.linspace(-5, 5, 400)
    logits = np.stack([xs, np.zeros_like(xs)], axis=1)
    probs = Softmax()(Tensor(logits)).data

    plt.figure(figsize=(8, 4))
    plt.plot(xs, probs[:, 0], label="P(类0)，分数=x")
    plt.plot(xs, probs[:, 1], label="P(类1)，分数=0")
    plt.axhline(0.5, color="gray", lw=0.6, ls="--")
    plt.axvline(0, color="gray", lw=0.6, ls="--")
    plt.title("两类 Softmax：logits=[x, 0]")
    plt.xlabel("x")
    plt.ylabel("概率")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)

    out = Path(__file__).resolve().parent / "softmax.png"
    plt.savefig(out, dpi=120)
    print(f"已保存 {out}")


if __name__ == "__main__":
    test_module()
    print("\n")
    analyze_activation_performance()
    print("\n")
    test_activations()
    print("\n")
    # plot_activations()
