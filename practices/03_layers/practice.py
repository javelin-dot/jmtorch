"""阶段草稿。实现写在这里，稳定后再迁入 src/jmtorch/。"""

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

rng = np.random.default_rng(7)

_act_path = Path(__file__).resolve().parent.parent / "02_activations" / "practice.py"
_spec = importlib.util.spec_from_file_location("activations_practice", _act_path)
_act_mod = importlib.util.module_from_spec(_spec)
sys.modules["activations_practice"] = _act_mod
_spec.loader.exec_module(_act_mod)
Tensor = _act_mod.Tensor
ReLU = _act_mod.ReLU
Sigmoid = _act_mod.Sigmoid

INIT_SCALE_FACTOR = 1.0
HE_SCALE_FACTOR = 2.0
DROPOUT_MIN_PROB = 0.0
DROPOUT_MAX_PROB = 1.0


def _show(label, tensor, *, data=True):
    parts = [f"shape={tensor.shape}", f"size={tensor.size}", f"ndim={tensor.ndim}", f"dtype={tensor.dtype}"]
    if data:
        parts.insert(0, f"data={tensor.data}")
    print(f"   {label}：{', '.join(parts)}")

# 层类抽象基类
class Layer:
    def forward(self, x):
        raise NotImplementedError(
            f"forward() 未在 {self.__class__.__name__} 中实现\n"
            f"  子类需要实现 forward()，定义输入如何被变换"
        )

    def __call__(self, x, *args, **kwargs):
        return self.forward(x, *args, **kwargs)

    def parameters(self):
        return []

    def __repr__(self):
        return f"{self.__class__.__name__}()"


# 线性层类
class Linear(Layer):
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features

        scale = np.sqrt(INIT_SCALE_FACTOR / in_features)
        weight_data = rng.standard_normal((in_features, out_features)) * scale
        self.weight = Tensor(weight_data)

        if bias:
            self.bias = Tensor(np.zeros(out_features))
        else:
            self.bias = None

    def forward(self, x):
        output = x.matmul(self.weight)
        if self.bias is not None:
            output = output + self.bias
        return output

    def parameters(self):
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params

    def __repr__(self):
        bias_str = f", bias={self.bias is not None}"
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}{bias_str})"

# 丢弃层类
class Dropout(Layer):
    def __init__(self, p=0.5):
        if not DROPOUT_MIN_PROB <= p <= DROPOUT_MAX_PROB:
            raise ValueError(
                f"非法 dropout 概率：{p}\n"
                f"  p 必须在 {DROPOUT_MIN_PROB} 和 {DROPOUT_MAX_PROB} 之间\n"
                f"  p=0.0 表示全部保留；p=0.5 随机丢掉一半；p=1.0 全部置零"
            )
        self.p = p

    def _should_apply_dropout(self, training):
        return training and self.p > DROPOUT_MIN_PROB

    def _generate_dropout_mask(self, shape):
        keep_prob = 1.0 - self.p
        binary_mask = (rng.random(shape) < keep_prob).astype(np.float32)
        scale = 1.0 / keep_prob
        return Tensor(binary_mask * scale)

    def forward(self, x, training=True):
        if not self._should_apply_dropout(training):
            return x
        if self.p == DROPOUT_MAX_PROB:
            return Tensor(np.zeros_like(x.data))
        mask = self._generate_dropout_mask(x.data.shape)
        return x * mask

    def __call__(self, x, training=True):
        return self.forward(x, training)

    def parameters(self):
        return []

    def __repr__(self):
        return f"Dropout(p={self.p})"

# 顺序容器类
class Sequential:
    # 初始化顺序容器
    def __init__(self, *layers):
        if len(layers) == 1 and isinstance(layers[0], (list, tuple)):
            self.layers = list(layers[0])
        else:
            self.layers = list(layers)

    # 前向传播
    def forward(self, x, training=True):
        for layer in self.layers:
            try:
                x = layer.forward(x, training=training)
            except TypeError:
                x = layer.forward(x)
        return x
    # 调用
    def __call__(self, x, training=True):
        return self.forward(x, training=training)

    # 获取参数
    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def __repr__(self):
        layer_reprs = ", ".join(repr(layer) for layer in self.layers)
        return f"Sequential({layer_reprs})"


def test_unit_linear_layer():
    print("🧪 单元测试：Linear 层...")

    layer = Linear(784, 256)
    assert layer.in_features == 784
    assert layer.out_features == 256
    assert layer.weight.shape == (784, 256)
    assert layer.bias.shape == (256,)
    print(f"   Linear(784, 256)：weight.shape={layer.weight.shape}，bias.shape={layer.bias.shape}")

    weight_std = np.std(layer.weight.data)
    expected_std = np.sqrt(INIT_SCALE_FACTOR / 784)
    assert 0.5 * expected_std < weight_std < 2.0 * expected_std, (
        f"权重标准差 {weight_std} 与期望 {expected_std} 相差过大"
    )
    print(f"   LeCun 初始化：weight_std={weight_std:.4f}，期望约 {expected_std:.4f}")
    assert np.allclose(layer.bias.data, 0), "偏置应初始化为 0"
    print(f"   bias 是否全零：{np.allclose(layer.bias.data, 0)}")

    x = Tensor(rng.standard_normal((32, 784)))
    y = layer.forward(x)
    assert y.shape == (32, 256), f"期望形状 (32, 256)，实际 {y.shape}"
    _show("输入 x (32, 784)", x, data=False)
    _show("输出 y = xW + b", y, data=False)

    layer_no_bias = Linear(10, 5, bias=False)
    assert layer_no_bias.bias is None
    params = layer_no_bias.parameters()
    assert len(params) == 1
    print(f"   Linear(10, 5, bias=False) 参数个数：{len(params)}")

    params = layer.parameters()
    assert len(params) == 2
    assert params[0] is layer.weight
    assert params[1] is layer.bias
    print(f"   Linear(784, 256) 参数个数：{len(params)}（weight + bias）")

    print("✅ Linear 层通过！")


def test_unit_edge_cases_linear():
    print("🧪 边界测试：Linear 层...")

    layer = Linear(10, 5)

    x_2d = Tensor(rng.standard_normal((1, 10)))
    y = layer.forward(x_2d)
    assert y.shape == (1, 5), "应能处理单样本"
    _show("单样本输入 (1, 10)", x_2d)
    _show("输出", y)

    x_empty = Tensor(rng.standard_normal((0, 10)))
    y_empty = layer.forward(x_empty)
    assert y_empty.shape == (0, 5), "应能处理空 batch"
    print(f"   空 batch：输入 {x_empty.shape} → 输出 {y_empty.shape}")

    layer_large = Linear(10, 5)
    layer_large.weight.data = np.ones((10, 5)) * 100
    x = Tensor(np.ones((1, 10)))
    y = layer_large.forward(x)
    assert not np.any(np.isnan(y.data)), "大权重不应产生 NaN"
    assert not np.any(np.isinf(y.data)), "大权重不应产生 Inf"
    _show("大权重前向", y)

    layer_no_bias = Linear(10, 5, bias=False)
    x = Tensor(rng.standard_normal((4, 10)))
    y = layer_no_bias.forward(x)
    assert y.shape == (4, 5), "无 bias 时应能前向"
    _show("无 bias 输入 (4, 10)", x, data=False)
    _show("无 bias 输出", y, data=False)

    print("✅ 边界情况通过！")


def test_unit_parameter_collection_linear():
    print("🧪 参数收集测试：Linear 层...")

    layer = Linear(10, 5)
    params = layer.parameters()
    assert len(params) == 2, "应返回 2 个参数（weight 和 bias）"
    assert params[0].shape == (10, 5), "第一个应是 weight"
    assert params[1].shape == (5,), "第二个应是 bias"
    print(f"   有 bias：{len(params)} 个参数，shapes={[p.shape for p in params]}")

    layer_no_bias = Linear(10, 5, bias=False)
    params_no_bias = layer_no_bias.parameters()
    assert len(params_no_bias) == 1, "无 bias 时应只返回 weight"
    print(f"   无 bias：{len(params_no_bias)} 个参数，shapes={[p.shape for p in params_no_bias]}")

    print("✅ 参数收集通过！")


def test_unit_should_apply_dropout():
    print("🧪 单元测试：Dropout 是否生效...")

    d = Dropout(0.5)
    train_on = d._should_apply_dropout(training=True)
    infer_off = d._should_apply_dropout(training=False)
    assert train_on is True, "Dropout(0.5) 训练时应生效"
    assert infer_off is False, "推理时不应 dropout"
    print(f"   Dropout(0.5) 训练={train_on}，推理={infer_off}")

    d_zero = Dropout(0.0)
    zero_train = d_zero._should_apply_dropout(training=True)
    assert zero_train is False, "Dropout(0.0) 永远不应生效"
    print(f"   Dropout(0.0) 训练={zero_train}")

    d_full = Dropout(1.0)
    full_train = d_full._should_apply_dropout(training=True)
    full_infer = d_full._should_apply_dropout(training=False)
    assert full_train is True, "Dropout(1.0) 训练时应生效"
    assert full_infer is False, "即使 p=1.0，推理时也不应 dropout"
    print(f"   Dropout(1.0) 训练={full_train}，推理={full_infer}")

    print("✅ Dropout 决策逻辑通过！")


def test_unit_generate_dropout_mask():
    print("🧪 单元测试：Dropout 掩码...")

    d = Dropout(0.5)
    mask = d._generate_dropout_mask((1000,))
    assert mask.shape == (1000,), f"期望形状 (1000,)，实际 {mask.shape}"

    unique_vals = set(np.unique(mask.data))
    assert unique_vals <= {0.0, 2.0}, f"掩码取值应为 {{0.0, 2.0}}，实际 {unique_vals}"

    non_zero = np.count_nonzero(mask.data)
    std_err = np.sqrt(1000 * 0.5 * 0.5)
    assert 500 - 3 * std_err < non_zero < 500 + 3 * std_err, f"期望约 500 个存活，实际 {non_zero}"
    print(f"   Dropout(0.5) 掩码 shape={mask.shape}，取值={unique_vals}，非零个数={non_zero}")

    d2 = Dropout(0.3)
    mask2 = d2._generate_dropout_mask((2000,))
    expected_scale = 1.0 / 0.7
    non_zero_vals = mask2.data[mask2.data != 0.0]
    assert np.allclose(non_zero_vals, expected_scale), (
        f"存活值应为 {expected_scale:.4f}，实际 {np.unique(non_zero_vals)}"
    )
    survival_rate = np.count_nonzero(mask2.data) / 2000
    assert 0.60 < survival_rate < 0.80, f"p=0.3 时期望约 70% 存活，实际 {survival_rate:.1%}"
    print(f"   Dropout(0.3) 缩放={expected_scale:.4f}，存活率={survival_rate:.1%}")

    print("✅ Dropout 掩码生成通过！")


def test_unit_dropout_layer():
    print("🧪 单元测试：Dropout 层...")

    dropout = Dropout(0.5)
    assert dropout.p == 0.5
    print(f"   Dropout(0.5)：p={dropout.p}")

    x = Tensor([1, 2, 3, 4])
    y_inference = dropout.forward(x, training=False)
    assert np.array_equal(x.data, y_inference.data), "推理应原样通过"
    _show("输入", x)
    _show("推理模式输出", y_inference)

    dropout_zero = Dropout(0.0)
    y_zero = dropout_zero.forward(x, training=True)
    assert np.array_equal(x.data, y_zero.data), "p=0 应原样通过"
    _show("p=0 训练输出", y_zero)

    dropout_full = Dropout(1.0)
    y_full = dropout_full.forward(x, training=True)
    assert np.allclose(y_full.data, 0), "p=1 应全部置零"
    _show("p=1 训练输出", y_full)

    x_large = Tensor(np.ones((1000,)))
    y_train = dropout.forward(x_large, training=True)
    non_zero_count = np.count_nonzero(y_train.data)
    expected = 500
    std_error = np.sqrt(1000 * 0.5 * 0.5)
    lower_bound = expected - 3 * std_error
    upper_bound = expected + 3 * std_error
    assert lower_bound < non_zero_count < upper_bound, (
        f"期望 {expected}±{3*std_error:.0f} 个存活，实际 {non_zero_count}"
    )
    surviving_values = y_train.data[y_train.data != 0]
    expected_value = 2.0
    assert np.allclose(surviving_values, expected_value), f"存活值应为 {expected_value}"
    print(f"   p=0.5 训练：非零 {non_zero_count}/1000，存活值={expected_value}")

    params = dropout.parameters()
    assert len(params) == 0, "Dropout 不应有可学习参数"
    print(f"   Dropout 参数个数：{len(params)}")

    try:
        Dropout(-0.1)
        assert False, "负概率应抛出 ValueError"
    except ValueError:
        print("   p=-0.1：捕获 ValueError")

    try:
        Dropout(1.1)
        assert False, "p>1 应抛出 ValueError"
    except ValueError:
        print("   p=1.1：捕获 ValueError")

    print("✅ Dropout 层通过！")


def analyze_layer_memory():
    print("📊 分析层的内存占用...")

    layer_configs = [
        (784, 256),
        (256, 256),
        (256, 10),
        (2048, 2048),
    ]

    print("\nLinear 层内存：")
    print("配置 (in, out) → 权重内存 → 偏置内存 → 合计")

    for in_feat, out_feat in layer_configs:
        weight_memory = in_feat * out_feat * 4
        bias_memory = out_feat * 4
        total_memory = weight_memory + bias_memory
        print(
            f"({in_feat:4d}, {out_feat:4d}) → {weight_memory/1024:7.1f} KB → "
            f"{bias_memory/1024:6.1f} KB → {total_memory/1024:7.1f} KB"
        )

    print("\n💡 多层模型内存随隐层变宽：")
    hidden_sizes = [128, 256, 512, 1024, 2048]

    for hidden_size in hidden_sizes:
        layer1_params = 784 * hidden_size + hidden_size
        layer2_params = hidden_size * (hidden_size // 2) + (hidden_size // 2)
        layer3_params = (hidden_size // 2) * 10 + 10
        total_params = layer1_params + layer2_params + layer3_params
        memory_mb = total_params * 4 / (1024 * 1024)
        print(f"隐层={hidden_size:4d}：{total_params:7,} 个参数 = {memory_mb:5.1f} MB")


def analyze_layer_performance():
    print("📊 分析层的计算量...")

    batch_sizes = [1, 32, 128, 512]
    layer = Linear(784, 256)

    print("\nLinear 层 MAC 分析：")
    print("batch → 矩阵乘 MAC → 加偏置 MAC → 合计 MAC")
    print("说明：FLOPs = 2 × MACs（一次乘加算一次 MAC）")

    for batch_size in batch_sizes:
        matmul_flops = batch_size * 784 * 256
        bias_flops = batch_size * 256
        total_flops = matmul_flops + bias_flops
        print(f"{batch_size:10d} → {matmul_flops:15,} → {bias_flops:13,} → {total_flops:11,}")

    print("\nLinear 层耗时：")
    print("batch → 耗时 (ms) → 吞吐 (样本/秒)")

    for batch_size in batch_sizes:
        x = Tensor(rng.standard_normal((batch_size, 784)))
        for _ in range(10):
            _ = layer.forward(x)

        iterations = 100
        start = time.perf_counter()
        for _ in range(iterations):
            _ = layer.forward(x)
        elapsed = time.perf_counter() - start

        time_per_forward = (elapsed / iterations) * 1000
        throughput = (batch_size * iterations) / elapsed
        print(f"{batch_size:10d} → {time_per_forward:8.3f} ms → {throughput:12,.0f} 样本/秒")

    print("\n💡 要点：")
    print("   Linear 复杂度：O(batch × in_features × out_features)")
    print("   内存随 batch 线性增长，随层宽近似平方增长")
    print("   Dropout 几乎只是逐元素运算，额外开销很小")
    print("   更大 batch 能摊掉固定开销，吞吐更高")


def test_module():
    print("🧪 正在运行模块集成测试")
    print("=" * 50)

    print("运行单元测试...")
    test_unit_linear_layer()
    test_unit_edge_cases_linear()
    test_unit_parameter_collection_linear()
    test_unit_should_apply_dropout()
    test_unit_generate_dropout_mask()
    test_unit_dropout_layer()

    print("\n运行集成场景...")
    print("🧪 集成测试：多层网络...")

    layer1 = Linear(784, 128)
    activation1 = ReLU()
    dropout1 = Dropout(0.5)
    layer2 = Linear(128, 64)
    activation2 = ReLU()
    dropout2 = Dropout(0.3)
    layer3 = Linear(64, 10)

    batch_size = 16
    x = Tensor(rng.standard_normal((batch_size, 784)))
    _show("输入", x, data=False)

    h = layer1.forward(x)
    _show("Linear(784, 128)", h, data=False)
    h = activation1.forward(h)
    _show("ReLU", h, data=False)
    h = dropout1.forward(h)
    _show("Dropout(0.5)", h, data=False)
    h = layer2.forward(h)
    _show("Linear(128, 64)", h, data=False)
    h = activation2.forward(h)
    _show("ReLU", h, data=False)
    h = dropout2.forward(h)
    _show("Dropout(0.3)", h, data=False)
    output = layer3.forward(h)
    _show("Linear(64, 10) 输出", output, data=False)

    assert output.shape == (batch_size, 10), f"期望输出形状 ({batch_size}, 10)，实际 {output.shape}"

    all_params = layer1.parameters() + layer2.parameters() + layer3.parameters()
    expected_params = 6
    assert len(all_params) == expected_params, f"期望 {expected_params} 个参数，实际 {len(all_params)}"
    print(f"   三层 Linear 参数个数：{len(all_params)}")

    test_x = Tensor(rng.standard_normal((4, 784)))
    dropout_test = Dropout(0.5)
    train_output = dropout_test.forward(test_x, training=True)
    infer_output = dropout_test.forward(test_x, training=False)
    assert np.array_equal(test_x.data, infer_output.data), "推理模式应原样通过"
    print(f"   Dropout 训练非零：{np.count_nonzero(train_output.data)}，推理与输入相同：{np.array_equal(test_x.data, infer_output.data)}")

    seq = Sequential(Linear(10, 8), ReLU(), Dropout(0.2), Linear(8, 3))
    seq_x = Tensor(rng.standard_normal((2, 10)))
    seq_y = seq(seq_x, training=False)
    _show("Sequential 输入", seq_x, data=False)
    _show("Sequential 推理输出", seq_y)
    print(f"   Sequential 参数个数：{len(seq.parameters())}")

    print("✅ 多层网络集成通过！")

    print("\n" + "=" * 50)
    print("🎉 全部测试通过！模块可以迁入。")


def test_layers():
    print("🎯 层在变换形状")
    print("=" * 45)

    layer = Linear(784, 10)
    batch = Tensor(rng.standard_normal((32, 784)))
    output = layer(batch)

    print(f"   输入形状：{batch.shape}  ← 32 张图，每张 784 像素")
    print(f"   输出形状：{output.shape}  ← 32 张图，每张 10 类")
    print(f"   参数量：  {784 * 10 + 10:,}（权重 + 偏置）")
    print("\n✨ Linear 层把图像特征变成类别分数。")


if __name__ == "__main__":
    test_module()
    print("\n")
    analyze_layer_memory()
    print("\n")
    analyze_layer_performance()
    print("\n")
    test_layers()
