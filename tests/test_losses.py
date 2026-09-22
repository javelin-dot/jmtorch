"""
验证第 04 章损失的数学结果、数值稳定性与输入约束。

正常输入使用手算或 NumPy 作为独立参照，边界输入检查公开接口的报错约定。
所有测试只覆盖前向损失；第 06 章的计算图和梯度传播不属于本章实现范围。
"""

# NumPy 用作独立数值参考，不依赖待测 Tensor 的算术实现。
import numpy as np
# pytest 提供参数化数据以及预期异常断言。
import pytest

# 直接使用练习包的公开损失接口，覆盖日常导入路径。
from practices.losses import CrossEntropyLoss, MSELoss, log_softmax, nll_loss
# 输入与输出均复用第 01 章的统一张量类型。
from practices.tensor import Tensor


def test_mse_matches_hand_calculation_and_returns_scalar_tensor():
    """二维数据应对全部元素平均，而不是仅对样本数量平均。"""
    # 四个位置的平方误差分别为 1、4、9、16。
    predictions = Tensor([[1, 2], [3, 4]])
    # 所有目标为零，手算均值为 (1+4+9+16)/4=7.5。
    result = MSELoss()(predictions, Tensor([[0, 0], [0, 0]]))
    # 前向结果必须继续支持同一套 Tensor 接口。
    assert type(result) is Tensor
    # 两个输入轴都归约掉，输出应为零维标量。
    assert result.shape == ()
    # 输出存储类型保持与前置张量模块一致。
    assert result.data.dtype == np.float32
    # 手算值能够识别错误的分母或未平方的误差。
    assert float(result.data) == 7.5


def test_mse_accepts_scalars_and_zero_error():
    """单个数也是合法回归输入；完全预测正确时损失为零。"""
    # 标量预测 4、目标 1，对应平方误差 9。
    assert float(MSELoss()(Tensor(4), Tensor(1)).data) == 9.0
    # 使用不全为零的数据，排除只会返回常量的实现。
    values = Tensor([[1.5, -2], [3.2, 0]])
    # 完全相同的输入逐元素差为零，平均值也为零。
    assert float(MSELoss().forward(values, values).data) == 0.0


def test_mse_matches_numpy_reference():
    """固定随机数据与独立 NumPy 公式一致，并且不修改输入。"""
    # 使用局部随机数发生器，使结果不依赖其他测试的随机调用顺序。
    random = np.random.default_rng(41)
    # 选用三维数据，验证实现没有把特定轴硬编码为样本轴。
    predictions = Tensor(random.normal(size=(2, 3, 4)))
    # 同形目标模拟每个位置都具有不同的回归目标。
    targets = Tensor(random.normal(size=(2, 3, 4)))
    # 保存原数组，以发现可能的原地减法或平方污染。
    original = predictions.data.copy()
    # 以实际 float32 输入转为双精度，独立计算参考答案。
    expected = np.mean(np.square(original.astype(np.float64) - targets.data))
    # 单独调用 forward，与前面测试的可调用对象接口互相补充。
    result = MSELoss().forward(predictions, targets)
    # 输出转回 float32 会产生正常舍入，使用相对误差容限比较。
    np.testing.assert_allclose(result.data, expected, rtol=1e-6)
    # 计算损失不应改变传入网络预测。
    np.testing.assert_array_equal(predictions.data, original)


# 列向量与一维向量虽然可广播，但不能代表本接口要求的一一配对。
@pytest.mark.parametrize("shapes", [((2, 1), (2,)), ((2,), (3,)), ((), (1,))])
def test_mse_rejects_mismatched_shapes(shapes):
    """提前拒绝形状不同的输入，防止静默广播产生错误目标。"""
    # 这里只关心输入形状，因此使用零值不会影响测试含义。
    predictions, targets = (Tensor(np.zeros(shape)) for shape in shapes)
    # 错误信息应指出形状约束，而不是暴露底层广播异常。
    with pytest.raises(ValueError, match="same shape"):
        # 不论 NumPy 是否能够广播，都应按本接口契约拒绝。
        MSELoss()(predictions, targets)


def test_mse_rejects_empty_inputs():
    """空集合不存在有效均值，必须报告错误而不是返回 NaN。"""
    # 保留二维外形也不能让零个元素成为有效输入。
    empty = Tensor(np.empty((0, 2)))
    # 同形但为空的情况由独立约束负责。
    with pytest.raises(ValueError, match="non-empty"):
        # 两个输入相同，确保本测试不会被形状校验提前拦住。
        MSELoss()(empty, empty)


# 同时覆盖正轴与负轴，防止实现固定沿最后一维归约。
@pytest.mark.parametrize("dim", [0, 1, -1])
def test_log_softmax_normalizes_along_requested_dimension(dim):
    """稳定对数概率应保持形状，并沿指定轴恢复出总和为一的概率。"""
    # 非对称数据能够区分按行归一化和按列归一化。
    logits = Tensor([[1, 2, 3], [-2, 0, 4]])
    # 单独测试基础操作，让错误可定位到归一化步骤。
    result = log_softmax(logits, dim)
    # 此操作只改值，不合并或删除任何轴。
    assert result.shape == logits.shape
    # 小范围数据可安全使用直接 softmax 作为独立参考。
    exponentials = np.exp(logits.data.astype(np.float64))
    # 这里显式计算概率后取对数，与实现的稳定公式不同。
    expected = np.log(exponentials / exponentials.sum(axis=dim, keepdims=True))
    # 对照每个类别的结果，不能只验证总和而忽略类别分配。
    np.testing.assert_allclose(result.data, expected, rtol=1e-6)
    # 还原出的每组概率之和应在 float32 精度内接近一。
    np.testing.assert_allclose(np.exp(result.data).sum(axis=dim), 1, rtol=1e-6)


# 轴越界、非整数轴以及布尔值都不属于合法的维度参数。
@pytest.mark.parametrize("dim", [2, -3, 0.5, True])
def test_log_softmax_rejects_invalid_dimensions(dim):
    """非法维度给出清楚的 ValueError，避免底层运算错误。"""
    # 二维输入只允许轴 0、1、-1、-2。
    logits = Tensor([[1, 2], [3, 4]])
    # 所有参数错误都应在指数运算之前发现。
    with pytest.raises(ValueError, match="dim"):
        # 每个参数值独立执行，以便失败信息能指出具体输入。
        log_softmax(logits, dim)


def test_cross_entropy_matches_known_probabilities_and_nll_composition():
    """概率对应的 logits 应得到手算负对数均值，且与拆分接口一致。"""
    # 第一条样本真实概率为 0.7，第二条样本真实概率为 0.5。
    probabilities = np.array([[0.7, 0.2, 0.1], [0.1, 0.4, 0.5]])
    # 对概率取对数得到合法 logits；各行 softmax 会恢复原分布。
    logits = Tensor(np.log(probabilities))
    # 浮点存储的整数值类别也是合法标签，这是当前 Tensor 的统一表示。
    targets = Tensor([0.0, 2.0])
    # 直接根据两个已知概率手算平均损失。
    expected = -(np.log(0.7) + np.log(0.5)) / 2
    # 对外使用 CrossEntropyLoss 可调用对象，不需要手动组合。
    result = CrossEntropyLoss()(logits, targets)
    # 两个辅助函数也能独立组合成同样的计算。
    composed = nll_loss(log_softmax(logits), targets)
    # 已有对数概率可直接交给 NLL，无需重复归一化。
    direct_nll = nll_loss(Tensor(np.log(probabilities)), targets)
    # 损失必须保持统一张量类及标量形状。
    assert type(result) is Tensor and result.shape == ()
    # 三条路径均与独立手算值比较，防止只检查两个错误实现彼此相等。
    np.testing.assert_allclose([result.data, composed.data, direct_nll.data], expected, rtol=1e-6)


def test_cross_entropy_uniform_scores_and_one_class():
    """均匀分布的损失为 log(C)，仅一个类别时损失为零。"""
    # 四个类别等分概率，每个类别都为四分之一。
    result = CrossEntropyLoss()(Tensor(np.zeros((3, 4))), Tensor([0, 1, 3]))
    # 与解析式 log(4) 比较，不依赖另一个 softmax 实现。
    np.testing.assert_allclose(result.data, np.log(4), rtol=1e-6)
    # 单类别模型总给类别零概率一，不受该类别原始得分影响。
    single_class = CrossEntropyLoss()(Tensor([[100], [-100]]), Tensor([0, 0]))
    # 负对数一恰好为零，也覆盖每行一个元素的边界。
    assert float(single_class.data) == 0.0


def test_cross_entropy_is_invariant_to_per_sample_constant_shift():
    """每条样本同时平移全部类别分数，不应改变概率或交叉熵。"""
    # 整数分数与偏移能被 float32 精确表示，隔离输入舍入的干扰。
    values = np.array([[1, 3, 5], [-4, 2, 1]], dtype=np.float32)
    # 两条样本使用不同目标位置，覆盖行与类别配对。
    targets = Tensor([1, 2])
    # 一个大正偏移和一个大负偏移分别检验平移不变性。
    offsets = np.array([[10000], [-10000]], dtype=np.float32)
    # 对原始分数求基准损失。
    original = CrossEntropyLoss()(Tensor(values), targets)
    # 同一行所有类别一起平移，属于 softmax 的不变变换。
    shifted = CrossEntropyLoss()(Tensor(values + offsets), targets)
    # 允许输出 float32 的舍入，但损失不应随得分基线变化。
    np.testing.assert_allclose(shifted.data, original.data, rtol=1e-6)


def test_cross_entropy_large_scores_avoid_overflow_and_underflow():
    """极不自信的真实类别仍有有限损失，不经过下溢到零的概率。"""
    # 直接 exp(1000) 会溢出，真实类别概率 exp(-2000) 又会下溢。
    logits = Tensor([[1000, -1000], [-1000, 1000]])
    # 特意选分数低的类别，使两条样本的正确类别负对数约为 2000。
    targets = Tensor([1, 0])
    # 本例稳定公式不需要对低概率显式求值。
    result = CrossEntropyLoss()(logits, targets)
    # 检查有限性才能排除两边都出现无穷时的误比较。
    assert np.isfinite(result.data)
    # log(1+exp(-2000)) 在机器精度下为零，因此损失为 2000。
    np.testing.assert_allclose(result.data, 2000, rtol=1e-6)


def test_nll_accumulates_large_finite_losses_without_float32_sum_overflow():
    """均值本身可表示时，多个大损失的累加不应先发生单精度溢出。"""
    # 每行正确类别的对数概率为 -2e38，仍处于 float32 表示范围。
    log_probs = Tensor([[-2e38, 0], [-2e38, 0]])
    # 两个样本都选择很低概率的类别。
    result = nll_loss(log_probs, Tensor([0, 0]))
    # 朴素 float32 累加会在除以二之前溢出。
    assert np.isfinite(result.data)
    # 正确均值等于其中一个值的绝对值，而不是无穷大。
    np.testing.assert_allclose(result.data, 2e38, rtol=1e-6)


# 这些输入分别覆盖小数、负值、上界、巨大值及非有限标签。
@pytest.mark.parametrize("bad_label", [1.2, -1, 3, 1e30, np.nan, np.inf, -np.inf])
def test_cross_entropy_and_nll_reject_invalid_labels(bad_label):
    """类别必须是有限、整数值且处于范围内，不能静默转换成其他标签。"""
    # 三个类别的合法编号只有 0、1、2。
    logits = Tensor([[0, 1, 2]])
    # 单元素一维标签保证此测试只检查标签值约束。
    targets = Tensor([bad_label])
    # 组合接口应在真实类别索引被使用之前拒绝非法值。
    with pytest.raises(ValueError):
        # 标签 1.2 尤其不能被 astype(int) 悄悄变成 1。
        CrossEntropyLoss()(logits, targets)
    # 独立使用 NLL 的调用方也应得到相同保护。
    with pytest.raises(ValueError):
        # 对数概率本身完全合法，只让目标标签触发错误。
        nll_loss(log_softmax(logits), targets)


# 依次覆盖一维、三维、空批次、空类别，以及目标秩或长度不匹配。
@pytest.mark.parametrize("score_shape,target_shape", [
    # 一维分数缺少显式批次轴。
    ((3,), (1,)),
    # 三维分数不属于本章的批分类接口。
    ((1, 2, 3), (1,)),
    # 零条样本无法进行平均。
    ((0, 3), (0,)),
    # 没有类别就无法构造概率分布。
    ((2, 0), (2,)),
    # 列向量标签必须由调用方显式压成一维。
    ((2, 3), (2, 1)),
    # 目标数量必须与批次中样本数一致。
    ((2, 3), (1,)),
    # 单条样本的目标也应使用一维数组，而非标量。
    ((1, 3), ()),
])
def test_cross_entropy_and_nll_reject_invalid_shapes(score_shape, target_shape):
    """输入形状错误应稳定报告 ValueError，避免隐式广播或索引异常。"""
    # 全零内容让合法位置均可视为类别零，只检验维度结构。
    scores, targets = Tensor(np.zeros(score_shape)), Tensor(np.zeros(target_shape))
    # 外层交叉熵接口先校验分数结构，再执行归一化。
    with pytest.raises(ValueError):
        # 空输入与额外轴都不应产生看似有效的标量结果。
        CrossEntropyLoss()(scores, targets)
    # NLL 独立入口必须同样校验结构，而非依赖上游调用。
    with pytest.raises(ValueError):
        # 此处 scores 当作对数概率，值为零不影响形状校验。
        nll_loss(scores, targets)


# 任意一个非有限分数都会污染整条样本的归一化结果。
@pytest.mark.parametrize("bad_score", [np.nan, np.inf, -np.inf])
def test_cross_entropy_and_log_softmax_reject_nonfinite_scores(bad_score):
    """在运算前明确拒绝 NaN 或无穷，避免把无效训练目标传播下去。"""
    # 只混入一个坏值，验证实现检查的是所有元素。
    scores = Tensor([[0, bad_score]])
    # 基础操作对直接调用者提供有限性约束。
    with pytest.raises(ValueError, match="finite"):
        # 不需要等到后续对数或减法产生警告才处理。
        log_softmax(scores)
    # 损失入口也通过相同保护检查原始 logits。
    with pytest.raises(ValueError, match="finite"):
        # 类别零是合法标签，因此仅由分数值触发失败。
        CrossEntropyLoss()(scores, Tensor([0]))
