"""04 损失函数练习；复用同一个 practices 包中的前置模块。"""

__all__ = ['rng', 'EPSILON', 'log_softmax', 'nll_loss', 'MSELoss', 'CrossEntropyLoss', 'BinaryCrossEntropyLoss']

import numpy as np
rng = np.random.default_rng(7)

# 直接复用本包中前面练习定义的张量、激活函数和线性层。
from .tensor import Tensor
from .activations import ReLU
from .layers import Linear

# 数值稳定性常量，用于将概率限制在开区间 (0, 1) 内。
EPSILON = 1e-7  # 避免对零取对数而产生无穷值。

def log_softmax(x: Tensor, dim: int = -1) -> Tensor:
    """
    使用数值稳定的方式计算指定维度上的 log-softmax。

    使用 log-sum-exp 技巧，避免先取指数时因输入过大而溢出。
    dim 默认为最后一维，输出形状与输入张量保持一致。

    计算步骤：
    1. 沿指定维度求最大值，并保留该维度以便广播。
    2. 输入减去最大值，将最大的指数输入移动到零。
    3. 对平移后的值取指数、求和，再取对数。
    4. 返回 input - max - log_sum_exp，得到各类别的对数概率。

    使用示例：
    >>> logits = Tensor([[1.0, 2.0, 3.0], [0.1, 0.2, 0.9]])
    >>> result = log_softmax(logits, dim=-1)
    >>> print(result.shape)
    (2, 3)

    学习提示：np.max(..., keepdims=True) 能保留维度，避免广播时形状不匹配。
    """
    # 归一化至少需要一条非空的类别轴；标量和空张量没有有效概率分布。
    if x.ndim == 0 or x.data.size == 0:
        # 提前拒绝无效输入，避免底层 max 抛出难以理解的归约错误。
        raise ValueError("log_softmax requires a non-empty, non-scalar tensor")
    # 轴必须是一个整数；bool 虽属于 int 子类，但不是这里期望的轴编号。
    if isinstance(dim, (bool, np.bool_)) or not isinstance(dim, (int, np.integer)):
        # 拒绝小数或元组，保证此处只沿一个维度归一化。
        raise ValueError("log_softmax dim must be an integer")
    # 正轴和负轴都遵循 NumPy 的维度编号范围。
    if not -x.ndim <= dim < x.ndim:
        # 提供输入秩，方便学习者检查是否选错了类别轴。
        raise ValueError(f"log_softmax dim {dim} is out of range for {x.ndim} dimensions")
    # NaN、正无穷和负无穷无法通过减最大值得到可靠的有限结果。
    if not np.isfinite(x.data).all():
        # 在运算之前报告数据问题，避免返回一整行 NaN。
        raise ValueError("log_softmax inputs must contain only finite values")

    # 第一步：中间计算提升至 float64，防止减法与指数求和先在 float32 中溢出。
    # 最终 Tensor 仍统一保存 float32；超出最终表示范围的结果仍无法表示。
    values = x.data.astype(np.float64)
    # 保留归约维度，取得每组输入的最大值，供后续自动广播。
    max_vals = np.max(values, axis=dim, keepdims=True)

    # 第二步：减去最大值，使后续指数运算的输入不大于零。
    shifted = values - max_vals

    # 第三步：求平移后指数和的对数，作为归一化项。
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=dim, keepdims=True))

    # 第四步：从平移后的输入中减去归一化项。
    result = shifted - log_sum_exp

    # 与前置练习使用相同的 Tensor 类型，输出形状与输入一致。
    return Tensor(result)


def nll_loss(log_probs: Tensor, targets: Tensor) -> Tensor:
    """
    从二维对数概率中选择真实类别，并返回平均负对数似然。

    log_probs 的形状必须为 (N, C)，targets 必须为 (N,)。
    每条样本仅使用一个类别编号，不接受 one-hot 或概率分布标签。
    Tensor 内部统一使用 float32，因此整数值浮点标签（如 1.0）也合法。
    先验证标签是有限整数值，再转换索引，避免把 1.2 静默截断成 1。

    与 log_softmax 分工后，本函数只负责选目标类别和求平均。
    因此已有对数概率的调用方也能复用，无需再次做归一化。
    """
    # 先约束为单标签批分类接口，避免广播或高维索引改变损失含义。
    if log_probs.ndim != 2:
        # 此处 N 表示样本数，C 表示可选类别数。
        raise ValueError("nll_loss log_probs must have shape (N, C)")
    # 空批次无法求平均，零类别也无法选择真实类别。
    if any(size == 0 for size in log_probs.shape):
        # 在取索引之前明确拒绝这两种空输入。
        raise ValueError("nll_loss requires non-empty samples and classes")
    # 每条样本恰好对应一个类别；二维列向量同样应显式调整为一维。
    if targets.shape != (log_probs.shape[0],):
        # 合并维度和数量校验，防止目标数量与样本数量不一致。
        raise ValueError("nll_loss targets must have shape (N,) matching the batch")
    # 对数概率中的 NaN 会污染均值，负无穷则表示不能有限表达的损失。
    if not np.isfinite(log_probs.data).all():
        # 本教学接口要求输入对数概率均为有限值。
        raise ValueError("nll_loss log_probs must contain only finite values")
    # 标签必须先检查有限性，不能直接把 NaN 或无穷大转换成整数。
    if not np.isfinite(targets.data).all():
        # 单独报错，方便区别非法数据与合法但越界的类别编号。
        raise ValueError("nll_loss targets must contain only finite values")
    # 对浮点存储的标签逐项检查整数值，保留 0.0、1.0 这种有效编号。
    if not np.equal(targets.data, np.floor(targets.data)).all():
        # 这项检查避免 NumPy 的整数类型转换静默丢弃小数部分。
        raise ValueError("nll_loss targets must contain integer class indices")
    # 在转换为机器整数之前校验范围，超大的浮点标签也能稳定报错。
    if np.any((targets.data < 0) | (targets.data >= log_probs.shape[1])):
        # NumPy 会把负索引用于倒数取值，不能让负标签进入数组索引。
        raise ValueError(f"nll_loss target index out of range [0, {log_probs.shape[1] - 1}]")
    # 前述有限性、整数值和范围检查通过后，转换为平台原生索引类型。
    target_indices = targets.data.astype(np.intp)
    # 行编号与目标编号逐项配对，每条样本只选择真实类别的一个对数概率。
    selected_log_probs = log_probs.data[np.arange(log_probs.shape[0]), target_indices]
    # 使用 float64 累加，避免多个很大的有限损失相加时先溢出。
    # 求均值后再转回 Tensor，保证公开返回类型与前置模块一致。
    return Tensor(-np.mean(selected_log_probs, dtype=np.float64))


class MSELoss:
    """用于回归任务的均方误差损失，对全部元素的误差平方取平均。"""

    def __init__(self):
        """初始化均方误差损失；当前实现无需维护额外状态。"""
        pass

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        计算预测值与目标值之间的均方误差，返回标量张量。

        预测与目标逐元素相减，较大的误差在平方后占据更大权重。
        最后对所有元素取平均，使损失反映整体预测偏差。

        计算步骤：
        1. 计算差值：predictions - targets。
        2. 对每个差值平方：diff²。
        3. 对所有元素的平方差取平均。

        使用示例：
        >>> loss_fn = MSELoss()
        >>> predictions = Tensor([1.0, 2.0, 3.0])
        >>> targets = Tensor([1.5, 2.5, 2.8])
        >>> loss = loss_fn(predictions, targets)
        >>> print(f"MSE Loss: {loss.data:.4f}")
        MSE Loss: 0.1800

        学习提示：
        - predictions.data - targets.data 会计算逐元素差值。
        - 可以使用 **2 或 np.power(diff, 2) 对差值平方。
        - np.mean() 不指定 axis 时会对全部元素取平均。
        """
        # 同形约束防止 (N, 1) 与 (N,) 意外广播成 (N, N)。
        if predictions.shape != targets.shape:
            # 一个预测位置必须对应一个目标位置，不能依赖隐式广播补齐。
            raise ValueError("MSELoss predictions and targets must have the same shape")
        # 空输入的平均值没有定义；零维标量包含一个元素，因此依然合法。
        if predictions.data.size == 0:
            # 显式报错比返回 NaN 更便于定位缺失数据。
            raise ValueError("MSELoss requires non-empty inputs")

        # 第一步：计算各个预测值相对于目标值的误差。
        # float64 中间值减少平方与求和的溢出风险，返回时仍统一为 float32。
        diff = predictions.data.astype(np.float64) - targets.data.astype(np.float64)

        # 第二步：平方使正负误差不会互相抵消。
        squared_diff = diff ** 2

        # 第三步：汇总所有元素，得到单个均方误差值。
        mse = np.mean(squared_diff)

        # 不保留样本轴或输出轴，形状为 () 的 Tensor 就是标量损失。
        return Tensor(mse)

    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """允许用 loss_fn(predictions, targets) 直接调用前向计算。"""
        return self.forward(predictions, targets)

    def backward(self) -> Tensor:
        """
        保留反向传播接口，梯度计算将在第 06 章自动求导练习中单独处理。

        当前方法仍是占位实现，学习前向损失计算时可暂时忽略。
        """
        pass

class CrossEntropyLoss:
    """用于多分类任务的交叉熵损失，目标值为正确类别的索引。"""

    def __init__(self):
        """初始化交叉熵损失；当前实现无需维护额外状态。"""
        pass

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        计算未归一化分类分数 logits 与目标类别索引之间的交叉熵。

        logits 必须是 (N, C)，targets 是 (N,) 的整数值类别索引。
        使用稳定的 log-softmax 直接计算对数概率，避免额外取对数。

        计算步骤：
        1. 使用数值稳定的 log-softmax 计算各类别的对数概率。
        2. 交给 nll_loss 验证标签、选择正确类别，并返回负对数概率均值。
        3. 两步组合等价于单标签交叉熵，但不显式计算可能下溢到零的概率。

        使用示例：
        >>> loss_fn = CrossEntropyLoss()
        >>> logits = Tensor([[2.0, 1.0, 0.1], [0.5, 1.5, 0.8]])  # 两个样本、三个类别
        >>> targets = Tensor([0, 1])  # 两个样本分别属于类别 0 和类别 1
        >>> loss = loss_fn(logits, targets)
        >>> print(f"Cross-Entropy Loss: {loss.data:.4f}")

        学习提示：
        - log_softmax() 能避免直接计算 softmax 后再取对数的数值问题。
        - nll_loss 先检查有限性、整数值和范围，再把目标转换为整数索引。
        - 类别数来自 logits.shape[-1]，必须先校验索引再访问数组。
          NumPy 的负索引会从末尾取值，可能静默选择错误类别；
          过大的索引则会直接触发 IndexError，缺少类别范围的说明。
        - np.arange(batch_size) 提供行索引，与各样本的目标类别配对取值。
        - 最终返回 -np.mean(selected_log_probs)，即平均负对数似然。
        """
        # 先固定批次与类别两个维度，避免把向量或空间张量误当作批分类输入。
        if logits.ndim != 2:
            # 对非标准形状的任务，应由调用方明确整理出 (N, C)。
            raise ValueError("CrossEntropyLoss logits must have shape (N, C)")
        # 第一步：减最大值后计算对数概率，内部同时检查空输入及非有限得分。
        log_probs = log_softmax(logits, dim=-1)
        # 第二步：只选择正确类别并求负均值；标签校验由可复用的 NLL 负责。
        # 拆分保留两步数学含义，也方便分别验证数值稳定性与类别索引逻辑。
        return nll_loss(log_probs, targets)

    def __call__(self, logits: Tensor, targets: Tensor) -> Tensor:
        """允许用 loss_fn(logits, targets) 直接调用前向计算。"""
        return self.forward(logits, targets)

    def backward(self) -> Tensor:
        """
        保留反向传播接口，梯度计算将在第 06 章自动求导练习中单独处理。

        当前方法仍是占位实现，学习前向损失计算时可暂时忽略。
        """
        pass

class BinaryCrossEntropyLoss:
    """用于二分类任务的交叉熵损失，预测值为概率，目标值为二元标签。"""

    def __init__(self):
        """初始化二分类交叉熵损失；当前实现无需维护额外状态。"""
        pass

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        计算预测概率与二元目标标签之间的二分类交叉熵。

        预测概率先裁剪到远离 0 和 1 的范围，防止对零取对数。
        每个样本同时使用正类概率与负类概率，对相应标签计算损失。

        计算步骤：
        1. 将预测概率裁剪到 [EPSILON, 1-EPSILON]。
        2. 计算 -(targets * log(predictions) + (1-targets) * log(1-predictions))。
        3. 对全部样本的损失取平均。

        使用示例：
        >>> loss_fn = BinaryCrossEntropyLoss()
        >>> predictions = Tensor([0.9, 0.1, 0.7, 0.3])  # 介于 0 和 1 之间的概率
        >>> targets = Tensor([1.0, 0.0, 1.0, 0.0])      # 二元标签
        >>> loss = loss_fn(predictions, targets)
        >>> print(f"Binary Cross-Entropy Loss: {loss.data:.4f}")

        学习提示：
        - np.clip(predictions.data, 1e-7, 1-1e-7) 可防止 log(0)。
        - 标签为 1 时仅保留 -log(preds)，标签为 0 时仅保留 -log(1-preds)。
        - np.mean() 汇总全部样本，使返回值为标量损失。
        """
        # 第一步：裁剪概率，使正类概率与负类概率都不会为零。
        eps = EPSILON
        clamped_preds = np.clip(predictions.data, eps, 1 - eps)

        # 第二步：分别计算正类与负类的对数概率。
        # 裁剪后的概率处于开区间 (0, 1)，这两个对数均为有限值。
        log_preds = np.log(clamped_preds)
        log_one_minus_preds = np.log(1 - clamped_preds)

        # 根据二元标签选择相应的负对数概率，得到各个样本的损失。
        bce_per_sample = -(targets.data * log_preds + (1 - targets.data) * log_one_minus_preds)

        # 第三步：对全部样本取平均，得到本批次的损失。
        bce_loss = np.mean(bce_per_sample)

        return Tensor(bce_loss)

    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """允许用 loss_fn(predictions, targets) 直接调用前向计算。"""
        return self.forward(predictions, targets)

    def backward(self) -> Tensor:
        """
        保留反向传播接口，梯度计算将在后续练习中单独处理。

        当前方法仍是占位实现，学习前向损失计算时可暂时忽略。
        """
        pass
