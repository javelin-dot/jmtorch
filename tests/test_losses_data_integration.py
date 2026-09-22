"""验证批次进入模型和损失后仍保持样本配对、数值与尾批权重正确。"""

# NumPy 公式提供独立数值参考，不用待测损失构造预期答案。
import numpy as np
# 使用两种任务和两种顺序，覆盖回归与分类的数据流。
import pytest

# 数据加载器与损失共同使用练习包的公开入口。
from practices.data import DataLoader, TensorDataset
# 已有线性层将真实批次转换成网络输出。
from practices.layers import Linear
# 两种监督目标分别处理连续数值和类别编号。
from practices.losses import CrossEntropyLoss, MSELoss
# 与所有前置章节保持同一个张量类型。
from practices.tensor import Tensor


# 无论按序还是洗牌，整轮每个样本都应恰好参与一次计算。
@pytest.mark.parametrize("shuffle", [False, True], ids=["ordered", "shuffled"])
# 分别验证回归目标与分类目标的形状约定。
@pytest.mark.parametrize("task", ["regression", "classification"])
def test_batches_feed_model_and_preserve_full_dataset_loss(shuffle, task):
    """按实际批大小加权后，分批损失应等于对全部样本一次计算的结果。"""
    # 五个样本分为 2、2、1，主动覆盖尾批与正常批次权重不同的情况。
    features = Tensor([[3, 1], [0, 2], [2, 0], [1, 3], [4, 1]])
    # 固定线性变换，避免网络随机初始化掩盖配对或损失错误。
    model = Linear(2, 2)
    # 两列使用不同的参数，验证输出不是简单照抄输入。
    model.weight.data[...] = [[1.0, -0.5], [0.25, 0.75]]
    # 非零偏置还会覆盖最后一批的广播行为。
    model.bias.data[...] = [0.5, -0.25]
    # 直接用 NumPy 做一次整批前向，生成独立参考分数。
    scores = features.data @ model.weight.data + model.bias.data
    # 回归的目标与预测保持完全相同的形状。
    if task == "regression":
        # 各样本目标不同，错配索引就会改变最终误差。
        targets = Tensor([[1, 0], [0, 2], [3, 1], [1, 4], [2, -1]])
        # 每个样本的输出维度相同，因此按批大小加权可还原全元素均值。
        expected = np.mean(np.square(scores.astype(np.float64) - targets.data))
        # 调用实际回归损失接口，验证与加载器的配合。
        loss_fn = MSELoss()
    else:
        # 分类标签使用一维类别编号，不能按回归形状广播。
        targets = Tensor([0, 1, 0, 1, 0])
        # 独立参考用 logaddexp.reduce 计算 log-sum-exp，避免复写被测步骤。
        normalizers = np.logaddexp.reduce(scores.astype(np.float64), axis=1)
        # 从每行选取真实类别分数，得到该样本的负对数似然。
        expected = np.mean(normalizers - scores[np.arange(5), targets.data.astype(int)])
        # 直接使用 logits，不在模型后重复归一化。
        loss_fn = CrossEntropyLoss()
    # 同种子保证失败可复现；开关控制是否重新排列样本。
    loader = DataLoader(TensorDataset(features, targets), batch_size=2, shuffle=shuffle, seed=17)
    # 同时记录尾批尺寸和损失总量，不能只验证最终输出类型。
    batch_sizes, weighted_loss = [], 0.0
    # 数据加载器返回的特征和标签应直接适配模型与损失函数。
    for batch_x, batch_y in loader:
        # 真实样本数来自批次，而非固定的配置参数。
        batch_sizes.append(batch_x.shape[0])
        # 三个组件串联后仍应返回标量 Tensor。
        loss = loss_fn(model(batch_x), batch_y)
        # 防止组批或损失阶段意外引入另一套张量定义。
        assert type(loss) is Tensor and loss.shape == ()
        # 恢复每批损失总量，在循环结束后统一求均值。
        weighted_loss += float(loss.data) * batch_x.shape[0]
    # 保留全部五个样本，最后一批不能被丢弃或重复填满。
    assert batch_sizes == [2, 2, 1]
    # 分批结果应在 float32 舍入范围内与整批独立参考一致。
    np.testing.assert_allclose(weighted_loss / len(features.data), expected, rtol=1e-6)
