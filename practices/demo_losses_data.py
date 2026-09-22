"""串起第 04、05 章：按批取样、执行前向计算并统计平均交叉熵。

从项目根目录运行 python -m practices.demo_losses_data。
示例固定模型参数以方便手算核对，尚不执行反向传播或参数更新。
"""

# 数据集按相同的样本索引保存特征和标签，加载器负责组成批次。
from .data import DataLoader, TensorDataset
# 使用前面章节的线性层，验证批次可以直接作为模型输入。
from .layers import Linear
# 回归与分类分别选择均方误差与交叉熵目标。
from .losses import CrossEntropyLoss, MSELoss
# 所有模块共用一个 Tensor 类，调用时无需转换成另一套对象。
from .tensor import Tensor


def main():
    """输出可复现的损失值、批次形状和按样本加权的整轮平均损失。"""
    # 先核对回归损失，三个位置的平方误差分别为 0.25、0.25、0.04。
    mse = MSELoss()(Tensor([1, 2, 3]), Tensor([1.5, 2.5, 2.8]))
    # 固定小数位让实际输出可以直接与学习报告中的手算值比较。
    print(f"MSELoss = {float(mse.data):.6f}")
    # 五个样本便于观察 batch_size=2 时的最后一个小批次。
    features = Tensor([[3, 1], [0, 2], [2, 0], [1, 3], [4, 1]])
    # 每行对应一个真实类别编号；内部 float32 存储不改变整数值语义。
    labels = Tensor([0, 1, 0, 1, 0])
    # 同一个索引同时选取特征和标签，洗牌也不会破坏配对。
    dataset = TensorDataset(features, labels)
    # 先按原顺序遍历，便于核对每批标签；训练时可改成 shuffle=True。
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    # 固定单位权重与零偏置，让模型输出与输入数字相同。
    model = Linear(2, 2)
    # 仅替换已有参数中的数值，参数本身仍使用原来的 Tensor 对象。
    model.weight.data[...] = [[1, 0], [0, 1]]
    # 固定偏置使该示例不依赖网络层的随机数生成顺序。
    model.bias.data[...] = [0, 0]
    # CrossEntropyLoss 接收原始 logits，调用前无需添加 Softmax。
    loss_fn = CrossEntropyLoss()
    # 累计损失总量和样本总数，避免把不同大小批次的均值等权平均。
    total_loss, sample_count = 0.0, 0
    # len(dataset) 是样本数，len(loader) 是包含尾批的一轮批次数。
    print(f"samples={len(dataset)}, batch_size=2, batches={len(loader)}")
    # for 会创建迭代器并反复调用 next，直到遇到 StopIteration。
    for batch_index, (batch_x, batch_y) in enumerate(loader, start=1):
        # 先前向计算得到 (B, C) 分数；这里不执行训练更新。
        logits = model(batch_x)
        # 标签形状为 (B,)，损失是该批 B 个样本的平均值。
        loss = loss_fn(logits, batch_y)
        # 最后一批 B=1，应使用实际形状而非配置的 batch_size。
        current_size = batch_x.shape[0]
        # 将批平均损失还原为当前批的损失总量，再加入整轮累计值。
        total_loss += float(loss.data) * current_size
        # 同时累计真实样本数，最后用于统一求平均。
        sample_count += current_size
        # 展示批次外形、对应标签及真实损失，便于验证数据流。
        print(f"batch={batch_index}, x.shape={batch_x.shape}, labels={batch_y.data.astype(int).tolist()}, loss={float(loss.data):.6f}")
    # 本示例数据非空，因此可直接按本轮实际样本数求平均。
    print(f"weighted_mean_cross_entropy={total_loss / sample_count:.6f}")


# 导入示例只注册函数；以模块方式运行时才打印演示结果。
if __name__ == "__main__":
    # 使用独立入口，其他测试和报告也能选择何时运行该流程。
    main()
