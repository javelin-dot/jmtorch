"""训练循环练习；先导入已有组件，再逐步补齐训练流程。"""

# 数据加载已经支持组批与洗牌；自动求导和优化器仍待实现。
# 后续可直接调用 data.DataLoader，并在梯度与优化器完成后补齐训练更新。
from . import autograd, data, nn, optim
# 复用激活函数，后续可以直接构建简单的回归或分类模型。
# 导入仅注册名称，不会自动创建或训练网络。
from .activations import ReLU, Sigmoid
# Linear 与 Sequential 已能做前向计算，训练阶段负责将它们组合起来。
# 参数更新仍须等待自动求导和优化器实现。
from .layers import Linear, Sequential
# 按任务选择损失：回归用 MSE，多分类用交叉熵，二分类可用二元交叉熵。
# 所有损失都复用前面章节的 Tensor 类型。
from .losses import BinaryCrossEntropyLoss, CrossEntropyLoss, MSELoss
# 训练输入、标签和预测保持一致的张量类型，避免跨文件类型判断失败。
from .tensor import Tensor
