# 04 损失函数

目标：学习均方误差、交叉熵与二元交叉熵，并考虑数值稳定性。

- 草稿：[`practices/losses.py`](../losses.py)
- 依赖：已有 `Tensor`、激活函数与网络层
- 迁入：`src/jmtorch/nn/`

使用 `from .tensor import Tensor`、`from .activations import ReLU`、`from .layers import Linear` 直接接入前面章节，继续完成损失函数练习。反向传播在第 06 章接入。
