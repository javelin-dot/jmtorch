# 神经网络集成入口

目标：汇集已有的网络层、激活函数与损失函数，随着后续章节完成再接入自动求导。

- 草稿：[`practices/nn.py`](../nn.py)
- 迁入：`src/jmtorch/nn/`
- 测试：`tests/test_nn.py`

复用 `practices.layers`、`practices.activations` 和 `practices.losses` 中的类，不重新定义一套实现。当前前向练习尚未接入反向传播。
