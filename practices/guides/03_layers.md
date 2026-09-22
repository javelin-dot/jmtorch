# 03 网络层

目标：理解线性层、Dropout、Sequential 的前向计算和参数组织。

- 草稿：[`practices/layers.py`](../layers.py)
- 依赖：第 01 章 `Tensor` 与第 02 章激活函数
- 迁入：`src/jmtorch/nn/`
- 测试：`tests/test_nn.py`

项目根目录运行 `python -m practices.layers`，其中性能分析会耗费较长时间。后续章节通过 `from practices.layers import Linear` 等语句复用已有层。
