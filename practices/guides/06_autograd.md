# 06 自动求导

目标：计算图、`backward`、梯度累积。

- 草稿：[`practices/autograd.py`](../autograd.py)
- 迁入：`src/jmtorch/autograd/`
- 测试：`tests/test_autograd.py`

直接复用 `practices.tensor.Tensor`，为已有张量运算逐步接入计算图与反向传播。当前模块仅预接张量依赖；原 `06_autograde` 对应的练习统一放在本章。
