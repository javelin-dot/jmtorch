# 08 训练循环

目标：取 batch → 前向 → 损失 → 反传 → 更新。损失应能下降。

- 草稿：[`practices/training.py`](../training.py)
- 迁入：`src/jmtorch/training/`
- 测试：`tests/test_training.py`
- 入口占位：`scripts/run_training.py`

组合前面章节的 `nn`、`losses`、`autograd`、`optim` 与 `data` 模块，不在这里重写层或优化器。当前仅预接依赖，训练流程需要在相关功能实现后补充。
