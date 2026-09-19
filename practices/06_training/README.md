# 06 训练循环

目标：取 batch → 前向 → 损失 → 反传 → 更新。损失应能下降。

- 草稿：本目录的 `practice.py`
- 迁入：`src/tinytorch/training/`
- 测试：`tests/test_training.py`
- 入口占位：`scripts/run_training.py`

只组合已经迁入的 nn、optim、data，不要在这里重写层或优化器。
