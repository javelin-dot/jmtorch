# 练习草稿

每个阶段一个目录。先在这里写，接口稳定后再迁入 `src/jmtorch/`。

| 目录 | 阶段 | 迁入 |
| --- | --- | --- |
| `01_tensor/` | 张量 | `src/jmtorch/tensor/` |
| `02_autograd/` | 自动求导 | `src/jmtorch/autograd/` |
| `03_nn/` | 神经网络 | `src/jmtorch/nn/` |
| `04_optim/` | 优化器 | `src/jmtorch/optim/` |
| `05_data/` | 数据管道 | `src/jmtorch/data/` |
| `06_training/` | 训练循环 | `src/jmtorch/training/` |
| `07_export/` | 导出 | `src/jmtorch/export/` |

每个目录里的 `practice.py` 是空的。不要从其他仓库或 `~/Downloads` 拷实现进来。顺序与依赖见 [docs/上手指南.md](../docs/上手指南.md)。
