# 练习草稿

每个阶段一个目录。先在这里写，接口稳定后再迁入 `src/tinytorch/`。

| 目录 | 阶段 | 迁入 |
| --- | --- | --- |
| `01_tensor/` | 张量 | `src/tinytorch/tensor/` |
| `02_autograd/` | 自动求导 | `src/tinytorch/autograd/` |
| `03_nn/` | 神经网络 | `src/tinytorch/nn/` |
| `04_optim/` | 优化器 | `src/tinytorch/optim/` |
| `05_data/` | 数据管道 | `src/tinytorch/data/` |
| `06_training/` | 训练循环 | `src/tinytorch/training/` |
| `07_export/` | 导出 | `src/tinytorch/export/` |

每个目录里的 `practice.py` 是空的。不要从官方 TinyTorch 或 `~/Downloads` 拷实现进来。顺序与依赖见 `docs/练习顺序.md`。
