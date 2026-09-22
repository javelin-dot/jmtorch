# 练习模块

每个阶段对应一个可以正常导入的 Python 模块。学习顺序放在本表和 `guides/` 文档中，避免数字目录和重复的 `practice.py` 文件名影响导入。接口稳定后再迁入 `src/jmtorch/`。

| 顺序 | 模块 | 指南 | 迁入 |
| --- | --- | --- | --- |
| 01 张量 | `practices.tensor` | [张量](guides/01_tensor.md) | `src/jmtorch/tensor/` |
| 02 激活函数 | `practices.activations` | [激活函数](guides/02_activations.md) | `src/jmtorch/nn/` |
| 03 网络层 | `practices.layers` | [网络层](guides/03_layers.md) | `src/jmtorch/nn/` |
| 04 损失函数 | `practices.losses` | [损失函数](guides/04_losses.md) | `src/jmtorch/nn/` |
| 05 数据管道 | `practices.data` | [数据管道](guides/05_data.md) | `src/jmtorch/data/` |
| 06 自动求导 | `practices.autograd` | [自动求导](guides/06_autograd.md) | `src/jmtorch/autograd/` |
| 07 优化器 | `practices.optim` | [优化器](guides/07_optim.md) | `src/jmtorch/optim/` |
| 08 训练循环 | `practices.training` | [训练循环](guides/08_training.md) | `src/jmtorch/training/` |
| 09 导出 | `practices.export` | [导出](guides/09_export.md) | `src/jmtorch/export/` |
| 集成入口 | `practices.nn` | [神经网络](guides/nn.md) | `src/jmtorch/nn/` |

在项目根目录中的其他代码或交互会话里，直接写：

```python
from practices.tensor import Tensor
from practices.activations import ReLU
from practices.layers import Linear

x = Tensor([[1.0, 2.0]])
y = ReLU()(Linear(2, 3)(x))
```

在 `practices` 包内复用前面的模块时，使用相对导入，例如：

```python
from .tensor import Tensor
from .activations import ReLU
from .layers import Linear
```

先在根目录执行 `pip install -e ".[dev]"` 安装开发依赖，再用模块方式运行：

```bash
python -m practices.tensor
python -m practices.activations
python -m practices.layers
```

`layers` 的演示包含性能分析，运行较慢。不要直接执行 `python practices/layers.py`：相对导入需要包上下文。

Windows 下若重定向输出时遇到 `UnicodeEncodeError`，使用 `python -X utf8 -m practices.activations` 等命令，让中文和表情符号按 UTF-8 输出。

04、05、06 及后续章节已预接已有类或模块；这些导入不代表自动求导、数据加载、优化器或训练流程已经实现。练习尚未迁入正式库时，也可以沿用上述路径继续开发。

可编辑安装仍只安装 `src/jmtorch`。`practices` 是仓库内的练习包，请从项目根目录运行；给其他项目使用的稳定接口继续放在 `jmtorch` 中。环境和迁入方式见 [上手指南](../docs/上手指南.md)。
