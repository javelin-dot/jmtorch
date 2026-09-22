# jmtorch

从零手写迷你 PyTorch 的学习项目。`practices/` 已包含张量、激活函数和网络层等练习，后续章节继续在这些实现上完善。

对外发行名和导入名都是 `jmtorch`。练习代码使用 `practices` 包组织；稳定接口再迁入 `src/jmtorch/`。

## 你要做什么

按阶段编辑 `practices/` 下的模块，稳定后再迁入 `src/jmtorch/`，并补上对应测试。推荐顺序：

1. 张量
2. 激活函数
3. 网络层
4. 损失函数
5. 数据管道
6. 自动求导
7. 优化器
8. 训练循环
9. 导出

练习之间直接导入已有模块，例如 `from practices.tensor import Tensor`；包内使用 `from .tensor import Tensor`。章节编号保留在 [练习指南](practices/README.md) 中，模块名不含数字前缀。`practices.nn` 汇集已有层与激活函数。细节见 [docs/上手指南.md](docs/上手指南.md)。

## 运行

需要 Python 3.10+。在仓库根目录：

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

或使用 uv：

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
uv run pytest
```

开发依赖包含 NumPy 和 pytest。在项目根目录运行练习：

```bash
python -m practices.tensor
python -m practices.activations
python -m practices.layers
```

网络层示例包含性能分析，运行时间相对较长。`pytest` 检查包与练习模块；后续功能实现后继续补充对应阶段测试。

## 文档

新人从 [docs/上手指南.md](docs/上手指南.md) 开始：环境、目录、阶段顺序、测试、调试、导出。

## 当前边界

- `src/jmtorch/` 下除 `__version__ = "0.1.0"` 外没有实现
- `practices/` 可以直接复用前面章节的实现；后续占位模块预先接好依赖
- 自动求导、数据加载、优化和训练等后续功能仍需实现
- `scripts/` 没有可运行流程
- 不要复制真实 PyTorch 源码

可编辑安装仅安装 `src/` 下的 `jmtorch` 包；`practices` 用于仓库内开发，请从项目根目录导入或运行。
