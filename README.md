# tinytorch-learn

从零手写迷你 PyTorch 的**目录与文档骨架**。灵感来自 [TinyTorch](https://mlsysbook.ai/tinytorch/)，**没有**官方源码、练习答案或神经网络实现。

发行名是 `tinytorch-learn`，导入名是 `tinytorch`。当前仓库只搭好路径；实现由你填写。

## 你要做什么

按阶段在 `practices/` 里草稿，稳定后再迁入 `src/tinytorch/`，并补上对应测试。推荐顺序：

1. 张量
2. 自动求导
3. 神经网络
4. 优化器
5. 数据管道
6. 训练循环
7. 导出

后一阶段只依赖已经迁入 `src/tinytorch/` 的接口。细节见 `docs/练习顺序.md` 与 `docs/目录地图.md`。

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

现在 pytest 只收集 `tests/test_package.py`（可导入与版本号）。各阶段测试与脚本都是占位，没有训练或写文件逻辑。

## 文档

| 文档 | 内容 |
| --- | --- |
| [docs/目录地图.md](docs/目录地图.md) | 顶层目录、写代码的位置 |
| [docs/练习顺序.md](docs/练习顺序.md) | 七个阶段与依赖关系 |
| [docs/运行指南.md](docs/运行指南.md) | 安装、pytest、脚本入口 |
| [docs/调试指南.md](docs/调试指南.md) | 按阶段调试、常见坑 |
| [docs/导出与复用.md](docs/导出与复用.md) | 给其他项目安装、公开表面 |

## 刻意留空

- `src/tinytorch/` 下除 `__version__ = "0.1.0"` 外没有实现
- `practices/` 只有 README、空 `practice.py`、`.gitkeep`
- 各阶段 `tests/test_*.py` 没有断言
- `scripts/` 没有可运行流程
- 不要复制官方 TinyTorch 或真实 PyTorch 源码

这是学习容器：目录已经搭好，实现由你填写。
