# 09 导出

目标：权重保存与加载；固定对外导入路径。

- 草稿：[`practices/export.py`](../export.py)
- 迁入：`src/jmtorch/export/`
- 测试：`tests/test_export.py`
- 入口占位：`scripts/export_checkpoint.py`

当前仅预接已有张量和网络定义，保存与加载仍待实现。实现后验证加载前后的前向输出一致；正式库的使用方式见 [上手指南](../../docs/上手指南.md)。
