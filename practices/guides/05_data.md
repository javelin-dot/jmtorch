# 05 数据管道

目标：`Dataset` / `DataLoader`。玩具数据即可。

- 草稿：[`practices/data.py`](../data.py)
- 迁入：`src/jmtorch/data/`
- 测试：`tests/test_data.py`

复用 `practices.tensor.Tensor` 表示数据。当前仅接入已有依赖，数据加载仍待实现；后续按 batch 取出时，形状要能对上训练循环。
