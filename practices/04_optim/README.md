# 04 优化器

目标：`step` / `zero_grad`，至少实现 SGD。

- 草稿：本目录的 `practice.py`
- 迁入：`src/jmtorch/optim/`
- 测试：`tests/test_optim.py`

只更新已接到 autograd 的参数。忘了 `zero_grad` 是常见坑，见 [docs/上手指南.md](../../docs/上手指南.md)「卡住了怎么查」。
