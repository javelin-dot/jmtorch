# 02 自动求导

目标：计算图、`backward`、梯度累积。

- 草稿：本目录的 `practice.py`
- 迁入：`src/tinytorch/autograd/`
- 测试：`tests/test_autograd.py`

只依赖已迁入的 `tinytorch.tensor`。张量运算要能留下可反传的痕迹。
