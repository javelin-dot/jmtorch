"""可互相复用的练习包；章节编号与学习顺序见 README.md。

在仓库根目录使用 ``from practices.tensor import Tensor`` 导入已有实现。
包内用 ``from .tensor import Tensor``，让各阶段共享同一个张量类。
本入口不提前加载子模块，使用 ``python -m practices.layers`` 时再运行对应练习。
"""
