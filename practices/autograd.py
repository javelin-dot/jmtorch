"""06 自动求导练习；计算图与反向传播待实现。"""

# 自动求导围绕已有张量记录运算与梯度，不重新定义另一套 Tensor。
# 当前 Tensor 仍只做前向计算，导入本模块不会自动为它增加 backward。
from .tensor import Tensor
