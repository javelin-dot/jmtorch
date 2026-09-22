"""神经网络集成练习；复用已有激活、层与损失函数。"""

# 集成阶段要接上自动求导；这里只引入模块，不假设 backward 已经存在。
# 前向组件仍来自各自的练习文件，修改一处即可供所有后续练习使用。
from . import autograd
# 激活函数没有可学习参数，用于组合已有网络层。
# 每个名称都指向第二章定义的同一个类，避免复制实现。
from .activations import GELU, ReLU, Sigmoid, Softmax, Tanh
# 层与容器负责参数管理和前向计算，当前还没有自动反向传播。
# 可以从本模块导入这些组件，也可以直接导入 practices.layers。
from .layers import Dropout, Layer, Linear, Sequential
# 损失函数接收网络预测与目标，输出供后续求导使用的标量。
# 目前这里只复用已完成的前向损失计算。
from .losses import BinaryCrossEntropyLoss, CrossEntropyLoss, MSELoss
# 对外输入、参数与返回值统一使用第一章的 Tensor。
from .tensor import Tensor
