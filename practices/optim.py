"""07 优化器练习；参数更新与梯度清零待实现。"""

# 优化器将消费自动求导产生的梯度；现阶段只导入模块，避免引用未实现的函数。
# 实现本阶段时再根据 autograd 中实际完成的接口补充 step 与 zero_grad。
from . import autograd
# 与网络层共用参数类型，后续就能直接接收 layer.parameters() 中的张量。
# 当前张量尚未保存梯度，因此此处只准备依赖，不执行参数更新。
from .tensor import Tensor
