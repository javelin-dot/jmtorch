"""05 数据管道：以样本索引组织数据，并按需生成可送入网络的批次。

Dataset 约定样本访问接口；TensorDataset 对齐内存中的特征和标签。
DataLoader 负责分批与洗牌，每次遍历保留不足 batch_size 的最后一批。
这里的全部批次复用第一章的 Tensor，因此可以直接交给网络层和损失函数。
"""

# 抽象基类要求子类提供长度和按索引取样这两个基本能力。
from abc import ABC, abstractmethod
# Integral 接受 Python 整数和 NumPy 整数，Real 用于检查数值样本。
from numbers import Integral, Real

# NumPy 负责沿批次轴堆叠数组，并提供不影响全局状态的局部随机生成器。
import numpy as np

# 所有阶段统一复用这一类，避免重复加载文件导致 isinstance 判断失效。
from .tensor import Tensor


class Dataset(ABC):
    """数据集的最小接口：给出样本数量，并根据索引返回一个样本。

    子类可以在内存、文件或数据库中保存数据；加载器只依赖这两个方法。
    自定义数据集无需提前读取所有内容，可以在 __getitem__ 中按需读取。
    """

    # 抽象方法阻止尚未实现完整接口的子类被实例化。
    @abstractmethod
    def __len__(self):
        """返回样本总数；空数据集应返回 0。"""
        # 具体数量由子类的数据存储方式决定。
        raise NotImplementedError

    # 使用同一个索引同时取得特征和标签，可以保持监督信号配对。
    @abstractmethod
    def __getitem__(self, index):
        """按样本索引返回 Tensor，或由多个数值字段组成的元组/列表。"""
        # 子类负责实际读取；接口本身不假设数据已经驻留内存。
        raise NotImplementedError


class TensorDataset(Dataset):
    """将多个 Tensor 的第 0 维视为同一组样本。

    例如 TensorDataset(features, labels)[i] 返回第 i 条特征和标签。
    输入至少包含一个张量；每个张量都必须有样本轴且样本数相同。
    长度为 0 的样本轴合法，标量张量则没有样本轴，不能作为数据集。
    """

    def __init__(self, *tensors):
        """验证各字段的样本轴，再保存原始张量供后续按索引访问。"""
        # 没有字段就无法定义样本数量，应在初始化时明确报错。
        if not tensors:
            # 与零条样本不同，零个字段不是有效的 TensorDataset。
            raise ValueError("TensorDataset requires at least one Tensor")
        # 只接收统一 Tensor 类型，避免静默改变其他对象的含义。
        if not all(isinstance(tensor, Tensor) for tensor in tensors):
            # 使用 TypeError 区分对象类型问题与形状问题。
            raise TypeError("TensorDataset inputs must all be Tensor objects")
        # ndim 为 0 时无法访问 shape[0]，因此先检查是否有样本轴。
        if any(tensor.ndim == 0 for tensor in tensors):
            # 标量标签可以是一个样本的字段，但不能表示整个数据集。
            raise ValueError("TensorDataset inputs must have a sample dimension")
        # 所有字段沿第 0 维对齐，才能让相同索引代表同一条样本。
        if any(tensor.shape[0] != tensors[0].shape[0] for tensor in tensors):
            # 拒绝缺失标签或多余特征，避免在某个批次才发生越界。
            raise ValueError("TensorDataset inputs must have the same sample count")
        # 保存引用而非提前复制全数据；Tensor 的普通索引负责生成单个样本。
        self.tensors = tensors

    def __len__(self):
        """返回第 0 维的长度；已验证其他字段具有相同长度。"""
        # 这里返回样本数量，不是所有字段的元素总数。
        return self.tensors[0].shape[0]

    def __getitem__(self, index):
        """使用同一索引取出各字段，并始终返回 Tensor 元组。"""
        # 负索引和越界行为沿用 Tensor/NumPy；单字段仍保留一元元组。
        return tuple(tensor[index] for tensor in self.tensors)


class DataLoader:
    """可重复遍历的数据加载器：按 batch_size 组批，可选择每轮洗牌。

    参数 seed 固定局部随机流；同种子的加载器产生相同的各轮排列。
    同一个加载器的新一轮会继续消耗随机流，因此通常得到不同排列。
    每次 iter(loader) 都返回独立迭代器，互不修改对方的游标和索引。
    样本在 next(iterator) 时读取；额外保留的样本数据仅为当前批次。
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, seed=None):
        """检查分批参数，保存数据集，并为此加载器建立局部随机状态。"""
        # bool 是 Python int 的子类，但 True/False 不应被误当作批大小。
        if isinstance(batch_size, bool) or not isinstance(batch_size, Integral):
            # 即便 2.0 数值为整数，也拒绝含糊的浮点参数。
            raise TypeError("batch_size must be a positive integer")
        # 空批次和负步长无法表达这里的分批语义。
        if batch_size <= 0:
            # 提前报错，避免在 range 切片时暴露难理解的异常。
            raise ValueError("batch_size must be a positive integer")
        # 要求显式布尔开关，避免字符串 'False' 被解释成真值。
        if not isinstance(shuffle, bool):
            # 拼写或配置解析错误应该在构造时就被发现。
            raise TypeError("shuffle must be a bool")
        # 接受实现相同协议的自定义对象，无需强制继承 Dataset。
        if not hasattr(dataset, "__len__") or not hasattr(dataset, "__getitem__"):
            # 迭代型生成器没有随机索引能力，不能直接用于本加载器。
            raise TypeError("dataset must support __len__ and __getitem__")
        # 保存对象引用；这里不会调用 dataset[index] 读取任何样本。
        self.dataset = dataset
        # 统一成 Python int，方便与 range 和普通序列配合。
        self.batch_size = int(batch_size)
        # 顺序模式在每轮使用 range，避免额外建立完整索引数组。
        self.shuffle = shuffle
        # default_rng 不调用 np.random.seed，也不会污染调用者的全局随机流。
        self._rng = np.random.default_rng(seed)

    def __len__(self):
        """返回一轮的批次数，使用向上取整以保留最后不足一批的样本。"""
        # N=0 时结果为 0；例如 N=5、batch_size=2 时结果为 3。
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        """为本轮固定索引顺序，并返回带独立游标的生成器迭代器。"""
        # 每次创建迭代器都读取当前长度，但不会提前读取实际样本。
        sample_count = len(self.dataset)
        # 洗牌只改变索引顺序，特征与标签始终用同一个索引读取。
        indices = self._rng.permutation(sample_count) if self.shuffle else range(sample_count)
        # 每次调用生成器方法都会产生新迭代器，嵌套遍历也不会重置旧游标。
        return self._iterate_batches(indices)

    def _iterate_batches(self, indices):
        """每次 next 只读取并合并当前批次；耗尽后自然抛出 StopIteration。"""
        # 不使用 yield 的 __iter__ 已固定本轮顺序，这里只推进自己的位置。
        for start in range(0, len(indices), self.batch_size):
            # 切片自然保留尾批；仅创建当前批次的样本列表。
            samples = [self.dataset[int(index)] for index in indices[start:start + self.batch_size]]
            # yield 把控制权交回训练循环，下次 next 才读取下一批。
            yield self._collate(samples)

    @staticmethod
    def _collate(samples):
        """将单字段样本或等长字段元组/列表，沿新增的第 0 维合成批次。"""
        # 一个元组通常对应 (features, labels)，需分别堆叠每个字段。
        if isinstance(samples[0], (tuple, list)):
            # 字段数量必须一致，否则 zip 会静默丢掉较长样本的字段。
            field_count = len(samples[0])
            # 空字段没有可送入网络的内容，同样属于无效样本。
            if field_count == 0 or any(not isinstance(sample, (tuple, list)) or len(sample) != field_count for sample in samples):
                # 在当前批次读取后立即报错，说明问题在样本结构而非矩阵运算。
                raise ValueError("all samples must contain the same nonzero number of fields")
            # 转置样本列表后，每一组 field_values 都是同一字段的一批样本。
            return tuple(DataLoader._stack(field_values) for field_values in zip(*samples))
        # 单字段数据集直接返回 Tensor，训练循环无需拆开一元元组。
        return DataLoader._stack(samples)

    @staticmethod
    def _stack(values):
        """将一批形状相同的 Tensor、NumPy 数组或实数堆叠成统一 Tensor。"""
        # 拒绝字典、嵌套字段等本练习尚未支持的结构，保留明确的接口范围。
        if any(not isinstance(value, (Tensor, np.ndarray, Real, np.bool_)) for value in values):
            # 同时防止混入字符串后被 NumPy 意外转换为数值。
            raise TypeError("sample fields must be Tensor, numpy arrays, or real numbers")
        # 已有 Tensor 取底层数组；其他数值字段交给 NumPy 统一处理。
        arrays = [value.data if isinstance(value, Tensor) else np.asarray(value) for value in values]
        # stack 要求每条样本形状一致，ragged 数据会得到明确的 ValueError。
        return Tensor(np.stack(arrays, axis=0))
