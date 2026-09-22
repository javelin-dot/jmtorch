"""验证 05 数据管道的样本对齐、分批、洗牌以及独立迭代器行为。

参考结果使用简单的样本编号和 NumPy 数组，避免依赖待测分批实现。
测试既检查常见训练用法，也覆盖空数据、尾批及非法输入的边界。
"""

# NumPy 提供独立的数组断言与预期编号序列。
import numpy as np
# pytest 参数化让同一组行为可以覆盖多个批大小和错误输入。
import pytest

# 数据管道使用本项目的公开练习模块，而不是课程参考代码。
from practices.data import DataLoader, Dataset, TensorDataset
# 输入与输出都必须维持项目统一的张量类型。
from practices.tensor import Tensor


def make_dataset(size=5):
    """建立可追踪样本编号的数据集，标签恒为编号的十倍。

    特征的形状为 (N, 1)，标签的形状为 (N,)。
    简单的对应关系能够直接检测洗牌后是否出现特征标签错配。
    """
    # 编号递增，便于同时验证不重不漏与顺序读取。
    numbers = np.arange(size, dtype=np.float32)
    # reshape 增加单特征轴，使批次形状与真实网络输入一致。
    return TensorDataset(Tensor(numbers.reshape(size, 1)), Tensor(numbers * 10))


def epoch_order(loader):
    """收集某一轮的特征编号，仅用于比较完整的样本排列。"""
    # 数据集在这些测试中非空，因此可以直接拼接所有批次。
    return np.concatenate([features.data[:, 0] for features, _ in loader])


def test_dataset_requires_both_abstract_methods():
    """基础接口和未实现完整接口的子类均不能实例化。"""
    # 没有实现两个抽象方法的 Dataset 不能误当作可读取的数据源。
    with pytest.raises(TypeError):
        # 这里验证 ABC 强制约束，而不是等到训练循环中再报错。
        Dataset()

    # 仅实现长度还不足以满足按索引读取的约定。
    class IncompleteDataset(Dataset):
        # 子类的长度实现不能替代缺少的 __getitem__。
        def __len__(self):
            # 任意合法长度都不应绕过抽象方法检查。
            return 3

    # 不完整子类同样应在构造时失败。
    with pytest.raises(TypeError):
        # 若能实例化，说明样本访问接口没有被强制实现。
        IncompleteDataset()


def test_tensor_dataset_keeps_aligned_fields_and_tensor_identity():
    """相同索引取得同一条样本的全部字段，并返回统一 Tensor。"""
    # 使用多字段数据集，覆盖特征、标签和额外数值信息。
    dataset = TensorDataset(Tensor([[1, 2], [3, 4]]), Tensor([10, 20]), Tensor([100, 200]))
    # 长度来自样本轴，不是字段数或元素数。
    assert len(dataset) == 2
    # 一条样本应返回固定顺序的元组。
    sample = dataset[1]
    # 所有字段均可继续交给前向模块，不能泄漏成裸数组。
    assert isinstance(sample, tuple) and all(type(field) is Tensor for field in sample)
    # 二维特征的一个样本是一维向量。
    np.testing.assert_array_equal(sample[0].data, [3, 4])
    # 标量标签不会被额外包成长度为一的向量。
    assert sample[1].shape == () and sample[1].data == 20
    # 负下标复用 Tensor/NumPy 的常规索引规则。
    assert dataset[-1][2].data == 200
    # 越过样本轴必须报错，不能悄悄返回最后一条。
    with pytest.raises(IndexError):
        # 数据集的有效正索引只有 0 和 1。
        dataset[2]


def test_tensor_dataset_single_field_still_returns_tuple():
    """只有一个字段时仍保持 TensorDataset 的元组样本协议。"""
    # 调用者可以始终写成 features, = dataset[index]。
    dataset = TensorDataset(Tensor([[1, 2], [3, 4]]))
    # 单字段不能隐式变成不同的样本结构。
    assert isinstance(dataset[0], tuple) and len(dataset[0]) == 1
    # DataLoader 保持字段结构，同时增加批次轴。
    (batch,) = next(iter(DataLoader(dataset, batch_size=2)))
    # 两条二维特征合并后为 (2, 2)。
    np.testing.assert_array_equal(batch.data, [[1, 2], [3, 4]])


# 输入类型、缺少样本轴和样本轴不一致分别有明确错误。
@pytest.mark.parametrize("tensors,error", [
    # 没有任何字段无法定义样本集合。
    ((), ValueError),
    # numpy 数组需要调用者先明确转换为项目 Tensor。
    ((np.ones((2, 1)),), TypeError),
    # 标量没有第 0 维，不能作为整个数据集。
    ((Tensor(1),), ValueError),
    # 特征和标签数量不一致会破坏对应关系。
    ((Tensor([[1], [2]]), Tensor([1])), ValueError),
])
def test_tensor_dataset_rejects_invalid_inputs(tensors, error):
    """数据集构造阶段拒绝无法可靠对齐的输入。"""
    # 提前给出异常，避免训练到某一批才发现缺失标签。
    with pytest.raises(error):
        # 展开元组模拟用户传入不同数量的字段。
        TensorDataset(*tensors)


# 分别覆盖大小为 1、存在尾批、超过样本总数以及刚好整除。
@pytest.mark.parametrize("size,batch_size,batch_lengths", [
    # 最小批大小保留每条样本。
    (5, 1, [1, 1, 1, 1, 1]),
    # 最后一条样本不能因不够两条而被丢弃。
    (5, 2, [2, 2, 1]),
    # 批大小超过数据集时应只产生一个小批次。
    (5, 10, [5]),
    # 恰好整除时不能再产生一个空批次。
    (6, 2, [2, 2, 2]),
])
def test_ordered_batches_keep_all_samples_and_tail(size, batch_size, batch_lengths):
    """顺序分批保持特征、标签和样本数，并给出正确的批次数。"""
    # 默认 shuffle=False，输入顺序应直接保留。
    loader = DataLoader(make_dataset(size), batch_size=batch_size)
    # len(loader) 表示批次数，而非数据集样本数。
    assert len(loader) == len(batch_lengths)
    # 消费完整一轮，准备核对全部批次。
    batches = list(loader)
    # shape[0] 是真实批大小，尾批可以较小。
    assert [features.shape[0] for features, _ in batches] == batch_lengths
    # 拼接后应恢复原始有序编号，证明不重复也不丢失。
    np.testing.assert_array_equal(np.concatenate([x.data[:, 0] for x, _ in batches]), np.arange(size))
    # 每个字段均以 Tensor 输出，保持网络和损失的兼容性。
    assert all(type(x) is Tensor and type(y) is Tensor for x, y in batches)
    # 标签必须使用与特征相同的分组与排列。
    np.testing.assert_array_equal(np.concatenate([y.data for _, y in batches]), np.arange(size) * 10)


# 顺序和洗牌模式都应正常处理零条样本。
@pytest.mark.parametrize("shuffle", [False, True])
def test_empty_dataset_has_zero_batches(shuffle):
    """空数据集不创建空张量批次，迭代器直接结束。"""
    # 合法的空样本轴与没有输入字段是不同情况。
    loader = DataLoader(make_dataset(0), batch_size=3, shuffle=shuffle, seed=7)
    # 向上取整公式在 N=0 时也应返回零。
    assert len(loader) == 0 and list(loader) == []
    # 手动使用迭代器协议同样得到标准的结束信号。
    with pytest.raises(StopIteration):
        # 空数据集不应调用 np.stack([]) 触发其他异常。
        next(iter(loader))


def test_iterator_protocol_repeated_epochs_and_independent_cursors():
    """iter/next/StopIteration 正常工作，重复和交错遍历不共享游标。"""
    # 三条样本产生两个批次，足以辨别游标是否被重置。
    loader = DataLoader(make_dataset(3), batch_size=2)
    # 第一个迭代器先消费一批。
    first = iter(loader)
    # Python 迭代器的 iter 必须返回它自身。
    assert iter(first) is first
    # 手动 next 得到前两条样本。
    np.testing.assert_array_equal(next(first)[0].data[:, 0], [0, 1])
    # 创建另一轮时不能把 first 的游标改回零。
    second = iter(loader)
    # 第二个迭代器从自己的第一个批次开始。
    np.testing.assert_array_equal(next(second)[0].data[:, 0], [0, 1])
    # 第一个迭代器继续读取其最后一条，而不是重复第一个批次。
    np.testing.assert_array_equal(next(first)[0].data[:, 0], [2])
    # 标准 for 循环依赖 StopIteration 来结束。
    with pytest.raises(StopIteration):
        # 第一个迭代器已经消费完全部数据。
        next(first)
    # 一个迭代器结束不会影响另一个迭代器的剩余内容。
    np.testing.assert_array_equal(next(second)[0].data[:, 0], [2])
    # 原加载器可以重复遍历多轮，不会永远停留在耗尽状态。
    np.testing.assert_array_equal(epoch_order(loader), [0, 1, 2])
    # 连续再跑一轮，验证 reset 发生在新迭代器而非加载器共享状态。
    np.testing.assert_array_equal(epoch_order(loader), [0, 1, 2])


def test_shuffle_preserves_every_sample_and_feature_label_pairs():
    """洗牌只重排样本，不重复、不遗漏，也不拆散特征与标签。"""
    # 足够长的固定排列便于辨别洗牌是否实际生效。
    loader = DataLoader(make_dataset(20), batch_size=6, shuffle=True, seed=42)
    # 读取四批，最后一批应为两条。
    batches = list(loader)
    # 拼接每条特征中的编号作为这一轮的访问顺序。
    order = np.concatenate([features.data[:, 0] for features, _ in batches])
    # 固定种子的这一轮应与原始顺序不同。
    assert not np.array_equal(order, np.arange(20))
    # 排序后完全恢复原始编号，证明每条样本恰好出现一次。
    np.testing.assert_array_equal(np.sort(order), np.arange(20))
    # 标签也必须随其所属样本一起移动。
    np.testing.assert_array_equal(np.concatenate([labels.data for _, labels in batches]), order * 10)
    # 保留尾批的规则在洗牌模式中也相同。
    assert [features.shape[0] for features, _ in batches] == [6, 6, 6, 2]


def test_seed_reproduces_epoch_sequence_and_each_epoch_reshuffles():
    """同种子的两次运行可复现各轮排列，同一运行的轮次继续洗牌。"""
    # 两个加载器各有独立 RNG，互不消耗对方的随机状态。
    first = DataLoader(make_dataset(20), batch_size=4, shuffle=True, seed=2026)
    # 相同参数应重现一整段训练过程的排列序列。
    second = DataLoader(make_dataset(20), batch_size=4, shuffle=True, seed=2026)
    # 保存第一轮，后续与下一轮比较。
    first_epoch = epoch_order(first)
    # 初始随机状态相同，应产生相同首轮排列。
    np.testing.assert_array_equal(first_epoch, epoch_order(second))
    # 再次遍历时继续使用推进后的随机状态。
    next_epoch = epoch_order(first)
    # 固定样例可确定这两轮排列不同。
    assert not np.array_equal(first_epoch, next_epoch)
    # 第二个加载器推进到相同轮次时也得到相同排列。
    np.testing.assert_array_equal(next_epoch, epoch_order(second))


def test_shuffled_iterators_capture_independent_index_orders():
    """新建洗牌迭代器不改变已有迭代器尚未消费的索引。"""
    # 参考加载器按正常顺序完整消费两轮。
    reference = DataLoader(make_dataset(20), batch_size=4, shuffle=True, seed=13)
    # 第一轮和第二轮分别构成独立预期。
    expected_first, expected_second = epoch_order(reference), epoch_order(reference)
    # 相同种子的加载器同时创建两个迭代器，再交错消费。
    loader = DataLoader(make_dataset(20), batch_size=4, shuffle=True, seed=13)
    # __iter__ 必须在创建时固定本轮索引，而非依赖后续 next 的顺序。
    first, second = iter(loader), iter(loader)
    # 先消费第二轮，验证它没有重用第一轮的索引容器。
    actual_second = np.concatenate([features.data[:, 0] for features, _ in second])
    # 再消费第一轮，其顺序仍应保持不变。
    actual_first = np.concatenate([features.data[:, 0] for features, _ in first])
    # 两组断言同时检查游标独立与索引顺序独立。
    np.testing.assert_array_equal(actual_first, expected_first)
    # 若共享索引数组，先后洗牌会让其中一组结果不正确。
    np.testing.assert_array_equal(actual_second, expected_second)


def test_loader_does_not_change_numpy_global_random_state():
    """创建和遍历局部 RNG 的加载器，不改变调用者的全局随机状态。"""
    # 直接快照当前全局状态，避免测试本身通过 seed 改动其他测试环境。
    before = np.random.get_state()
    # 创建并消费多轮，覆盖初始化与实际洗牌两个阶段。
    loader = DataLoader(make_dataset(20), shuffle=True, seed=31)
    # 完整遍历才能确保洗牌路径真实执行。
    list(loader)
    # 重复洗牌也不应调用全局 np.random 接口。
    list(loader)
    # 取得遍历后的状态，用于逐部分核对。
    after = np.random.get_state()
    # 算法标识及游标等标量元数据必须保持一致。
    assert before[0] == after[0] and before[2:] == after[2:]
    # 随机状态数组需要专门的数组断言，不能直接使用元组相等。
    np.testing.assert_array_equal(before[1], after[1])


def test_loader_reads_samples_only_when_a_batch_is_requested():
    """构造和 iter 不读取样本，每次 next 只访问当前批次。"""
    # 日志数据集记录实际发生的样本读取，模拟需要磁盘访问的数据源。
    class LoggingDataset(Dataset):
        # 每个数据集实例拥有自己的访问记录。
        def __init__(self):
            # 从空日志开始，不在构造时预加载任何样本。
            self.reads = []

        # 数据集长度可以在不访问样本内容时取得。
        def __len__(self):
            # 五条样本确保第二批之后还存在尾批。
            return 5

        # 每次实际读取都记录索引，便于检查是否提前消耗未来批次。
        def __getitem__(self, index):
            # 记录读取副作用，测试可直接观察惰性访问。
            self.reads.append(index)
            # 自定义数据集可以返回单个 Tensor。
            return Tensor([index])

    # 构造对象与加载器，此时应没有任何实际读取。
    dataset = LoggingDataset()
    # 一批两条数据，每次 next 恰好增加两个记录，尾批除外。
    loader = DataLoader(dataset, batch_size=2)
    # 创建迭代器仅固定顺序，不组装首批。
    iterator = iter(loader)
    # len(loader) 也只读取元信息。
    assert len(loader) == 3 and dataset.reads == []
    # 请求首批才发生第一次样本访问。
    np.testing.assert_array_equal(next(iterator).data, [[0], [1]])
    # 如果预加载全数据，这里会出现 0 到 4 的全部编号。
    assert dataset.reads == [0, 1]
    # 请求第二批后仅新增其两个编号。
    np.testing.assert_array_equal(next(iterator).data, [[2], [3]])
    # 最后一条尚未读取，证明仍是按批次推进。
    assert dataset.reads == [0, 1, 2, 3]


def test_custom_dataset_collates_array_and_number_fields():
    """普通列表数据源中的 NumPy 特征与数字标签可直接合成 Tensor 批次。"""
    # 固定长度列表被视为样本的多个字段，与元组具有相同语义。
    dataset = [[np.array([1, 2]), 3], [np.array([4, 5]), 6]]
    # 协议兼容的数据源无需继承 Dataset。
    features, labels = next(iter(DataLoader(dataset, batch_size=2)))
    # 新增批次轴后为 (2, 2)，原始数组内容按行保留。
    np.testing.assert_array_equal(features.data, [[1, 2], [4, 5]])
    # 数字标签从标量堆叠为一维批次。
    np.testing.assert_array_equal(labels.data, [3, 6])
    # 输出全部使用同一 Tensor，方便接入 MSE/交叉熵等后续模块。
    assert type(features) is Tensor and type(labels) is Tensor


# 同时覆盖 Python bool、浮点数和非数字类型的误用。
@pytest.mark.parametrize("batch_size", [True, False, 2.0, "2", None])
def test_batch_size_requires_integer_type(batch_size):
    """只有正整数可以表示本加载器的批大小。"""
    # TypeError 提示调用者先修正参数类型。
    with pytest.raises(TypeError, match="batch_size"):
        # 无效参数必须在初始化时被拒绝。
        DataLoader(make_dataset(), batch_size=batch_size)


# 数值类型正确但不大于零时，错误类别改为 ValueError。
@pytest.mark.parametrize("batch_size", [0, -1])
def test_batch_size_requires_positive_value(batch_size):
    """零和负数不能形成有效批次。"""
    # 在遍历前报错，避免 range 产生空结果或非法步长。
    with pytest.raises(ValueError, match="batch_size"):
        # 这两个参数类型都是整数，问题出在数值范围。
        DataLoader(make_dataset(), batch_size=batch_size)


def test_numpy_integer_batch_size_is_accepted():
    """NumPy 的整数配置可以直接用作批大小。"""
    # Integral 接口应兼容 np.int64，避免过度限制合法参数。
    loader = DataLoader(make_dataset(), batch_size=np.int64(2))
    # 五条样本形成三批，语义与普通 Python int 相同。
    assert len(loader) == 3


# 防止 truthy 值误开启洗牌，尤其是字符串 'False'。
@pytest.mark.parametrize("shuffle", [0, 1, "False", None])
def test_shuffle_requires_boolean(shuffle):
    """洗牌开关必须由调用者显式给出 bool。"""
    # 这里不隐式调用 bool(shuffle)，以便尽早暴露配置错误。
    with pytest.raises(TypeError, match="shuffle"):
        # 非布尔参数应在创建加载器时失败。
        DataLoader(make_dataset(), shuffle=shuffle)


def test_dataset_must_support_length_and_indexing():
    """只有迭代能力的对象不满足按索引洗牌的最小接口。"""
    # 普通生成器无法获知总长度或随机读取某条样本。
    with pytest.raises(TypeError, match="dataset"):
        # 无需开始遍历就应给出接口错误。
        DataLoader(iter([1, 2, 3]))


# 不同字段数量与空字段都应失败，不能被 zip 静默截断。
@pytest.mark.parametrize("dataset", [[(1, 2), (3,)], [(), ()]])
def test_collate_rejects_inconsistent_or_empty_fields(dataset):
    """每个样本需要相同且非零的字段数量。"""
    # 结构检查按批进行，保持加载器的惰性读取约定。
    with pytest.raises(ValueError, match="fields"):
        # 在首批合并时发现不完整的数据。
        next(iter(DataLoader(dataset, batch_size=2)))


def test_collate_rejects_ragged_shapes_and_unsupported_fields():
    """不等长特征和字典样本均不属于本练习的自动组批范围。"""
    # stack 要求同一字段内各样本形状一致。
    with pytest.raises(ValueError):
        # 变长序列需要调用者先做补齐或其他处理。
        next(iter(DataLoader([Tensor([1]), Tensor([2, 3])], batch_size=2)))
    # 字典需要自定义组批策略，本接口明确拒绝而非猜测字段顺序。
    with pytest.raises(TypeError, match="fields"):
        # 错误消息说明自动组批当前支持的类型。
        next(iter(DataLoader([{"feature": 1}], batch_size=1)))
