# 阅读路线：先看构造 Tensor 怎样保存数字和形状，再看四则运算、矩阵乘法与变形，最后看求和/平均/最大值。
# Python 零基础速查：= 是赋值（把右边的结果交给左边变量）；== 才比较相等；冒号后缩进的行属于上面的函数或条件。
# class 定义一种对象；Tensor([1, 2]) 创建该类的实例；方法里的 self 就是正在操作的这个实例。
# 函数名后面的 (...) 用来传参数并执行；点号如 self.data 表示读取对象的属性；return 把结果交回调用处。
# [1, 2] 是列表，(2, 3) 是元组；shape=(2, 3) 是两行三列，shape=(3,) 是一维、含三个数。
# 神经网络把一批输入保存为多维数字数组（张量）；这里只提供前向数字运算，没有梯度、损失函数与自动学习。
# 先认三类：Tensor(5) 是 0 维标量、shape=()；Tensor([5, 6]) 是 1 维向量、shape=(2,)。
# Tensor([[1, 2, 3], [4, 5, 6]]) 是 2 维矩阵、shape=(2, 3)；数字总数 size=2*3=6。
# 批量输入常写成 (批量条数, 每条特征数)，如 (32,784) 是 32 条，每条包含 784 个数字。
# 下标从 0 开始：x[0] 取第一项；负下标从末尾开始：shape[-1] 取最后一个维度。
# Python 的函数定义 def ...: 后只注册计算规则；执行函数须写名字和括号，如 x.reshape(2,3)。
# if/else 让程序按条件选择一条分支；for 每次取出一个值反复执行缩进代码块。
# 程序出错时先看异常类型：TypeError 通常指对象类型不对，ValueError 通常指值或形状不合理。
# 本类把 NumPy 数组装进对象并提供运算符接口，初学时可把张量看成“带形状的一组数字”。
"""阶段草稿。实现写在这里，稳定后再迁入 src/jmtorch/。"""

# import 引入 Python 标准库的计时工具；下面的性能练习会用到。
import time

# as np 给第三方 NumPy 库起简短别名；真正计算数字的数组是它的 ndarray。
import numpy as np
# 固定随机种子为 7，使按相同顺序生成的随机数可复现；不是每次调用都拿到同一个数。
rng = np.random.default_rng(7)

# 大写名字约定表示常量；float32 每个数字占 4 字节，供后续估算内存用。
BYTES_PER_FLOAT32 = 4
# 一 MiB 有 1024 * 1024 字节；这里变量名虽然写 MB，除以它得到的严格单位是 MiB。
MB_TO_BYTES = 1024 * 1024

# 本项目自己写的 Tensor 是对 NumPy 数组的包装，不等于 PyTorch 的 torch.Tensor。
class Tensor:
    # __init__ 是创建 Tensor(...) 时自动执行的初始化方法；self 指当前这个新对象。
    # 同一类的两个实例各有自己的 self.data，例如 x=Tensor([1])、y=Tensor([2]) 通常存着不同的数组。
    def __init__(self, data):
        # isinstance(x, 类型) 检查对象类别；and 要三个条件全为真。先查长度避免访问空列表的 data[0]。
        # 如果传入的是非空 Tensor 列表/元组，将各自数组叠起来，例如两个 (3,) 张量得到 (2, 3)。
        if isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], Tensor):
            # [t.data for t in data] 是列表推导式，逐个取出底层数组；np.stack 沿新增的第一维堆叠。
            data = np.stack([t.data for t in data])
        # np.array 将标量、列表或现有数组转换为 NumPy 数组，并统一为 float32；self.data 保存在实例上。
        # float32 表示常用的 32 位浮点数：1、1.5 都转为浮点表示；小数计算可能存在舍入误差。
        self.data = np.array(data, dtype=np.float32)
        # shape 是各轴长度组成的元组，例如 (32, 784) 表示 32 条样本、每条 784 个特征。
        # shape=(2,3) 的第一轴可看作 2 行，第二轴可看作每行 3 列；shape=() 是一个标量而非空列表。
        self.shape = self.data.shape
        # size 是所有位置的数字总数，如 32 * 784；不是占用多少字节。
        self.size = self.data.size
        # dtype 是数据类型，此处初始化后通常是 float32；这些属性只在构造时缓存。
        # 若后来直接替换/改变 self.data，shape、size、dtype 缓存不会自动重新同步。
        self.dtype = self.data.dtype

    # __repr__ 定义调试环境中 repr(tensor) 的详细文字表示，通常包含数据和形状。
    def __repr__(self):
        # f"...{表达式}..." 把值插入字符串；return 返回文字，并不执行神经网络计算。
        return f"Tensor(data={self.data}, shape={self.shape})"

    # __str__ 定义 print(tensor) 或 str(tensor) 显示的较简短文字。
    def __str__(self):
        # 实例的 .data 是 NumPy 数组；这里只把其内容格式化为文字。
        return f"Tensor({self.data})"

    # 用方法取回 NumPy 数组；括号意味着必须调用，如 x.numpy()。
    def numpy(self):
        # 返回的是内部同一个数组对象，不是副本；修改返回数组也会改变 Tensor 的实际数据。
        # 例：a=x.numpy(); a[0]=99 会使 x 的第一个数变为 99；只想观察时不要改 a。
        return self.data

    # 此方法测量数字数组实际存储的字节数，不含 Python 对象和其他开销。
    def memory_footprint(self):
        # 对 float32 数组，一般是元素数 * 4 字节；nbytes 是 NumPy 数组自己的属性。
        return self.data.nbytes

    # @property 把下面的方法变为只读式属性：写 x.ndim，而不是 x.ndim()。
    @property
    # ndim 是维度/轴的数量；(32, 784) 有两维，而 (784,) 有一维。
    def ndim(self):
        # len(shape) 是元组中的轴数；0 维标量的 shape 是空元组 ()，所以 ndim 为 0。
        return len(self.shape)

    # numel 是“number of elements”，读取张量总元素数的方法。
    def numel(self):
        # 返回构造时缓存的 size；与用于表达内存的 nbytes 概念不同。
        return self.size

    # 连续数组在内存中按规定顺序排布；有些底层计算要求这种布局。
    def contiguous(self):
        # C 顺序的连续布局通常意味着先顺次保存一行里的数字，再接着保存下一行。
        # NumPy 保证得到连续布局，再由 Tensor(...) 包装为新的 Tensor 对象。
        return Tensor(np.ascontiguousarray(self.data))

    # *shape 把不定数量的位置参数收成元组：view(2, 3) 收到 (2, 3)。
    def view(self, *shape):
        # 这里实际转调 reshape，而 Tensor(...) 会复制数组；因此不像某些框架的 view 保证共享底层数据。
        return self.reshape(*shape)

    # 掩码指定要替换的位置；True 替换成 value，False 保留原数字。
    def masked_fill(self, mask, value):
        # 三元表达式“真值 if 条件 else 假值”；Tensor 掩码读 .data，其他输入交给 np.asarray。
        # astype(bool) / dtype=bool 把 0 变 False、非零变 True；掩码形状须能用于 NumPy 索引。
        mask_array = mask.data.astype(bool) if isinstance(mask, Tensor) else np.asarray(mask, dtype=bool)
        # copy() 先复制输入数组，避免对当前 Tensor 的原数据进行原地修改。
        result = self.data.copy()
        # 布尔索引 result[mask_array] 只选取掩码为 True 的位置，并用 value 覆盖。
        result[mask_array] = value
        # 返回包装好的新 Tensor；例如 [1, 2, 3] 掩码 [False, True, False] 填 0 得 [1, 0, 3]。
        return Tensor(result)

    # __add__ 是运算符 + 对应的特殊方法：a + b 会尝试 a.__add__(b)。
    # 加、减、乘、除返回新的 Tensor；即使表达式写 x + 1，也不会自动把 x 自己改掉。
    def __add__(self, other):
        # 若右边也是 Tensor，必须取它内部的 NumPy 数组才能逐位置相加。
        if isinstance(other, Tensor):
            # 数组相加遵循广播规则：例如 (2, 3) + (3,) 会把一行三个数用于每一行。
            # 具体例：[[1,2,3],[4,5,6]] + [10,20,30] -> [[11,22,33],[14,25,36]]。
            return Tensor(self.data + other.data)
        # else 对应上面的 if：右边是数字或 NumPy 可处理的其他值时走这里。
        else:
            # Tensor(...) 把运算结果重新封装，当前 self 不被这次加法改变。
            return Tensor(self.data + other)

    # __radd__ 处理左边不是 Tensor 的反向加法，如 3 + tensor。
    def __radd__(self, other):
        # 加法可交换两侧位置，因此复用前面的实现。
        return self.__add__(other)

    # __sub__ 对应 tensor - other；减法顺序有意义，不能随便交换。
    # 广播只在维度可配对时成立；例如 (2,3) 与 (4,) 不兼容，NumPy 会抛异常。
    def __sub__(self, other):
        # 条件判断右边是否有 .data 属性所代表的 Tensor 包装。
        if isinstance(other, Tensor):
            # NumPy 逐元素相减；不匹配的形状会按广播规则尝试，无法广播就报错。
            return Tensor(self.data - other.data)
        # 右侧是标量等非 Tensor 对象时使用另一支。
        else:
            # 每个元素减掉 other，例如 [4, 5] - 2 得 [2, 3]。
            return Tensor(self.data - other)

    # __rsub__ 对应 other - tensor；与 tensor - other 不同，必须保持原来的运算顺序。
    def __rsub__(self, other):
        # 这里也检查 Tensor，虽然普通 Tensor - Tensor 通常会先调用左侧的 __sub__。
        if isinstance(other, Tensor):
            # other.data 放左侧，使结果正确体现“右边这个 Tensor 被左边减去”。
            return Tensor(other.data - self.data)
        # 例如 10 - Tensor([2, 3]) 得 Tensor([8, 7])，不能沿用 __sub__ 的相反结果。
        return Tensor(other - self.data)

    # __mul__ 对应 *，这里实现逐元素乘法，不是神经网络权重常用的矩阵乘法。
    # 例：Tensor([2,3]) * Tensor([4,5]) -> [8,15]；而矩阵乘 @ 会做乘积后求和。
    def __mul__(self, other):
        # 当两边都为 Tensor，取两边底层数组相乘。
        if isinstance(other, Tensor):
            # NumPy 自动广播兼容形状；同形状则对应位置相乘。
            return Tensor(self.data * other.data)
        # 非 Tensor 走这个分支，常见形式是张量乘一个数字。
        else:
            # 例如 Tensor([2, 3]) * 4 得 [8, 12]。
            return Tensor(self.data * other)

    # __rmul__ 让 4 * tensor 与 tensor * 4 均可工作。
    def __rmul__(self, other):
        # 普通数字乘法可以交换顺序，因此可复用逐元素乘法实现。
        return self.__mul__(other)

    # __truediv__ 对应 /；Python 中 / 是真除法，5 / 2 得 2.5。
    # NumPy 中浮点除零可能得到 inf（无穷）或 nan（非数），它们仍可能被包装为 Tensor。
    def __truediv__(self, other):
        # 若分母是另一 Tensor，按其每个位置的数字进行逐元素相除。
        if isinstance(other, Tensor):
            # 例如 [6, 8] / [2, 4] 得 [3, 2]；除以零可能产生 inf 或 nan 警告。
            return Tensor(self.data / other.data)
        # 右侧是普通数字等对象时使用另一种 NumPy 除法。
        else:
            # 单个数字会广播到张量所有位置；结果封装成新 Tensor。
            return Tensor(self.data / other)

    # __rtruediv__ 对应 other / tensor；分子和分母不可交换。
    def __rtruediv__(self, other):
        # 两边都为 Tensor 时保留 other 在分子的位置。
        if isinstance(other, Tensor):
            # 逐位置计算 other.data / self.data。
            return Tensor(other.data / self.data)
        # 例如 12 / Tensor([3, 4]) 得 [4, 3]；self.data 是分母。
        return Tensor(other / self.data)

    # 矩阵乘要求两边的内侧维度相等：形状 (M, K) @ (K, N) 才得到 (M, N)。
    # 以下辅助方法优先给初学者明确报错；后续的 NumPy 仍负责完整的广播与一维维度检查。
    # 手算：A=[[1,2],[3,4]]、B=[[10],[20]]，则 A@B=[[1*10+2*20],[3*10+4*20]]=[[50],[110]]。
    # A 的形状 (2,2)，B 的形状 (2,1)：内侧两个 2 对得上，外侧保留 (2,1)。
    # 神经网络线性层常令 x=(批量数,输入特征数)、W=(输入特征数,输出特征数)，于是 x@W 给整批输出。
    def _validate_matmul_shapes(self, other):
        # not 反转布尔结果；类型不对时 raise 主动抛异常，后面的运算不会继续。
        if not isinstance(other, Tensor):
            # TypeError 表示数据类型用错；如 tensor @ [1, 2] 要先包装为 Tensor。
            raise TypeError(
                # f 字符串中的 {type(other).__name__} 动态写入实参的类型名称；\n 是换行符。
                f"Matrix multiplication requires Tensor, got {type(other).__name__}\n"
                # 括号中的相邻字符串自动合并，分几行写只是为了读起来清楚。
                f"  ❌ Cannot perform: Tensor @ {type(other).__name__}\n"
                f"  💡 Matrix multiplication (@) only works between two Tensors\n"
                f"  🔧 Wrap your data: Tensor({other}) @ other_tensor"
            # 此右括号结束上面的 TypeError(...) 构造调用。
            )
        # len(shape) 为 0 表示标量 Tensor（shape=()）；or 只需有一边为标量就拒绝 @。
        if len(self.shape) == 0 or len(other.shape) == 0:
            # 0 维数值应按普通乘法 * 来算；@ 定义的是向量/矩阵乘法。
            raise ValueError(
                # ValueError 是数值/形状不满足规则的错误，与上面的 TypeError 不同。
                f"Matrix multiplication requires at least 1D tensors\n"
                f"  ❌ Got shapes: {self.shape} @ {other.shape}\n"
                f"  💡 Scalars (0D tensors) cannot be matrix-multiplied; use * for element-wise\n"
                f"  🔧 Reshape scalar to 1D: tensor.reshape(1) or use tensor * scalar"
            )
        # 仅当两边都至少二维时，先明确检查最后一轴与倒数第二轴。
        if len(self.shape) >= 2 and len(other.shape) >= 2:
            # 负数索引从末尾算：[-1] 是末轴，[-2] 是倒数第二轴；!= 意思是不相等。
            if self.shape[-1] != other.shape[-2]:
                # 例如 (32, 784) @ (10, 5) 的 784 != 10，输入特征数对不上。
                raise ValueError(
                    # 多个 f 字符串描述错误、原因、规则和可能修改方向；只用于提示，不做计算。
                    f"Matrix multiplication shape mismatch: {self.shape} @ {other.shape}\n"
                    f"  ❌ Inner dimensions don't match: {self.shape[-1]} vs {other.shape[-2]}\n"
                    f"  💡 For A @ B, A's last dim must equal B's second-to-last dim\n"
                    # [::-1] 会把元组的全部轴顺序反过来，只是报错中的建议；高维 transpose() 实际仅交换最后两轴。
                    f"  🔧 Try: other.transpose() to get shape {other.shape[::-1]}, or reshape self"
                )

    # 真正计算矩阵乘；神经网络线性层的 x @ W 使用这里，而 x * W 是逐位置乘。
    # 一个输出数字会依赖输入行的多个数字，权重矩阵的某一列决定这个输出怎样混合输入特征。
    # 这里只有乘与加；权重如何随训练数据变化，需要另外实现梯度与优化器。
    def matmul(self, other):
        # 调用 self 上的方法先检查输入；若抛异常，本函数其余行不再运行。
        self._validate_matmul_shapes(other)

        # a 和 b 是两边 Tensor 里的 NumPy 数组，并没有复制其数值。
        a = self.data
        b = other.data

        # 两个数组都是二维时走下面显式循环，帮助看清矩阵乘的行乘列原理。
        if len(a.shape) == 2 and len(b.shape) == 2:
            # 元组解包：a.shape=(M, K) 时分别把行数、列数交给 M、K。
            M, K = a.shape
            # b.shape=(K, N)；_ 表示这里暂不需要第一个维度值，维度兼容性前面已检查。
            _, N = b.shape
            # 先开一张 (M, N) 的全零结果表；每个输出位置对应一行与一列的点积。
            result_data = np.zeros((M, N), dtype=a.dtype)

            # range(M) 生成 0 至 M-1；外层 for 每次处理一条输入行。
            for i in range(M):
                # 内层 for 每次处理一个输出列；两个嵌套循环总共填 M*N 个位置。
                for j in range(N):
                    # a[i, :] 取第 i 行全部 K 个数；b[:, j] 取第 j 列全部 K 个数。
                    # np.dot 做对应元素乘积之和：C[i,j]=sum(a[i,k]*b[k,j])，结果写回该位置。
                    # 例如 a[i,:]=[1,2]、b[:,j]=[10,20]，点积为 1*10+2*20=50。
                    # 这个循环按输出位置在 Python 层迭代，通常比直接使用一次 NumPy 矩阵乘法慢。
                    result_data[i, j] = np.dot(a[i, :], b[:, j])
        # 一维向量或高维批次不走手写二维循环，交给 NumPy 的现成矩阵乘法。
        else:
            # NumPy 会执行更完整的维度匹配、广播和一维向量规则；非法形状仍可能在这里报错。
            result_data = np.matmul(a, b)

        # 用 Tensor 再包装结果；假设 a=(32,784)、b=(784,10)，输出为 (32,10)。
        return Tensor(result_data)

    # Python 的 @ 运算符会调用左侧对象的 __matmul__；可以写 x @ W。
    def __matmul__(self, other):
        # 与 x.matmul(W) 完全相同，只是写法更贴近数学公式。
        return self.matmul(other)

    # 方括号索引 x[key] 调用 __getitem__；可选单个元素、切片、某一行等。
    # 实例：二维 x=[[1,2],[3,4]]，x[0] 取第一行 [1,2]，x[0,1] 取数字 2。
    # 冒号在索引里代表切片：x[:,0] 选所有行的第一列，x[0:2] 选从第 0 行到第 2 行之前。
    def __getitem__(self, key):
        # key 如 0、1:3、(0, 2)；NumPy 负责解释这些索引规则。
        result_data = self.data[key]
        # 单个位置可能返回 NumPy 标量而非 ndarray，先判断对象类型。
        if not isinstance(result_data, np.ndarray):
            # 将标量变为 0 维 NumPy 数组，便于统一交给 Tensor 构造函数。
            result_data = np.array(result_data)
        # Tensor 构造调用 np.array，得到新数组；即使原 NumPy 切片可共享内存，这里的输出也不共享。
        return Tensor(result_data)

    # reshape 改变形状但保持总元素数；*shape 允许 x.reshape(2,3) 或 x.reshape((2,3))。
    # 例：[[1,2,3],[4,5,6]] 由 (2,3) 变 (3,2) 后通常为 [[1,2],[3,4],[5,6]]。
    # reshape 只改变数字被分组的方式；transpose 则按轴互换数字的排列，结果通常不一样。
    # 例如同一个 (2,3) 数组转置成 (3,2) 是 [[1,4],[2,5],[3,6]]。
    def reshape(self, *shape):
        # 只传了一个列表/元组时，把该序列当作完整目标形状。
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            # tuple(...) 转为元组，如 [2,3] 变 (2,3)；shape[0] 是第一个参数。
            new_shape = tuple(shape[0])
        # 否则多个位置参数已经被 *shape 收成元组。
        else:
            # 例如 reshape(2,3) 的 shape 就是 (2,3)。
            new_shape = shape
        # -1 表示请推断这一维的长度，例如 6 个数字 reshape(2,-1) 推断成 (2,3)。
        if -1 in new_shape:
            # count(-1) 统计出现次数；有多个未知数无法确定各自应取多大。
            if new_shape.count(-1) > 1:
                # 明确抛 ValueError，禁止 reshape(-1, -1) 这样的歧义。
                raise ValueError(
                    # 下列字符串组成包含形状、出错原因与修改提示的多行错误信息。
                    f"Cannot reshape {self.shape} with multiple unknown dimensions\n"
                    f"  ❌ Found {new_shape.count(-1)} dimensions set to -1 in {new_shape}\n"
                    f"  💡 Only one dimension can be inferred; others must be specified\n"
                    f"  🔧 Replace all but one -1 with explicit sizes (total elements: {self.size})"
                )
            # 乘法的起始值应为 1；后面把所有已知维度乘起来。
            known_size = 1
            # index(-1) 给出 -1 在目标形状里的位置，如 (2,-1) 中的位置 1。
            unknown_idx = new_shape.index(-1)
            # enumerate 同时提供位置 i 和维度值 dim；例如依次拿到 (0,2)、(1,-1)。
            for i, dim in enumerate(new_shape):
                # 仅计算已知的维度，排除需要推断的那一项。
                if i != unknown_idx:
                    # *= 是 known_size = known_size * dim 的简写。
                    known_size *= dim
            # % 是取余；6 % 2 == 0，说明 6 个元素能分成每组 2 个。
            # 边界提醒：例如 reshape(0,-1) 可能使 known_size 为 0，这里会先触发除零错误；当前实现未特别处理。
            # 与之相对，元素数为 0 的空数组做 reshape(0,3) 时，没有 -1 就能通过元素数检查。
            if self.size % known_size != 0:
                # 若不能整除，目标维度不会是整数，形状不可能成立。
                raise ValueError(
                    f"Cannot infer -1 dimension: {self.size} elements is not "
                    # 相邻字符串自动合并；报错文字中的 % 是说明计算结果。
                    f"divisible by the known dimensions product {known_size}\n"
                    f"  ❌ {self.size} % {known_size} = {self.size % known_size}\n"
                    f"  💡 The -1 dimension must be a whole number"
                )
            # // 是向下整除；已确认可整除，故这里正好是未知维度的整数值。
            unknown_dim = self.size // known_size
            # 元组不能修改元素，因此先转成可修改的 list。
            new_shape = list(new_shape)
            # 用求出的尺寸替换原本 -1 所在的位置。
            new_shape[unknown_idx] = unknown_dim
            # 再转回形状使用的元组格式。
            new_shape = tuple(new_shape)
        # 即便没有 -1，总元素数也必须完全一致；np.prod 计算各目标维度之积。
        if np.prod(new_shape) != self.size:
            # int(...) 把 NumPy 返回的整数转换为普通 Python 整数，用在报错文字中。
            target_size = int(np.prod(new_shape))
            # 例如 (2,3) 的 6 个元素不能无端 reshape 成 (2,4) 的 8 个元素。
            raise ValueError(
                f"Cannot reshape {self.shape} to {new_shape}\n"
                f"  ❌ Element count mismatch: {self.size} elements vs {target_size} elements\n"
                f"  💡 Reshape preserves data, so total elements must stay the same\n"
                # 条件表达式选最后一维作为报错提示；若形状为空元组则改用 1。
                f"  🔧 Use -1 to infer a dimension: reshape(-1, {new_shape[-1] if len(new_shape) > 0 else 1}) lets NumPy calculate"
            )
        # NumPy 按目标形状重排数组；例如 [1,2,3,4,5,6] -> 两行三列。
        reshaped_data = np.reshape(self.data, new_shape)
        # Tensor(...) 构造时会复制为 float32 数组，所以本方法的返回值不与原 Tensor 共享数据。
        return Tensor(reshaped_data)

    # transpose 交换两个轴；轴是 shape 中各位置，二维矩阵默认把行、列互换。
    # None 是 Python 的“没有值”：两个参数都不传时用默认的最后两轴，也可同时提供轴号。
    # 比如图片批次的形状为 (批量,高,宽)，默认 transpose() 得 (批量,宽,高)，批量轴不动。
    # transpose(0,2) 则交换批量轴与宽轴；即使轴长碰巧相等，数字索引的含义也改变了。
    def transpose(self, dim0=None, dim1=None):
        # is None 判断对象是否就是 None；and 要求两个参数均未给出。
        if dim0 is None and dim1 is None:
            # 一维向量或标量不足两轴可以互换，所以返回原内容的新 Tensor。
            if len(self.shape) < 2:
                # copy() 拷贝底层 NumPy 数组；Tensor(...) 构造时也会包装新的对象。
                return Tensor(self.data.copy())
            # 二维及更高维开始计算轴的顺序。
            else:
                # range(n) 产生 0 到 n-1 的轴编号，再用 list 变为可修改列表。
                axes = list(range(len(self.shape)))
                # 同时赋值交换最后两项：二维 (行,列) 变 (列,行)；高维只交换最后两轴。
                axes[-2], axes[-1] = axes[-1], axes[-2]
                # 按新的轴次序转置数组，例如 (2,3,4) 默认变为 (2,4,3)。
                transposed_data = np.transpose(self.data, axes)
        # 若用户传入轴号，就走指定轴交换分支。
        else:
            # 只指定一个轴不能知道要与谁交换；or 表示有一个没给就报错。
            if dim0 is None or dim1 is None:
                # 三元表达式生成提示文字，显示用户给的是 dim0 还是 dim1。
                provided = f"dim0={dim0}" if dim1 is None else f"dim1={dim1}"
                # 再指出缺少的另一项参数的名字。
                missing = "dim1" if dim1 is None else "dim0"
                # 抛出 ValueError 后结束执行；括号内相邻的 f 字符串组成一条多行提示。
                raise ValueError(
                    f"Transpose requires both dimensions to be specified\n"
                    f"  ❌ Got {provided}, but {missing} is None\n"
                    f"  💡 Either provide both dims or neither (default swaps last two)\n"
                    # f 字符串里的 if/else 只是生成建议文字，并不会自动修好输入。
                    f"  🔧 Use transpose({dim0 if dim0 is not None else 0}, {dim1 if dim1 is not None else 1}) or just transpose()"
                )
            # 建立原有各轴的编号表，例如三维是 [0,1,2]。
            axes = list(range(len(self.shape)))
            # 对列表指定的两个位置做同时赋值交换；越界的轴号会由 Python 报 IndexError。
            axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
            # 真正交给 NumPy 按新轴顺序排列；错误/重复轴号仍可能在 NumPy 中报错。
            transposed_data = np.transpose(self.data, axes)
        # 最终重新包装为 Tensor；构造过程复制数组，结果的值改变不影响原 Tensor。
        return Tensor(transposed_data)

    # sum 是“归约”：把若干位置加起来；在神经网络中可用于聚合特征或损失。
    # axis=None 表示沿所有轴合成标量；axis=0/1 指沿相应方向合并；keepdims=True 保留被压缩的轴为长度 1。
    # 对 x=[[1,2],[3,4]]，sum() 是 10，结果形状 ()；sum(axis=0) 是 [4,6]，形状 (2,)。
    # sum(axis=0,keepdims=True) 则为 [[4,6]]、形状 (1,2)，便于后面与 (批量,2) 数组广播。
    def sum(self, axis=None, keepdims=False):
        # 如 [[1,2],[3,4]] 沿 axis=0 求和得到 [4,6]，沿 axis=1 得到 [3,7]。
        result = np.sum(self.data, axis=axis, keepdims=keepdims)
        # NumPy 可能返回一个数或一个数组，统一用 Tensor 包装。
        return Tensor(result)

    # mean 求平均数，接口与 sum 相同；常见于批次的平均指标、平均损失。
    # 上例的 mean(axis=0) 是 [2,3]：第一列 (1+3)/2，第二列 (2+4)/2。
    # 对一批样本沿 axis=0 求平均，就是对同一特征在多条样本上的数字取平均。
    def mean(self, axis=None, keepdims=False):
        # 如 [2,4,6] 的平均是 4；对空数据求平均可能发出警告并得到 nan。
        result = np.mean(self.data, axis=axis, keepdims=keepdims)
        # 输出仍为本项目的 Tensor；没有实现反向传播，不能用它自动训练权重。
        return Tensor(result)

    # max 取最大数，可对全部数字或沿指定轴取最大值；不是“取最大值所在位置”的 argmax。
    # 上例 max() 是 4；max(axis=0) 是 [3,4]，分别是两列的最大数字。
    # 分类网络的最大原始分数可以对应预测类别，但这个 max() 不返回其类别位置。
    def max(self, axis=None, keepdims=False):
        # axis 和 keepdims 的含义与 sum 一样；对空轴求最大值不存在答案，NumPy 会报 ValueError。
        result = np.max(self.data, axis=axis, keepdims=keepdims)
        # 重新封装为 Tensor，保证调用者拿到的仍是该类对象。
        return Tensor(result)


# 阅读下面的测试时，可以把一个 Tensor 想成“数字 + 形状”的组合。
# 神经网络接收数字并按层运算；数字组成向量或矩阵后，形状决定哪些运算合法。
# Python 中的变量（如 x）是对象的名字，赋值 x = Tensor(...) 是让名字指向对象。
# 列表用 [1, 2] 表示，元组用 (1, 2) 表示；形状通常用元组表达。
# 注意 Tensor 的数据虽来自 Python 列表，内部保存的是 NumPy 数组。
# 下面的函数名以 test_ 开头，是用小例子检查实现，不会训练任何参数。
# 这个辅助函数只负责展示张量，不会改变张量的数据或形状。
# def 用于定义函数；单独的 * 表示 data 必须以 data=False 这样的关键字形式传入。
def _show(label, tensor, *, data=True):
    # 方括号创建列表；f"...{表达式}..." 会把表达式的值插入字符串。
    # shape 是各维度的长度，size 是数字总数，ndim 是维度数，dtype 是每个数字的类型。
    parts = [f"shape={tensor.shape}", f"size={tensor.size}", f"ndim={tensor.ndim}", f"dtype={tensor.dtype}"]
    # if 的条件为真时才执行缩进的部分；默认 data=True，会展示具体数字。
    if data:
        # insert(0, 内容) 把文字插入列表开头；tensor.data 是底层 NumPy 数组。
        parts.insert(0, f"data={tensor.data}")
    # join 把列表里的文字用逗号和空格连接；print 将结果输出到终端。
    print(f"   {label}：{', '.join(parts)}")


# 测试函数以 test_ 开头，只是命名约定；执行到调用它时，函数内的代码才运行。
def test_unit_tensor_creation():
    # 字符串是引号中的文字；这一行只是显示标题。
    print("🧪 单元测试：张量创建...")

    # = 是赋值；Tensor(5.0) 把一个数字包装为零维张量，也叫标量。
    # 这里的 5.0 属于带小数点的 float 字面量，与字符串 "5.0" 不同。
    scalar = Tensor(5.0)
    # assert 会在条件不成立时抛 AssertionError；== 是比较两个值，不是赋值。
    assert scalar.data == 5.0
    # 零维标量的 shape 是空元组 ()，不是 (1,)；它依然有一个数字。
    assert scalar.shape == ()
    # size 表示元素数量；标量也含 1 个元素。
    assert scalar.size == 1
    # float32 是用 32 位存储的浮点数；Tensor 构造函数统一转为此类型。
    assert scalar.dtype == np.float32
    # 调用前面定义的 _show；第一个参数是显示标题，第二个是张量对象。
    _show("标量 Tensor(5.0)", scalar)

    # 一层方括号是 Python 列表；包装后成为含 3 个数字的一维向量。
    # 例如未来一条样本若有 3 项测量值，也可以用这样的向量表示。
    vector = Tensor([1, 2, 3])
    # np.array 建立预期数组；np.array_equal 比较形状和每个数字是否完全相同。
    assert np.array_equal(vector.data, np.array([1, 2, 3], dtype=np.float32))
    # (3,) 中的逗号表示单元素元组：向量有一条长度为 3 的轴。
    assert vector.shape == (3,)
    # 一维向量的元素总数为 3。
    assert vector.size == 3
    # 显示向量的数字、形状、元素数量、维度数量和类型。
    _show("向量 Tensor([1, 2, 3])", vector)

    # 两层列表是一张有 2 行、每行 2 列的二维数字表，即矩阵。
    # matrix.data[0, 1] 读出第 0 行第 1 列的数字 2；下标从 0 开始。
    matrix = Tensor([[1, 2], [3, 4]])
    # 用二维 NumPy 数组核对矩阵的四个元素及形状。
    assert np.array_equal(matrix.data, np.array([[1, 2], [3, 4]], dtype=np.float32))
    # 矩阵的形状是 (行数, 列数)，这里为 2 行 2 列。
    assert matrix.shape == (2, 2)
    # size 是所有轴的长度相乘：2*2=4。
    assert matrix.size == 4
    # 打印二维矩阵的信息。
    _show("矩阵 Tensor([[1, 2], [3, 4]])", matrix)

    # 三层嵌套列表可以看成两个 2*2 的矩阵叠放起来。
    # tensor_3d.data[1, 0, 1] 指向第二组、第一行、第二列，取到 6。
    tensor_3d = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    # 三个 2 分别表示外层有 2 组、每组 2 行、每行 2 个数字。
    assert tensor_3d.shape == (2, 2, 2)
    # 所有数字的数量为 2*2*2=8，而不是“有 3 个数字”。
    assert tensor_3d.size == 8
    # 显示三维张量的层次与数据。
    _show("三维 Tensor", tensor_3d)

    # ndim 表示轴的数量：零维标量没有轴。
    assert scalar.ndim == 0, "Scalar should be 0-dimensional"
    # assert 后的逗号和字符串提供失败时的提示，不影响成功时的结果。
    assert vector.ndim == 1, "Vector should be 1-dimensional"
    # 2 行 2 列意味着有两个轴，与总元素数 4 是不同概念。
    assert matrix.ndim == 2, "Matrix should be 2-dimensional"
    # 三层嵌套的张量有 3 个轴。
    assert tensor_3d.ndim == 3, "3D tensor should be 3-dimensional"

    # 点号调用对象的方法；numel() 的意思是元素总数，等于 size。
    assert scalar.numel() == 1, "Scalar has 1 element"
    # 向量含 3 个标量数字。
    assert vector.numel() == 3, "Vector has 3 elements"
    # 二维矩阵含 2*2 个标量数字。
    assert matrix.numel() == 4, "2x2 matrix has 4 elements"
    # 一句 f-string 能同时显示多个表达式的运行结果。
    print(f"   numel：标量={scalar.numel()}，向量={vector.numel()}，矩阵={matrix.numel()}，三维={tensor_3d.numel()}")

    # contiguous() 在这里创建数据布局连续的新 Tensor，即便原矩阵已连续也会复制。
    # 同一块内存意味着一个对象的原始数组变化可能影响另一个；复制可隔离这种影响。
    contig = matrix.contiguous()
    # 数据内容相同，但这不足以证明它们指向同一个对象。
    assert np.array_equal(contig.data, matrix.data)
    # is 比较“是否同一个对象”；is not 确认底层数组不是原矩阵数组。
    assert contig.data is not matrix.data, "contiguous() should return a copy"
    # 打印两类比较：内容相等是 True，对象相同是 False。
    print(f"   contiguous()：data 相同={np.array_equal(contig.data, matrix.data)}，是否同一块内存={contig.data is matrix.data}")

    # 运行至此说明前面的断言未失败。
    print("✅ 张量创建通过！")

# 这一组演示逐元素算术运算；* 在这里不是矩阵乘法。
# “逐元素”意味着位置 0 对位置 0、位置 1 对位置 1 分别计算。
# 把整个输入一起送进网络时，通常第 0 轴是样本数量，也叫 batch 维。
# 广播能让形状不同的数组照样计算，但要先想清楚是否符合问题的含义。
def test_unit_arithmetic_operations():
    # 输出算术测试标题。
    print("🧪 单元测试：算术运算...")

    # 创建两个形状同为 (3,) 的向量，便于逐位置比较。
    a = Tensor([1, 2, 3])
    # b 的第 1、2、3 个数字分别是 4、5、6。
    b = Tensor([4, 5, 6])
    # 以下两行展示运算前的输入，方便对照结果。
    _show("a", a)
    _show("b", b)

    # a+b 调用 Tensor 的加法：同位置相加得到 [5,7,9]。
    result = a + b
    # 核对算出的三个数字；array_equal 要求精确相等，此处是简单整数运算。
    assert np.array_equal(result.data, np.array([5, 7, 9], dtype=np.float32))
    # result 是新 Tensor；重新给它赋值不修改 a 或 b。
    _show("a + b", result)

    # 标量 10 广播至每个位置，相当于 [1+10,2+10,3+10]。
    result = a + 10
    # 三个结果应为 [11,12,13]。
    assert np.array_equal(result.data, np.array([11, 12, 13], dtype=np.float32))
    # 显示向量与标量相加后的值。
    _show("a + 10", result)

    # 标量写在左边也能加；Tensor.__radd__ 处理这种右侧对象的运算。
    result = 10 + a
    # 加法交换左右顺序，结果仍为 [11,12,13]。
    assert np.array_equal(result.data, np.array([11, 12, 13], dtype=np.float32))
    # 显示结果。
    _show("10 + a", result)

    # 2 行 2 列的矩阵形状为 (2,2)。
    matrix = Tensor([[1, 2], [3, 4]])
    # 向量形状为 (2,)；可对齐矩阵每一行的两个位置。
    vector = Tensor([10, 20])
    # NumPy 广播把 [10,20] 分别加在两行：并未复制或学出网络权重。
    # 具体算式是 [[1+10,2+20],[3+10,4+20]]，不是矩阵乘法。
    result = matrix + vector
    # 第一行变 [11,22]，第二行变 [13,24]。
    expected = np.array([[11, 22], [13, 24]], dtype=np.float32)
    # 比较实际二维数组与预期二维数组。
    assert np.array_equal(result.data, expected)
    # 连续展示输入和输出，更容易看到广播是按“列位置”重复加法。
    _show("矩阵", matrix)
    _show("向量", vector)
    _show("矩阵 + 向量（广播）", result)

    # 机器学习中常见形状：(4,3) 是四条样本，每条三个输出。
    # 第一轴取值 0 到 3 表示样本编号，第二轴取值 0 到 2 表示输出位置。
    predictions = Tensor(np.ones((4, 3)))
    # zeros 创建同形状的全零目标；只有做损失运算时才会使用它。
    targets_good = Tensor(np.zeros((4, 3)))
    # (3,) 少了表示四条样本的 batch 轴，广播仍可能让运算运行。
    targets_bad = Tensor(np.zeros((3,)))
    # 通过 == 比较完整的 shape 元组，要求预测与目标逐样本对齐。
    assert predictions.shape == targets_good.shape, "Matching shapes — safe"
    # != 表示不相等；此处只检查形状，不真的执行预测减错误目标。
    # 若错误地把 (3,) 当目标来相减，它会被应用于所有四条样本，损失可能失真。
    assert predictions.shape != targets_bad.shape, (
        # 括号允许把较长的报错消息分多行；两个相邻 f-string 会拼成一条消息。
        f"Shape mismatch: {predictions.shape} vs {targets_bad.shape}. "
        f"NumPy broadcasts silently — this is almost always a bug in ML code."
    )
    # shape 相等说明每条样本的三个预测都能对应三个目标。
    print(f"   预测 shape={predictions.shape}，正确目标 shape={targets_good.shape}（对齐）")
    # 错误目标会被重复给四条样本，可能使训练结果悄悄出错。
    print(f"   错误目标 shape={targets_bad.shape}（缺 batch 维，广播会静默发生）")

    # 减法按位置做：b-a 为 [4-1,5-2,6-3]。
    result = b - a
    # 三个位置都等于 3。
    assert np.array_equal(result.data, np.array([3, 3, 3], dtype=np.float32))
    # 显示减法结果。
    _show("b - a", result)

    # 星号乘数字 2 是逐元素缩放，不是线性代数里的矩阵乘法。
    result = a * 2
    # 得到 [2,4,6]。
    assert np.array_equal(result.data, np.array([2, 4, 6], dtype=np.float32))
    # 显示逐元素乘法结果。
    _show("a * 2", result)

    # 交换顺序 2*a，Tensor.__rmul__ 会处理这种写法。
    result = 2 * a
    # 仍为 [2,4,6]。
    assert np.array_equal(result.data, np.array([2, 4, 6], dtype=np.float32))
    # 显示从左侧乘标量的结果。
    _show("2 * a", result)

    # / 是普通除法；向量每个位置各自除以 2。
    result = b / 2
    # [4/2,5/2,6/2] 为 [2,2.5,3]，结果可以有小数。
    assert np.array_equal(result.data, np.array([2.0, 2.5, 3.0], dtype=np.float32))
    # 显示结果。
    _show("b / 2", result)

    # 左侧标量 12 分别除以 b 的数字；不是先对整个向量求和。
    result = 12 / b
    # 12/5=2.4 涉及浮点小数；allclose 允许很小的舍入误差。
    # 浮点数在计算机里以有限位数存放，小数结果可能不能精确表示。
    assert np.allclose(result.data, np.array([3.0, 12.0 / 5.0, 2.0], dtype=np.float32))
    # 显示 [3,2.4,2]。
    _show("12 / b", result)

    # 减法不满足交换律：10-a 是 [9,8,7]，不是 a-10。
    result = 10 - a
    # 检验 __rsub__ 处理了左边是普通数字的情况。
    assert np.array_equal(result.data, np.array([9, 8, 7], dtype=np.float32))
    # 显示结果。
    _show("10 - a", result)

    # 括号规定计算顺序：先逐元素减 2，再逐元素除 2。
    normalized = (a - 2) / 2
    # 正规化后 [1,2,3] 变 [-0.5,0,0.5]，这里只是演示变换，不涉及训练。
    expected = np.array([-0.5, 0.0, 0.5], dtype=np.float32)
    # 实数运算一般用 allclose 判断足够接近，而非要求字节完全一样。
    assert np.allclose(normalized.data, expected)
    # 显示变换后的向量。
    _show("(a - 2) / 2", normalized)

    # 前面的断言全部通过才打印该消息。
    print("✅ 算术运算通过！")

# 矩阵乘法先检查输入是不是 Tensor、维度是否允许、相接的长度是否相等。
# 形状公式：(m,k) @ (k,n) -> (m,n)，中间 k 必须相等。
# 例如一批 4 条样本各 3 个特征，可与形状 (3,2) 的权重相乘得到 (4,2)。
# 输入的 4 表示样本数，权重无需为每条样本重新创建一次。
def test_unit_validate_matmul_shapes():
    # 标题只供人阅读。
    print("🧪 单元测试：矩阵乘法形状校验...")

    # 两个 2*2 的矩阵，a 的列数 2 与 b 的行数 2 相同。
    a = Tensor([[1, 2], [3, 4]])
    # 右侧矩阵仍为 (2,2)；这里还没做真正的矩阵乘法。
    b = Tensor([[5, 6], [7, 8]])
    # 前导下划线只是“内部方法”的命名惯例；此处调用方法只检查形状。
    a._validate_matmul_shapes(b)
    # 打印合法输入；f-string 的花括号会填入 shape 元组。
    print(f"   合法：{a.shape} @ {b.shape}")

    # c 的 shape 为 (1,3)，一行三个数字。
    c = Tensor([[1, 2, 3]])
    # d 的 shape 为 (3,1)，三行一列。
    d = Tensor([[1], [2], [3]])
    # 中间两个 3 对齐，结果理论上是 shape (1,1) 的矩阵。
    c._validate_matmul_shapes(d)
    # 打印另一组合法形状。
    print(f"   合法：{c.shape} @ {d.shape}")

    # try/except 用于预期可能报错的代码，这里故意测试“右侧不是 Tensor”。
    # 冒号之后缩进的代码属于 try；一旦抛错，流程跳到相应的 except。
    try:
        # 普通嵌套列表虽然表示二维数据，但矩阵乘接口明确要求先包装为 Tensor。
        a._validate_matmul_shapes([[1, 2], [3, 4]])
        # 理应已在上一行抛错；若没抛错，就用 assert False 让测试失败。
        assert False, "Should have raised TypeError for non-Tensor"
    # 只捕获 TypeError，as e 把异常对象保存到变量 e，供后面检查。
    except TypeError as e:
        # str(e) 把异常转换成说明文字，in 检查文字是否包含指定片段。
        assert "requires Tensor" in str(e)
    # 报错也应说明收到的右侧对象是 list。
    # 异常文字检查保证用户看到的是能帮助定位输入类型的消息。
        assert "list" in str(e)
        # type(e).__name__ 取得异常的类型名称；有报错才会走到这里。
        print(f"   非法（右侧不是 Tensor）：捕获 {type(e).__name__}")

    # 第二次 try 故意用零维标量参与矩阵乘，应触发 ValueError。
    try:
        # 标量 shape 为 ()，没有可用于矩阵乘法的轴。
        scalar = Tensor(5.0)
        # 标量若想逐位置相乘应使用 *，而不是矩阵乘法 @。
        scalar._validate_matmul_shapes(a)
        # 若未报错，这条断言使测试失败；它抛 AssertionError，不会被下方捕获。
        assert False, "Should have raised ValueError for 0D tensor"
    # ValueError 表示值或形状不符合该操作要求。
    # TypeError 偏向“给错对象类型”，ValueError 偏向“类型对但取值不合法”。
    except ValueError as e:
        # 报错文字应提示矩阵乘法至少需要一维输入。
        assert "at least 1D" in str(e)
        # 显示零维形状 () 与右侧矩阵形状 (2,2)。
        print(f"   非法（0 维标量 {scalar.shape} @ {a.shape}）：捕获 {type(e).__name__}")

    # 第三次 try 右侧确实是 Tensor，但两边内维不对齐。
    try:
        # shape (1,2)：左矩阵只有 2 列。
        incompatible_a = Tensor([[1, 2]])
        # shape (3,1)：右矩阵有 3 行。
        incompatible_b = Tensor([[1], [2], [3]])
        # 矩阵乘 (1,2) @ (3,1) 需要 2==3，但条件不成立，因此应报错。
        incompatible_a._validate_matmul_shapes(incompatible_b)
        # 如果上一步没报错，则显式让测试失败。
        assert False, "Should have raised ValueError for shape mismatch"
    # 捕获不兼容形状所抛的 ValueError。
    except ValueError as e:
        # 核对异常说明确实提到“内部维度不匹配”。
        assert "Inner dimensions don't match" in str(e)
        # 同时核对它提供了 2 与 3 的具体数字。
        assert "2 vs 3" in str(e)
        # 打印实际捕获到的错误类型名称。
        print(f"   非法（内维不匹配 {incompatible_a.shape} @ {incompatible_b.shape}）：捕获 {type(e).__name__}")

    # 合法和非法输入都按预期处理后，显示通过。
    print("✅ 矩阵乘法形状校验通过！")

# 矩阵乘法不是对应位置相乘：左边的每一行与右边的每一列做乘积之和。
# 神经网络中的线性层常用“输入矩阵 @ 权重矩阵 + 偏置”处理整批样本。
# 这里仅验证乘法，没有偏置、损失、求梯度或优化器。
# 区分 A*B（按位置相乘）与 A@B（行与列相乘）是读网络公式的关键。
def test_unit_matrix_multiplication():
    # 输出当前测试标题。
    print("🧪 单元测试：矩阵乘法...")

    # 左右形状均为 (2,2)，输出也为 (2,2)。
    a = Tensor([[1, 2], [3, 4]])
    # b 的第一列为 [5,7]、第二列为 [6,8]。
    b = Tensor([[5, 6], [7, 8]])
    # 计算：左上角 1*5+2*7=19；右上角 1*6+2*8=22。
    result = a.matmul(b)
    # 第二行结果：3*5+4*7=43，3*6+4*8=50。
    expected = np.array([[19, 22], [43, 50]], dtype=np.float32)
    # 精确核对测试中这些小整数计算的结果。
    assert np.array_equal(result.data, expected)
    # 以下三行依次显示左右输入和结果，便于对照手算。
    _show("A", a)
    _show("B", b)
    _show("A.matmul(B)", result)

    # 第二组左矩阵 (2,3)、右矩阵 (3,2)，中间维度都是 3。
    c = Tensor([[1, 2, 3], [4, 5, 6]])
    # d 有 3 行 2 列，结果的形状应取外侧的 (2,2)。
    d = Tensor([[7, 8], [9, 10], [11, 12]])
    # 例如左上角 1*7+2*9+3*11=58。
    result = c.matmul(d)
    # 四个输出数字分别为 58、64、139、154。
    expected = np.array([[58, 64], [139, 154]], dtype=np.float32)
    # 比较矩阵乘出的完整二维结果。
    assert np.array_equal(result.data, expected)
    # 显示这一组输入与输出；标题中的乘号只是文字。
    _show("C (2×3)", c)
    _show("D (3×2)", d)
    _show("C.matmul(D)", result)

    # 矩阵的每一行将与同一个一维向量求点积。
    matrix = Tensor([[1, 2, 3], [4, 5, 6]])
    # 向量形状为 (3,)，没有第二个轴。
    vector = Tensor([1, 2, 3])
    # 输出只有每行一个数字，形状为 (2,)：1*1+2*2+3*3=14。
    result = matrix.matmul(vector)
    # 第二行计算：4*1+5*2+6*3=32。
    expected = np.array([14, 32], dtype=np.float32)
    # 检查一维结果而不是误判为 shape (2,1) 的二维列矩阵。
    # shape 相近不等于相同：(2,) 有一个轴，(2,1) 有两个轴。
    assert np.array_equal(result.data, expected)
    # 依次打印矩阵、向量和结果。
    _show("矩阵 (2×3)", matrix)
    _show("向量 (3,)", vector)
    _show("矩阵 @ 向量", result)

    # @ 是 Python 的矩阵乘法运算符；a @ b 会调用 Tensor.__matmul__。
    result_at = a @ b
    # @ 与 a.matmul(b) 应得到同样的数字。
    assert np.array_equal(result_at.data, np.array([[19, 22], [43, 50]], dtype=np.float32))
    # 打印 @ 运算结果。
    _show("A @ B", result_at)

    # 所有矩阵乘测试均通过后输出提示。
    print("✅ 矩阵乘法通过！")


# 改变形状时不能丢掉或凭空增加元素；每种形状都表示同一批数字的组织方式。
# reshape 与 transpose 不同：前者按顺序重新分组，后者交换轴及对应元素坐标。
# 对图像样本展平时，要保留第 0 轴的 batch 数量，避免把不同样本混在一起。
def test_unit_shape_manipulation():
    # 打印本组测试的名称。
    print("🧪 单元测试：形状操作...")

    # 六个数字先按一维向量存储，其 shape 为 (6,)。
    tensor = Tensor([1, 2, 3, 4, 5, 6])
    # 显示原向量，便于和重排后的数字顺序比较。
    _show("原向量", tensor)
    # reshape(2,3) 把 6 个元素按 2 行 3 列排列；这里不做加减乘除。
    reshaped = tensor.reshape(2, 3)
    # 形状由 (6,) 变成 (2,3)，总元素数依然为 6。
    assert reshaped.shape == (2, 3)
    # NumPy 数组给出按原顺序分组后的预期二维数据。
    expected = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    # 比较重排后的形状和数字；这里用 array_equal 核对精确内容。
    assert np.array_equal(reshaped.data, expected)
    # 显示 (2,3) 矩阵。
    _show("reshape(2, 3)", reshaped)

    # reshape 也允许把形状装进一个元组 (3,2) 再作为单个参数传入。
    reshaped2 = tensor.reshape((3, 2))
    # 三行两列的形状总共仍是 3*2=6 个数字。
    assert reshaped2.shape == (3, 2)
    # 相邻的两个数字组成一行，因此预期是三行。
    expected2 = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    # 数据顺序保持 1 到 6。
    assert np.array_equal(reshaped2.data, expected2)
    # 显示另一种重排后的矩阵。
    _show("reshape((3, 2))", reshaped2)

    # -1 表示“让程序推断这一轴长度”；总数 6/已知行数 2=3。
    # -1 在这里不是“倒数第一个下标”；reshape 只能有一个待推断维度。
    auto_reshaped = tensor.reshape(2, -1)
    # 推断出来的形状是 (2,3)，不是有 -1 个元素。
    assert auto_reshaped.shape == (2, 3)
    # 打印实际推断结果。
    _show("reshape(2, -1)", auto_reshaped)

    # 故意测试不合法的重排：原有 6 个数字，2*2 却只能放 4 个。
    try:
        # reshape 会主动抛 ValueError 来告知元素数量不一致。
        tensor.reshape(2, 2)
        # 若上一行没有抛异常，这个 assert False 将让测试失败。
        assert False, "Should have raised ValueError"
    # 只接住预期的 ValueError，并用 e 引用这次异常。
    except ValueError as e:
        # 检查异常文字包含元素数量不匹配的说明。
        assert "Element count mismatch" in str(e)
        # 同时检查诊断中包含“6 个元素与 4 个元素”的具体数字。
        assert "6 elements vs 4 elements" in str(e)
        # 打印捕获的错误类型；错误发生正是此项测试要验证的行为。
        print(f"   reshape(2, 2) 非法：6 个元素装不进 4，捕获 {type(e).__name__}")

    # 用一个 2 行 3 列的矩阵展示转置。
    matrix = Tensor([[1, 2, 3], [4, 5, 6]])
    # 不给参数时，该实现交换最后两个轴；二维矩阵即行列互换。
    transposed = matrix.transpose()
    # 行数和列数交换，输出从 (2,3) 变成 (3,2)。
    assert transposed.shape == (3, 2)
    # 新的第一行 [1,4] 来自旧的第一列；依次类推。
    # 如果旧元素在 [0,1]（数字 2），交换轴后它就在 [1,0]。
    expected = np.array([[1, 4], [2, 5], [3, 6]], dtype=np.float32)
    # 核对转置后的数据内容，不能只检查形状。
    assert np.array_equal(transposed.data, expected)
    # 连续展示转置前与转置后的数字排列。
    _show("矩阵 (2, 3)", matrix)
    _show("transpose() → (3, 2)", transposed)

    # 一维向量只有一个轴，不能像二维矩阵那样交换行和列。
    vector = Tensor([1, 2, 3])
    # 对一维向量调用 transpose() 会得到数值不变的新 Tensor。
    vector_t = vector.transpose()
    # 验证数字没有因转置发生改变；它仍是 shape (3,) 的一维对象。
    assert np.array_equal(vector.data, vector_t.data)
    # 打印转置结果以观察一维向量形状。
    _show("一维 transpose（应不变）", vector_t)

    # np.arange(24) 生成 0 到 23；reshape 把它们排成 (2,3,4) 的三维数组。
    tensor_3d = Tensor(np.arange(24).reshape(2, 3, 4))
    # transpose(0,2) 交换第 0 轴和第 2 轴；Python 的轴编号从 0 开始。
    swapped = tensor_3d.transpose(0, 2)
    # 原 (2,3,4) 交换首尾轴后成为 (4,3,2)。
    assert swapped.shape == (4, 3, 2), (
        # 条件失败时在 f-string 中显示实际形状，帮助定位问题。
        f"transpose(0, 2) on (2,3,4) should give (4,3,2), got {swapped.shape}"
    )
    # 三层 for 对所有旧坐标 (i,j,k) 逐一检查；range(2) 得 0 和 1。
    # 内层循环每完成四次，j 增加 1；j 完成三轮，i 才增加 1。
    for i in range(2):
        # 第二个索引 j 可以取 0、1、2。
        for j in range(3):
            # 第三个索引 k 可以取 0、1、2、3；缩进表示循环嵌套。
            for k in range(4):
                # 下标可写成 [i,j,k]；交换首尾轴后原位置 [i,j,k] 变成 [k,j,i]。
                assert swapped.data[k, j, i] == tensor_3d.data[i, j, k], (
                    # f-string 只构造失败提示，不会改变实际断言比较。
                    f"Data mismatch at [{k},{j},{i}]: expected {tensor_3d.data[i,j,k]}, "
                    f"got {swapped.data[k,j,i]}"
                )
    # 所有元素都验证后，展示原张量和轴交换后的张量。
    _show("三维原张量", tensor_3d)
    _show("transpose(0, 2)", swapped)

    # 假设 2 条样本各有 3*4 个数字；批次维是开头的 2。
    batch_images = Tensor(rng.random((2, 3, 4)))
    # 只固定批次维 2，-1 自动算出剩余的 3*4=12 个特征。
    flattened = batch_images.reshape(2, -1)
    # “展平”每条样本得到 shape (2,12)，两条样本不会合成一条。
    # 例如一张 3 行 4 列的图像可当作 12 个输入特征，再送给线性层。
    assert flattened.shape == (2, 12)
    # data=False 只显示摘要，避免把随机生成的每个数字都打印出来。
    _show("batch (2, 3, 4)", batch_images, data=False)
    # 打印展平后的形状；神经网络的全连接层常使用这种二维输入。
    _show("flatten → (2, 12)", flattened, data=False)

    # 所有合法和非法重排检查通过后显示结果。
    print("✅ 形状操作通过！")

# 归约运算把一组数字“归并”为更少的数字，例如求和、平均或最大值。
# axis=0 是第一个轴（例子中为行），axis=1 是第二个轴（列）。
# 指定 axis 表示把该轴的多个位置合并，结果一般不再包含这条轴。
# keepdims=True 则保留这条轴，但其长度变成 1，便于后续广播。
def test_unit_reduction_operations():
    # 打印本组测试的标题。
    print("🧪 单元测试：归约运算...")

    # 准备 2 行 3 列的矩阵，便于手算每行和每列的结果。
    matrix = Tensor([[1, 2, 3], [4, 5, 6]])
    # 先展示原数据，以后所有归约都以它为输入。
    _show("矩阵", matrix)

    # 不指定 axis 时，对所有 6 个数字求和：1+2+3+4+5+6=21。
    total = matrix.sum()
    # 标量张量的 data 可与普通数字 21.0 比较。
    assert total.data == 21.0
    # 所有轴都归并掉，结果是一个数字，因此 shape 为零维的 ()。
    assert total.shape == ()
    # 打印总和。
    _show("sum()", total)

    # axis=0 表示沿“行所在的第 0 轴”合并：上下两行逐列相加。
    col_sum = matrix.sum(axis=0)
    # 三列分别得到 1+4=5、2+5=7、3+6=9。
    expected_col = np.array([5, 7, 9], dtype=np.float32)
    # 核对列和内容。
    assert np.array_equal(col_sum.data, expected_col)
    # 行轴消失，只留下 3 个列位置，因此结果 shape 为 (3,)。
    # “沿第 0 轴求和”看似求列和，正是因为第 0 轴表示上下两行。
    assert col_sum.shape == (3,)
    # 打印列和。
    _show("sum(axis=0) 列和", col_sum)

    # axis=1 表示沿“列所在的第 1 轴”合并：每行内部相加。
    row_sum = matrix.sum(axis=1)
    # 第一行 1+2+3=6，第二行 4+5+6=15。
    expected_row = np.array([6, 15], dtype=np.float32)
    # 核对两个行和。
    assert np.array_equal(row_sum.data, expected_row)
    # 列轴消失，留下 2 个行位置，因此 shape 为 (2,)。
    assert row_sum.shape == (2,)
    # 显示行和。
    _show("sum(axis=1) 行和", row_sum)

    # mean() 对所有数字求平均，即总和 21 除以元素数量 6，得到 3.5。
    avg = matrix.mean()
    # np.isclose 用于比较单个浮点数是否足够接近目标值。
    assert np.isclose(avg.data, 3.5)
    # 所有数字归并为一个平均值后，shape 仍是标量的 ()。
    assert avg.shape == ()
    # 展示整体平均值。
    _show("mean()", avg)

    # 沿第 0 轴求均值：相同列的上下数字各取平均。
    col_mean = matrix.mean(axis=0)
    # 三列均值分别是 (1+4)/2=2.5、3.5、4.5。
    expected_mean = np.array([2.5, 3.5, 4.5], dtype=np.float32)
    # np.allclose 用于逐位置近似比较浮点数组，允许微小的舍入误差。
    assert np.allclose(col_mean.data, expected_mean)
    # 打印各列的均值。
    _show("mean(axis=0)", col_mean)

    # 不指定轴时，在全部六个数字中找最大值。
    maximum = matrix.max()
    # 最大数字为 6.0。
    assert maximum.data == 6.0
    # 总体最大值是一个标量，所以没有轴。
    assert maximum.shape == ()
    # 显示最大数字。
    _show("max()", maximum)

    # axis=1 表示分别在每一行的三列中寻找最大值。
    row_max = matrix.max(axis=1)
    # 第一行最大为 3，第二行最大为 6。
    expected_max = np.array([3, 6], dtype=np.float32)
    # 检验每行的最大值。
    assert np.array_equal(row_max.data, expected_max)
    # 打印结果，形状为 (2,)。
    _show("max(axis=1)", row_max)

    # keepdims=True 表示归约后保留原轴，但把它的长度改成 1。
    sum_keepdims = matrix.sum(axis=1, keepdims=True)
    # 两行仍在，每行合并三列后只余 1 列，shape 为 (2,1)。
    assert sum_keepdims.shape == (2, 1)
    # 内容是两行 [[6],[15]]；不同于一维向量 [6,15]。
    expected_keepdims = np.array([[6], [15]], dtype=np.float32)
    # 检查二维数据和预期一致；保留维度便于后面按形状广播。
    assert np.array_equal(sum_keepdims.data, expected_keepdims)
    # 展示保留轴后的结果。
    _show("sum(axis=1, keepdims=True)", sum_keepdims)

    # 三维 shape (2,2,2)：第一轴可看成两条样本，后两轴是各样本的 2*2 个数。
    tensor_3d = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    # axis=(1,2) 是含两个轴编号的元组；各样本内部的四个数字分别求平均。
    spatial_mean = tensor_3d.mean(axis=(1, 2))
    # 两条样本各剩一个均值，shape 为 (2,)；具体均值是 [2.5,6.5]。
    # 第一组均值 (1+2+3+4)/4=2.5，第二组 (5+6+7+8)/4=6.5。
    assert spatial_mean.shape == (2,)
    # 先展示原始三维张量，再看每条样本的均值。
    _show("三维", tensor_3d)
    _show("mean(axis=(1, 2)) 空间均值", spatial_mean)

    # 以上断言都通过才会输出此提示；这里没有训练网络或更新权重。
    print("✅ 归约运算通过！")

# 这一节计时比较同一个二维数组按行和按列读取；它只做前向数据访问，不训练模型。
def analyze_memory_layout():
    # print 把标题写到终端；字符串乘 60 表示重复该字符 60 次，不是张量乘法。
    print("📊 分析内存访问模式...")
    print("=" * 60)

    # size 是矩阵每条边的元素数；2000 × 2000 一共有 400 万个数字。
    size = 2000
    # rng.random((行, 列)) 生成随机二维 NumPy 数组；Tensor(...) 将其数据转成 float32。
    # 这里创建的是常规按行连续的数组；后面的行列访问比较以这个具体布局为前提。
    matrix = Tensor(rng.random((size, size)))

    # matrix.size 是元素个数，每个 float32 占 4 字节；除以 1024² 得到 MiB，原输出写 MB 是宽泛叫法。
    # f 字符串中的 {表达式:.1f} 会算出并显示一位小数；\n 代表显示时换行。
    print(f"\n测试矩阵：{size}×{size}（{matrix.size * BYTES_PER_FLOAT32 / MB_TO_BYTES:.1f} MB）")
    # shape 表示各轴长度，size 是元素总数，dtype 是单个数字的数据类型；点号读取对象属性。
    print(f"   shape={matrix.shape}，size={matrix.size}，dtype={matrix.dtype}")
    # 显示分隔线；这两个 print 只展示信息，不参与计时和计算。
    print("-" * 60)

    # 访问一整行：同一行的相邻 float32 通常也相邻存放，适合顺序读取。
    print("\n测试 1：按行访问（对缓存友好）")
    # time.time() 返回秒数；用结束时间减开始时间估计这段代码的墙钟耗时。
    start = time.time()
    # [] 创建空列表；之后依次记录每一行的和。
    row_sums = []
    # for ... in range(2000) 让 i 依次取 0 到 1999；缩进部分每行重复执行一次。
    for i in range(size):
        # [i, :] 选第 i 行、该行全部列；NumPy 的 sum() 把这 2000 个值相加。
        row_sum = matrix.data[i, :].sum()
        # append 追加一个结果，最终列表长度为 2000。
        row_sums.append(row_sum)
    # 上面的循环、切片、求和及列表追加都算在内；不是纯内存硬件读取时间。
    row_time = time.time() - start
    # *1000 把秒转毫秒；[:3] 只拿列表最前面 3 个结果用于展示。
    print(f"   耗时：{row_time*1000:.1f}ms，前 3 个行和={row_sums[:3]}")
    # 这句话描述当前行优先数组的常见访问方式，并非在验证所有数组的布局。
    print("   访问模式：顺序访问（顺着内存布局）")

    # 与按行相对，下面固定列号，遍历这个数组的所有行。
    print("\n测试 2：按列访问（对缓存不友好）")
    # 重新记起点，分别测量第二段循环。
    start = time.time()
    # 建立新列表，保存各列的和；变量名不同于上一段的 row_sums。
    col_sums = []
    # j 是列下标，依次取 0 到 1999。
    for j in range(size):
        # [:, j] 是所有行的第 j 列；在这个按行连续的矩阵里，相邻列元素相隔一整行。
        col_sum = matrix.data[:, j].sum()
        # 只记录列和数字，避免把 2000 列都打印出来。
        col_sums.append(col_sum)
    # 算出这一段循环总墙钟时间；测量会受机器负载、NumPy 实现及缓存状态影响。
    col_time = time.time() - start
    # 与上面的行计时采用相同毫秒显示方式；只输出前 3 个列和。
    print(f"   耗时：{col_time*1000:.1f}ms，前 3 个列和={col_sums[:3]}")
    # 对这个 C 风格的 float32 方阵，理论相邻行中同一列地址差 size * 4 字节。
    # 更通用的真实步幅可读 matrix.data.strides，不能对任意布局都套用这个公式。
    print(f"   访问模式：跨步访问（每个元素跳 {size * BYTES_PER_FLOAT32} 字节）")

    # / 是普通除法；如列循环耗时 0.2 秒、行循环 0.1 秒，比例为 2.0。
    # 这只比较两段整体代码，不能量化缓存未命中本身造成了多少开销。
    slowdown = col_time / row_time
    # "\n" + 字符串使用 + 拼接文字，print 输出下一节标题。
    print("\n" + "=" * 60)
    print("📊 性能影响：")
    # :.2f 显示两位小数；两个分数实质上是同一个列/行耗时比。
    print(f"   变慢倍数：{slowdown:.2f}×（慢 {col_time/row_time:.1f} 倍）")
    # 注意：此百分比仅是 (列耗时 / 行耗时 - 1) × 100 的算术结果。
    # 原输出把差额全部归为缓存未命中，但此实验没有隔离 Python 循环、NumPy、硬件及先后顺序的影响。
    print(f"   缓存未命中导致约 {(slowdown-1)*100:.0f}% 的性能损失")

    # 以下是解释性文字，不是新的测量数据；适用于当前按行连续的矩阵访问场景。
    print("\n💡 要点：")
    # C 风格表示最后一维的相邻元素通常在内存里相邻；F 风格通常先沿第一维连续。
    print("   1. 内存布局很重要：行优先（C 风格）存储是连续的")
    # 缓存行大小因处理器而异；64 字节只是常见数值，不是 Python 规定。
    print("   2. 缓存行大约 64 字节：按行访问会把邻近元素「顺便」载入")
    # 按列访问可能因步幅大而不利于缓存；也可能命中缓存，不能断言每次都重新从 DRAM 加载。
    print("   3. 按列访问容易缓存未命中：每次都要从 DRAM 重新加载")
    # 大 O 描述输入规模增大时的增长趋势；下面的时间倍数只来自这一轮具体测试。
    print(f"   4. 算法仍是 O(n)，墙钟时间却差了 {slowdown:.1f} 倍！")

    # 以下应用场景只是背景知识；这几行 print 不会真的执行图像处理或矩阵分块。
    print("\n🚀 实际影响：")
    print("   • 图像处理库会选用特定内存格式，以利用缓存")
    # 分块是让一部分数据在较快的缓存中重复利用，不等于此处已经实现分块矩阵乘。
    print("   • 矩阵乘法常用分块，把数据切进缓存大小的块里")
    # 注意：上面的实验没有对 transpose() 计时；slowdown 不能当作转置耗时倍数。
    # 当前 Tensor.transpose() 还会构造新的 Tensor，不能断言返回结果一直是共享数据的非连续视图。
    print(f"   • transpose 较贵（{slowdown:.1f}×）：非连续视图，缓存局部性差")
    # 硬件库通常会考虑内存布局，但实际加速幅度取决于数据大小和具体实现。
    print("   • 硬件优化库会顺着内存布局来提升性能")

    # 最后再显示分隔线，函数执行到末尾后自动返回 None；它不返回测量结果。
    print("\n" + "=" * 60)

# 把各项小测试串起来运行，再用两次线性变换、形状调整、广播检查它们能否协同工作。
# 这里的「通过」表示断言中的情况成立，不代表张量已具备自动求导、反向传播或训练能力。
def test_module():
    # 输出标题；字符串字面量是给人看的说明，不是网络运算。
    print("🧪 正在运行模块集成测试")
    # 字符串 * 整数重复文字，形成分隔线。
    print("=" * 50)

    # 运行先前用 def 定义的函数；只有调用函数名后加 () 时才执行函数内语句。
    print("运行单元测试...")
    # 逐项检查从普通数/列表创建 Tensor 后的值、形状和数据类型。
    test_unit_tensor_creation()
    # 检查逐元素加减乘除等算术行为。
    test_unit_arithmetic_operations()
    # 检查矩阵乘法所需的类型与形状，不兼容时应给出对应错误。
    test_unit_validate_matmul_shapes()
    # 检查矩阵乘法的结果数值和形状。
    test_unit_matrix_multiplication()
    # 检查 reshape、transpose 以及元素总数保持不变等形状行为。
    test_unit_shape_manipulation()
    # 检查 sum、mean、max 的整体或按轴归约结果。
    test_unit_reduction_operations()

    # "\n" 是换行符，使后面的集成场景与上面的测试结果隔开。
    print("\n运行集成场景...")

    # 以下人为指定权重与偏置，模拟一小批样本的前向计算；没有自动学习权重。
    print("🧪 集成测试：两阶段线性变换...")

    # 外层列表有两行：两条样本；每行 3 个数：每条样本的三个输入特征。
    # Tensor(...) 将这些 Python 列表转换成 float32 数组，所以 x.shape 为 (2, 3)。
    x = Tensor([[1, 2, 3], [4, 5, 6]])

    # W1 有 3 行、4 列；第 1 行对应第 1 个输入特征对 4 个隐藏输出的权重。
    # 括号未关闭时，Python 允许把一个长列表分成多行，下面三行仍是同一个表达式。
    W1 = Tensor([[0.1, 0.2, 0.3, 0.4],
                 # 这一行对应第 2 个输入特征；逗号分隔四个浮点数。
                 [0.5, 0.6, 0.7, 0.8],
                 # 这一行对应第 3 个输入特征；整体 W1.shape 是 (3, 4)。
                 [0.9, 1.0, 1.1, 1.2]])
    # b1 是一维的四个偏置值，每个隐藏输出位置各用一个，shape 为 (4,)。
    b1 = Tensor([0.1, 0.2, 0.3, 0.4])
    # _show 只打印数据与形状，便于跟踪以下每一步；不修改张量。
    _show("输入 x", x)
    # 大写 W 是权重的命名习惯；名字大小写敏感，W1 与 w1 是两个不同标识符。
    _show("W1", W1)
    # b 是偏置的命名习惯。
    _show("b1", b1)

    # x (2,3) 乘 W1 (3,4) 得 (2,4)：内侧的 3 对齐，两条样本各得 4 个隐藏数字。
    # + b1 把 (4,) 的四个偏置广播到两行；例如第一条的首个输出是 1*0.1+2*0.5+3*0.9+0.1=3.9。
    # 调用 .matmul(W1) 是矩阵乘；与 x * W1 所代表的逐元素乘法不同。
    hidden = x.matmul(W1) + b1
    # assert 检查条件；若形状不符，抛出 AssertionError，逗号后 f 字符串是错误提示。
    # == 比较是否相等，= 则是赋值；这里 (2, 4) 是包含两个整数的元组。
    assert hidden.shape == (2, 4), f"Expected (2, 4), got {hidden.shape}"
    # 名为 hidden 是因为它是输入与最后输出之间的一层结果；右侧标题中的 @ 是矩阵乘的常见记法。
    _show("hidden = x @ W1 + b1", hidden)

    # 第二组权重有 4 行、2 列：前面的四个隐藏值映射为两个输出值。
    W2 = Tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
    # 最终每个输出位置各加一个偏置，所以 b2.shape 是 (2,)。
    b2 = Tensor([0.1, 0.2])
    # 依次观察第二层的权重与偏置，核实对接隐藏张量的形状。
    _show("W2", W2)
    # 只打印内容，不会在此时计算第二层输出。
    _show("b2", b2)

    # hidden (2,4) 乘 W2 (4,2) 得 (2,2)，再将 b2 (2,) 加到每条样本的两个输出上。
    # 每个输出都是“输入乘权重的和，再加偏置”；此处没有激活函数，两个线性/仿射步骤可合并成一个。
    # 结果只是两个原始数值，不自动变为概率、类别预测，更不会自动更新 W1/W2。
    output = hidden.matmul(W2) + b2
    # 检查两条样本各得到两个输出；如果中间维度接不上，matmul 会先抛出形状错误。
    assert output.shape == (2, 2), f"Expected (2, 2), got {output.shape}"

    # np.isnan 为每个数判断是否 NaN（不是有效数字），.any() 问其中是否至少一个为真。
    # not 把判断结果取反；此断言要求所有输出都不是 NaN。
    assert not np.isnan(output.data).any(), "Output contains NaN values"
    # np.isfinite 判断各数是否为有限值，.all() 要求每个位置都通过；可排除正负无穷和 NaN。
    assert np.isfinite(output.data).all(), "Output contains infinite values"
    # 打印最终结果，以便对照两步前向公式。
    _show("output = hidden @ W2 + b2", output)

    # 能执行到这里说明上面的断言都成立；这只是本例的数值正确性检查。
    print("✅ 两阶段线性变换通过！")

    # 下面专测同样 12 个数如何用不同形状表示；reshape 不会凭空增减数字。
    print("🧪 集成测试：复杂形状操作...")
    # 一维列表含 12 个数；Tensor 的 shape 为 (12,)。
    data = Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    # 先看原始一维输入，后面的变形均从 data 或 tensor_3d 得来。
    _show("原数据", data)

    # 2*2*3=12；拆成 (2,2,3)，可视作「2 条样本，每条含 2 行、每行 3 个数」。
    tensor_3d = data.reshape(2, 2, 3)
    # 形状检查的是各维长度，不验证某一维的现实含义。
    assert tensor_3d.shape == (2, 2, 3)
    # 展示三维数据；每条样本的六个数仍是原来顺序的数字。
    _show("reshape(2, 2, 3)", tensor_3d)

    # axis=(1,2) 表示一起对第 1、2 轴求平均，只留下第 0 轴「两条样本」。
    # 第一条均值为 (1+2+3+4+5+6)/6=3.5，第二条为 (7+...+12)/6=9.5。
    # 神经网络里的池化也会在空间位置汇聚数值；这里仅是简单算术平均。
    pooled = tensor_3d.mean(axis=(1, 2))
    # 因为每条样本最后只剩一个均值，结果是一维的 (2,)，不是二维 (2,1)。
    assert pooled.shape == (2,), f"Expected (2,), got {pooled.shape}"
    # _show 显示 [3.5, 9.5] 以及现在的维度、元素数。
    _show("空间均值 mean(axis=(1, 2))", pooled)

    # 仍从原三维张量变形：(2,-1) 保留 2 条样本，用 -1 让程序自动推导另一维=12/2=6。
    # 扁平化常用于把图像的多维特征送进只接受二维输入的全连接层。
    flattened = tensor_3d.reshape(2, -1)
    # 每条样本 6 个数，形状应是 (2,6)。
    assert flattened.shape == (2, 6)
    # 看扁平化结果；没有把不同样本混到同一行。
    _show("flatten reshape(2, -1)", flattened)

    # 不给轴参数的 transpose() 在本类中交换最后两个轴：(2,2,3) 变成 (2,3,2)。
    # 元素总数仍是 12，但索引与布局会变化；这不是两个神经网络层的交换。
    transposed = tensor_3d.transpose()
    # 检查交换轴之后的长度；第 0 轴的两条样本保持不变。
    assert transposed.shape == (2, 3, 2)
    # 显示交换轴之后的内容，观察不同索引顺序。
    _show("transpose() 交换最后两维", transposed)

    # 前面的若干形状断言都没报错，此时打印通过。
    print("✅ 复杂形状操作通过！")

    # 广播就是在形状兼容时，把较少维度的数字按规则用于更多位置；以下验证两个常见例子。
    print("🧪 集成测试：广播边界情况...")

    # 单个数 5.0 是零维标量，shape 为 ()，注意不是只含一个值的一维列表 [5.0]。
    scalar = Tensor(5.0)
    # 三个数字构成一维向量，shape 是 (3,)。
    vector = Tensor([1, 2, 3])
    # scalar + vector 借助广播等价于 [5+1, 5+2, 5+3]，新结果形状为 (3,)。
    result = scalar + vector
    # 创建用来核对的预期 NumPy 数组；dtype=np.float32 指定一致的浮点数据类型。
    expected = np.array([6, 7, 8], dtype=np.float32)
    # np.array_equal 要求形状和各位置数字完全一致；这里只用于确定的简单加法。
    assert np.array_equal(result.data, expected)
    # 三次 _show 分别展示广播前的两个对象和广播后的结果。
    _show("标量", scalar)
    # 这里显示向量的原数据；加法没有把原向量修改为 [6,7,8]。
    _show("向量", vector)
    # 显示本次加法返回的新 Tensor。
    _show("标量 + 向量", result)

    # 二维矩阵 shape=(2,2)：两行，每行两个数。
    matrix = Tensor([[1, 2], [3, 4]])
    # 一维 vec shape=(2,)：两个数，按最后一维与矩阵的列数对齐。
    vec = Tensor([10, 20])
    # 第一行 [1,2]+[10,20]=[11,22]；第二行 [3,4]+[10,20]=[13,24]。
    result = matrix + vec
    # 再创建预期答案用于逐位置比较；覆盖上一个 expected 变量名不会影响已完成的断言。
    expected = np.array([[11, 22], [13, 24]], dtype=np.float32)
    # 假如某一列没有正确广播，断言会让测试失败。
    assert np.array_equal(result.data, expected)
    # 以下三行把输入、广播向量、输出依次打印出来。
    _show("矩阵", matrix)
    # 原向量仍保存 [10,20]；这个 + 返回新 Tensor，并未原地加到 vec。
    _show("向量", vec)
    # 结果应是两行；NumPy 的广播沿缺失的第 0 轴扩展 vec。
    _show("矩阵 + 向量", result)

    # 两种广播结果都符合预期，输出本节通过文字。
    print("✅ 广播边界情况通过！")

    # 这些 print 只负责呈现报告；没有检查损失是否降低，也没有训练循环。
    print("\n" + "=" * 50)
    # 「模块可以迁入」是脚本作者的提示，并非程序已经自动迁移代码或证明所有情况正确。
    print("🎉 全部测试通过！模块可以迁入。")

# 给初学者看 Tensor 的逐元素算术如何与 NumPy 数组结果对齐；这段不涉及矩阵乘法。
def demo_tensor():
    # print(...) 打印文字；这个标题不代表整个 NumPy 接口都已实现。
    print("🎯 你的 Tensor 可以像 NumPy 一样用")
    # 生成 45 个等号作为视觉分隔。
    print("=" * 45)

    # 先以 np.array([...]) 从 Python 列表创建 NumPy 数组，再交给自定义 Tensor 包装。
    # a 和 b 都是一维 shape=(3,)；变量名只是引用对象的名字。
    a = Tensor(np.array([1, 2, 3]))
    # 第二个对象的值是 4、5、6；构造时 Tensor 会统一为 float32。
    b = Tensor(np.array([4, 5, 6]))
    # _show 显示三个数字、shape、size、ndim 和 dtype；此处默认 data=True。
    _show("a", a)
    # 和上一行相同，展示第二个张量。
    _show("b", b)

    # a + b 触发 Tensor.__add__，各对应位置相加，得到 [5,7,9]。
    # Python 自动把加号对应到特殊方法；a 和 b 的原数据不会因此改变。
    tensor_sum = a + b
    # a * b 调用 Tensor.__mul__ 做逐元素乘法，得到 [4,10,18]；* 不是矩阵乘法。
    tensor_prod = a * b

    # 用原生 NumPy 数组做一遍相同运算，当作独立的参照答案。
    np_sum = np.array([1, 2, 3]) + np.array([4, 5, 6])
    # NumPy 数组上的 * 也按对应位置相乘；这里两边长度一样。
    np_prod = np.array([1, 2, 3]) * np.array([4, 5, 6])

    # f 字符串中的 {tensor_sum.data} 读取张量内部 NumPy 数组，插入可见文字。
    print(f"   Tensor a + b：{tensor_sum.data}")
    # 同样显示原生 NumPy 相加后的参照数组。
    print(f"   NumPy  a + b：{np_sum}")
    # np.allclose 用容差比较浮点数组；True 表示这两个示例结果在容差内相同。
    print(f"   是否一致：{np.allclose(tensor_sum.data, np_sum)}")

    # 展示自定义 Tensor 的逐元素乘法结果。
    print(f"   Tensor a * b：{tensor_prod.data}")
    # 展示原生 NumPy 的乘法结果，方便逐位置肉眼对照。
    print(f"   NumPy  a * b：{np_prod}")
    # 仅检查这三个具体数字的乘法，并不能证明所有运算、所有形状都和 NumPy 一致。
    print(f"   是否一致：{np.allclose(tensor_prod.data, np_prod)}")

    # 此处展示性结语只是打印一句话，不会自动进入后续模块。
    print("\n✨ 你的 Tensor 已与 NumPy 对齐，可以继续做后续模块！")

# __name__ 是 Python 给模块设的特殊名字；直接运行本文件时等于 "__main__"。
# 被别的文件 import 时通常是模块名，因此下面的脚本演示不会自动启动。
if __name__ == "__main__":
    # 按顺序跑所有单元检查和集成场景；函数名后的 () 表示真正调用。
    test_module()
    # print("\n") 插入空白行，让终端报告分段显示。
    print("\n")
    # 在这里运行 2000×2000 数组的行列计时；所需内存与时间高于上面的演示。
    analyze_memory_layout()
    # 再打印分隔用的空白行。
    print("\n")
    # 最后运行三元素数组的加法和乘法演示。
    demo_tensor()
