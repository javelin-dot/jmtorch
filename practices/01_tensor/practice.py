"""阶段草稿。实现写在这里，稳定后再迁入 src/jmtorch/。"""

import time

import numpy as np
rng = np.random.default_rng(7)

BYTES_PER_FLOAT32 = 4
MB_TO_BYTES = 1024 * 1024

class Tensor:
    def __init__(self, data):
        if isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], Tensor):
            data = np.stack([t.data for t in data])
        self.data = np.array(data, dtype=np.float32)
        self.shape = self.data.shape
        self.size = self.data.size
        self.dtype = self.data.dtype

    def __repr__(self):
        return f"Tensor(data={self.data}, shape={self.shape})"

    def __str__(self):
        return f"Tensor({self.data})"

    def numpy(self):
        return self.data

    def memory_footprint(self):
        return self.data.nbytes

    @property
    def ndim(self):
        return len(self.shape)

    def numel(self):
        return self.size

    def contiguous(self):
        return Tensor(np.ascontiguousarray(self.data))

    def view(self, *shape):
        return self.reshape(*shape)

    def masked_fill(self, mask, value):
        mask_array = mask.data.astype(bool) if isinstance(mask, Tensor) else np.asarray(mask, dtype=bool)
        result = self.data.copy()
        result[mask_array] = value
        return Tensor(result)

    def __add__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data + other.data)
        else:
            return Tensor(self.data + other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data - other.data)
        else:
            return Tensor(self.data - other)

    def __rsub__(self, other):
        if isinstance(other, Tensor):
            return Tensor(other.data - self.data)
        return Tensor(other - self.data)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data * other.data)
        else:
            return Tensor(self.data * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data / other.data)
        else:
            return Tensor(self.data / other)

    def __rtruediv__(self, other):
        if isinstance(other, Tensor):
            return Tensor(other.data / self.data)
        return Tensor(other / self.data)

    def _validate_matmul_shapes(self, other):
        if not isinstance(other, Tensor):
            raise TypeError(
                f"Matrix multiplication requires Tensor, got {type(other).__name__}\n"
                f"  ❌ Cannot perform: Tensor @ {type(other).__name__}\n"
                f"  💡 Matrix multiplication (@) only works between two Tensors\n"
                f"  🔧 Wrap your data: Tensor({other}) @ other_tensor"
            )
        if len(self.shape) == 0 or len(other.shape) == 0:
            raise ValueError(
                f"Matrix multiplication requires at least 1D tensors\n"
                f"  ❌ Got shapes: {self.shape} @ {other.shape}\n"
                f"  💡 Scalars (0D tensors) cannot be matrix-multiplied; use * for element-wise\n"
                f"  🔧 Reshape scalar to 1D: tensor.reshape(1) or use tensor * scalar"
            )
        if len(self.shape) >= 2 and len(other.shape) >= 2:
            if self.shape[-1] != other.shape[-2]:
                raise ValueError(
                    f"Matrix multiplication shape mismatch: {self.shape} @ {other.shape}\n"
                    f"  ❌ Inner dimensions don't match: {self.shape[-1]} vs {other.shape[-2]}\n"
                    f"  💡 For A @ B, A's last dim must equal B's second-to-last dim\n"
                    f"  🔧 Try: other.transpose() to get shape {other.shape[::-1]}, or reshape self"
                )

    def matmul(self, other):
        self._validate_matmul_shapes(other)

        a = self.data
        b = other.data

        if len(a.shape) == 2 and len(b.shape) == 2:
            M, K = a.shape
            _, N = b.shape
            result_data = np.zeros((M, N), dtype=a.dtype)

            for i in range(M):
                for j in range(N):
                    result_data[i, j] = np.dot(a[i, :], b[:, j])
        else:
            result_data = np.matmul(a, b)

        return Tensor(result_data)

    def __matmul__(self, other):
        return self.matmul(other)

    def __getitem__(self, key):
        result_data = self.data[key]
        if not isinstance(result_data, np.ndarray):
            result_data = np.array(result_data)
        return Tensor(result_data)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            new_shape = tuple(shape[0])
        else:
            new_shape = shape
        if -1 in new_shape:
            if new_shape.count(-1) > 1:
                raise ValueError(
                    f"Cannot reshape {self.shape} with multiple unknown dimensions\n"
                    f"  ❌ Found {new_shape.count(-1)} dimensions set to -1 in {new_shape}\n"
                    f"  💡 Only one dimension can be inferred; others must be specified\n"
                    f"  🔧 Replace all but one -1 with explicit sizes (total elements: {self.size})"
                )
            known_size = 1
            unknown_idx = new_shape.index(-1)
            for i, dim in enumerate(new_shape):
                if i != unknown_idx:
                    known_size *= dim
            if self.size % known_size != 0:
                raise ValueError(
                    f"Cannot infer -1 dimension: {self.size} elements is not "
                    f"divisible by the known dimensions product {known_size}\n"
                    f"  ❌ {self.size} % {known_size} = {self.size % known_size}\n"
                    f"  💡 The -1 dimension must be a whole number"
                )
            unknown_dim = self.size // known_size
            new_shape = list(new_shape)
            new_shape[unknown_idx] = unknown_dim
            new_shape = tuple(new_shape)
        if np.prod(new_shape) != self.size:
            target_size = int(np.prod(new_shape))
            raise ValueError(
                f"Cannot reshape {self.shape} to {new_shape}\n"
                f"  ❌ Element count mismatch: {self.size} elements vs {target_size} elements\n"
                f"  💡 Reshape preserves data, so total elements must stay the same\n"
                f"  🔧 Use -1 to infer a dimension: reshape(-1, {new_shape[-1] if len(new_shape) > 0 else 1}) lets NumPy calculate"
            )
        reshaped_data = np.reshape(self.data, new_shape)
        return Tensor(reshaped_data)

    def transpose(self, dim0=None, dim1=None):
        if dim0 is None and dim1 is None:
            if len(self.shape) < 2:
                return Tensor(self.data.copy())
            else:
                axes = list(range(len(self.shape)))
                axes[-2], axes[-1] = axes[-1], axes[-2]
                transposed_data = np.transpose(self.data, axes)
        else:
            if dim0 is None or dim1 is None:
                provided = f"dim0={dim0}" if dim1 is None else f"dim1={dim1}"
                missing = "dim1" if dim1 is None else "dim0"
                raise ValueError(
                    f"Transpose requires both dimensions to be specified\n"
                    f"  ❌ Got {provided}, but {missing} is None\n"
                    f"  💡 Either provide both dims or neither (default swaps last two)\n"
                    f"  🔧 Use transpose({dim0 if dim0 is not None else 0}, {dim1 if dim1 is not None else 1}) or just transpose()"
                )
            axes = list(range(len(self.shape)))
            axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
            transposed_data = np.transpose(self.data, axes)
        return Tensor(transposed_data)

    def sum(self, axis=None, keepdims=False):
        result = np.sum(self.data, axis=axis, keepdims=keepdims)
        return Tensor(result)

    def mean(self, axis=None, keepdims=False):
        result = np.mean(self.data, axis=axis, keepdims=keepdims)
        return Tensor(result)

    def max(self, axis=None, keepdims=False):
        result = np.max(self.data, axis=axis, keepdims=keepdims)
        return Tensor(result)


def _show(label, tensor, *, data=True):
    parts = [f"shape={tensor.shape}", f"size={tensor.size}", f"ndim={tensor.ndim}", f"dtype={tensor.dtype}"]
    if data:
        parts.insert(0, f"data={tensor.data}")
    print(f"   {label}：{', '.join(parts)}")


def test_unit_tensor_creation():
    print("🧪 单元测试：张量创建...")

    scalar = Tensor(5.0)
    assert scalar.data == 5.0
    assert scalar.shape == ()
    assert scalar.size == 1
    assert scalar.dtype == np.float32
    _show("标量 Tensor(5.0)", scalar)

    vector = Tensor([1, 2, 3])
    assert np.array_equal(vector.data, np.array([1, 2, 3], dtype=np.float32))
    assert vector.shape == (3,)
    assert vector.size == 3
    _show("向量 Tensor([1, 2, 3])", vector)

    matrix = Tensor([[1, 2], [3, 4]])
    assert np.array_equal(matrix.data, np.array([[1, 2], [3, 4]], dtype=np.float32))
    assert matrix.shape == (2, 2)
    assert matrix.size == 4
    _show("矩阵 Tensor([[1, 2], [3, 4]])", matrix)

    tensor_3d = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert tensor_3d.shape == (2, 2, 2)
    assert tensor_3d.size == 8
    _show("三维 Tensor", tensor_3d)

    assert scalar.ndim == 0, "Scalar should be 0-dimensional"
    assert vector.ndim == 1, "Vector should be 1-dimensional"
    assert matrix.ndim == 2, "Matrix should be 2-dimensional"
    assert tensor_3d.ndim == 3, "3D tensor should be 3-dimensional"

    assert scalar.numel() == 1, "Scalar has 1 element"
    assert vector.numel() == 3, "Vector has 3 elements"
    assert matrix.numel() == 4, "2x2 matrix has 4 elements"
    print(f"   numel：标量={scalar.numel()}，向量={vector.numel()}，矩阵={matrix.numel()}，三维={tensor_3d.numel()}")

    contig = matrix.contiguous()
    assert np.array_equal(contig.data, matrix.data)
    assert contig.data is not matrix.data, "contiguous() should return a copy"
    print(f"   contiguous()：data 相同={np.array_equal(contig.data, matrix.data)}，是否同一块内存={contig.data is matrix.data}")

    print("✅ 张量创建通过！")

def test_unit_arithmetic_operations():
    print("🧪 单元测试：算术运算...")

    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])
    _show("a", a)
    _show("b", b)

    result = a + b
    assert np.array_equal(result.data, np.array([5, 7, 9], dtype=np.float32))
    _show("a + b", result)

    result = a + 10
    assert np.array_equal(result.data, np.array([11, 12, 13], dtype=np.float32))
    _show("a + 10", result)

    result = 10 + a
    assert np.array_equal(result.data, np.array([11, 12, 13], dtype=np.float32))
    _show("10 + a", result)

    matrix = Tensor([[1, 2], [3, 4]])
    vector = Tensor([10, 20])
    result = matrix + vector
    expected = np.array([[11, 22], [13, 24]], dtype=np.float32)
    assert np.array_equal(result.data, expected)
    _show("矩阵", matrix)
    _show("向量", vector)
    _show("矩阵 + 向量（广播）", result)

    predictions = Tensor(np.ones((4, 3)))
    targets_good = Tensor(np.zeros((4, 3)))
    targets_bad = Tensor(np.zeros((3,)))
    assert predictions.shape == targets_good.shape, "Matching shapes — safe"
    assert predictions.shape != targets_bad.shape, (
        f"Shape mismatch: {predictions.shape} vs {targets_bad.shape}. "
        f"NumPy broadcasts silently — this is almost always a bug in ML code."
    )
    print(f"   预测 shape={predictions.shape}，正确目标 shape={targets_good.shape}（对齐）")
    print(f"   错误目标 shape={targets_bad.shape}（缺 batch 维，广播会静默发生）")

    result = b - a
    assert np.array_equal(result.data, np.array([3, 3, 3], dtype=np.float32))
    _show("b - a", result)

    result = a * 2
    assert np.array_equal(result.data, np.array([2, 4, 6], dtype=np.float32))
    _show("a * 2", result)

    result = 2 * a
    assert np.array_equal(result.data, np.array([2, 4, 6], dtype=np.float32))
    _show("2 * a", result)

    result = b / 2
    assert np.array_equal(result.data, np.array([2.0, 2.5, 3.0], dtype=np.float32))
    _show("b / 2", result)

    result = 12 / b
    assert np.allclose(result.data, np.array([3.0, 12.0 / 5.0, 2.0], dtype=np.float32))
    _show("12 / b", result)

    result = 10 - a
    assert np.array_equal(result.data, np.array([9, 8, 7], dtype=np.float32))
    _show("10 - a", result)

    normalized = (a - 2) / 2
    expected = np.array([-0.5, 0.0, 0.5], dtype=np.float32)
    assert np.allclose(normalized.data, expected)
    _show("(a - 2) / 2", normalized)

    print("✅ 算术运算通过！")

def test_unit_validate_matmul_shapes():
    print("🧪 单元测试：矩阵乘法形状校验...")

    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[5, 6], [7, 8]])
    a._validate_matmul_shapes(b)
    print(f"   合法：{a.shape} @ {b.shape}")

    c = Tensor([[1, 2, 3]])
    d = Tensor([[1], [2], [3]])
    c._validate_matmul_shapes(d)
    print(f"   合法：{c.shape} @ {d.shape}")

    try:
        a._validate_matmul_shapes([[1, 2], [3, 4]])
        assert False, "Should have raised TypeError for non-Tensor"
    except TypeError as e:
        assert "requires Tensor" in str(e)
        assert "list" in str(e)
        print(f"   非法（右侧不是 Tensor）：捕获 {type(e).__name__}")

    try:
        scalar = Tensor(5.0)
        scalar._validate_matmul_shapes(a)
        assert False, "Should have raised ValueError for 0D tensor"
    except ValueError as e:
        assert "at least 1D" in str(e)
        print(f"   非法（0 维标量 {scalar.shape} @ {a.shape}）：捕获 {type(e).__name__}")

    try:
        incompatible_a = Tensor([[1, 2]])
        incompatible_b = Tensor([[1], [2], [3]])
        incompatible_a._validate_matmul_shapes(incompatible_b)
        assert False, "Should have raised ValueError for shape mismatch"
    except ValueError as e:
        assert "Inner dimensions don't match" in str(e)
        assert "2 vs 3" in str(e)
        print(f"   非法（内维不匹配 {incompatible_a.shape} @ {incompatible_b.shape}）：捕获 {type(e).__name__}")

    print("✅ 矩阵乘法形状校验通过！")

def test_unit_matrix_multiplication():
    print("🧪 单元测试：矩阵乘法...")

    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[5, 6], [7, 8]])
    result = a.matmul(b)
    expected = np.array([[19, 22], [43, 50]], dtype=np.float32)
    assert np.array_equal(result.data, expected)
    _show("A", a)
    _show("B", b)
    _show("A.matmul(B)", result)

    c = Tensor([[1, 2, 3], [4, 5, 6]])
    d = Tensor([[7, 8], [9, 10], [11, 12]])
    result = c.matmul(d)
    expected = np.array([[58, 64], [139, 154]], dtype=np.float32)
    assert np.array_equal(result.data, expected)
    _show("C (2×3)", c)
    _show("D (3×2)", d)
    _show("C.matmul(D)", result)

    matrix = Tensor([[1, 2, 3], [4, 5, 6]])
    vector = Tensor([1, 2, 3])
    result = matrix.matmul(vector)
    expected = np.array([14, 32], dtype=np.float32)
    assert np.array_equal(result.data, expected)
    _show("矩阵 (2×3)", matrix)
    _show("向量 (3,)", vector)
    _show("矩阵 @ 向量", result)

    result_at = a @ b
    assert np.array_equal(result_at.data, np.array([[19, 22], [43, 50]], dtype=np.float32))
    _show("A @ B", result_at)

    print("✅ 矩阵乘法通过！")

def test_unit_shape_manipulation():
    print("🧪 单元测试：形状操作...")

    tensor = Tensor([1, 2, 3, 4, 5, 6])
    _show("原向量", tensor)
    reshaped = tensor.reshape(2, 3)
    assert reshaped.shape == (2, 3)
    expected = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    assert np.array_equal(reshaped.data, expected)
    _show("reshape(2, 3)", reshaped)

    reshaped2 = tensor.reshape((3, 2))
    assert reshaped2.shape == (3, 2)
    expected2 = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    assert np.array_equal(reshaped2.data, expected2)
    _show("reshape((3, 2))", reshaped2)

    auto_reshaped = tensor.reshape(2, -1)
    assert auto_reshaped.shape == (2, 3)
    _show("reshape(2, -1)", auto_reshaped)

    try:
        tensor.reshape(2, 2)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Element count mismatch" in str(e)
        assert "6 elements vs 4 elements" in str(e)
        print(f"   reshape(2, 2) 非法：6 个元素装不进 4，捕获 {type(e).__name__}")

    matrix = Tensor([[1, 2, 3], [4, 5, 6]])
    transposed = matrix.transpose()
    assert transposed.shape == (3, 2)
    expected = np.array([[1, 4], [2, 5], [3, 6]], dtype=np.float32)
    assert np.array_equal(transposed.data, expected)
    _show("矩阵 (2, 3)", matrix)
    _show("transpose() → (3, 2)", transposed)

    vector = Tensor([1, 2, 3])
    vector_t = vector.transpose()
    assert np.array_equal(vector.data, vector_t.data)
    _show("一维 transpose（应不变）", vector_t)

    tensor_3d = Tensor(np.arange(24).reshape(2, 3, 4))
    swapped = tensor_3d.transpose(0, 2)
    assert swapped.shape == (4, 3, 2), (
        f"transpose(0, 2) on (2,3,4) should give (4,3,2), got {swapped.shape}"
    )
    for i in range(2):
        for j in range(3):
            for k in range(4):
                assert swapped.data[k, j, i] == tensor_3d.data[i, j, k], (
                    f"Data mismatch at [{k},{j},{i}]: expected {tensor_3d.data[i,j,k]}, "
                    f"got {swapped.data[k,j,i]}"
                )
    _show("三维原张量", tensor_3d)
    _show("transpose(0, 2)", swapped)

    batch_images = Tensor(rng.random((2, 3, 4)))
    flattened = batch_images.reshape(2, -1)
    assert flattened.shape == (2, 12)
    _show("batch (2, 3, 4)", batch_images, data=False)
    _show("flatten → (2, 12)", flattened, data=False)

    print("✅ 形状操作通过！")

def test_unit_reduction_operations():
    print("🧪 单元测试：归约运算...")

    matrix = Tensor([[1, 2, 3], [4, 5, 6]])
    _show("矩阵", matrix)

    total = matrix.sum()
    assert total.data == 21.0
    assert total.shape == ()
    _show("sum()", total)

    col_sum = matrix.sum(axis=0)
    expected_col = np.array([5, 7, 9], dtype=np.float32)
    assert np.array_equal(col_sum.data, expected_col)
    assert col_sum.shape == (3,)
    _show("sum(axis=0) 列和", col_sum)

    row_sum = matrix.sum(axis=1)
    expected_row = np.array([6, 15], dtype=np.float32)
    assert np.array_equal(row_sum.data, expected_row)
    assert row_sum.shape == (2,)
    _show("sum(axis=1) 行和", row_sum)

    avg = matrix.mean()
    assert np.isclose(avg.data, 3.5)
    assert avg.shape == ()
    _show("mean()", avg)

    col_mean = matrix.mean(axis=0)
    expected_mean = np.array([2.5, 3.5, 4.5], dtype=np.float32)
    assert np.allclose(col_mean.data, expected_mean)
    _show("mean(axis=0)", col_mean)

    maximum = matrix.max()
    assert maximum.data == 6.0
    assert maximum.shape == ()
    _show("max()", maximum)

    row_max = matrix.max(axis=1)
    expected_max = np.array([3, 6], dtype=np.float32)
    assert np.array_equal(row_max.data, expected_max)
    _show("max(axis=1)", row_max)

    sum_keepdims = matrix.sum(axis=1, keepdims=True)
    assert sum_keepdims.shape == (2, 1)
    expected_keepdims = np.array([[6], [15]], dtype=np.float32)
    assert np.array_equal(sum_keepdims.data, expected_keepdims)
    _show("sum(axis=1, keepdims=True)", sum_keepdims)

    tensor_3d = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    spatial_mean = tensor_3d.mean(axis=(1, 2))
    assert spatial_mean.shape == (2,)
    _show("三维", tensor_3d)
    _show("mean(axis=(1, 2)) 空间均值", spatial_mean)

    print("✅ 归约运算通过！")

def analyze_memory_layout():
    print("📊 分析内存访问模式...")
    print("=" * 60)

    size = 2000
    matrix = Tensor(rng.random((size, size)))

    print(f"\n测试矩阵：{size}×{size}（{matrix.size * BYTES_PER_FLOAT32 / MB_TO_BYTES:.1f} MB）")
    print(f"   shape={matrix.shape}，size={matrix.size}，dtype={matrix.dtype}")
    print("-" * 60)

    print("\n测试 1：按行访问（对缓存友好）")
    start = time.time()
    row_sums = []
    for i in range(size):
        row_sum = matrix.data[i, :].sum()
        row_sums.append(row_sum)
    row_time = time.time() - start
    print(f"   耗时：{row_time*1000:.1f}ms，前 3 个行和={row_sums[:3]}")
    print("   访问模式：顺序访问（顺着内存布局）")

    print("\n测试 2：按列访问（对缓存不友好）")
    start = time.time()
    col_sums = []
    for j in range(size):
        col_sum = matrix.data[:, j].sum()
        col_sums.append(col_sum)
    col_time = time.time() - start
    print(f"   耗时：{col_time*1000:.1f}ms，前 3 个列和={col_sums[:3]}")
    print(f"   访问模式：跨步访问（每个元素跳 {size * BYTES_PER_FLOAT32} 字节）")

    slowdown = col_time / row_time
    print("\n" + "=" * 60)
    print("📊 性能影响：")
    print(f"   变慢倍数：{slowdown:.2f}×（慢 {col_time/row_time:.1f} 倍）")
    print(f"   缓存未命中导致约 {(slowdown-1)*100:.0f}% 的性能损失")

    print("\n💡 要点：")
    print("   1. 内存布局很重要：行优先（C 风格）存储是连续的")
    print("   2. 缓存行大约 64 字节：按行访问会把邻近元素「顺便」载入")
    print("   3. 按列访问容易缓存未命中：每次都要从 DRAM 重新加载")
    print(f"   4. 算法仍是 O(n)，墙钟时间却差了 {slowdown:.1f} 倍！")

    print("\n🚀 实际影响：")
    print("   • 图像处理库会选用特定内存格式，以利用缓存")
    print("   • 矩阵乘法常用分块，把数据切进缓存大小的块里")
    print(f"   • transpose 较贵（{slowdown:.1f}×）：非连续视图，缓存局部性差")
    print("   • 硬件优化库会顺着内存布局来提升性能")

    print("\n" + "=" * 60)

def test_module():
    print("🧪 正在运行模块集成测试")
    print("=" * 50)

    print("运行单元测试...")
    test_unit_tensor_creation()
    test_unit_arithmetic_operations()
    test_unit_validate_matmul_shapes()
    test_unit_matrix_multiplication()
    test_unit_shape_manipulation()
    test_unit_reduction_operations()

    print("\n运行集成场景...")

    print("🧪 集成测试：两阶段线性变换...")

    x = Tensor([[1, 2, 3], [4, 5, 6]])

    W1 = Tensor([[0.1, 0.2, 0.3, 0.4],
                 [0.5, 0.6, 0.7, 0.8],
                 [0.9, 1.0, 1.1, 1.2]])
    b1 = Tensor([0.1, 0.2, 0.3, 0.4])
    _show("输入 x", x)
    _show("W1", W1)
    _show("b1", b1)

    hidden = x.matmul(W1) + b1
    assert hidden.shape == (2, 4), f"Expected (2, 4), got {hidden.shape}"
    _show("hidden = x @ W1 + b1", hidden)

    W2 = Tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
    b2 = Tensor([0.1, 0.2])
    _show("W2", W2)
    _show("b2", b2)

    output = hidden.matmul(W2) + b2
    assert output.shape == (2, 2), f"Expected (2, 2), got {output.shape}"

    assert not np.isnan(output.data).any(), "Output contains NaN values"
    assert np.isfinite(output.data).all(), "Output contains infinite values"
    _show("output = hidden @ W2 + b2", output)

    print("✅ 两阶段线性变换通过！")

    print("🧪 集成测试：复杂形状操作...")
    data = Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    _show("原数据", data)

    tensor_3d = data.reshape(2, 2, 3)
    assert tensor_3d.shape == (2, 2, 3)
    _show("reshape(2, 2, 3)", tensor_3d)

    pooled = tensor_3d.mean(axis=(1, 2))
    assert pooled.shape == (2,), f"Expected (2,), got {pooled.shape}"
    _show("空间均值 mean(axis=(1, 2))", pooled)

    flattened = tensor_3d.reshape(2, -1)
    assert flattened.shape == (2, 6)
    _show("flatten reshape(2, -1)", flattened)

    transposed = tensor_3d.transpose()
    assert transposed.shape == (2, 3, 2)
    _show("transpose() 交换最后两维", transposed)

    print("✅ 复杂形状操作通过！")

    print("🧪 集成测试：广播边界情况...")

    scalar = Tensor(5.0)
    vector = Tensor([1, 2, 3])
    result = scalar + vector
    expected = np.array([6, 7, 8], dtype=np.float32)
    assert np.array_equal(result.data, expected)
    _show("标量", scalar)
    _show("向量", vector)
    _show("标量 + 向量", result)

    matrix = Tensor([[1, 2], [3, 4]])
    vec = Tensor([10, 20])
    result = matrix + vec
    expected = np.array([[11, 22], [13, 24]], dtype=np.float32)
    assert np.array_equal(result.data, expected)
    _show("矩阵", matrix)
    _show("向量", vec)
    _show("矩阵 + 向量", result)

    print("✅ 广播边界情况通过！")

    print("\n" + "=" * 50)
    print("🎉 全部测试通过！模块可以迁入。")

def demo_tensor():
    print("🎯 你的 Tensor 可以像 NumPy 一样用")
    print("=" * 45)

    a = Tensor(np.array([1, 2, 3]))
    b = Tensor(np.array([4, 5, 6]))
    _show("a", a)
    _show("b", b)

    tensor_sum = a + b
    tensor_prod = a * b

    np_sum = np.array([1, 2, 3]) + np.array([4, 5, 6])
    np_prod = np.array([1, 2, 3]) * np.array([4, 5, 6])

    print(f"   Tensor a + b：{tensor_sum.data}")
    print(f"   NumPy  a + b：{np_sum}")
    print(f"   是否一致：{np.allclose(tensor_sum.data, np_sum)}")

    print(f"   Tensor a * b：{tensor_prod.data}")
    print(f"   NumPy  a * b：{np_prod}")
    print(f"   是否一致：{np.allclose(tensor_prod.data, np_prod)}")

    print("\n✨ 你的 Tensor 已与 NumPy 对齐，可以继续做后续模块！")

if __name__ == "__main__":
    test_module()
    print("\n")
    analyze_memory_layout()
    print("\n")
    demo_tensor()
