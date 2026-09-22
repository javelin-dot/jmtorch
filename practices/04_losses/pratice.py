import numpy as np
rng = np.random.default_rng(7)
from typing import Optional

# __file__ 是本脚本的路径；resolve() 化为绝对路径，两个 .parent 回到 practices/ 目录。
# Path 的 / 是路径拼接，不是数字除法；下划线开头表示作者把名字当作模块内部变量。
_tensor_path = Path(__file__).resolve().parent.parent / "01_tensor" / "practice.py"
# 按指定文件路径建立加载说明；"tensor_practice" 是给这个模块起的名字，此时尚未执行文件。
_spec = importlib.util.spec_from_file_location("tensor_practice", _tensor_path)
# 根据加载说明先创建模块对象；这里的“模块”可以理解为一个装着函数和类的文件对象。
_tensor_mod = importlib.util.module_from_spec(_spec)
# 方括号 ["tensor_practice"] 用字符串作键，把模块对象记入 sys.modules。
sys.modules["tensor_practice"] = _tensor_mod
# 真正执行 01 阶段的 practice.py，执行完后模块对象才拥有那里定义的 Tensor 等名字。
_spec.loader.exec_module(_tensor_mod)
# 从模块中取出 Tensor 类，在本文件中也用 Tensor 这个名字；它不是 PyTorch 的 torch.Tensor。
Tensor = _tensor_mod.Tensor

from .activations import ReLU
from .layers import Linear

# Constants for numerical stability
EPSILON = 1e-7  # Small value to prevent log(0) and numerical instability

# %% ../../modules/04_losses/losses.ipynb #a141f26d
def log_softmax(x: Tensor, dim: int = -1) -> Tensor:
    """
    Compute log-softmax with numerical stability.

    TODO: Implement numerically stable log-softmax using the log-sum-exp trick

    APPROACH:
    1. Find maximum along dimension (for stability)
    2. Subtract max from input (prevents overflow)
    3. Compute log(sum(exp(shifted_input)))
    4. Return input - max - log_sum_exp

    EXAMPLE:
    >>> logits = Tensor([[1.0, 2.0, 3.0], [0.1, 0.2, 0.9]])
    >>> result = log_softmax(logits, dim=-1)
    >>> print(result.shape)
    (2, 3)

    HINT: Use np.max(x.data, axis=dim, keepdims=True) to preserve dimensions
    """
    ### BEGIN SOLUTION
    # Step 1: Find max along dimension for numerical stability
    max_vals = np.max(x.data, axis=dim, keepdims=True)

    # Step 2: Subtract max to prevent overflow
    shifted = x.data - max_vals

    # Step 3: Compute log(sum(exp(shifted)))
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=dim, keepdims=True))

    # Step 4: Return log_softmax = input - max - log_sum_exp
    result = x.data - max_vals - log_sum_exp

    return Tensor(result)
    ### END SOLUTION

# %% ../../modules/04_losses/losses.ipynb #4d316fbd
class MSELoss:
    """Mean Squared Error loss for regression tasks."""

    def __init__(self):
        """Initialize MSE loss function."""
        pass

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Compute mean squared error between predictions and targets.

        TODO: Implement MSE loss calculation

        APPROACH:
        1. Compute difference: predictions - targets
        2. Square the differences: diff²
        3. Take mean across all elements

        EXAMPLE:
        >>> loss_fn = MSELoss()
        >>> predictions = Tensor([1.0, 2.0, 3.0])
        >>> targets = Tensor([1.5, 2.5, 2.8])
        >>> loss = loss_fn(predictions, targets)
        >>> print(f"MSE Loss: {loss.data:.4f}")
        MSE Loss: 0.1800

        HINTS:
        - Use (predictions.data - targets.data) for element-wise difference
        - Square with **2 or np.power(diff, 2)
        - Use np.mean() to average over all elements
        """
        ### BEGIN SOLUTION
        # Step 1: Compute element-wise difference
        diff = predictions.data - targets.data

        # Step 2: Square the differences
        squared_diff = diff ** 2

        # Step 3: Take mean across all elements
        mse = np.mean(squared_diff)

        return Tensor(mse)
        ### END SOLUTION

    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Allows the loss function to be called like a function."""
        return self.forward(predictions, targets)

    def backward(self) -> Tensor:
        """
        Compute gradients (placeholder — gradient computation is separate).

        For now, this is a stub that students can ignore.
        """
        pass

# %% ../../modules/04_losses/losses.ipynb #6efdd415
class CrossEntropyLoss:
    """Cross-entropy loss for multi-class classification."""

    def __init__(self):
        """Initialize cross-entropy loss function."""
        pass

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Compute cross-entropy loss between logits and target class indices.

        TODO: Implement cross-entropy loss with numerical stability

        APPROACH:
        1. Compute log-softmax of logits (numerically stable)
        2. Check every target index is a real class, 0 <= t < num_classes,
           and raise ValueError if any is not
        3. Select log-probabilities for correct classes
        4. Return negative mean of selected log-probabilities

        EXAMPLE:
        >>> loss_fn = CrossEntropyLoss()
        >>> logits = Tensor([[2.0, 1.0, 0.1], [0.5, 1.5, 0.8]])  # 2 samples, 3 classes
        >>> targets = Tensor([0, 1])  # First sample is class 0, second is class 1
        >>> loss = loss_fn(logits, targets)
        >>> print(f"Cross-Entropy Loss: {loss.data:.4f}")

        HINTS:
        - Use log_softmax() for numerical stability
        - targets.data.astype(int) ensures integer indices
        - num_classes is logits.shape[-1]; validate before indexing, because
          NumPy would let a negative target silently select the wrong class
          and would raise a bare IndexError for one that is too large
        - Use np.arange(batch_size) for row indexing: log_probs[np.arange(batch_size), targets]
        - Return negative mean: -np.mean(selected_log_probs)
        """
        ### BEGIN SOLUTION
        # Step 1: Compute log-softmax for numerical stability
        log_probs = log_softmax(logits, dim=-1)

        # Step 2: Select log-probabilities for correct classes
        batch_size = logits.shape[0]
        num_classes = logits.shape[-1]
        target_indices = targets.data.astype(int)

        out_of_range = (target_indices < 0) | (target_indices >= num_classes)
        if np.any(out_of_range):
            bad_values = np.unique(target_indices[out_of_range])
            raise ValueError(
                f"CrossEntropyLoss target index out of range: {bad_values.tolist()}\n"
                f"  Valid range for {num_classes} classes is [0, {num_classes - 1}]"
            )

        # Select correct class log-probabilities using advanced indexing
        selected_log_probs = log_probs.data[np.arange(batch_size), target_indices]

        # Step 3: Return negative mean (cross-entropy is negative log-likelihood)
        cross_entropy = -np.mean(selected_log_probs)

        return Tensor(cross_entropy)
        ### END SOLUTION

    def __call__(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Allows the loss function to be called like a function."""
        return self.forward(logits, targets)

    def backward(self) -> Tensor:
        """
        Compute gradients (placeholder — gradient computation is separate).

        For now, this is a stub that students can ignore.
        """
        pass

# %% ../../modules/04_losses/losses.ipynb #9b512f01
class BinaryCrossEntropyLoss:
    """Binary cross-entropy loss for binary classification."""

    def __init__(self):
        """Initialize binary cross-entropy loss function."""
        pass

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Compute binary cross-entropy loss.

        TODO: Implement binary cross-entropy with numerical stability

        APPROACH:
        1. Clamp predictions to avoid log(0) and log(1)
        2. Compute: -(targets * log(predictions) + (1-targets) * log(1-predictions))
        3. Return mean across all samples

        EXAMPLE:
        >>> loss_fn = BinaryCrossEntropyLoss()
        >>> predictions = Tensor([0.9, 0.1, 0.7, 0.3])  # Probabilities between 0 and 1
        >>> targets = Tensor([1.0, 0.0, 1.0, 0.0])      # Binary labels
        >>> loss = loss_fn(predictions, targets)
        >>> print(f"Binary Cross-Entropy Loss: {loss.data:.4f}")

        HINTS:
        - Use np.clip(predictions.data, 1e-7, 1-1e-7) to prevent log(0)
        - Binary cross-entropy: -(targets * log(preds) + (1-targets) * log(1-preds))
        - Use np.mean() to average over all samples
        """
        ### BEGIN SOLUTION
        # Step 1: Clamp predictions to avoid numerical issues with log(0) and log(1)
        eps = EPSILON
        clamped_preds = np.clip(predictions.data, eps, 1 - eps)

        # Step 2: Compute binary cross-entropy
        # BCE = -(targets * log(preds) + (1-targets) * log(1-preds))
        log_preds = np.log(clamped_preds)
        log_one_minus_preds = np.log(1 - clamped_preds)

        bce_per_sample = -(targets.data * log_preds + (1 - targets.data) * log_one_minus_preds)

        # Step 3: Return mean across all samples
        bce_loss = np.mean(bce_per_sample)

        return Tensor(bce_loss)
        ### END SOLUTION

    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Allows the loss function to be called like a function."""
        return self.forward(predictions, targets)

    def backward(self) -> Tensor:
        """
        Compute gradients (placeholder — gradient computation is separate).

        For now, this is a stub that students can ignore.
        """
        pass
