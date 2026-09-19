> **文档来源**：`src/06_autograd/06_autograd.py` 中的 Markdown 教学说明（英文原版，本模块尚无 `*_zh.py`）。  
> 下文保留原模块一级标题；组件主题下的从属小节已下调一级，避免同级标题重复。

# Module 06: Autograd - The Gradient Engine

Welcome to Module 06! Today you'll awaken the gradient engine and unlock automatic differentiation.

## 🔗 Prerequisites & Progress
**You've Built**: Tensor operations, activations, layers, losses, and DataLoader
**You'll Build**: The autograd system that computes gradients automatically
**You'll Enable**: Learning! Training! The ability to optimize neural networks!

**Connection Map**:
```
Modules 01-05 → Autograd → Optimizers → Training
(forward pass)  (Module 06)  (Module 07)  (Module 08)
```

## 🎯 Learning Objectives
By the end of this module, you will:
1. **Enhance Tensor** with automatic differentiation capabilities
2. **Build computation graphs** that track operations for gradient flow
3. **Implement backward()** method for reverse-mode differentiation
4. **Create Function classes** for operation-specific gradient rules
5. **Test gradient correctness** with mathematical validation

**CRITICAL**: This module enhances the existing Tensor class - no new wrapper classes needed!

## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/06_autograd/autograd_dev.py`
**Building Side:** Code exports to `tinytorch.core.autograd`

```python
# How to use this module:
from tinytorch.core.autograd import Function, enable_autograd
```

**Why this matters:**
- **Learning:** Complete autograd system enabling automatic differentiation
- **Production:** PyTorch-style computational graph and backward pass
- **Consistency:** All gradient operations in core.autograd
- **Integration:** Enhances existing Tensor without breaking anything

Let's get started!

## 💡 Introduction: What is Automatic Differentiation?

Automatic differentiation (autograd) is the magic that makes neural networks learn. Instead of manually computing gradients for every parameter, autograd tracks operations and automatically computes gradients via the chain rule.

### The Challenge
In previous modules, you implemented layers and loss functions. To train a model, you need:
```
Loss = f(W₃, f(W₂, f(W₁, x)))
∂Loss/∂W₁ = ?  ∂Loss/∂W₂ = ?  ∂Loss/∂W₃ = ?
```

Manual gradient computation becomes impossible for complex models with millions of parameters.

### The Solution: Computational Graphs
```
Forward Pass:  x → Linear₁ → ReLU → Linear₂ → Loss
Backward Pass: ∇x ← ∇Linear₁ ← ∇ReLU ← ∇Linear₂ ← ∇Loss
```

**Complete Autograd Process Visualization:**
```
┌─ FORWARD PASS ─────────────────────────────────────────────────┐
│                                                                │
│ x ──┬── W₁ ──┐                                                 │
│     │        ├──[Linear₁]──→ z₁ ──[ReLU]──→ a₁ ──┬── W₂ ──┐    │
│     └── b₁ ──┘                               │        ├─→ Loss │
│                                              └── b₂ ──┘        │
│                                                                │
└─ COMPUTATION GRAPH BUILT ──────────────────────────────────────┘
                             │
                             ▼
┌─ BACKWARD PASS ─────────────────────────────────────────────┐
│                                                             │
│∇x ←┬← ∇W₁ ←┐                                                │
│    │       ├←[Linear₁]←─ ∇z₁ ←[ReLU]← ∇a₁ ←┬← ∇W₂ ←┐        │
│    └← ∇b₁ ←┘                             │       ├← ∇Loss   │
│                                          └← ∇b₂ ←┘          │
│                                                             │
└─ GRADIENTS COMPUTED ────────────────────────────────────────┘

Key Insight: Each [operation] stores how to compute its backward pass.
The chain rule automatically flows gradients through the entire graph.
```

Each operation records how to compute its backward pass. The chain rule connects them all.

## 📐 Foundations: The Chain Rule in Action

### Mathematical Foundation
For composite functions: f(g(x)), the derivative is:
```
df/dx = (df/dg) × (dg/dx)
```

### Computational Graph Example
```
Simple computation: L = (x * y + 5)²

Forward Pass:
  x=2 ──┐
        ├──[×]──→ z=6 ──[+5]──→ w=11 ──[²]──→ L=121
  y=3 ──┘

Backward Pass (Chain Rule in Action):
  ∂L/∂x = ∂L/∂w × ∂w/∂z × ∂z/∂x
        = 2w  ×  1  ×  y
        = 2(11) × 1 × 3 = 66

  ∂L/∂y = ∂L/∂w × ∂w/∂z × ∂z/∂y
        = 2w  ×  1  ×  x
        = 2(11) × 1 × 2 = 44

Gradient Flow Visualization:
  ∇x=66 ←──┐
           ├──[×]←── ∇z=22 ←──[+]←── ∇w=22 ←──[²]←── ∇L=1
  ∇y=44 ←──┘
```

### Memory Layout During Backpropagation
```
Computation Graph Memory Structure:
┌─────────────────────────────────────────────────────────┐
│ Forward Pass (stored for backward)                      │
├─────────────────────────────────────────────────────────┤
│ Node 1: x=2 (leaf, requires_grad=True) │ grad: None→66  │
│ Node 2: y=3 (leaf, requires_grad=True) │ grad: None→44  │
│ Node 3: z=x*y (MulFunction)            │ grad: None→22  │
│         saved: (x=2, y=3)              │ inputs: [x,y]  │
│ Node 4: w=z+5 (AddFunction)            │ grad: None→22  │
│         saved: (z=6, 5)                │ inputs: [z]    │
│ Node 5: L=w² (PowFunction)             │ grad: 1        │
│         saved: (w=11)                  │ inputs: [w]    │
└─────────────────────────────────────────────────────────┘

Memory Cost: 2× parameters (data + gradients) + graph overhead
```

## 🏗️ Implementation: Building the Autograd Engine

Let's implement the autograd system step by step. We'll enhance the existing Tensor class and create supporting infrastructure.

### The Function Architecture

Every differentiable operation needs two things:
1. **Forward pass**: Compute the result
2. **Backward pass**: Compute gradients for inputs

```
Function Class Design:
┌─────────────────────────────────────┐
│ Function (Base Class)               │
├─────────────────────────────────────┤
│ • saved_tensors    ← Store data     │
│ • apply()          ← Compute grads  │
└─────────────────────────────────────┘
          ↑
    ┌─────┴─────┬─────────┬──────────┐
    │           │         │          │
┌───▼────┐ ┌────▼───┐ ┌───▼────┐ ┌───▼────┐
│  Add   │ │  Mul   │ │ Matmul │ │  Sum   │
│Backward│ │Backward│ │Backward│ │Backward│
└────────┘ └────────┘ └────────┘ └────────┘
```

Each operation inherits from Function and implements specific gradient rules.

### Function Base Class - The Foundation of Autograd

The Function class is the foundation that makes autograd possible. Every differentiable operation (addition, multiplication, etc.) inherits from this class.

**Why Functions Matter:**
- They remember inputs needed for backward pass
- They implement gradient computation via apply()
- They connect to form computation graphs
- They enable the chain rule to flow gradients

**The Pattern:**
```
Forward:  inputs → Function.forward() → output
Backward: grad_output → Function.apply() → grad_inputs
```

This pattern enables the chain rule to flow gradients through complex computations.

### Operation Functions - Implementing Gradient Rules

Now we'll implement specific operations that compute gradients correctly. Each operation has mathematical rules for how gradients flow backward.

**Gradient Flow Visualization:**
```
Addition (z = a + b):
    ∂z/∂a = 1    ∂z/∂b = 1

    a ──┐           grad_a ←──┐
        ├─[+]─→ z          ├─[+]←── grad_z
    b ──┘           grad_b ←──┘

Multiplication (z = a * b):
    ∂z/∂a = b    ∂z/∂b = a

    a ──┐           grad_a = grad_z * b
        ├─[×]─→ z
    b ──┘           grad_b = grad_z * a

Matrix Multiplication (Z = A @ B):
    ∂Z/∂A = grad_Z @ B.T
    ∂Z/∂B = A.T @ grad_Z

    A ──┐           grad_A = grad_Z @ B.T
        ├─[@]─→ Z
    B ──┘           grad_B = A.T @ grad_Z
```

Each operation stores the inputs it needs for computing gradients.

#### 🔍 Understanding Broadcasting in Gradients

Before implementing gradient operations, we need to understand a critical challenge:
**Broadcasting in Forward Pass vs. Gradient Reduction in Backward Pass**

#### The Broadcasting Problem

NumPy automatically broadcasts tensors of different shapes during forward operations:

```
Forward Pass (Broadcasting):
┌─────────────────────────────────────────────────────────────┐
│ Example: Adding bias to batched data                        │
│                                                              │
│ x:    (32, 128)  ← Batch of 32 samples, 128 features        │
│ bias: (128,)     ← Just 128 features (no batch dimension)   │
│                                                              │
│ Forward: y = x + bias                                        │
│          NumPy broadcasts bias from (128,) to (32, 128)     │
│          Result shape: (32, 128)                             │
└─────────────────────────────────────────────────────────────┘

Backward Pass (Gradient Reduction):
┌─────────────────────────────────────────────────────────────┐
│ grad_output: (32, 128)  ← Gradient from upstream            │
│                                                              │
│ grad_x:    (32, 128)    ← Same shape as x ✓                 │
│ grad_bias: (128,)       ← Must match bias shape!            │
│                                                              │
│ Problem: grad_output is (32, 128) but bias is (128,)        │
│ Solution: Sum gradients over batch dimension                │
│           grad_bias = grad_output.sum(axis=0)               │
│           Result: (128,) ✓                                   │
└─────────────────────────────────────────────────────────────┘
```

#### Why Gradient Reduction is Necessary

**Mathematical Intuition:**
When bias broadcasts to multiple samples, it contributes to each sample's loss.
The total gradient w.r.t. bias is the SUM of gradients from all samples.

**Concrete Example:**
```
x = [[1, 2],      bias = [0.1, 0.2]
     [3, 4]]

Forward:
  y[0] = [1, 2] + [0.1, 0.2] = [1.1, 2.2]  ← bias[0]=0.1 affects sample 0
  y[1] = [3, 4] + [0.1, 0.2] = [3.1, 4.2]  ← bias[0]=0.1 affects sample 1

Backward:
  grad_output = [[1, 1],   ← ∂Loss/∂y[0]
                 [1, 1]]   ← ∂Loss/∂y[1]
  
  grad_bias[0] = ∂Loss/∂bias[0] 
               = ∂Loss/∂y[0,0] + ∂Loss/∂y[1,0]  ← Chain rule: sum contributions
               = 1 + 1 = 2
  
  grad_bias = [2, 2]  ← Sum over batch dimension
```

#### Broadcasting Scenarios to Handle

```
Scenario 1: Batch Dimension Broadcasting
  x:    (32, 128)  +  bias: (128,)    →  y: (32, 128)
  grad: (32, 128)  →  grad_bias = sum(grad, axis=0) → (128,)

Scenario 2: Multiple Dimension Broadcasting  
  x:    (32, 10, 5)  +  y: (10, 1)    →  z: (32, 10, 5)
  grad: (32, 10, 5)  →  
    1. Sum over extra dim: sum(grad, axis=0) → (10, 5)
    2. Sum over singleton: sum(grad, axis=2, keepdims=True) → (10, 1)

Scenario 3: Scalar Broadcasting
  x:    (32, 128)  +  scalar: ()      →  y: (32, 128)
  grad: (32, 128)  →  grad_scalar = sum(grad) → scalar
```

#### The General Algorithm

To reduce gradient back to original input shape:
1. **Remove extra dimensions**: Sum over leading dimensions that weren't in input
2. **Collapse singleton dimensions**: Sum over dimensions where input had size 1
3. **Preserve shape**: Use keepdims=True when collapsing to maintain dimensionality

This ensures gradients flow correctly regardless of broadcasting patterns!

#### 🔬 Unit Test: Broadcast Gradient Reduction

This test validates our gradient reduction helper handles all broadcasting scenarios.

**What we're testing**: Correct shape reduction for gradients after broadcasting
**Why it matters**: Wrong gradient shapes cause crashes or silent bugs in training
**Expected**: Reduced gradients match original tensor shapes for all broadcasting patterns

### AddBackward - Gradient Rules for Addition

Addition is the simplest gradient operation: gradients flow unchanged to both inputs.

**Mathematical Principle:**
```
If z = a + b, then:
∂z/∂a = 1  (gradient of z w.r.t. a)
∂z/∂b = 1  (gradient of z w.r.t. b)

By chain rule:
∂Loss/∂a = ∂Loss/∂z × ∂z/∂a = grad_output × 1 = grad_output
∂Loss/∂b = ∂Loss/∂z × ∂z/∂b = grad_output × 1 = grad_output
```

**Broadcasting Challenge:**
When tensors have different shapes, NumPy broadcasts automatically in forward pass,
but we must "unbroadcast" gradients in backward pass to match original shapes.

### MulBackward - Gradient Rules for Element-wise Multiplication

Element-wise multiplication follows the product rule of calculus.

**Mathematical Principle:**
```
If z = a * b (element-wise), then:
∂z/∂a = b  (gradient w.r.t. a equals the other input)
∂z/∂b = a  (gradient w.r.t. b equals the other input)

By chain rule:
∂Loss/∂a = grad_output * b
∂Loss/∂b = grad_output * a
```

**Visual Example:**
```
Forward:  a=[2,3] * b=[4,5] = z=[8,15]
Backward: grad_z=[1,1]
          grad_a = grad_z * b = [1,1] * [4,5] = [4,5]
          grad_b = grad_z * a = [1,1] * [2,3] = [2,3]
```

### SubBackward - Gradient Rules for Subtraction

Subtraction is mathematically simple but important for operations like normalization.

**Mathematical Principle:**
```
If z = a - b, then:
∂z/∂a = 1
∂z/∂b = -1
```

**Key Insight:** Gradient flows forward to the first operand, but **negated** to the second.
This is crucial for operations like centering data (`x - mean`).

### DivBackward - Gradient Rules for Division

Division requires the quotient rule from calculus.

**Mathematical Principle:**
```
If z = a / b, then:
∂z/∂a = 1/b
∂z/∂b = -a/b²
```

**Quotient Rule:** For z = f/g, dz = (g·df - f·dg)/g²

### MatmulBackward - Gradient Rules for Matrix Multiplication

Matrix multiplication has more complex gradient rules based on matrix calculus.

**Mathematical Principle:**
```
If Z = A @ B (matrix multiplication), then:
∂Z/∂A = grad_Z @ B.T
∂Z/∂B = A.T @ grad_Z
```

**Why These Rules Work:**
```
For element Z[i,j] = Σ_k A[i,k] * B[k,j]
∂Z[i,j]/∂A[i,k] = B[k,j]  ← This gives us grad_Z @ B.T
∂Z[i,j]/∂B[k,j] = A[i,k]  ← This gives us A.T @ grad_Z
```

**Dimension Analysis:**
```
Forward:  A(m×k) @ B(k×n) = Z(m×n)
Backward: grad_Z(m×n) @ B.T(n×k) = grad_A(m×k) ✓
          A.T(k×m) @ grad_Z(m×n) = grad_B(k×n) ✓
```

### SumBackward - Gradient Rules for Reduction Operations

Sum operations reduce tensor dimensions, so gradients must be broadcast back.

**Mathematical Principle:**
```
If z = sum(a), then ∂z/∂a[i] = 1 for all i
Gradient is broadcasted from scalar result back to input shape.
```

**Gradient Broadcasting Examples:**
```
Case 1: Full sum
  Forward:  a=[1,2,3] → sum() → z=6 (scalar)
  Backward: grad_z=1 → broadcast → grad_a=[1,1,1]

Case 2: Axis sum
  Forward:  a=[[1,2],[3,4]] → sum(axis=0) → z=[4,6]
  Backward: grad_z=[1,1] → broadcast → grad_a=[[1,1],[1,1]]
```

#### 🔬 Unit Test: Function Classes

This test validates our Function classes compute gradients correctly.

**What we're testing**: Forward and backward passes for each operation
**Why it matters**: These are the building blocks of autograd
**Expected**: Correct gradients that satisfy mathematical definitions

#### 🔬 Unit Test: Broadcasting in Gradients
This test validates that gradient reduction works correctly when operations
involve broadcasting. This is crucial for real-world training where bias terms,
normalization parameters, and other operations broadcast over batches.

**What we're testing**: Gradient shape correctness with broadcasting
**Why it matters**: Without proper reduction, bias gradients would have wrong shapes
**Expected**: Gradients match original tensor shapes after reduction

## 🏗️ Enhancing Tensor with Autograd Capabilities

Now we'll enhance the existing Tensor class to use these gradient functions and build computation graphs automatically.

**Computation Graph Formation:**
```
Before Autograd:             After Autograd:
  x → operation → y           x → [Function] → y
                                     ↓
                               Stores operation
                               for backward pass
```

**The Enhancement Strategy:**
1. **Add backward() method** - Triggers gradient computation
2. **Enhance operations** - Replace simple ops with gradient-tracking versions
3. **Track computation graphs** - Each tensor remembers how it was created
4. **Maintain compatibility** - All existing code continues to work

**Critical Design Decision:**
We enhance the EXISTING Tensor class rather than creating a new one.
This means:
- ✅ All previous modules continue working unchanged
- ✅ No import changes needed
- ✅ Gradients are "opt-in" via requires_grad=True
- ✅ No confusion between Tensor types

### The enable_autograd() Function

This function is the magic that brings gradients to life! It enhances the existing Tensor class with autograd capabilities by:

1. **Monkey-patching operations** - Replaces `__add__`, `__mul__`, etc. with gradient-aware versions
2. **Adding backward() method** - Implements reverse-mode automatic differentiation
3. **Maintaining compatibility** - All existing code continues to work unchanged

**The Pattern:**
```
Original: x + y → simple addition
Enhanced: x + y → addition + gradient tracking (if requires_grad=True)
```

This approach follows PyTorch 2.0 style - clean, modern, and educational.

### Helper: Numerically Stable Softmax

Computing softmax naively as `exp(x) / sum(exp(x))` overflows for large values.
The fix is to subtract the maximum value first, which is mathematically equivalent
but numerically stable.

```
Naive (overflows):     softmax(x) = exp(x) / sum(exp(x))
Stable (safe):         softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

Why it works:
  exp(x - max(x)) / sum(exp(x - max(x)))
= exp(x) * exp(-max(x)) / (sum(exp(x)) * exp(-max(x)))
= exp(x) / sum(exp(x))
```

This helper is used by CrossEntropyBackward to convert logits to probabilities.

#### Unit Test: Stable Softmax Helper

**What we're testing**: Numerically stable softmax computation
**Why it matters**: Unstable softmax causes NaN/Inf in cross-entropy gradients
**Expected**: Probabilities sum to 1.0, correct values, no overflow on large inputs

### Helper: One-Hot Encoding

Converts class indices to one-hot vectors. This is needed by the cross-entropy
gradient formula: `grad = softmax - one_hot`.

```
Indices: [0, 2, 1]  with 3 classes

One-hot:
  [[1, 0, 0],    ← class 0
   [0, 0, 1],    ← class 2
   [0, 1, 0]]    ← class 1
```

#### Unit Test: One-Hot Encoding Helper

**What we're testing**: Conversion from class indices to one-hot vectors
**Why it matters**: Incorrect one-hot encoding produces wrong cross-entropy gradients
**Expected**: Each row has exactly one 1.0, at the correct class position

### CrossEntropyBackward - Gradient Rules for Cross-Entropy Loss

The cross-entropy gradient combines three sub-computations:
1. **Stable softmax**: Convert raw logits to probabilities
2. **One-hot encoding**: Convert target indices to indicator vectors
3. **Gradient formula**: `(softmax - one_hot) / batch_size`

```
Logits: [2.0, 1.0, 0.1]     Target: class 0

Step 1 - Softmax:   [0.659, 0.242, 0.099]
Step 2 - One-hot:   [1.000, 0.000, 0.000]
Step 3 - Gradient:  [0.659-1, 0.242-0, 0.099-0] / 1 = [-0.341, 0.242, 0.099]
```

The gradient is simply "how far off each class probability is from the target."
This is one of the most elegant results in machine learning.

## ⚠️ DANGER: In-Place Operations Break Autograd

**THIS IS THE MOST COMMON SILENT FAILURE IN TINYTORCH!**

### Critical Rule: Never Modify Tensors In-Place When requires_grad=True

**WRONG ❌ - This Corrupts the Gradient Graph:**
```python
x = Tensor([1, 2, 3], requires_grad=True)
y = x * 2
x.data[0] = 999  # ❌ CORRUPTS GRADIENT GRAPH WITHOUT ERROR!
y.backward()     # ❌ Wrong gradients or crash
```

**RIGHT ✅ - Create New Tensors Instead:**
```python
x = Tensor([1, 2, 3], requires_grad=True)
y = x * 2
x = Tensor([999, 2, 3], requires_grad=True)  # ✅ New tensor, safe
y.backward()  # ✅ Correct gradients
```

### Why This Breaks Everything

Autograd records operations on the **original tensor values**. When you modify `.data` directly:

1. **Forward pass** records: "y = x * 2" where x = [1, 2, 3]
2. **You corrupt**: x.data[0] = 999, so x = [999, 2, 3]
3. **Backward pass** uses: corrupted x values, causing wrong gradients or crashes

**The computation graph becomes inconsistent** - forward used [1, 2, 3], backward uses [999, 2, 3].

### Common In-Place Operations to AVOID

```python
# ❌ FORBIDDEN - Direct index assignment
x.data[0] = value
x.data[:, 0] = values
x.data[mask] = values

# ❌ FORBIDDEN - In-place arithmetic
x.data += other
x.data *= scalar
x.data -= value

# ❌ FORBIDDEN - NumPy in-place operations
np.fill(x.data, value)
np.add(x.data, other, out=x.data)
x.data.fill(value)

# ✅ CORRECT - Create new tensors
x = x + other              # Creates new tensor
x = Tensor(x.data + other) # Explicit new tensor
x = Tensor([new_values])   # Complete replacement
```

### Real-World Example: Parameter Update Gone Wrong

```python
# ❌ WRONG - This is a common mistake in custom optimizers
W = Tensor([[0.5, 0.3]], requires_grad=True)
y = x.matmul(W.T)
loss = compute_loss(y, target)
loss.backward()

# Student writes custom optimizer:
W.data -= 0.01 * W.grad  # ❌ CORRUPTS GRAPH! Next forward pass is broken!

# ✅ CORRECT - Create new parameter tensor
W = Tensor(W.data - 0.01 * W.grad, requires_grad=True)  # ✅ Safe
```

### How to Debug In-Place Corruption

If your gradients look wrong or you get mysterious errors:

1. **Search your code** for `.data[` assignments
2. **Search for** in-place operators: `+=`, `-=`, `*=`, `/=` on `.data`
3. **Check custom functions** that modify tensors
4. **Verify** all parameter updates create new tensors

### Why PyTorch Has torch.no_grad()

PyTorch explicitly disables gradient tracking during parameter updates to allow safe in-place operations.
TinyTorch now supports this via the `no_grad` context manager:

```python
# PyTorch pattern
with torch.no_grad():
    W -= 0.01 * W.grad  # Safe inside no_grad context

# TinyTorch equivalent
from tinytorch.core.autograd import no_grad
with no_grad():
    W -= 0.01 * W.grad  # No graph built, safe for parameter updates
```

**How it works**: `no_grad()` sets a global flag that all tracked operations check.
When the flag is off, operations skip graph construction entirely -- the result tensor
will have `requires_grad=False` regardless of its inputs.

### Memory Impact

**Question**: "Doesn't creating new tensors waste memory?"

**Answer**: Gradient tracking already stores intermediate tensors for backprop. Creating new tensors is negligible compared to the computation graph memory overhead. Correctness > premature perf.

**Bottom Line**: If a tensor has `requires_grad=True`, treat it as **immutable**. Always create new tensors instead of modifying in-place.

---

### 🔬 Unit Test: Tensor Autograd Enhancement

This test validates our enhanced Tensor class computes gradients correctly.

**What we're testing**: Gradient computation and chain rule implementation
**Why it matters**: This is the core of automatic differentiation
**Expected**: Correct gradients for various operations and computation graphs

## 📊 Systems Analysis: Computation Graph Memory

Let's understand ONE key systems concept: **computation graph memory overhead**.

This single analysis reveals why gradient tracking is expensive and why frameworks make gradient tracking opt-in.

## 🧪 Module Integration Test

Final validation that everything works together correctly.

## 🤔 ML Systems Reflection Questions

Before we wrap up, reflect on these systems-level questions. Use only knowledge from Modules 01-05 (no forward references to concepts you haven't learned yet).

### Question 1: Computational Graph Memory
**Scenario**: A 10-layer neural network processes a single sample. Each layer performs matrix multiplication (matmul) and addition (bias).

**Question**: How much memory does the computation graph use compared to just storing the weights?

**Consider**:
- What tensors must be saved during forward pass for backward pass?
- If weights take 10MB total, estimate graph memory overhead
- When is the graph freed?

---

### Question 2: Gradient Accumulation
**Scenario**: A weight matrix is shared between two computation paths in a network (like a tied-weights architecture).

**Question**: Why does gradient accumulation (`grad = grad + new_grad`) save memory during training? What's the trade-off?

**Consider**:
- What happens if you process a large batch all at once vs. multiple smaller batches?
- Memory usage: storing intermediate activations vs. recomputing forward passes
- Training behavior: does gradient accumulation change what the model learns?

---

### Question 3: Backward Pass Cost
**Scenario**: A forward pass through a 3-layer MLP takes 10ms.

**Question**: Is the backward pass faster, slower, or the same speed as the forward pass? Why?

**Consider**:
- Operations in forward pass: matmul, activation, addition
- Operations in backward pass: matmul (for gradients), element-wise multiplication (chain rule)
- Number of matmul operations: forward vs. backward
- Memory access patterns: reading vs. writing gradients

**Hint**: Think about matrix multiplication gradients:
```
Forward:  y = x @ W       (one matmul)
Backward: grad_x = grad_y @ W.T     (one matmul)
          grad_W = x.T @ grad_y     (another matmul)
```

---

### Question 4: Graph Retention
**Scenario**: You're training a language model that processes sequences of varying lengths.

**Question**: When should you call `.zero_grad()`? What happens if you forget?

**Consider**:
- Gradient accumulation behavior (Question 2)
- Memory growth over multiple iterations
- Training correctness: what values do parameters see?

**Example**:
```python
for batch in dataloader:
    # Should zero_grad() go here?
    loss = model(batch)
    loss.backward()
    optimizer.step()
    # Or should zero_grad() go here?
```

---

### Question 5: Production Pattern
**Scenario**: PyTorch and TensorFlow use `requires_grad` flags instead of always tracking gradients for every tensor.

**Question**: Why? What's the performance benefit of making gradient tracking opt-in?

**Consider**:
- Memory: What gets stored when requires_grad=True vs. False?
- Compute: What operations are skipped when requires_grad=False?
- Typical model: What percentage of tensors need gradients?
  - Inputs (data): requires_grad = ?
  - Weights: requires_grad = ?
  - Intermediate activations: requires_grad = ?
  - Targets (labels): requires_grad = ?

**Hint**: In a typical training loop, think about:
- How many tensors are created per forward pass?
- How many of those tensors are actually parameters that need updates?
- What's the memory multiplier for gradient tracking?

---

### Reflection Prompts

After answering these questions, consider:
1. **Which surprised you most?** What behavior was counterintuitive?
2. **What trade-offs exist?** Memory vs. compute? Simplicity vs. efficiency?
3. **How does this connect to Module 01?** Why did we include requires_grad, grad, and backward() from the start?
4. **What production patterns emerged?** What choices would you make differently for a research prototype vs. production system?

These questions prepare you for Module 07 (Optimizers), where you'll use these gradients to actually update parameters and train models!

## ⭐ Aha Moment: Gradients Flow Automatically

**What you built:** An autograd engine that computes gradients through computation graphs.

**Why it matters:** Before autograd, you had to derive and code gradients by hand for every
operation—error-prone and tedious. Your engine does this automatically! When you call
`backward()`, gradients flow from the loss back through every operation to every parameter.

This is the magic behind deep learning. PyTorch, TensorFlow, and JAX all have autograd
engines at their core. You just built one yourself!

## 🚀 MODULE SUMMARY: Autograd Engine

Congratulations! You've built the gradient engine that makes neural networks learn!

### Key Accomplishments
- **Enhanced Tensor class** with backward() method (no new wrapper classes!)
- **Built computation graph tracking** for automatic differentiation
- **Implemented Function classes** (Add, Mul, Matmul, Sum) with correct gradients
- **Created enable_autograd()** function that activates gradients globally
- **Tested complex multi-layer** computation graphs with gradient propagation
- **All tests pass** (validated by `test_module()`)

### Systems Insights Discovered
- **Memory overhead**: Computation graphs store tensors for backward pass (2x memory)
- **Gradient accumulation**: Allows processing large batches in smaller chunks
- **Backward pass cost**: Approximately same as forward pass (similar number of matmuls)
- **Graph retention**: Must call zero_grad() to prevent gradient accumulation across iterations

### Ready for Next Steps
Your autograd implementation enables optimization!
Export with: `tito module complete 06`

**Next**: Module 07 will add optimizers (SGD, Adam) that use these gradients to actually train neural networks!
