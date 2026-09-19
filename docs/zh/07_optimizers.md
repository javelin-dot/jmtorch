> **文档来源**：`src/07_optimizers/07_optimizers.py` 中的 Markdown 教学说明（英文原版，本模块尚无 `*_zh.py`）。  
> 下文保留原模块一级标题；组件主题下的从属小节已下调一级，避免同级标题重复。

# Module 07: Optimizers - Sophisticated Learning Algorithms

Welcome to Module 07! You'll build optimizers that enable neural networks to learn from gradients using sophisticated algorithms.

## 🔗 Prerequisites & Progress
**You've Built**: Tensor with gradients (Modules 01-06)
**You'll Build**: SGD, Adam, and AdamW optimizers with sophisticated momentum and adaptive learning
**You'll Enable**: Modern optimization algorithms that power state-of-the-art neural networks

**Connection Map**:
```
Gradients → Optimizers → Training
(Module 06)  (Module 07)  (Module 08)
```

## 🎯 Learning Objectives
By the end of this module, you will:
1. Implement SGD with momentum for stable gradient descent
2. Build Adam optimizer with adaptive learning rates
3. Create AdamW optimizer with decoupled weight decay
4. Understand memory and computational trade-offs in optimization algorithms

Let's get started!

## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `src/07_optimizers/07_optimizers.py`
**Building Side:** Code exports to `tinytorch.core.optimizers`

```python
# How to use this module:
from tinytorch.core.optimizers import SGD, Adam, AdamW
```

**Why this matters:**
- **Learning:** Complete optimization system for modern neural network training
- **Production:** Proper organization like PyTorch's torch.optim with all optimization algorithms together
- **Consistency:** All optimization logic and parameter updating in core.optimizers
- **Integration:** Works seamlessly with gradients from Module 06 for complete training capability

## 🏗️ Implementation: Building Optimizers

Now we'll implement each optimizer step by step, following the pattern: understand the algorithm → implement it → test it immediately. Each optimizer builds on the foundation of the previous one.

### Implementation Strategy

```
Optimizer Base Class
    ↓
SGD (foundation algorithm)
    ↓
SGD + Momentum (reduce oscillations)
    ↓
Adam (adaptive learning rates)
    ↓
AdamW (proper weight decay)
```

### 🏗️ Gradient Extraction - Handling Tensor vs NumPy Gradients

When autograd computes gradients, they can arrive as either a `Tensor` object
(wrapping a NumPy array in `.data`) or as a raw NumPy array. Every optimizer
needs to normalize this before doing math on the gradient.

```
param.grad
    │
    ├── Tensor?  ──→  return grad.data  (unwrap the NumPy array)
    │
    └── ndarray? ──→  return grad       (already NumPy, use directly)
```

This helper lives in the base `Optimizer` class so SGD, Adam, and AdamW
all share the same extraction logic.

#### 🧪 Unit Test: Gradient Extraction

This test validates that `_extract_gradient` correctly handles both
Tensor-wrapped gradients and raw NumPy array gradients.

**What we're testing**: Gradient normalization across storage formats
**Why it matters**: Every optimizer needs raw NumPy data for update math
**Expected**: NumPy array output regardless of input format

#### 🧪 Unit Test: Base Optimizer

This test validates our base Optimizer class works correctly.

**What we're testing**: Parameter validation and zero_grad functionality
**Why it matters**: Foundation for all specific optimizer implementations
**Expected**: Proper parameter storage and gradient clearing

#### 🧪 Unit Test: SGD Optimizer

This test validates our SGD implementation works correctly.

**What we're testing**: SGD updates with and without momentum
**Why it matters**: Core optimization algorithm used in neural network training
**Expected**: Correct parameter updates following SGD formulas

## 🏗️ Adam - Adaptive Moment Estimation

Adam solves a fundamental problem with SGD: different parameters often need different learning rates. Think of tuning a complex system where some knobs need gentle adjustments and others need bold changes.

### The Parameter Scaling Problem

Consider a neural network with both first layer weights and output weights:

```
Parameter Sensitivity Landscape:

    first_layer_weight              output_weight
           ↑                               ↑
           |                               |
           |  🐌 gentle slope              |  ⛰️ steep cliff
           |  (needs big steps)            |  (needs tiny steps)
           |                               |
        ━━━●━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━●━━━→

Same learning rate = disaster!
• Small LR: output weights learn fast, first layer crawls
• Large LR: first layer learns well, output weights explode
```

### Adam's Adaptive Solution

Adam automatically adjusts learning rates by tracking two statistics:

```
1. MOMENTUM (first moment): "Which way am I usually going?"
   m = 0.9 * old_direction + 0.1 * current_gradient

   Visualization:
   old: →→→→
   new:     ↗️
   m:   →→→↗️  (weighted average)

2. SCALE (second moment): "How big are my steps usually?"
   v = 0.999 * old_scale + 0.001 * (current_gradient)²

   Big gradients → bigger v → smaller effective steps
   Small gradients → smaller v → bigger effective steps

3. ADAPTIVE UPDATE:
   step = momentum / √scale
   param = param - learning_rate * step
```

### Bias Correction: The Cold Start Problem

Adam starts with m=0 and v=0, which creates a bias toward zero initially:

```
Without bias correction:    With bias correction:

Step 1: m = 0.9*0 + 0.1*g    Step 1: m̂ = m / (1-0.9¹) = m / 0.1
       = 0.1*g (too small!)           = g (correct!)

Step 2: m = 0.9*0.1*g + 0.1*g Step 2: m̂ = m / (1-0.9²) = m / 0.19
       = 0.19*g (still small)         ≈ g (better!)
```

**Key Insight:** Adam is like having an automatic transmission that adjusts gear ratios for each parameter individually.

### 🏗️ Moment Updates - EMA and Bias Correction

Adam tracks two running statistics per parameter: a first moment (mean of
gradients) and a second moment (mean of squared gradients). Both use
exponential moving averages (EMA) and need bias correction because they
start from zero.

```
grad_data ──→ m = β₁ * m + (1-β₁) * grad       (direction EMA)
         └──→ v = β₂ * v + (1-β₂) * grad²      (magnitude EMA)

             m̂ = m / (1 - β₁^t)                 (bias-corrected mean)
             v̂ = v / (1 - β₂^t)                 (bias-corrected variance)
```

This helper isolates the EMA + bias correction math so that `step()`
only has to compose: extract gradient, update moments, apply update.

#### 🧪 Unit Test: Adam Moment Updates

This test validates that `_update_moments` correctly computes exponential
moving averages and bias correction for Adam's first and second moments.

**What we're testing**: EMA computation and bias correction math
**Why it matters**: Incorrect moments produce wrong adaptive learning rates
**Expected**: Bias-corrected moments match hand-calculated values

### 🏗️ Adam Step - Composing Gradient Extraction, Moments, and Update

The `step()` method now composes three focused operations:
1. `_extract_gradient()` -- normalize Tensor/ndarray gradient to NumPy
2. `_update_moments()` -- EMA + bias correction for adaptive scaling
3. Parameter update -- `param -= lr * m_hat / (sqrt(v_hat) + eps)`

```
For each parameter:
    param.grad ──→ _extract_gradient() ──→ grad_data
                                               │
                   (optional weight decay)  ←──┘
                                               │
                   _update_moments(i, grad) ──→ (m_hat, v_hat)
                                                     │
                   param.data -= lr * m_hat / (√v_hat + ε)
```

#### 🧪 Unit Test: Adam Optimizer

This test validates our Adam implementation works correctly.

**What we're testing**: Adam updates with adaptive learning rates and bias correction
**Why it matters**: Most popular optimizer for modern neural networks
**Expected**: Correct parameter updates following Adam formulas

## 🏗️ AdamW - Adam with Decoupled Weight Decay

AdamW fixes a subtle but important bug in Adam's weight decay implementation. The bug affects how regularization interacts with adaptive learning rates.

### The Adam Weight Decay Bug

In standard Adam, weight decay is added to gradients before the adaptive scaling:

```
Adam's approach (problematic):
1. gradient = computed_gradient + weight_decay * parameter
2. m = β₁ * m + (1-β₁) * gradient
3. v = β₂ * v + (1-β₂) * gradient²
4. step = m / √v
5. parameter = parameter - learning_rate * step

Problem: Weight decay gets "adapted" by the learning rate scaling!
```

### Why This Matters

Weight decay should be a consistent regularization force, but Adam makes it inconsistent:

```
Parameter Update Comparison:

Large gradients → small adaptive LR → weak weight decay effect
Small gradients → large adaptive LR → strong weight decay effect

This is backwards! We want consistent regularization.
```

### AdamW's Fix: Decoupled Weight Decay

AdamW separates gradient-based updates from weight decay:

```
AdamW's approach (correct):
1. m = β₁ * m + (1-β₁) * pure_gradient  ← NO weight decay here
2. v = β₂ * v + (1-β₂) * pure_gradient²
3. step = m / √v
4. parameter = parameter - learning_rate * step           ← gradient update
5. parameter = parameter * (1 - lr * weight_decay)       ← separate decay

Result: Consistent regularization independent of gradient magnitudes!
```

Note: Step 5 uses the "decoupled" form where weight decay is scaled by the
learning rate (`1 - lr * weight_decay`), not the simpler `(1 - weight_decay)`.
This ensures the regularization strength scales consistently with the gradient
update step size.

### Visual Comparison

```
Adam weight decay:                  AdamW weight decay:

gradient ──┐                        gradient ──→ adaptive ──→ param
           ├─→ adaptive ──→ param                  update
weight ────┘   scaling
decay
                                    weight ─────────→ param
                                    decay           shrinkage

Coupled (inconsistent)          Decoupled (consistent)
```

**Key Insight:** AdamW treats optimization and regularization as separate, independent processes, leading to better training dynamics and generalization.

### 🏗️ AdamW Moment Updates - Same EMA, Different Context

AdamW uses identical moment update math as Adam (EMA + bias correction).
The critical difference is that AdamW passes **pure gradients** to moment
updates -- weight decay is applied separately to parameters, not mixed
into the gradient signal.

```
AdamW flow:
    grad_data ──→ _update_moments(i, grad_data)  ← pure gradient, no decay
                        │
                   (m_hat, v_hat)
                        │
    param -= lr * m_hat / (√v_hat + ε)           ← gradient update
    param *= (1 - lr * weight_decay)              ← separate decay step
```

#### 🧪 Unit Test: AdamW Moment Updates

This test validates that AdamW's `_update_moments` computes the same EMA
and bias correction as Adam (the math is identical; the decoupling
difference is in how `step()` uses these results).

**What we're testing**: EMA computation for AdamW
**Why it matters**: Correct moments are needed for adaptive learning rates
**Expected**: Same bias-corrected values as Adam for identical inputs

### 🏗️ AdamW Step - Decoupled Weight Decay Composition

AdamW's `step()` composes the same helpers as Adam, but with one critical
difference: weight decay is applied **after** the gradient update, directly
to the parameter values, rather than being mixed into the gradient.

```
For each parameter:
    param.grad ──→ _extract_gradient() ──→ grad_data
                                               │
                   _update_moments(i, grad) ──→ (m_hat, v_hat)
                                                     │
                   param.data -= lr * m_hat / (√v_hat + ε)   ← gradient step
                   param.data *= (1 - lr * weight_decay)      ← separate decay
```

#### 🧪 Unit Test: AdamW Optimizer

This test validates our AdamW implementation with decoupled weight decay.

**What we're testing**: AdamW updates with proper weight decay decoupling
**Why it matters**: State-of-the-art optimizer for modern neural networks
**Expected**: Correct separation of gradient updates and weight decay

## 🔧 Integration: Bringing It Together

Now let's see how our optimizers perform in realistic scenarios. We'll compare their behavior on the same optimization problem to understand their different characteristics.

### Optimizer Behavior Comparison

Each optimizer takes a different approach to the same problem:

```
Optimization Problem: Find minimum of f(x) = x²

SGD approach:        Adam approach:        AdamW approach:
  ↓                    ↓                     ↓
 x ──→ minimize       x ──→ minimize       x ──→ minimize
  ↑                    ↑                     ↑
fixed LR           adaptive LR          adaptive LR + decay
```

## 📊 Systems Analysis: Optimizer Performance and Memory

Different optimizers have very different resource requirements. Understanding these trade-offs is crucial for production ML systems.

### Memory Usage Patterns

```
Optimizer Memory Requirements (per parameter):

SGD:           Adam/AdamW:
┌────────┐     ┌────────┐
│ param  │     │ param  │
├────────┤     ├────────┤
│momentum│     │   m    │ ← first moment
└────────┘     ├────────┤
               │   v    │ ← second moment
               └────────┘

2× memory       3× memory
```

### Computational Complexity

```
Per-step Operations:

SGD:                     Adam:
• 1 multiplication       • 3 multiplications
• 1 addition            • 4 additions
• 1 subtraction         • 1 subtraction
                        • 1 square root
                        • 1 division

O(n) simple ops         O(n) complex ops
```

## 🧪 Module Integration Test

Final validation that everything works together correctly.

## 🤔 ML Systems Reflection Questions

Answer these to deepen your understanding of optimizer operations and their systems implications:

### 1. Memory vs Performance
**Question**: You've implemented SGD (2x memory) and Adam (3x memory). For a model with 10 billion parameters at float32 (4 bytes each):

**Consider**:
- How much total memory does each optimizer require?
- At what model size does Adam's extra 50% memory overhead become prohibitive?
- What real-world constraints might force you to choose SGD over Adam?

**Calculate**:
- Parameters: 10 x 10^9
- Bytes per float32: 4
- SGD memory (2x): ___________GB
- Adam memory (3x): ___________GB

---

### 2. Learning Rate Sensitivity
**Question**: SGD uses a fixed learning rate for all parameters, while Adam adapts per-parameter.

**Consider**:
- Why might Adam converge faster on problems with parameters at different scales?
- When might SGD's uniform learning rate actually be an advantage?
- How does momentum in SGD relate to Adam's first moment estimation?

**Real-world context**: Deep neural networks often have early layers with very different gradient magnitudes than later layers. Adam's adaptive rates help balance these naturally.

---

### 3. Optimizer State Management
**Question**: Adam and AdamW maintain momentum buffers (m, v) that persist across training steps.

**Consider**:
- What happens to these buffers when you checkpoint during training?
- If you resume training with different hyperparameters, should you restore the old buffers?
- How does optimizer state affect training when you restart from a checkpoint?

**Think about**:
- Checkpoint size: Adam stores 2 additional tensors per parameter
- Resume behavior: Warm vs cold restart implications
- Resume behavior: What happens to momentum buffers if you change learning rate mid-training?

---

### 4. Weight Decay Trade-offs
**Question**: AdamW decouples weight decay from gradient updates.

**Consider**:
- Why does Adam's coupled weight decay behave inconsistently?
- In what scenarios would AdamW's consistent regularization matter most?
- How does weight decay interact with learning rate schedules?

**Real-world context**: The AdamW paper showed that proper decoupling leads to better generalization on ImageNet and language models.

---

### 5. Production Scale: Memory Requirements
**Question**: For training a GPT-scale model with 1 billion parameters, calculate the memory requirements:

**Calculate**:
- Parameters: 1 x 10^9
- Bytes per float32: 4
- Parameter memory: ___________GB

**With Adam optimizer (3x memory)**:
- Total: ___________GB

**Real-world implications**:
- Why do we need multiple GPUs for training large models?
- Why is understanding optimizer memory crucial for choosing hardware?
- When would you choose SGD over Adam despite slower convergence?

---

### Bonus Challenge: Optimization Analysis

**Scenario**: You're training a deep neural network and observing the following:
- Loss decreases rapidly for first 1000 steps
- Loss plateaus between steps 1000-5000
- Loss suddenly increases at step 5000

**Questions**:
1. What might cause the plateau? How would momentum help?
2. What might cause the sudden increase? Is this an optimizer issue?
3. How would you diagnose whether this is a learning rate problem vs. data problem?
4. Would switching from Adam to AdamW help in this scenario?

**Key insight**: Optimization is not just about algorithms - it's about understanding the interaction between data, model architecture, and training dynamics.

## ⭐ Aha Moment: Optimizers Update Weights

**What you built:** Optimization algorithms (SGD, Adam) that update neural network weights.

**Why it matters:** Gradients tell us which direction reduces the loss, but someone has to
actually move the weights. That's what optimizers do! SGD takes simple steps, while Adam
adapts the learning rate for each parameter—like having a personal trainer for each weight.

In the next module, you'll combine optimizers with a training loop to actually train networks!

## 🚀 MODULE SUMMARY: Optimizers

Congratulations! You've built sophisticated optimization algorithms that power modern neural network training!

### Key Accomplishments
- **Built SGD optimizer** with momentum for stable gradient descent and oscillation reduction
- **Implemented Adam optimizer** with adaptive learning rates and bias correction for different parameter scales
- **Created AdamW optimizer** with decoupled weight decay for proper regularization
- **Analyzed memory trade-offs**: SGD (2x), Adam/AdamW (3x parameter memory)
- **All tests pass** (validated by `test_module()`)

### Systems Insights Discovered
- **Memory scaling**: SGD needs 2x parameter memory (momentum), Adam needs 3x (two moment buffers)
- **Adaptive learning**: Adam automatically adjusts step sizes per parameter for faster convergence
- **Weight decay coupling**: AdamW fixes Adam's inconsistent regularization by decoupling weight decay
- **State management**: Optimizer buffers must be checkpointed for training resume

### Ready for Next Steps
Your optimizer implementations enable sophisticated neural network training! With gradients from Module 06 and optimizers from Module 07, you're ready to build complete training loops.

Export with: `tito module complete 07`

**Next**: Module 08 will add training loops, learning rate scheduling, and checkpointing for complete end-to-end neural network training!
