> **文档来源**：`src/13_transformers/13_transformers.py` 中的 Markdown 教学说明（英文原版，本模块尚无 `*_zh.py`）。  
> 下文保留原模块一级标题；组件主题下的从属小节已下调一级，避免同级标题重复。

# Module 13: Transformers - Complete Transformer Architecture

Welcome to Module 13! You're about to build the complete transformer architecture that powers modern language models like GPT, Claude, and ChatGPT.

## 🔗 Prerequisites & Progress
**You've Built**: Tokenization, embeddings, attention mechanisms, and all foundational components
**You'll Build**: TransformerBlock, complete GPT architecture, and autoregressive generation
**You'll Enable**: Full language model training and text generation capabilities

**Connection Map**:
```
Tokenization + Embeddings + Attention → Transformers → Language Generation
(text→numbers)  (learnable vectors) (sequence modeling)  (complete models)
```

## 🎯 Learning Objectives
By the end of this module, you will:
1. Implement complete TransformerBlock with attention, MLP, and layer normalization
2. Build full GPT architecture with multiple transformer blocks
3. Add autoregressive text generation capability
4. Understand parameter scaling in large language models
5. Test transformer components and generation pipeline

Let's get started!

## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/13_transformers/transformers_dev.py`
**Building Side:** Code exports to `tinytorch.core.transformers`

```python
# How to use this module:
from tinytorch.core.transformers import LayerNorm, MLP, TransformerBlock, GPT
```

**Why this matters:**
- **Learning:** Complete transformer architecture in one focused module for deep understanding
- **Production:** Proper organization like PyTorch's torch.nn with transformer components
- **Consistency:** All transformer building blocks (LayerNorm, MLP, TransformerBlock, GPT) in core.transformers
- **Integration:** Works seamlessly with attention, embeddings, and tokenization for complete language models

## 📦 Exported Interfaces: TransformerBlock and TinyGPT

**Learning Side:** You work in `modules/13_transformers/transformers_dev.py`
**Building Side:** Code exports to `tinytorch.core.transformers`

```python
# How to use this module:
from tinytorch.core.transformers import TransformerBlock, TinyGPT, LayerNorm, MLP
```

**Why this matters:**
- **Learning:** Complete transformer system showcasing how all components work together
- **Production:** Matches PyTorch's transformer implementation with proper model organization
- **Consistency:** All transformer components and generation logic in core.transformer
- **Integration:** Demonstrates the power of modular design by combining all previous modules

## 📋 Module Dependencies

**Prerequisites**: Modules 01-12 must be complete

**External Dependencies**:
- `numpy` (for array operations and numerical computing)

**TinyTorch Dependencies**:
- `tinytorch.core.tensor` (Module 01: Tensor foundation)
- `tinytorch.core.activations` (Module 03: GELU activation)
- `tinytorch.core.layers` (Module 04: Linear layers)
- `tinytorch.core.embeddings` (Module 11: Embedding layers)
- `tinytorch.core.attention` (Module 12: MultiHeadAttention)

**Dependency Flow**:
```
Tensor → Activations → Layers → Attention → Embeddings → Transformers
  ↓         ↓           ↓         ↓            ↓            ↓
 data    GELU act    Linear    MultiHead    Token+Pos   Complete GPT
```

Students completing this module will have built the complete
transformer architecture that powers modern language models.

## 💡 Introduction: What are Transformers?

Transformers are the revolutionary architecture that powers modern AI language models like GPT, ChatGPT, and Claude. The key breakthrough is **self-attention**, which allows every token in a sequence to directly interact with every other token, creating rich contextual understanding.

### The Transformer Revolution

Before transformers, language models used RNNs or CNNs that processed text sequentially or locally. Transformers changed everything by processing all positions in parallel while maintaining global context.

### Complete GPT Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  COMPLETE GPT ARCHITECTURE: From Text to Generation             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  INPUT: "Hello world"  →  Token IDs: [15496, 1917]              │
│                                ↓                                │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                EMBEDDING LAYER                            │  │
│  │                                                           │  │
│  │  ┌─────────────┐       ┌─────────────────────────────┐    │  │
│  │  │Token Embed  │   +   │ Positional Embedding        │    │  │
│  │  │15496→[0.1,  │       │ pos_0→[0.05, -0.02, ...]    │    │  │
│  │  │     0.3,..]│       │ pos_1→[0.12,  0.08, ...]     │    │  │
│  │  │1917→[0.2,   │       │                             │    │  │
│  │  │    -0.1,..]│       │                              │    │  │
│  │  └─────────────┘       └─────────────────────────────┘    │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                ↓                                │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              TRANSFORMER BLOCK 1                          │  │
│  │                                                           │  │
│  │  x → LayerNorm → MultiHeadAttention → + x → result        │  │
│  │  │                                      ↑                 │  │
│  │  │              residual connection     │                 │  │
│  │  └──────────────────────────────────────┘                 │  │
│  │  │                                                        │  │
│  │  result → LayerNorm → MLP (Feed Forward) → + result       │  │
│  │  │                                           ↑            │  │
│  │  │                residual connection        │            │  │
│  │  └───────────────────────────────────────────┘            │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                ↓                                │
│              TRANSFORMER BLOCK 2 (same pattern)                 │
│                                ↓                                │
│                      ... (more blocks) ...                      │
│                                ↓                                │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                   OUTPUT HEAD                             │  │
│  │                                                           │  │
│  │  final_hidden → LayerNorm → Linear(embed_dim, vocab_size) │  │
│  │                              ↓                            │  │
│  │               Vocabulary Logits: [0.1, 0.05, 0.8, ...]    │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                ↓                                │
│  OUTPUT: Next Token Probabilities                               │
│  "Hello" → 10%,  "world" → 5%,  "!" → 80%,  ...                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Why Transformers Dominate

**Parallel Processing**: Unlike RNNs that process tokens one by one, transformers process all positions simultaneously. This makes training much faster.

**Global Context**: Every token can directly attend to every other token in the sequence, capturing long-range dependencies that RNNs struggle with.

**Scalability**: Performance predictably improves with more parameters and data. This enabled the scaling laws that led to GPT-3, GPT-4, and beyond.

**Residual Connections**: Allow training very deep networks (100+ layers) by providing gradient highways.

### The Building Blocks We'll Implement

1. **LayerNorm**: Stabilizes training by normalizing activations
2. **Multi-Layer Perceptron (MLP)**: Provides non-linear transformation
3. **TransformerBlock**: Combines attention + MLP with residuals
4. **GPT**: Complete model with embeddings and generation capability

## 📐 Foundations: Essential Transformer Mathematics

### Layer Normalization: The Stability Engine

Layer Normalization is crucial for training deep transformer networks. Unlike batch normalization (which normalizes across the batch), layer norm normalizes across the feature dimension for each individual sample.

```
Mathematical Formula:
output = (x - μ) / σ * γ + β

where:
  μ = mean(x, axis=features)     # Mean across feature dimension
  σ = sqrt(var(x) + ε)          # Standard deviation + small epsilon
  γ = learnable scale parameter  # Initialized to 1.0
  β = learnable shift parameter  # Initialized to 0.0
```

**Why Layer Norm Works:**
- **Independence**: Each sample normalized independently (good for variable batch sizes)
- **Stability**: Prevents internal covariate shift that breaks training
- **Gradient Flow**: Helps gradients flow better through deep networks

### Residual Connections: The Gradient Highway

Residual connections are the secret to training deep networks. They create "gradient highways" that allow information to flow directly through the network.

```
┌─────────────────────────────────────────────────────────────────┐
│  RESIDUAL CONNECTIONS: The Gradient Highway System              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  PRE-NORM ARCHITECTURE (Modern Standard):                       │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                ATTENTION SUB-LAYER                        │  │
│  │                                                           │  │
│  │  Input (x) ────┬─→ LayerNorm ─→ MultiHeadAttention ─┐     │  │
│  │                │                                    │     │  │
│  │                │         ┌──────────────────────────┘     │  │
│  │                │         ▼                                │  │
│  │                └────→ ADD ─→ Output to next sub-layer     │  │
│  │                      (x + attention_output)               │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                ↓                                │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                   MLP SUB-LAYER                           │  │
│  │                                                           │  │
│  │  Input (x) ────┬─→ LayerNorm ─→ MLP (Feed Forward)  ─┐    │  │
│  │                │                                     │    │  │
│  │                │         ┌───────────────────────────┘    │  │
│  │                │         ▼                                │  │
│  │                └────→ ADD ─→ Final Output                 │  │
│  │                      (x + mlp_output)                     │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  KEY INSIGHT: Each sub-layer ADDS to the residual stream        │
│  rather than replacing it, preserving information flow!         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Gradient Flow Visualization:**
```
Backward Pass Without Residuals:    With Residuals:
Loss                                Loss
 │ gradients get smaller             │ gradients stay strong
 ↓ at each layer                    ↓ via residual paths
Layer N  ← tiny gradients          Layer N  ← strong gradients
 │                                  │     ↗ (direct path)
 ↓                                  ↓   ↗
Layer 2  ← vanishing                Layer 2  ← strong gradients
 │                                  │     ↗
 ↓                                  ↓   ↗
Layer 1  ← gone!                   Layer 1  ← strong gradients
```

### Feed-Forward Network (MLP): The Thinking Layer

The MLP provides the actual "thinking" in each transformer block. It's a simple two-layer network with a specific expansion pattern.

```
MLP Architecture:
Input (embed_dim) → Linear → GELU → Linear → Output (embed_dim)
       512           2048      2048    512
                   (4x expansion)

Mathematical Formula:
FFN(x) = Linear₂(GELU(Linear₁(x)))
       = W₂ · GELU(W₁ · x + b₁) + b₂

where:
  W₁: (embed_dim, 4*embed_dim)  # Expansion matrix
  W₂: (4*embed_dim, embed_dim)  # Contraction matrix
  GELU: smooth activation function (better than ReLU for language)
```

**Why 4x Expansion?**
- **Capacity**: More parameters = more representation power
- **Non-linearity**: GELU activation creates complex transformations
- **Information Bottleneck**: Forces the model to compress useful information

### The Complete Transformer Block Data Flow

```
Input Tensor (batch, seq_len, embed_dim)
          ↓
    ┌─────────────────────────────────────┐
    │ ATTENTION SUB-LAYER                 │
    │                                     │
    │ x₁ = LayerNorm(x₀)                  │
    │ attention_out = MultiHeadAttn(x₁)   │
    │ x₂ = x₀ + attention_out  (residual) │
    └─────────────────────────────────────┘
          ↓
    ┌─────────────────────────────────────┐
    │ MLP SUB-LAYER                       │
    │                                     │
    │ x₃ = LayerNorm(x₂)                  │
    │ mlp_out = MLP(x₃)                   │
    │ x₄ = x₂ + mlp_out    (residual)     │
    └─────────────────────────────────────┘
          ↓
Output Tensor (batch, seq_len, embed_dim)
```

**Key Insight**: Each sub-layer (attention and MLP) gets a "clean" normalized input but adds its contribution to the residual stream. This creates a stable training dynamic.

## 🏗️ Implementation: Building Transformer Components

Now we'll implement each transformer component with a clear understanding of their role in the overall architecture. We'll follow the pattern: **Explanation → Implementation → Test** for each component.

Each component serves a specific purpose:
- **LayerNorm**: Stabilizes training and normalizes activations
- **MLP**: Provides non-linear transformation and "thinking" capacity
- **TransformerBlock**: Combines attention with MLP using residual connections
- **GPT**: Complete autoregressive language model for text generation

### Understanding Layer Normalization

Layer Normalization is the foundation of stable transformer training. Unlike batch normalization, it normalizes each sample independently across its feature dimensions.

#### Why Layer Norm is Essential

Without normalization, deep networks suffer from "internal covariate shift" - the distribution of inputs to each layer changes during training, making learning unstable.

#### Layer Norm Visualization

```
┌─────────────────────────────────────────────────────────────────┐
│  LAYER NORMALIZATION: Stabilizing Deep Networks                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  INPUT TENSOR: (batch=2, seq=3, features=4)                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Sample 1: [[1.0,  2.0,  3.0,  4.0],     ← Position 0      │  │
│  │            [5.0,  6.0,  7.0,  8.0],     ← Position 1      │  │
│  │            [9.0, 10.0, 11.0, 12.0]]     ← Position 2      │  │
│  │                                                           │  │
│  │ Sample 2: [[13., 14., 15., 16.],         ← Position 0     │  │
│  │            [17., 18., 19., 20.],         ← Position 1     │  │
│  │            [21., 22., 23., 24.]]         ← Position 2     │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                ↓                                │
│           NORMALIZE ACROSS FEATURES (per position)              │
│                                ↓                                │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ AFTER NORMALIZATION: Each position → mean=0, std=1        │  │
│  │                                                           │  │
│  │ Sample 1: [[-1.34, -0.45,  0.45,  1.34],                  │  │
│  │            [-1.34, -0.45,  0.45,  1.34],                  │  │
│  │            [-1.34, -0.45,  0.45,  1.34]]                  │  │
│  │                                                           │  │
│  │ Sample 2: [[-1.34, -0.45,  0.45,  1.34],                  │  │
│  │            [-1.34, -0.45,  0.45,  1.34],                  │  │
│  │            [-1.34, -0.45,  0.45,  1.34]]                  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                ↓                                │
│            APPLY LEARNABLE PARAMETERS: γ * norm + β             │
│                                ↓                                │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ FINAL OUTPUT: Model can learn any desired distribution    │  │
│  │ γ (scale) and β (shift) are learned during training       │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  KEY INSIGHT: Unlike batch norm, each sample normalized         │
│  independently - perfect for variable-length sequences!         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Key Properties
- **Per-sample normalization**: Each sequence position normalized independently
- **Learnable parameters**: γ (scale) and β (shift) allow the model to recover any desired distribution
- **Gradient friendly**: Helps gradients flow smoothly through deep networks

### 🧪 Unit Test: Layer Normalization

This test validates our LayerNorm implementation works correctly.

**What we're testing**: Normalization statistics and parameter learning
**Why it matters**: Essential for transformer stability and training
**Expected**: Mean approximately 0, std approximately 1 after normalization, learnable parameters work

### Understanding the Multi-Layer Perceptron (MLP)

The MLP is where the "thinking" happens in each transformer block. It's a simple feed-forward network that provides non-linear transformation capacity.

#### The Role of MLP in Transformers

While attention handles relationships between tokens, the MLP processes each position independently, adding computational depth and non-linearity.

#### MLP Architecture and Information Flow

```
Information Flow Through MLP:

Input: (batch, seq_len, embed_dim=512)
         ↓
┌─────────────────────────────────────────────┐
│ Linear Layer 1: Expansion                   │
│ Weight: (512, 2048)  Bias: (2048,)          │
│ Output: (batch, seq_len, 2048)              │
└─────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────┐
│ GELU Activation                             │
│ Smooth, differentiable activation           │
│ Better than ReLU for language modeling      │
└─────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────┐
│ Linear Layer 2: Contraction                 │
│ Weight: (2048, 512)  Bias: (512,)           │
│ Output: (batch, seq_len, 512)               │
└─────────────────────────────────────────────┘
         ↓
Output: (batch, seq_len, embed_dim=512)
```

#### Why 4x Expansion?

```
Parameter Count Analysis:

Embed Dim: 512
MLP Hidden: 2048 (4x expansion)

Parameters:
- Linear1: 512 × 2048 + 2048 = 1,050,624
- Linear2: 2048 × 512 + 512 = 1,049,088
- Total MLP: ~2.1M parameters

For comparison:
- Attention (same embed_dim): ~1.5M parameters
- MLP has MORE parameters → more computational capacity
```

#### GELU vs ReLU

```
Activation Function Comparison:

ReLU(x) = max(0, x)        # Hard cutoff at 0
         ┌────
         │
    ─────┘
         0

GELU(x) ≈ x * Φ(x)         # Smooth, probabilistic
         ╭────
        ╱
    ───╱
      ╱
     0

GELU is smoother and provides better gradients for language modeling.
```

### 🧪 Unit Test: MLP (Feed-Forward Network)

This test validates our MLP implementation works correctly.

**What we're testing**: Shape preservation and parameter counting
**Why it matters**: MLP provides the non-linear transformation in transformers
**Expected**: Input/output shapes match, correct parameter count

### Understanding the Complete Transformer Block

The TransformerBlock is the core building unit of GPT and other transformer models. It combines self-attention with feed-forward processing using a carefully designed residual architecture.

#### Pre-Norm vs Post-Norm Architecture

Modern transformers use "pre-norm" architecture where LayerNorm comes BEFORE the sub-layers, not after. This provides better training stability.

```
Pre-Norm Architecture (What We Implement):
┌────────────────────────────────────────────────────────┐
│                     INPUT (x)                          │
│                       │                                │
│       ┌───────────────┴───────────────┐                │
│       │                               │                │
│       ▼                               │                │
│  LayerNorm                            │                │
│       │                               │                │
│       ▼                               │                │
│ MultiHeadAttention                    │                │
│       │                               │                │
│       └───────────────┬───────────────┘                │
│                       │          (residual connection) │
│                       ▼                                │
│                  x + attention                         │
│                       │                                │
│       ┌───────────────┴───────────────┐                │
│       │                               │                │
│       ▼                               │                │
│  LayerNorm                            │                │
│       │                               │                │
│       ▼                               │                │
│      MLP                              │                │
│       │                               │                │
│       └───────────────┬───────────────┘                │
│                       │          (residual connection) │
│                       ▼                                │
│                   x + mlp                              │
│                       │                                │
│                       ▼                                │
│                    OUTPUT                              │
└────────────────────────────────────────────────────────┘
```

#### Why Pre-Norm Works Better

**Training Stability**: LayerNorm before operations provides clean, normalized inputs to attention and MLP layers.

**Gradient Flow**: Residual connections carry gradients directly from output to input, bypassing the normalized operations.

**Deeper Networks**: Pre-norm enables training much deeper networks (100+ layers) compared to post-norm.

#### Information Processing in Transformer Block

```
Step-by-Step Data Transformation:

1. Input Processing:
   x₀: (batch, seq_len, embed_dim) # Original input

2. Attention Sub-layer:
   x₁ = LayerNorm(x₀)               # Normalize input
   attn_out = MultiHeadAttn(x₁)     # Self-attention
   x₂ = x₀ + attn_out               # Residual connection

3. MLP Sub-layer:
   x₃ = LayerNorm(x₂)               # Normalize again
   mlp_out = MLP(x₃)                # Feed-forward
   x₄ = x₂ + mlp_out                # Final residual

4. Output:
   return x₄                        # Ready for next block
```

#### Residual Stream Concept

Think of the residual connections as a "stream" that carries information through the network:

```
Residual Stream Flow:

Layer 1: [original embeddings] ─┐
                                 ├─→ + attention info ─┐
Attention adds information ──────┘                      │
                                                        ├─→ + MLP info ─┐
MLP adds information ───────────────────────────────────┘               │
                                                                        │
Layer 2: carries accumulated information ───────────────────────────────┘
```

Each layer adds information to this stream rather than replacing it, creating a rich representation.

### 🧪 Unit Test: Transformer Block

This test validates our complete TransformerBlock implementation.

**What we're testing**: Shape preservation, residual connections, parameter counting
**Why it matters**: This is the core component that will be stacked to create GPT
**Expected**: Input/output shapes match, all components work together

### 🧪 Unit Test: GPT Model

This test validates our complete GPT implementation.

**What we're testing**: Model forward pass, shape consistency, generation capability
**Why it matters**: This is the complete language model that ties everything together
**Expected**: Correct output shapes, generation works, parameter counting

### 🧪 Unit Test: Token Sampling

This test validates the `_sample_next_token` helper that handles temperature-controlled
token sampling, separated from the generation loop for clarity.

```
Token Sampling Pipeline:
Raw logits: [1.0, 2.0, 3.0]
       |
  temperature scaling (divide by T)
       |
  softmax (numerical stability via max subtraction)
       |
  probability distribution: [0.09, 0.24, 0.67]
       |
  rng.choice -> sampled token index
```

**What we're testing**: Temperature scaling, softmax probability output, valid token range
**Why it matters**: Sampling quality controls generation coherence and creativity
**Expected**: Valid token indices, probabilities sum to 1, temperature affects distribution

## 🔧 Integration: Complete Transformer Workflow

Now that we've built all the components, let's see how they work together in a complete language modeling pipeline. This demonstrates the full power of the transformer architecture.

### The Language Modeling Pipeline

```
Complete Workflow Visualization:

1. Text Input:
   "hello world" → Tokenization → [15496, 1917]

2. Model Processing:
   [15496, 1917]
        ↓ Token Embedding
   [[0.1, 0.5, ...], [0.3, -0.2, ...]]  # Vector representations
        ↓ + Position Embedding
   [[0.2, 0.7, ...], [0.1, -0.4, ...]]  # With position info
        ↓ Transformer Block 1
   [[0.3, 0.2, ...], [0.5, -0.1, ...]]  # After attention + MLP
        ↓ Transformer Block 2
   [[0.1, 0.9, ...], [0.7, 0.3, ...]]   # Further processed
        ↓ Final LayerNorm + LM Head
   [[0.1, 0.05, 0.8, ...], [...]]       # Probability over vocab

3. Generation:
   Model predicts next token: "!" (token 33)
   New sequence: "hello world!"
```

This integration demo will show:
- **Character-level tokenization** for simplicity
- **Forward pass** through all components
- **Autoregressive generation** in action
- **Temperature effects** on creativity

## 📊 Systems Analysis: Parameter Scaling and Memory

Transformer models scale dramatically with size, leading to both opportunities and challenges. Let's analyze the computational and memory requirements to understand why training large language models requires massive infrastructure.

### The Scaling Laws Revolution

One of the key discoveries in modern AI is that transformer performance follows predictable scaling laws:

```
Scaling Laws Pattern:
Performance ∝ Parameters^α × Data^β × Compute^γ

where α ≈ 0.7, β ≈ 0.8, γ ≈ 0.5

This means:
- 10× more parameters → ~5× better performance
- 10× more data → ~6× better performance
- 10× more compute → ~3× better performance
```

### Memory Scaling Analysis

Memory requirements grow in different ways for different components:

```
Memory Scaling by Component:

1. Parameter Memory (Linear with model size):
   - Embeddings: vocab_size × embed_dim
   - Transformer blocks: ~4 × embed_dim²
   - Total: O(embed_dim²)

2. Attention Memory (Quadratic with sequence length):
   - Attention matrices: batch × heads × seq_len²
   - This is why long context is expensive!
   - Total: O(seq_len²)

3. Activation Memory (Linear with batch size):
   - Forward pass activations for backprop
   - Scales with: batch × seq_len × embed_dim
   - Total: O(batch_size)
```

### The Attention Memory Wall

```
┌─────────────────────────────────────────────────────────────────┐
│  ATTENTION MEMORY WALL: Why Long Context is Expensive           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  MEMORY USAGE BY SEQUENCE LENGTH (Quadratic Growth):            │
│                                                                 │
│  1K tokens:   [▓] 16 MB                ← Manageable             │
│  2K tokens:   [▓▓▓▓] 64 MB             ← 4× memory (quadratic)  │
│  4K tokens:   [▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓] 256 MB   ← 16× memory          │
│  8K tokens:   [████████████████████████████████] 1 GB           │
│  16K tokens:  [████████████████████████████████████████] 4 GB   │
│  32K tokens:  [████████████████████████████████████████████] →  │
│               ← extends to 16 GB (off the chart!)               │
│                                                                 │
│  REAL-WORLD CONTEXT LIMITS:                                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ GPT-3:     2K tokens  (limited by memory)                 │  │
│  │ GPT-4:     8K tokens  (32K with optimizations)            │  │
│  │ Claude-3:  200K tokens (special techniques required!)     │  │
│  │ GPT-4o:    128K tokens (efficient attention)              │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  MATHEMATICAL SCALING:                                          │
│  Memory = batch_size × num_heads × seq_len² × 4 bytes           │
│                                   ↑                             │
│                          This is the killer!                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 🧪 Module Integration Test

Final validation that everything works together correctly.

## 🤔 ML Systems Reflection Questions

Now that you've built a complete transformer architecture, let's think about its systems-level implications. Understanding memory scaling, gradient flow, and parameter distribution helps you make informed decisions when building and deploying production language models.

### Question 1: Attention Memory Complexity

You implemented multi-head attention that computes attention matrices of size (batch, heads, seq_len, seq_len).

**Consider**:
- For a model with seq_len=1024, batch_size=4, num_heads=8, how many elements in the attention matrix?
- If each element is 4 bytes (float32), how much memory per layer?
- Why does doubling sequence length quadruple attention memory?

**Key Insight**: Attention memory scales quadratically with sequence length, which is why long-context models require specialized attention efficiency techniques (KV caching, sparse attention).

### Question 2: Residual Connection Benefits

Your TransformerBlock uses residual connections (x + attention_output, x + mlp_output).

**Consider**:
- What happens to gradients during backpropagation without residual connections?
- How do residual connections help train deeper networks?
- Why is pre-norm (LayerNorm before operations) preferred over post-norm?

**Key Insight**: Residual connections create "gradient highways" that allow gradients to flow directly from loss to early layers. This enables training networks with 100+ layers that would otherwise suffer from vanishing gradients.

### Question 3: Parameter Scaling Analysis

Your GPT model combines embeddings, transformer blocks, and output projection.

**Calculate** (for embed_dim=512, vocab_size=10000, num_layers=6):
- Token embedding parameters: vocab_size x embed_dim = 5.12M
- Parameters per transformer block: approximately 12 x embed_dim^2 = 3.15M
- Total model parameters: approximately 29.7M

**Key Insight**: Most parameters in large language models come from the embedding layers (for large vocabularies) and the MLP layers (which use 4x expansion). Attention parameters are relatively small.

### Question 4: Autoregressive Generation Efficiency

Your generate() method processes the full sequence for each new token.

**Consider**:
- Why is this inefficient for long sequences?
- How does computation scale as you generate more tokens?
- Can you spot the redundant work?

**Key Insight**: Generation involves significant redundant work that grows with sequence length. KV caching eliminates this redundancy by storing previously computed key-value pairs.

## ⭐ Aha Moment: Transformer Processes Sequences

**What you built:** A complete transformer block with attention, MLPs, and residual connections.

**Why it matters:** This is THE architecture behind GPT, Claude, LLaMA, and every modern
language model. The transformer block combines attention (for relationships) with MLPs
(for processing) and residual connections (for trainability).

In the milestones, you'll stack these blocks to build a working language model!

## 🚀 MODULE SUMMARY: Transformers

Congratulations! You've built the complete transformer architecture that powers modern language models like GPT, Claude, and ChatGPT!

### Key Accomplishments
- Built LayerNorm for stable training across deep transformer networks
- Implemented MLP (feed-forward) networks with GELU activation and 4x expansion
- Created complete TransformerBlock with self-attention, residual connections, and pre-norm architecture
- Built full GPT model with embeddings, positional encoding, and autoregressive generation
- Discovered attention memory scaling and parameter distribution patterns
- All tests pass ✅ (validated by `test_module()`)

### Ready for Next Steps
Your transformer implementation is the capstone of the language modeling pipeline.
Export with: `tito module complete 13`

**Next**: Module 14 will add profiling and optimization techniques to make your transformers production-ready!
