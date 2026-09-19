> **文档来源**：`src/16_compression/16_compression.py` 中的 Markdown 教学说明（英文原版，本模块尚无 `*_zh.py`）。  
> 下文保留原模块一级标题；组件主题下的从属小节已下调一级，避免同级标题重复。

# Module 16: Compression - Pruning and Model Optimization

Welcome to Module 16! You're about to build model compression techniques that make neural networks smaller and more efficient while preserving their intelligence.

## 🔗 Prerequisites & Progress
**You've Built**: Complete optimization pipeline with profiling (14) and quantization (15)
**You'll Build**: Pruning (magnitude & structured), knowledge distillation, and low-rank approximation
**You'll Enable**: Compressed models that maintain accuracy while using dramatically less storage and memory

**Connection Map**:
```
Profiling (14) → Quantization (15) → Compression (16) → Advanced Optimization
(measure size)   (reduce precision)  (remove weights)   (next modules)
```

## 🎯 Learning Objectives
By the end of this module, you will:
1. Implement magnitude-based and structured pruning
2. Build knowledge distillation for model compression
3. Create low-rank approximations of weight matrices
4. Measure compression ratios and sparsity levels
5. Understand structured vs unstructured sparsity trade-offs

Let's get started!

## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in modules/16_compression/compression_dev.py
**Building Side:** Code exports to tinytorch.perf.compression

```python
# Final package structure:
from tinytorch.perf.compression import magnitude_prune, structured_prune, measure_sparsity
```

**Why this matters:**
- **Learning:** Complete compression system in one focused module for deep understanding
- **Production:** Proper organization like real compression libraries with all techniques together
- **Consistency:** All compression operations and sparsity management in perf.compression
- **Integration:** Works seamlessly with models and quantization for complete optimization pipeline

## 📋 Module Dependencies

**Prerequisites**: Modules 01-15 must be completed (especially 14 Profiling, 15 Quantization)

**External Dependencies**:
- `numpy` (for array operations and numerical computing)
- `copy` (for model duplication during compression)

**TinyTorch Dependencies**:
- `tinytorch.core.tensor` (Tensor class)
- `tinytorch.core.layers` (Linear, Sequential)
- `tinytorch.core.activations` (ReLU)
- `tinytorch.perf.profiling` (Profiler, analyze_weight_distribution)

**Dependency Flow**:
```
Module 14 (Profiling) → Module 15 (Quantization) → Module 16 (Compression)
       ↓                        ↓                         ↓
  Measure size           Reduce precision           Remove weights
```

Students completing this module will have built compression techniques
that integrate with profiling and quantization for complete model optimization.

### Note on Sequential Usage in This Module

This module uses `Sequential` as a parameter container for compression operations.
`Sequential` is familiar from Module 03 and is used here as a convenience — it gives
compression wrappers a clean way to hold the original model's layers and expose them
for inspection, pruning, and quantization. The compression logic itself remains explicit
and visible throughout the module.

## 💡 Motivation: Why Compression Matters

Before we learn compression, let's profile a model to analyze its weight
distribution. We'll discover that many weights are tiny and might not matter much!

## 💡 Introduction: Model Compression Concepts

Imagine you have a massive library with millions of books, but you only reference 10% of them regularly. Model compression is like creating a curated collection that keeps the essential knowledge while dramatically reducing storage space.

Model compression reduces the size and computational requirements of neural networks while preserving their intelligence. It's the bridge between powerful research models and practical deployment.

### Why Compression Matters in ML Systems

**The Storage Challenge:**
- Modern language models: 100GB+ (GPT-3 scale)
- Mobile devices: <1GB available for models
- Edge devices: <100MB realistic limits
- Network bandwidth: Slow downloads kill user experience

**The Speed Challenge:**
- Research models: Designed for accuracy, not efficiency
- Production needs: Sub-second response times
- Battery life: Energy consumption matters for mobile
- Cost scaling: Inference costs grow with model size

### The Compression Landscape

```
Neural Network Compression Techniques:

┌───────────────────────────────────────────────────────────────────────┐
│                         COMPRESSION METHODS                           │
├───────────────────────────────────────────────────────────────────────┤
│  WEIGHT-BASED                       │  ARCHITECTURE-BASED             │
│  ┌────────────────────────────────┐ │  ┌────────────────────────────┐ │
│  │ Magnitude Pruning              │ │  │ Knowledge Distillation     │ │
│  │ • Remove small weights         │ │  │ • Teacher → Student        │ │
│  │ • 90% sparsity achievable      │ │  │ • 10x size reduction       │ │
│  │                                │ │  │                            │ │
│  │ Structured Pruning             │ │  │ Neural Architecture        │ │
│  │ • Remove entire channels       │ │  │ Search (NAS)               │ │
│  │ • Hardware-friendly            │ │  │ • Automated design         │ │
│  │                                │ │  │                            │ │
│  │ Low-Rank Approximation         │ │  │ Early Exit                 │ │
│  │ • Matrix factorization         │ │  │ • Adaptive compute         │ │
│  │ • SVD decomposition            │ │  │                            │ │
│  └────────────────────────────────┘ │  └────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────┘
```

Think of compression like optimizing a recipe - you want to keep the essential ingredients that create the flavor while removing anything that doesn't contribute to the final dish.

## 📐 Foundations: Mathematical Background

Understanding the mathematics behind compression helps us choose the right technique for each situation and predict their effects on model performance.

### Magnitude-Based Pruning: The Simple Approach

The core insight: small weights contribute little to the final prediction. Magnitude pruning removes weights based on their absolute values.

```
Mathematical Foundation:
For weight w_ij in layer l:
    If |w_ij| < threshold_l → w_ij = 0

Threshold Selection:
- Global: One threshold for entire model
- Layer-wise: Different threshold per layer
- Percentile-based: Remove bottom k% of weights

Sparsity Calculation:
    Sparsity = (Zero weights / Total weights) × 100%
```

### Structured Pruning: Hardware-Friendly Compression

Unlike magnitude pruning which creates scattered zeros, structured pruning removes entire computational units (neurons, channels, attention heads).

```
Channel Importance Metrics:

Method 1: L2 Norm
    Importance(channel_i) = ||W[:,i]||₂ = √(Σⱼ W²ⱼᵢ)

Method 2: Gradient-based
    Importance(channel_i) = |∂Loss/∂W[:,i]|

Method 3: Activation-based
    Importance(channel_i) = E[|activations_i|]

Pruning Decision:
    Remove bottom k% of channels based on importance ranking
```

### Knowledge Distillation: Learning from Teachers

Knowledge distillation transfers knowledge from a large "teacher" model to a smaller "student" model. The student learns not just the correct answers, but the teacher's reasoning process.

```
Distillation Loss Function:
    L_total = α × L_soft + (1-α) × L_hard

Where:
    L_soft = KL_divergence(σ(z_s/T), σ(z_t/T))  # Soft targets
    L_hard = CrossEntropy(σ(z_s), y_true)        # Hard targets

    σ(z/T) = Softmax with temperature T
    z_s = Student logits, z_t = Teacher logits
    α = Balance parameter (typically 0.7)
    T = Temperature parameter (typically 3-5)

Temperature Effect:
    T=1: Standard softmax (sharp probabilities)
    T>1: Softer distributions (reveals teacher's uncertainty)
```

### Low-Rank Approximation: Matrix Compression

Large weight matrices often have redundancy that can be captured with lower-rank approximations using Singular Value Decomposition (SVD).

```
SVD Decomposition:
    W_{m×n} = U_{m×k} × Σ_{k×k} × V^T_{k×n}

Parameter Reduction:
    Original: m × n parameters
    Compressed: (m × k) + k + (k × n) = k(m + n + 1) parameters

    Compression achieved when: k < mn/(m+n+1)

Reconstruction Error:
    ||W - W_approx||_F = √(Σᵢ₌ₖ₊₁ʳ σᵢ²)

    Where σᵢ are singular values, r = rank(W)
```

## 🏗️ Implementation: Sparsity Measurement

Before we can compress models, we need to understand how dense they are. Sparsity measurement tells us what percentage of weights are zero (or effectively zero).

### Understanding Sparsity

Sparsity is like measuring how much of a parking lot is empty. A 90% sparse model means 90% of its weights are zero - only 10% of the "parking spaces" are occupied.

```
Sparsity Visualization:

Dense Matrix (0% sparse):           Sparse Matrix (75% sparse):
┌─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┐    ┌─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┐
│ 2.1 1.3 0.8 1.9 2.4 1.1 0.7 │    │ 2.1 0.0 0.0 1.9 0.0 0.0 0.0 │
│ 1.5 2.8 1.2 0.9 1.6 2.2 1.4 │    │ 0.0 2.8 0.0 0.0 0.0 2.2 0.0 │
│ 0.6 1.7 2.5 1.1 0.8 1.3 2.0 │    │ 0.0 0.0 2.5 0.0 0.0 0.0 2.0 │
│ 1.9 1.0 1.6 2.3 1.8 0.9 1.2 │    │ 1.9 0.0 0.0 2.3 0.0 0.0 0.0 │
└─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┘    └─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┘
All weights active                   Only 7/28 weights active
Storage: 28 values                   Storage: 7 values + indices
```

Why this matters: Sparsity directly relates to memory savings, but achieving speedup requires special sparse computation libraries.

### 🧪 Unit Test: Sparsity Measurement

This test validates our sparsity measurement function works correctly.

**What we're testing**: Zero weight counting and percentage calculation
**Why it matters**: Accurate sparsity measurement is essential for compression evaluation
**Expected**: Correct sparsity percentages for dense and sparse models

## 🏗️ Implementation: Magnitude-Based Pruning

Magnitude pruning is the simplest and most intuitive compression technique. It's based on the observation that weights with small magnitudes contribute little to the model's output.

### How Magnitude Pruning Works

Think of magnitude pruning like editing a document - you remove words that don't significantly change the meaning. In neural networks, we remove weights that don't significantly affect predictions.

```
Magnitude Pruning Process:

Step 1: Collect All Weights (20 total across 2 layers)
┌──────────────────────────────────────────────────┐
│ Layer 1: [2.1, 0.08, -1.8, 0.04, 3.2,             │
│           -0.02, 1.5, -0.03, 2.8, 0.06]           │
│ Layer 2: [0.7, 2.4, -0.05, 1.9, 0.01,             │
│           -1.3, 0.03, 2.1, -0.07, 0.09]           │
└──────────────────────────────────────────────────┘
                    ↓
Step 2: Calculate Magnitudes
┌──────────────────────────────────────────────────┐
│ Sorted: [0.01, 0.02, 0.03, 0.03, 0.04, 0.05,     │
│          0.06, 0.07, 0.08, 0.09, 0.7, 1.3,       │
│          1.5, 1.8, 1.9, 2.1, 2.1, 2.4, 2.8, 3.2] │
└──────────────────────────────────────────────────┘
                    ↓
Step 3: Find Threshold (e.g., 50th percentile)
┌──────────────────────────────────────────────────┐
│ 20 values → 50th pctile between 10th and 11th    │ Threshold ≈ 0.4
│ Values ≤ 0.4: ten small weights get zeroed        │ (50% of weights removed)
└──────────────────────────────────────────────────┘
                    ↓
Step 4: Apply Pruning Mask
┌──────────────────────────────────────────────────┐
│ Layer 1: [2.1, 0.0, -1.8, 0.0, 3.2,              │
│           0.0, 1.5, 0.0, 2.8, 0.0]               │ 50% weights → 0
│ Layer 2: [0.7, 2.4, 0.0, 1.9, 0.0,               │ 50% preserved
│           -1.3, 0.0, 2.1, 0.0, 0.0]              │
└──────────────────────────────────────────────────┘

Memory Impact:
- Dense storage: 20 values × 4 bytes = 80 bytes
- Sparse storage: 10 values + 10 indices = 80 bytes (no savings!)
- At 90% sparsity: 2 values + 2 indices = 16 bytes (80% savings)
```

### Why Global Thresholding Works

Global thresholding treats the entire model as one big collection of weights, finding a single threshold that achieves the target sparsity across all layers.

**Advantages:**
- Simple to implement and understand
- Preserves overall model capacity
- Works well for uniform network architectures

**Disadvantages:**
- May over-prune some layers, under-prune others
- Doesn't account for layer-specific importance
- Can hurt performance if layers have very different weight distributions

### 🧪 Unit Test: Magnitude Pruning

This test validates magnitude-based pruning works correctly with threshold selection.

**What we're testing**: Weight removal based on magnitude threshold
**Why it matters**: Core technique for creating sparse neural networks
**Expected**: Achieves target sparsity with smallest weights removed

## 🏗️ Implementation: Structured Pruning

While magnitude pruning creates scattered zeros throughout the network, structured pruning removes entire computational units (channels, neurons, heads). This creates sparsity patterns that modern hardware can actually accelerate.

### Why Structured Pruning Matters

Think of the difference between removing random words from a paragraph versus removing entire sentences. Structured pruning removes entire "sentences" (channels) rather than random "words" (individual weights).

```
Unstructured vs Structured Sparsity:

UNSTRUCTURED (Magnitude Pruning):
┌─────────────────────────────────────────────┐
│ Channel 0: [2.1, 0.0, 1.8, 0.0, 3.2]        │ ← Sparse weights
│ Channel 1: [0.0, 2.8, 0.0, 2.1, 0.0]        │ ← Sparse weights
│ Channel 2: [1.5, 0.0, 2.4, 0.0, 1.9]        │ ← Sparse weights
│ Channel 3: [0.0, 1.7, 0.0, 2.0, 0.0]        │ ← Sparse weights
└─────────────────────────────────────────────┘
Issues: Irregular memory access, no hardware speedup

STRUCTURED (Channel Pruning):
┌─────────────────────────────────────────────┐
│ Channel 0: [2.1, 1.3, 1.8, 0.9, 3.2]        │ ← Fully preserved
│ Channel 1: [0.0, 0.0, 0.0, 0.0, 0.0]        │ ← Fully removed
│ Channel 2: [1.5, 2.2, 2.4, 1.1, 1.9]        │ ← Fully preserved
│ Channel 3: [0.0, 0.0, 0.0, 0.0, 0.0]        │ ← Fully removed
└─────────────────────────────────────────────┘
Benefits: Regular patterns, hardware acceleration possible
```

### Channel Importance Ranking

How do we decide which channels to remove? We rank them by importance using various metrics:

```
Channel Importance Metrics:

Method 1: L2 Norm (Most Common)
    For each output channel i:
    Importance_i = ||W[:, i]||_2 = √(Σⱼ w²ⱼᵢ)

    Intuition: Channels with larger weights have bigger impact

Method 2: Activation-Based
    Importance_i = E[|activation_i|] over dataset

    Intuition: Channels that activate more are more important

Method 3: Gradient-Based
    Importance_i = |∂Loss/∂W[:, i]|

    Intuition: Channels with larger gradients affect loss more

Ranking Process:
    1. Calculate importance for all channels
    2. Sort channels by importance (ascending)
    3. Remove bottom k% (least important)
    4. Zero out entire channels, not individual weights
```

### Hardware Benefits of Structured Sparsity

Structured sparsity enables real hardware acceleration because:

1. **Memory Coalescing**: Accessing contiguous memory chunks is faster
2. **SIMD Operations**: Can process multiple remaining channels in parallel
3. **No Indexing Overhead**: Don't need to track locations of sparse weights
4. **Cache Efficiency**: Better spatial locality of memory access

### 🧪 Unit Test: Structured Pruning

This test validates structured pruning removes entire channels correctly.

**What we're testing**: Channel-wise pruning based on L2 norm importance
**Why it matters**: Creates hardware-friendly sparsity patterns
**Expected**: Entire channels zeroed, not scattered weights

## 🏗️ Implementation: Low-Rank Approximation

Low-rank approximation discovers that large weight matrices often contain redundant information that can be captured with much smaller matrices through mathematical decomposition.

### The Intuition Behind Low-Rank Approximation

Imagine you're storing a massive spreadsheet where many columns are highly correlated. Instead of storing all columns separately, you could store a few "basis" columns and coefficients for how to combine them to recreate the original data.

```
Low-Rank Decomposition Visualization:

Original Matrix W (large):           Factorized Form (smaller):
┌─────────────────────────┐         ┌──────┐    ┌──────────────┐
│ 2.1  1.3  0.8  1.9  2.4 │         │ 1.1  │    │ 1.9  1.2  0.7│
│ 1.5  2.8  1.2  0.9  1.6 │    ≈    │ 2.4  │ @  │ 0.6  1.2  0.5│
│ 0.6  1.7  2.5  1.1  0.8 │         │ 0.8  │    │ 1.4  2.1  0.9│
│ 1.9  1.0  1.6  2.3  1.8 │         │ 1.6  │    │ 0.5  0.6  1.1│
└─────────────────────────┘         └──────┘    └──────────────┘
    W (4×5) = 20 params           U (4×2)=8  +  V (2×5)=10  = 18 params

Parameter Reduction:
- Original: 4 × 5 = 20 parameters
- Compressed: (4 × 2) + (2 × 5) = 18 parameters
- Compression ratio: 18/20 = 0.9 (10% savings)

For larger matrices, savings become dramatic:
- W (1000×1000): 1M parameters → U (1000×100) + V (100×1000): 200K parameters
- Compression ratio: 0.2 (80% savings)
```

### SVD: The Mathematical Foundation

Singular Value Decomposition (SVD) finds the optimal low-rank approximation by identifying the most important "directions" in the data:

```
SVD Decomposition:
    W = U × Σ × V^T

Where:
    U: Left singular vectors (input patterns)
    Σ: Singular values (importance weights)
    V^T: Right singular vectors (output patterns)

Truncated SVD (Rank-k approximation):
    W ≈ U[:,:k] × Σ[:k] × V^T[:k,:]

Quality vs Compression Trade-off:
    Higher k → Better approximation, less compression
    Lower k → More compression, worse approximation

Choosing Optimal Rank:
    Method 1: Fixed ratio (k = ratio × min(m,n))
    Method 2: Energy threshold (keep 90% of singular value energy)
    Method 3: Error threshold (reconstruction error < threshold)
```

#### When Low-Rank Works Best

Low-rank approximation works well when:
- **Matrices are large**: Compression benefits scale with size
- **Data has structure**: Correlated patterns enable compression
- **Moderate accuracy loss acceptable**: Some precision traded for efficiency

It works poorly when:
- **Matrices are already small**: Overhead exceeds benefits
- **Data is random**: No patterns to exploit
- **High precision required**: SVD introduces approximation error

#### 🧪 Unit Test: Low-Rank Approximation

This test validates SVD-based matrix factorization for compression.

**What we're testing**: Truncated SVD decomposition and reconstruction
**Why it matters**: Enables significant parameter reduction for large matrices
**Expected**: Correct factorization with acceptable reconstruction error

## 🏗️ Implementation: Knowledge Distillation

Knowledge distillation is like having an expert teacher simplify complex concepts for a student. The large "teacher" model shares its knowledge with a smaller "student" model, achieving similar performance with far fewer parameters.

### The Teacher-Student Learning Process

Unlike traditional training where models learn from hard labels (cat/dog), knowledge distillation uses "soft" targets that contain richer information about the teacher's decision-making process.

```
Knowledge Distillation Process:

                    TEACHER MODEL (Large)
                    ┌─────────────────────┐
Input Data ────────→│ 100M parameters     │
                    │ 95% accuracy        │
                    │ 500ms inference     │
                    └─────────────────────┘
                             │
                             ↓ Soft Targets
                    ┌─────────────────────┐
                    │  Logits: [2.1, 0.3, │
                    │           0.8, 4.2] │ ← Rich information
                    └─────────────────────┘
                             │
                             ↓ Distillation Loss
                    ┌─────────────────────┐
Input Data ────────→│ STUDENT MODEL       │
Hard Labels ───────→│ 10M parameters      │ ← 10x smaller
                    │ 93% accuracy        │ ← 2% loss
                    │ 50ms inference      │ ← 10x faster
                    └─────────────────────┘

Benefits:
• Size: 10x smaller models
• Speed: 10x faster inference
• Accuracy: Only 2-5% degradation
• Knowledge transfer: Student learns teacher's "reasoning"
```

### Temperature Scaling: Softening Decisions

Temperature scaling is a key innovation that makes knowledge distillation effective. It "softens" the teacher's confidence, revealing uncertainty that helps the student learn.

```
Temperature Effect on Probability Distributions:

Without Temperature (T=1):           With Temperature (T=3):
Teacher Logits: [1.0, 2.0, 0.5]    Teacher Logits: [1.0, 2.0, 0.5]
                       ↓                               ↓ ÷ 3
Softmax: [0.09, 0.67, 0.24]         Logits/T: [0.33, 0.67, 0.17]
         ^      ^      ^                       ↓
      Low   High   Med              Softmax: [0.21, 0.42, 0.17]
                                             ^      ^      ^
Sharp decisions (hard to learn)           Soft   decisions (easier to learn)

Why Soft Targets Help:
1. Reveal teacher's uncertainty about similar classes
2. Provide richer gradients for student learning
3. Transfer knowledge about class relationships
4. Reduce overfitting to hard labels
```

### Loss Function Design

The distillation loss balances learning from both the teacher's soft knowledge and the ground truth hard labels:

```
Combined Loss Function:

L_total = α × L_soft + (1-α) × L_hard

Where:
    L_soft = KL_divergence(Student_soft, Teacher_soft)
             │
             └─ Measures how well student mimics teacher

    L_hard = CrossEntropy(Student_predictions, True_labels)
             │
             └─ Ensures student still learns correct answers

Balance Parameter α:
• α = 0.7: Focus mainly on teacher (typical)
• α = 0.9: Almost pure distillation
• α = 0.3: Balance teacher and ground truth
• α = 0.0: Ignore teacher (regular training)

Temperature T:
• T = 1: No softening (standard softmax)
• T = 3-5: Good balance (typical range)
• T = 10+: Very soft (may lose information)
```

### 🧪 Unit Test: Knowledge Distillation

This test validates teacher-student knowledge transfer with temperature scaling.

**What we're testing**: Distillation loss combining soft and hard targets
**Why it matters**: Enables training small models with teacher knowledge
**Expected**: Valid loss computation for teacher-student training

## 🔧 Integration: Complete Compression Pipeline

Now let's combine all our compression techniques into a unified system that can apply multiple methods and track their cumulative effects.

### Compression Strategy Design

Real-world compression often combines multiple techniques in sequence, each targeting different types of redundancy:

```
Multi-Stage Compression Pipeline:

Original Model (100MB, 100% accuracy)
         │
         ↓ Stage 1: Magnitude Pruning (remove 80% of small weights)
Sparse Model (20MB, 98% accuracy)
         │
         ↓ Stage 2: Structured Pruning (remove 30% of channels)
Compact Model (14MB, 96% accuracy)
         │
         ↓ Stage 3: Low-Rank Approximation (compress large layers)
Factorized Model (10MB, 95% accuracy)
         │
         ↓ Stage 4: Knowledge Distillation (train smaller architecture)
Student Model (5MB, 93% accuracy)

Final Result: 20x size reduction, 7% accuracy loss
```

### Compression Configuration

Different deployment scenarios require different compression strategies:

```
Deployment Scenarios and Strategies:

MOBILE APP (Aggressive compression needed):
┌─────────────────────────────────────────┐
│ Target: <10MB, <100ms inference         │
│ Strategy:                               │
│ • Magnitude pruning: 95% sparsity       │
│ • Structured pruning: 50% channels      │
│ • Knowledge distillation: 10x reduction │
│ • Quantization: 8-bit weights           │
└─────────────────────────────────────────┘

EDGE DEVICE (Balanced compression):
┌─────────────────────────────────────────┐
│ Target: <50MB, <200ms inference         │
│ Strategy:                               │
│ • Magnitude pruning: 80% sparsity       │
│ • Structured pruning: 30% channels      │
│ • Low-rank: 50% rank reduction          │
│ • Quantization: 16-bit weights          │
└─────────────────────────────────────────┘

CLOUD SERVICE (Minimal compression):
┌─────────────────────────────────────────┐
│ Target: Maintain accuracy, reduce cost  │
│ Strategy:                               │
│ • Magnitude pruning: 50% sparsity       │
│ • Structured pruning: 10% channels      │
│ • Dynamic batching optimization         │
│ • Mixed precision inference             │
└─────────────────────────────────────────┘
```

### 🧪 Unit Test: Comprehensive Model Compression

This test validates the complete compression pipeline with multiple techniques.

**What we're testing**: Sequential application of compression techniques
**Why it matters**: Real deployments combine multiple compression methods
**Expected**: Cumulative sparsity increase and tracking of applied techniques

## 📊 Systems Analysis: Compression Trade-offs

Understanding the real-world effectiveness of different compression techniques through systematic measurement and comparison.

The fundamental challenge in model compression is balancing three competing objectives: model size, inference speed, and prediction accuracy.

## 📊 Measuring Compression Impact with Profiler

Now let's use the **Profiler** tool from Module 14 to measure the actual parameter reduction from pruning. This demonstrates the complete workflow: profile baseline (M14) → apply compression (M16) → measure impact (M14+M16).

This is the production workflow: measure → prune → validate → deploy.

### Comparing Compression Techniques

Let's analyze compression ratios across different techniques systematically.

### Knowledge Distillation Analysis

Now let's analyze how knowledge distillation compares to other compression techniques for different compression ratios and accuracy preservation goals.

## 🔧 Consolidated Compression Classes for Export

Now that we've implemented all compression techniques, let's create a consolidated class
for export to the tinytorch package. This allows milestones to use the complete compression system.

## 🔧 Verification: Prove Pruning Works

Before running the full integration test, let's create a verification function that
proves pruning actually creates zeros using real zero counting.

## 🧪 Module Integration Test

Final validation that everything works together correctly before module completion.

## 🤔 ML Systems Reflection Questions

Answer these to deepen your understanding of compression techniques and their systems implications:

### 1. Compression Trade-offs
**Question**: You implemented magnitude pruning that removes 90% of weights from a 10M parameter model.

**Consider**:
- How many parameters remain active? _____ M parameters
- If the original model was 40MB, what's the theoretical minimum storage? _____ MB
- Why might actual speedup be less than 10x?

**Real-world context**: Sparse matrix formats have indexing overhead, and many hardware accelerators cannot efficiently exploit unstructured sparsity.

---

### 2. Structured vs Unstructured Sparsity
**Question**: Your structured pruning removes entire channels, while magnitude pruning creates scattered zeros.

**Consider**:
- Which enables better hardware acceleration?
- Which preserves accuracy better at high sparsity?
- Which creates more predictable memory access patterns?

**Think about**: How would you choose between these approaches for different deployment targets (GPU, CPU, mobile)?

---

### 3. Knowledge Distillation Efficiency
**Question**: A teacher model has 100M parameters, student has 10M parameters, both achieve 85% accuracy.

**Calculate**:
- Compression ratio: _____x
- If teacher inference takes 100ms, student takes 15ms, what's the speedup? _____x
- Why is the speedup greater than the compression ratio?

**Real-world context**: Smaller models often have better cache locality and fewer memory bottlenecks.

---

### 4. Low-Rank Decomposition
**Question**: You approximate a (512, 256) weight matrix with rank 64 using SVD.

**Calculate**:
- Original parameter count: _____ parameters
- Decomposed parameter count: (512 x 64) + 64 + (64 x 256) = _____ parameters
- Compression ratio: _____x
- At what rank does compression become ineffective? rank > _____

**Trade-offs to consider**: Reconstruction error vs. compression ratio, and the overhead of two matrix multiplications vs. one.

---

### 5. Pruning Strategy Selection
**Question**: For deploying on a mobile device with 50MB model limit and 100ms latency requirement:

**Consider**:
- Which pruning strategy optimizes for memory? [magnitude/structured/both]
- Which pruning strategy optimizes for speed? [magnitude/structured/both]
- What order should you apply compression techniques?

**Real-world context**: Mobile devices have limited memory bandwidth, making structured sparsity more beneficial for latency.

## ⭐ Aha Moment: Pruning Removes Unimportant Weights

**What you built:** Pruning that zeros out small weights, creating sparse models.

**Why it matters:** Most neural network weights are close to zero—and removing them barely
affects accuracy! At 50% sparsity, half your weights are gone, but the model still works.
This is how you make models faster and smaller without retraining.

Combined with quantization, pruning can shrink models 8× or more.

## 🚀 MODULE SUMMARY: Compression

Congratulations! You've built a comprehensive model compression system that can dramatically reduce model size while preserving intelligence!

### Key Accomplishments
- Built magnitude-based and structured pruning techniques with clear sparsity patterns
- Implemented knowledge distillation for teacher-student compression with temperature scaling
- Created low-rank approximation using SVD decomposition for matrix factorization
- Developed sparsity measurement and comprehensive compression pipeline
- All tests pass (validated by `test_module()`)

### Systems Insights Discovered
- **Structured vs Unstructured**: Hardware-friendly patterns vs maximum compression
- **Compression Cascading**: Multiple techniques compound but need careful sequencing
- **Memory vs Speed**: Parameter reduction needs sparse libraries for speedup
- **Deployment Strategy**: Different scenarios require different compression approaches

### Ready for Next Steps
Your compression implementation enables efficient model deployment across diverse hardware constraints!
Export with: `tito module complete 16`

**Next**: Module 17 will add acceleration techniques including vectorization and kernel fusion!
