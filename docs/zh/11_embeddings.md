> **文档来源**：`src/11_embeddings/11_embeddings.py` 中的 Markdown 教学说明（英文原版，本模块尚无 `*_zh.py`）。  
> 下文保留原模块一级标题；组件主题下的从属小节已下调一级，避免同级标题重复。

# Module 11: Embeddings - Converting Tokens to Learnable Representations

Welcome to Module 11! You're about to build embedding layers that convert discrete tokens into dense, learnable vectors - the foundation of all modern NLP models.

## 🔗 Prerequisites & Progress
**You've Built**: Tensors, layers, tokenization (discrete text processing)
**You'll Build**: Embedding lookups and positional encodings for sequence modeling
**You'll Enable**: Foundation for attention mechanisms and transformer architectures

**Connection Map**:
```
Tokenization → Embeddings → Positional Encoding → Attention
(discrete)     (dense)      (position-aware)     (context-aware)
```

## 🎯 Learning Objectives
By the end of this module, you will:
1. Implement embedding layers for token-to-vector conversion
2. Understand learnable vs fixed positional encodings
3. Build both sinusoidal and learned position encodings
4. Analyze embedding memory requirements and lookup performance

Let's transform tokens into intelligence!

## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/11_embeddings/embeddings_dev.py`
**Building Side:** Code exports to `tinytorch.text.embeddings`

```python
# How to use this module:
from tinytorch.core.embeddings import Embedding, PositionalEncoding, create_sinusoidal_embeddings
```

**Why this matters:**
- **Learning:** Complete embedding system for converting discrete tokens to continuous representations
- **Production:** Essential component matching PyTorch's torch.nn.Embedding with positional encoding patterns
- **Consistency:** All embedding operations and positional encodings in text.embeddings
- **Integration:** Works seamlessly with tokenizers for complete text processing pipeline

## 📋 Module Dependencies

**Prerequisites**: Modules 01-10 (especially Tensor foundation)

**External Dependencies**:
- `numpy` (for array operations and numerical computing)
- `math` (for mathematical constants and functions)

**TinyTorch Dependencies**:
- `tinytorch.core.tensor.Tensor` (from Module 01)

**Dependency Flow**:
```
Module 01 (Tensor) → Module 11 (Embeddings)
     ↓                       ↓
  Foundation        Token-to-Vector
```

Students completing this module will have built the embedding system
that converts discrete tokens into continuous representations for transformers.

## 💡 Introduction: Why Embeddings?

Neural networks operate on dense vectors, but language consists of discrete tokens. Embeddings are the crucial bridge that converts discrete tokens into continuous, learnable vector representations that capture semantic meaning.

### The Token-to-Vector Challenge

Consider the tokens from our tokenizer: [1, 42, 7] - how do we turn these discrete indices into meaningful vectors that capture semantic relationships?

```
┌─────────────────────────────────────────────────────────────────┐
│  EMBEDDING PIPELINE: Discrete Tokens → Dense Vectors            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input (Token IDs):     [1, 42, 7]                              │
│           │                                                     │
│           ├─ Step 1: Lookup in embedding table                  │
│           │         Each ID → vector of learned features        │
│           │                                                     │
│           ├─ Step 2: Add positional information                 │
│           │         Same word at different positions → different│
│           │                                                     │
│           ├─ Step 3: Create position-aware representations      │
│           │         Ready for attention mechanisms              │
│           │                                                     │
│           └─ Step 4: Enable semantic understanding              │
│                     Similar words → similar vectors             │
│                                                                 │
│  Output (Dense Vectors): [[0.1, 0.4, ...], [0.7, -0.2, ...]]    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### The Four-Layer Embedding System

Modern embedding systems combine multiple components:

**1. Token embeddings** - Learn semantic representations for each vocabulary token
**2. Positional encoding** - Add information about position in sequence
**3. Optional scaling** - Normalize embedding magnitudes (Transformer convention)
**4. Integration** - Combine everything into position-aware representations

### Why This Matters

The choice of embedding strategy dramatically affects:
- **Semantic understanding** - How well the model captures word meaning
- **Memory requirements** - Embedding tables can be gigabytes in size
- **Position awareness** - Whether the model understands word order
- **Extrapolation** - How well the model handles longer sequences than training

## 📐 Foundations: Embedding Strategies

Different embedding approaches make different trade-offs between memory, semantic understanding, and computational efficiency.

### Token Embedding Lookup Process

**Approach**: Each token ID maps to a learned dense vector

```
┌──────────────────────────────────────────────────────────────┐
│ TOKEN EMBEDDING LOOKUP PROCESS                               │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Step 1: Build Embedding Table (vocab_size × embed_dim)      │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Token ID  │  Embedding Vector (learned features)       │  │
│  ├────────────────────────────────────────────────────────┤  │
│  │    0      │  [0.2, -0.1,  0.3, 0.8, ...]  (<UNK>)      │  │
│  │    1      │  [0.1,  0.4, -0.2, 0.6, ...]  ("the")      │  │
│  │   42      │  [0.7, -0.2,  0.1, 0.4, ...]  ("cat")      │  │
│  │    7      │  [-0.3, 0.1,  0.5, 0.2, ...]  ("sat")      │  │
│  │   ...     │             ...                            │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  Step 2: Lookup Process (O(1) per token)                     │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Input: Token IDs [1, 42, 7]                           │  │
│  │                                                        │  │
│  │   ID 1  → embedding[1]  → [0.1,  0.4, -0.2, ...]       │  │
│  │   ID 42 → embedding[42] → [0.7, -0.2,  0.1, ...]       │  │
│  │   ID 7  → embedding[7]  → [-0.3, 0.1,  0.5, ...]       │  │
│  │                                                        │  │
│  │  Output: Matrix (3 × embed_dim)                        │  │
│  │  [[0.1,  0.4, -0.2, ...],                              │  │
│  │   [0.7, -0.2,  0.1, ...],                              │  │
│  │   [-0.3, 0.1,  0.5, ...]]                              │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  Step 3: Training Updates Embeddings                         │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Gradients flow back to embedding table                │  │
│  │                                                        │  │
│  │  Similar words learn similar vectors:                  │  │
│  │  "cat" and "dog" → closer in embedding space           │  │
│  │  "the" and "a"   → closer in embedding space           │  │
│  │  "sat" and "run" → farther in embedding space          │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**Pros**:
- Dense representation (every dimension meaningful)
- Learnable (captures semantic relationships through training)
- Efficient lookup (O(1) time complexity)
- Scales to large vocabularies

**Cons**:
- Memory intensive (vocab_size × embed_dim parameters)
- Requires training to develop semantic relationships
- Fixed vocabulary (new tokens need special handling)

### Positional Encoding Strategies

Since embeddings by themselves have no notion of order, we need positional information:

```
Position-Aware Embeddings = Token Embeddings + Positional Encoding

Learned Approach:     Fixed Mathematical Approach:
Position 0 → [learned]     Position 0 → [sin/cos pattern]
Position 1 → [learned]     Position 1 → [sin/cos pattern]
Position 2 → [learned]     Position 2 → [sin/cos pattern]
...                        ...
```

**Learned Positional Encoding**:
- Trainable position embeddings
- Can learn task-specific patterns
- Limited to maximum training sequence length

**Sinusoidal Positional Encoding**:
- Mathematical sine/cosine patterns
- No additional parameters
- Can extrapolate to longer sequences

### Strategy Comparison

```
Text: "cat sat on mat" → Token IDs: [42, 7, 15, 99]

Token Embeddings:    [vec_42, vec_7, vec_15, vec_99]  # Same vectors anywhere
Position-Aware:      [vec_42+pos_0, vec_7+pos_1, vec_15+pos_2, vec_99+pos_3]
                      ↑ Now "cat" at position 0 ≠ "cat" at position 1
```

The combination enables transformers to understand both meaning and order!

## 🏗️ Implementation: Building Embedding Systems

Let's implement embedding systems from basic token lookup to sophisticated position-aware representations. We'll start with the core embedding layer and work up to complete systems.

### Gradient Computation for Embedding Lookups

Now that you understand how embedding lookups work (index → row of the weight matrix),
let's think about how gradients flow backward through this operation.

The forward pass is a **gather** — we select rows from the weight matrix by index.
The backward pass is a **scatter** — we distribute gradients back to the rows that were selected.

```
Forward (gather):                    Backward (scatter):
Weight Table:                        Gradient Table:
  Row 0: [0.1, 0.2]  ← selected       Row 0: [2, 2]  ← accumulated (selected twice!)
  Row 1: [0.3, 0.4]                    Row 1: [0, 0]  ← not selected
  Row 2: [0.5, 0.6]  ← selected       Row 2: [1, 1]  ← selected once

Indices: [0, 2, 0]                   grad_output: [[1,1], [1,1], [1,1]]
Output:  [[0.1, 0.2],               Row 0 gets grad[0] + grad[2] = [2, 2]
          [0.5, 0.6],               Row 2 gets grad[1] = [1, 1]
          [0.1, 0.2]]
```

**Key insight**: When the same token appears multiple times in a sequence (like word "the"),
its embedding row accumulates gradients from every position. This is why `np.add.at` is
essential — standard indexing would overwrite instead of accumulating.

### 🧪 Unit Test: Embedding.__init__

**What we're testing**: Weight matrix initialization with correct shape and Xavier scaling
**Why it matters**: Bad initialization causes vanishing/exploding gradients from the start
**Expected**: Weight shape is (vocab_size, embed_dim), values are within Xavier bounds

### 🧪 Unit Test: Embedding.forward

This test validates our Embedding class works correctly with various token indices and batch configurations.

**What we're testing**: Token embedding lookup and parameter management
**Why it matters**: Foundation for all NLP models - if embedding fails, nothing works
**Expected**: Correct shape output, consistent lookups, proper parameter access

### Learned Positional Encoding

Trainable position embeddings that can learn position-specific patterns. This approach treats each position as a learnable parameter, similar to token embeddings.

```
Learned Position Embedding Process:

Step 1: Initialize Position Embedding Table
┌───────────────────────────────────────────────────────────────┐
│ Position  │  Learnable Vector (trainable parameters)          │
├───────────────────────────────────────────────────────────────┤
│    0      │ [0.1, -0.2,  0.4, ...]  ← learns "start" patterns │
│    1      │ [0.3,  0.1, -0.1, ...]  ← learns "second" patterns│
│    2      │ [-0.1, 0.5,  0.2, ...]  ← learns "third" patterns │
│   ...     │        ...                                        │
│  511      │ [0.4, -0.3,  0.1, ...]  ← learns "late" patterns  │
└───────────────────────────────────────────────────────────────┘

Step 2: Add to Token Embeddings
Input: ["The", "cat", "sat"] → Token IDs: [1, 42, 7]

Token embeddings:     Position embeddings:     Combined:
[1]  → [0.1, 0.4, ...] + [0.1, -0.2, ...] = [0.2, 0.2, ...]
[42] → [0.7, -0.2, ...] + [0.3, 0.1, ...] = [1.0, -0.1, ...]
[7]  → [-0.3, 0.1, ...] + [-0.1, 0.5, ...] = [-0.4, 0.6, ...]

Result: Position-aware embeddings that can learn task-specific patterns!
```

**Why learned positions work**: The model can discover that certain positions have special meaning (like sentence beginnings, question words, etc.) and learn specific representations for those patterns.

### Implementing Learned Positional Encoding

Let's build trainable positional embeddings that can learn position-specific patterns for our specific task.

### 🧪 Unit Test: PositionalEncoding.__init__

**What we're testing**: Position embedding matrix initialization with correct shape
**Why it matters**: Wrong shape or scale breaks the additive position signal
**Expected**: Matrix shape is (max_seq_len, embed_dim), values are small (additive)

### 🧪 Unit Test: PositionalEncoding.forward

This test validates our PositionalEncoding class works correctly with various sequence lengths and configurations.

**What we're testing**: Position embedding consistency and shape handling
**Why it matters**: Position awareness is critical for sequence understanding
**Expected**: Consistent encodings, correct shapes, proper parameter management

### Sinusoidal Positional Encoding

Mathematical position encoding that creates unique signatures for each position using trigonometric functions. This approach requires no additional parameters and can extrapolate to sequences longer than seen during training.

```
┌───────────────────────────────────────────────────────────────────────┐
│ SINUSOIDAL POSITION ENCODING: Mathematical Position Signatures        │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│ MATHEMATICAL FORMULA:                                                 │
│ ┌───────────────────────────────────────────────────────────────────┐ │
│ │ PE(pos, 2i)   = sin(pos / 10000^(2i/embed_dim))  # Even dims      │ │
│ │ PE(pos, 2i+1) = cos(pos / 10000^(2i/embed_dim))  # Odd dims       │ │
│ │                                                                   │ │
│ │ Where:                                                            │ │
│ │   pos = position in sequence (0, 1, 2, ...)                       │ │
│ │   i = dimension pair index (0, 1, 2, ...)                         │ │
│ │   10000 = base frequency (creates different wavelengths)          │ │
│ └───────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│ FREQUENCY PATTERN ACROSS DIMENSIONS:                                  │
│ ┌───────────────────────────────────────────────────────────────────┐ │
│ │ Dimension:  0     1     2     3     4     5     6     7           │ │
│ │ Frequency:  High  High  Med   Med   Low   Low   VLow  VLow        │ │
│ │ Function:   sin   cos   sin   cos   sin   cos   sin   cos         │ │
│ │                                                                   │ │
│ │ pos=0:    [0.00, 1.00, 0.00, 1.00, 0.00, 1.00, 0.00, 1.00]        │ │
│ │ pos=1:    [0.84, 0.54, 0.01, 1.00, 0.00, 1.00, 0.00, 1.00]        │ │
│ │ pos=2:    [0.91,-0.42, 0.02, 1.00, 0.00, 1.00, 0.00, 1.00]        │ │
│ │ pos=3:    [0.14,-0.99, 0.03, 1.00, 0.00, 1.00, 0.00, 1.00]        │ │
│ │                                                                   │ │
│ │ Each position gets a unique mathematical "fingerprint"!           │ │
│ └───────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│ WHY THIS WORKS:                                                       │
│ ┌───────────────────────────────────────────────────────────────────┐ │
│ │ Wave Pattern Visualization:                                       │ │
│ │                                                                   │ │
│ │ Dim 0: ∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿  (rapid oscillation)                  │ │
│ │ Dim 2: ∿---∿---∿---∿---∿---∿  (medium frequency)                  │ │
│ │ Dim 4: ∿-----∿-----∿-----∿--  (low frequency)                     │ │
│ │ Dim 6: ∿----------∿----------  (very slow changes)                │ │
│ │                                                                   │ │
│ │ • High frequency dims change rapidly between positions            │ │
│ │ • Low frequency dims change slowly                                │ │
│ │ • Combination creates unique signature for each position          │ │
│ │ • Similar positions have similar (but distinct) encodings         │ │
│ └───────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│ KEY ADVANTAGES:                                                       │
│ • Zero parameters (no memory overhead)                                │
│ • Infinite sequence length (can extrapolate)                          │
│ • Smooth transitions (nearby positions are similar)                   │
│ • Mathematical elegance (interpretable patterns)                      │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

**Why this matters**: The mathematical structure creates unique positional signatures and enables smooth interpolation to longer sequences. Attention mechanisms leverage these properties to distinguish token positions.

### Computing the Sinusoidal Table

The core of sinusoidal positional encoding is building a table of sin/cos values
where each dimension oscillates at a different frequency. This helper computes
the raw numpy array that both `create_sinusoidal_embeddings` and other components
can reuse.

```
Sinusoidal Table Construction:

Step 1: Position column vector     Step 2: Frequency row vector
  [0]                                [high_freq, ..., low_freq]
  [1]     (max_len, 1)               (embed_dim//2,)
  [2]
  [...]

Step 3: Outer product → angles     Step 4: Interleave sin/cos
  positions * frequencies            pe[:, 0::2] = sin(angles)
  = (max_len, embed_dim//2)          pe[:, 1::2] = cos(angles)
                                     = (max_len, embed_dim)
```

### 🧪 Unit Test: Sinusoidal Table Computation

This test validates the helper that builds the raw sin/cos table before it gets
wrapped in a Tensor.

**What we're testing**: Correct sin/cos alternation and frequency decay across dimensions
**Why it matters**: The table is the mathematical core of sinusoidal positional encoding
**Expected**: sin(0)=0 at even dims, cos(0)=1 at odd dims, higher dims change slower

### Implementing Sinusoidal Positional Encodings

Now we compose the table computation into the public API that returns a Tensor
ready for use in embedding pipelines.

### 🧪 Unit Test: Sinusoidal Embeddings

This test validates our sinusoidal positional encoding function creates correct mathematical patterns.

**What we're testing**: Sinusoidal pattern generation and frequency properties
**Why it matters**: Enables position awareness without trainable parameters
**Expected**: Correct sin/cos patterns, unique positions, frequency decay

## 🔧 Integration: Bringing It Together

Now let's build the complete embedding system that combines token and positional embeddings into a production-ready component. This is the same pattern used in modern language models.

```
Complete Embedding Pipeline:

1. Token Lookup → 2. Position Encoding → 3. Combination → 4. Ready for Attention
     ↓                     ↓                   ↓                  ↓
  sparse IDs         position info       dense vectors      context-aware
```

### Complete Embedding System Architecture

The production embedding layer that powers modern transformers combines multiple components into an efficient, flexible pipeline.

```
┌───────────────────────────────────────────────────────────────────────────┐
│ COMPLETE EMBEDDING SYSTEM: Token + Position → Position-Aware Representations│
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│ INPUT: Token IDs [1, 42, 7, 99]                                           │
│         │                                                                 │
│         ├─ STEP 1: TOKEN EMBEDDING LOOKUP                                 │
│         │  ┌─────────────────────────────────────────────────────────┐    │
│         │  │   Token Embedding Table (vocab_size × embed_dim)        │    │
│         │  │                                                         │    │
│         │  │   ID 1  → [0.1,  0.4, -0.2, ...]  (semantic features)   │    │
│         │  │   ID 42 → [0.7, -0.2,  0.1, ...]  (learned meaning)     │    │
│         │  │   ID 7  → [-0.3, 0.1,  0.5, ...]  (dense vector)        │    │
│         │  │   ID 99 → [0.9, -0.1,  0.3, ...]  (context-free)        │    │
│         │  └─────────────────────────────────────────────────────────┘    │
│         │                                                                 │
│         ├─ STEP 2: POSITIONAL ENCODING (Choose Strategy)                  │
│         │  ┌─────────────────────────────────────────────────────────┐    │
│         │  │ Strategy A: Learned PE                                  │    │
│         │  │   pos 0 → [trainable vector] (learns patterns)          │    │
│         │  │   pos 1 → [trainable vector] (task-specific)            │    │
│         │  │   pos 2 → [trainable vector] (fixed max length)         │    │
│         │  │                                                         │    │
│         │  │ Strategy B: Sinusoidal PE                               │    │
│         │  │   pos 0 → [sin/cos pattern] (mathematical)              │    │
│         │  │   pos 1 → [sin/cos pattern] (no parameters)             │    │
│         │  │   pos 2 → [sin/cos pattern] (infinite length)           │    │
│         │  │                                                         │    │
│         │  │ Strategy C: No PE                                       │    │
│         │  │   positions ignored (order-agnostic)                    │    │
│         │  └─────────────────────────────────────────────────────────┘    │
│         │                                                                 │
│         ├─ STEP 3: ELEMENT-WISE ADDITION                                  │
│         │  ┌─────────────────────────────────────────────────────────┐    │
│         │  │ Token + Position = Position-Aware Representation        │    │
│         │  │                                                         │    │
│         │  │ [0.1, 0.4, -0.2] + [pos0] = [0.1+p0, 0.4+p0, ...]       │    │
│         │  │ [0.7, -0.2, 0.1] + [pos1] = [0.7+p1, -0.2+p1, ...]      │    │
│         │  │ [-0.3, 0.1, 0.5] + [pos2] = [-0.3+p2, 0.1+p2, ...]      │    │
│         │  │ [0.9, -0.1, 0.3] + [pos3] = [0.9+p3, -0.1+p3, ...]      │    │
│         │  └─────────────────────────────────────────────────────────┘    │
│         │                                                                 │
│         ├─ STEP 4: OPTIONAL SCALING (Transformer Convention)              │
│         │  ┌─────────────────────────────────────────────────────────┐    │
│         │  │ Scale by √embed_dim for gradient stability              │    │
│         │  │ Helps balance token and position magnitudes             │    │
│         │  └─────────────────────────────────────────────────────────┘    │
│         │                                                                 │
│         └─ OUTPUT: Position-Aware Dense Vectors                           │
│            Ready for attention mechanisms and transformers!               │
│                                                                           │
│ INTEGRATION FEATURES:                                                     │
│ • Flexible position encoding (learned/sinusoidal/none)                    │
│ • Efficient batch processing with variable sequence lengths               │
│ • Memory optimization (shared position encodings)                         │
│ • Production patterns (matches PyTorch/HuggingFace)                       │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

**Why this architecture works**: By separating token semantics from positional information, the model can learn meaning and order independently, then combine them optimally for the specific task.

### EmbeddingLayer Initialization

The `__init__` method assembles the sub-components: a token `Embedding` for
vocabulary lookup and one of three positional encoding strategies.

```
EmbeddingLayer.__init__ assembles sub-components:

  vocab_size, embed_dim, pos_encoding
         │
         ├─ self.token_embedding = Embedding(vocab_size, embed_dim)
         │
         └─ self.pos_encoding =
              'learned'    → PositionalEncoding(max_seq_len, embed_dim)
              'sinusoidal'  → create_sinusoidal_embeddings(max_seq_len, embed_dim)
              None          → None  (no positional information)
```

### 🧪 Unit Test: EmbeddingLayer Initialization

This test validates that `__init__` correctly assembles sub-components for each
positional encoding strategy.

**What we're testing**: Sub-component creation and configuration storage
**Why it matters**: Incorrect initialization cascades into broken forward passes
**Expected**: Correct component types, parameter counts, and error on invalid strategy

### EmbeddingLayer Forward Pass

The `forward` method composes the full embedding pipeline: token lookup,
optional scaling, positional encoding addition, and batch dimension handling.

```
EmbeddingLayer.forward pipeline:

  tokens (batch, seq) or (seq,)
         │
         ├─ 1D? Add batch dim → (1, seq)
         │
         ├─ Token lookup → (batch, seq, embed)
         │
         ├─ Scale by √embed_dim? (optional)
         │
         ├─ Add positional encoding
         │    learned:    pos_encoding.forward(token_embeds)
         │    sinusoidal: token_embeds + sinusoidal_table[:seq_len]
         │    None:       pass through
         │
         └─ Squeeze batch if added → output
```

### 🧪 Unit Test: EmbeddingLayer Forward Pass

This test validates the forward composition: token lookup + scaling + positional
encoding addition across all three PE strategies.

**What we're testing**: Token + positional embedding integration, scaling, and batch processing
**Why it matters**: Production transformers use this exact pattern
**Expected**: Correct shapes, proper scaling, flexible position encoding support

### 🧪 Unit Test: Complete Embedding System

This test validates our EmbeddingLayer combines all components correctly for production use.

**What we're testing**: Token + positional embedding integration, scaling, and batch processing
**Why it matters**: Production transformers use this exact pattern
**Expected**: Correct shapes, proper scaling, flexible position encoding support

## 📊 Systems Analysis: Embedding Trade-offs

Understanding the performance implications of different embedding strategies is crucial for building efficient NLP systems that scale to production workloads.

## 🧪 Module Integration Test

Final validation that everything works together correctly before module completion.

## 🤔 ML Systems Reflection Questions

Answer these to deepen your understanding of embedding systems and their implications:

### 1. Memory Scaling
You implemented an embedding layer with vocab_size=50,000 and embed_dim=512.
- How many parameters does this embedding table contain? _____ million
- If using FP32 (4 bytes per parameter), how much memory does this use? _____ MB
- If you double the embedding dimension to 1024, what happens to memory usage? _____ MB

---

### 2. Lookup Complexity
Your embedding layer performs table lookups for token indices.
- What is the time complexity of looking up a single token? O(_____)
- For a batch of 32 sequences, each of length 128, how many lookup operations? _____
- Why doesn't vocabulary size affect individual lookup performance? _____

---

### 3. Positional Encoding Trade-offs
You implemented both learned and sinusoidal positional encodings.
- Learned PE for max_seq_len=2048, embed_dim=512 adds how many parameters? _____
- What happens if you try to process a sequence longer than max_seq_len with learned PE? _____
- Which type of PE can handle sequences longer than seen during training? _____

---

### 4. Production Implications
Your complete EmbeddingLayer combines token and positional embeddings.
- In GPT-3 (vocab_size≈50K, embed_dim≈12K), approximately what percentage of total parameters are in the embedding table? _____%
- If you wanted to reduce memory usage by 50%, which would be more effective: halving vocab_size or halving embed_dim? _____
- Why might sinusoidal PE be preferred for models that need to handle variable sequence lengths? _____

## ⭐ Aha Moment: Tokens Become Vectors

**What you built:** An embedding layer that converts token IDs to dense vectors.

**Why it matters:** Tokens are just integers (like word IDs), but embeddings give them meaning!
Each token gets a learned vector that captures its semantic properties. Similar words end up
with similar vectors—this is how models understand language.

In the next module, you'll use attention to let these embeddings interact with each other.

## 🚀 MODULE SUMMARY: Embeddings

Congratulations! You've built a complete embedding system that transforms discrete tokens into learnable representations!

### Key Accomplishments
- **Built Embedding class** with efficient token-to-vector lookup and Xavier initialization
- **Implemented PositionalEncoding** for learnable position-specific patterns
- **Created sinusoidal embeddings** using the Transformer paper formula for extrapolation
- **Developed EmbeddingLayer** combining token and positional embeddings (production-ready)
- **All tests pass** (validated by `test_module()`)

### Systems Insights Discovered
- **Memory scaling**: Embedding tables grow linearly with vocab_size x embed_dim
- **Lookup efficiency**: O(1) per token regardless of vocabulary size
- **Positional trade-offs**: Learned PE is task-specific; sinusoidal PE extrapolates to longer sequences
- **Production patterns**: GPT-3's embedding table alone uses ~2.4GB of memory

### Ready for Next Steps
Your embeddings implementation enables attention mechanisms and transformer architectures.
Export with: `tito module complete 11`

**Next**: Module 12 will add attention mechanisms for context-aware representations!
