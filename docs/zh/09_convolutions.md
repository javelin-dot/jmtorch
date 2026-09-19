> **文档来源**：`src/09_convolutions/09_convolutions.py` 中的 Markdown 教学说明（英文原版，本模块尚无 `*_zh.py`）。  
> 下文保留原模块一级标题；组件主题下的从属小节已下调一级，避免同级标题重复。

# Module 09: Convolutions - Processing Images with Convolutions

Welcome to Module 09! You'll implement spatial operations that transform machine learning from working with simple vectors to understanding images and spatial patterns.

## 🔗 Prerequisites & Progress
**You've Built**: Complete training pipeline with MLPs, optimizers, and data loaders
**You'll Build**: Spatial operations - Conv2d, MaxPool2d, AvgPool2d for image processing
**You'll Enable**: Convolutional Neural Networks (CNNs) for computer vision

**Connection Map**:
```
Training Pipeline → Spatial Operations → CNN (Milestone 03)
    (MLPs)            (Conv/Pool)        (Computer Vision)
```

## 🎯 Learning Objectives
By the end of this module, you will:
1. Implement Conv2d with explicit loops to understand O(N²M²K²) complexity
2. Build pooling operations (Max and Average) for spatial reduction
3. Understand receptive fields and spatial feature extraction
4. Analyze memory vs computation trade-offs in spatial operations

Let's get started!

## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in `modules/09_convolutions/convolutions_dev.py`
**Building Side:** Code exports to `tinytorch.core.spatial`

```python
# How to use this module:
from tinytorch.core.spatial import Conv2d, MaxPool2d, AvgPool2d
```

**Why this matters:**
- **Learning:** Complete spatial processing system in one focused module for deep understanding
- **Production:** Proper organization like PyTorch's torch.nn.Conv2d with all spatial operations together
- **Consistency:** All convolution and pooling operations in core.spatial
- **Integration:** Works seamlessly with existing layers for complete CNN architectures

## 📋 Module Dependencies

**Prerequisites**: Modules 01-08 must be complete

**External Dependencies**:
- `numpy` (for array operations and numerical computing)
- `time` (for performance measurements)

**TinyTorch Dependencies**:
- `tinytorch.core.tensor` (Tensor class from Module 01)
- `tinytorch.core.autograd` (gradient tracking from Module 06)

**Important**: This module builds on the complete training pipeline.
Spatial operations will integrate with your existing layers and training system.

**Dependency Flow**:
```
Training Pipeline (Modules 01-08) → Spatial Operations (Module 09) → CNNs (Milestone 03)
         ↓
  Complete training system enables CNN development
```

Students completing this module will have built the spatial processing
foundation that powers computer vision applications.

## 💡 Introduction: What are Spatial Operations?

Spatial operations transform machine learning from working with simple vectors to understanding images and spatial patterns. When you look at a photo, your brain naturally processes spatial relationships, including edges, textures, and objects. Spatial operations give neural networks this same capability.

### The Two Core Spatial Operations

**Convolution**: Detects local patterns by sliding filters across the input
**Pooling**: Reduces spatial dimensions while preserving important features

### Visual Example: How Convolution Works

```
Input Image (5×5):     Kernel (3×3):      Output (3×3):
┌───────────────┐      ┌──────────┐       ┌─────────┐
│ 1  2  3  4  5 │      │  1  0 -1 │       │ ?  ?  ? │
│ 6  7  8  9  0 │  *   │  1  0 -1 │   =   │ ?  ?  ? │
│ 1  2  3  4  5 │      │  1  0 -1 │       │ ?  ?  ? │
│ 6  7  8  9  0 │      └──────────┘       └─────────┘
│ 1  2  3  4  5 │
└───────────────┘

At each valid position, the kernel and overlapping image patch are multiplied
elementwise and summed.
Sliding Window Process:

Position (0,0): [1,2,3]      Position (0,1): [2,3,4]       Position (0,2): [3,4,5]
                [6,7,8]                      [7,8,9]                       [8,9,0]
                [1,2,3]                      [2,3,4]                       [3,4,5]
                = Output[0,0]                = Output[0,1]                 = Output[0,2]
```

Each output pixel summarizes a local neighborhood, allowing the network to
detect edges, corners, and textures while preserving their spatial relationships.

### Why Spatial Operations Transform ML

```
Without Convolution:                       With Convolution:
32×32×3 image = 3,072 inputs               32×32×3 → Conv → 30×30×16
        ↓                                             ↓
Dense(3072 → 1000) = 3M parameters         16 shared 3×3×3 kernels = 432 weights
        ↓                                             ↓
Memory explosion + no spatial awareness    Efficient + preserves spatial structure
```

Convolution achieves dramatic parameter reduction (more than 1000×!) while preserving the spatial relationships that matter for visual understanding.

## 📐 Mathematical Foundations

### Understanding Convolution Step by Step

Convolution sounds complex, but it's just a "sliding window dot product".
Let's see exactly how it works:

<pre>
Step 1: Position the kernel over input
Input:          Kernel:
┌─────────┐     ┌─────┐
│ <span style="color:blue">1 2</span> 3 4 │     │ 1 0 │  ← If we place kernel at position (0,0)
│ <span style="color:blue">5 6</span> 7 8 │  ×  │ 0 1 │
│ 9 0 1 2 │     └─────┘
└─────────┘

Step 2: Multiply corresponding elements
Overlap:        Computation:
┌─────┐         1×1 + 2×0 + 5×0 + 6×1 = 1 + 0 + 0 + 6 = 7
│ 1 2 │
│ 5 6 │
└─────┘

Step 3: Slide kernel and repeat
Position (0,1):  Position (0,2):
   ┌─────┐          ┌─────┐
   │ 2 3 │          │ 3 4 │
   │ 6 7 │          │ 7 8 │
   └─────┘          └─────┘
   Result: 9        Result: 11

Position (1,0):  Position (1,1):  Position (1,2):
   ┌─────┐          ┌─────┐          ┌─────┐
   │ 5 6 │          │ 6 7 │          │ 7 8 │
   │ 9 0 │          │ 0 1 │          │ 1 2 │
   └─────┘          └─────┘          └─────┘
   Result: 5        Result: 7        Result: 9

Final Output:  ┌────────┐
               │ 7 9 11 │
               │ 5 7 9  │
               └────────┘
</pre>

### The Mathematical Formula

Suppose `n` is the kernel height (rows) and `m` the kernel width (columns).
Then, for a 2D convolution, we slide kernel K across input I:
```
O[i,j] = Σ Σ I[i+n, j+m] × K[n,m]
         n m
```

This formula captures the "multiply and sum" operation for each kernel position.

### Pooling: Spatial Summarization

```
Max Pooling Example (2×2 window):
Input:             Output:
┌────────────┐  ┌───────┐
│ 1  3  2  4 │  │ 6   8 │  ← max([1,3,5,6])=6, max([2,4,7,8])=8
│ 5  6  7  8 │  │ 9   9 │  ← max([2,9,0,1])=9, max([1,3,9,3])=9
│ 2  9  1  3 │  └───────┘
│ 0  1  9  3 │
└────────────┘

Average Pooling (same window):
┌────────────┐
│ 3.75  5.25 │  ← avg([1,3,5,6])=3.75, avg([2,4,7,8])=5.25
│ 3.0   4.0  │  ← avg([2,9,0,1])=3.0, avg([1,3,9,3])=4.0
└────────────┘
```

#### Why This Complexity Matters

For convolution with input (1, 3, 224, 224) and kernel (64, 3, 3, 3):
- **Operations**: 1 × 64 × 3 × 3 × 3 × 224 × 224 = 86.7 million multiply-adds
- **Memory**: Input (600KB) + Weights (6.9KB) + Output (12.8MB) = ~13.4MB

This is why kernel size matters enormously - a 7×7 kernel would require ~5.4× more computation!

#### Key Properties That Enable Deep Learning

**Translation Equivariance**: Move the cat → detection moves the same way
**Parameter Sharing**: Same edge detector works everywhere in the image
**Local Connectivity**: Each output only looks at nearby inputs (like human vision)
**Hierarchical Features**: Early layers detect edges → later layers detect objects

## 🏗️ Implementation: Building Spatial Operations

Now we'll implement convolution step by step, using explicit loops so you can see and feel the computational complexity. This helps you understand why modern optimizations matter!

### Conv2d: Detecting Patterns with Sliding Windows

Convolution slides a small filter (kernel) across the entire input, computing weighted sums at each position. Think of it like using a template to find matching patterns everywhere in an image.

```
Convolution Visualization:
Input (4×4):          Kernel (3×3):           Output (2×2):
┌─────────┐           ┌──────────┐             ┌───────┐
│ a b c d │           │ k1 k2 k3 │             │ o1 o2 │
│ e f g h │     ×     │ k4 k5 k6 │      =      │ o3 o4 │
│ i j k l │           │ k7 k8 k9 │             └───────┘
│ m n o p │           └──────────┘
└─────────┘

Computation Details:
o1 = a×k1 + b×k2 + c×k3 + e×k4 + f×k5 + g×k6 + i×k7 + j×k8 + k×k9
o2 = b×k1 + c×k2 + d×k3 + f×k4 + g×k5 + h×k6 + j×k7 + k×k8 + l×k9
o3 = e×k1 + f×k2 + g×k3 + i×k4 + j×k5 + k×k6 + m×k7 + n×k8 + o×k9
o4 = f×k1 + g×k2 + h×k3 + j×k4 + k×k5 + l×k6 + n×k7 + o×k8 + p×k9
```

#### The Six Nested Loops of Convolution

Our implementation will use explicit loops to show exactly where the computational cost comes from:

```
for batch in range(B):          # Loop 1: Process each sample
    for out_ch in range(C_out):     # Loop 2: Generate each output channel
        for out_h in range(H_out):      # Loop 3: Each output row
            for out_w in range(W_out):      # Loop 4: Each output column
                for k_h in range(K_h):          # Loop 5: Each kernel row
                    for k_w in range(K_w):          # Loop 6: Each kernel column
                        for in_ch in range(C_in):       # Loop 7: Each input channel
                            # The actual multiply-accumulate operation
                            result += input[...] * kernel[...]
```

Total operations: B × C_out × H_out × W_out × K_h × K_w × C_in

For typical values (B=32, C_out=64, H_out=224, W_out=224, K_h=3, K_w=3, C_in=3):
That's 32 × 64 × 224 × 224 × 3 × 3 × 3 = **2.8 billion operations** per forward pass!

#### Shared Input Validation

All spatial operations (Conv2d, MaxPool2d, AvgPool2d) require 4D inputs shaped
as (batch, channels, height, width). Rather than duplicating this validation
logic three times, we define it once here.

This is NOT a student task -- it is shared infrastructure.

### Conv2d Implementation - Building the Core of Computer Vision

Conv2d is the workhorse of computer vision. It slides learned filters across images to detect patterns like edges, textures, and eventually complex objects.

#### How Conv2d Transforms Machine Learning

```
Before Conv2d (Dense Only):        After Conv2d (Spatial Aware):
Input: 32×32×3 = 3,072 values      Input: 32×32×3 structured as image
         ↓                                   ↓
Dense(3072→1000) = 3M params       Conv2d(3→16, 3×3) = 448 params
         ↓                                   ↓
No spatial awareness               Preserves spatial relationships
Massive parameter count            Parameter sharing across space
```

#### Weight Initialization: He Initialization for ReLU Networks

Our Conv2d uses He initialization, specifically designed for ReLU activations:
- **Problem**: Wrong initialization → vanishing/exploding gradients
- **Solution**: std = sqrt(2 / fan_in) where fan_in = channels × kernel_height × kernel_width
- **Why it works**: Maintains variance through ReLU nonlinearity

#### The 6-Loop Implementation Strategy

We'll implement convolution with explicit loops to show the true computational cost:

```
Nested Loop Structure:
for batch:             ← Process each sample in parallel (in practice)
  for out_channel:     ← Generate each output feature map
    for out_h:         ← Each row of output
      for out_w:       ← Each column of output
        for k_h:       ← Each row of kernel
          for k_w:     ← Each column of kernel
            for in_ch: ← Accumulate across input channels
              result += input[...] * weight[...]
```

This reveals why convolution is expensive: O(B×C_out×H×W×K_h×K_w×C_in) operations!

#### Implementation Strategy: Decomposed Helpers

We break Conv2d.forward into focused helpers so you can build and test each concept
independently:

1. **`_compute_output_shape(in_h, in_w)`** -- Apply the output dimension formula
2. **`_apply_padding(x_data)`** -- Zero-pad spatial dimensions
3. **`_convolve_loops(padded, batch, oh, ow)`** -- The sliding window dot products
4. **`forward(x)`** -- Compose all helpers together

Each helper teaches ONE concept. Build them in order, test each before moving on.

#### Step 1: Output Shape Formula

Before running convolution, you must know the output dimensions. The formula is:

```
out_height = (in_height + 2 * padding - kernel_h) // stride + 1
out_width  = (in_width  + 2 * padding - kernel_w) // stride + 1
```

Examples with 32x32 input and 3x3 kernel:
- padding=0, stride=1: (32 + 0 - 3) // 1 + 1 = 30 (shrinks)
- padding=1, stride=1: (32 + 2 - 3) // 1 + 1 = 32 (same size)
- padding=0, stride=2: (32 + 0 - 3) // 2 + 1 = 15 (downsamples)

#### Step 2: Zero Padding

Padding adds zeros around the spatial borders of the input, which controls
whether convolution preserves the spatial size:

```
Before padding (3x3):      After padding=1 (5x5):
┌─────────┐                ┌───────────────┐
│ 1  2  3 │                │ 0  0  0  0  0 │
│ 4  5  6 │    -->         │ 0  1  2  3  0 │
│ 7  8  9 │                │ 0  4  5  6  0 │
└─────────┘                │ 0  7  8  9  0 │
                           │ 0  0  0  0  0 │
                           └───────────────┘
```

Only spatial dimensions (height, width) are padded. Batch and channel dims
stay unchanged.

#### Step 3: The Convolution Loops

This is the core computation. For each output position, you extract a patch
from the input and compute its dot product with the kernel:

```
Convolution = Sliding Window Dot Products:
┌────────────────────────────────────┐
│ For EACH output position:          │
│  Input patch      Kernel           │
│  ┌───┬───┬───┐   ┌────┬────┬────┐  │
│  │ a │ b │ c │   │ w₁ │ w₂ │ w₃ │  │
│  ├───┼───┼───┤   ├────┼────┼────┤  │
│  │ d │ e │ f │ × │ w₄ │ w₅ │ w₆ │  │
│  ├───┼───┼───┤   ├────┼────┼────┤  │
│  │ g │ h │ i │   │ w₇ │ w₈ │ w₉ │  │
│  └───┴───┴───┘   └────┴────┴────┘  │
│ output = a·w₁ + b·w₂ + ... + i·w₉  │
└────────────────────────────────────┘
```

The 7 nested loops iterate over:
- batch (each image in the batch)
- output channel (each filter)
- output height and width (each spatial position)
- kernel height, kernel width, and input channel (the dot product)

#### 🧪 Unit Test: Conv2d Output Shape Computation

This test validates that `_compute_output_shape` correctly applies the
convolution output dimension formula.

```
Output size formula:
  out = (in + 2 * padding - kernel) // stride + 1

Examples:
  (32 + 2*1 - 3) // 1 + 1 = 32  (same padding)
  (32 + 2*0 - 3) // 1 + 1 = 30  (no padding)
  (32 + 2*0 - 3) // 2 + 1 = 15  (stride 2)
```

**What we're testing**: Output dimension formula for various configurations
**Why it matters**: Wrong dimensions cause silent shape bugs in CNNs
**Expected**: Matches hand-calculated values

#### 🧪 Unit Test: Conv2d Padding

This test validates that `_apply_padding` correctly zero-pads the spatial
dimensions while leaving batch and channel dimensions untouched.

```
Before padding (1, 1, 3, 3):       After padding=1 (1, 1, 5, 5):
                                    ┌─────────────────┐
┌─────────┐                         │ 0  0  0  0  0   │
│ 1  2  3 │                         │ 0  1  2  3  0   │
│ 4  5  6 │          -->            │ 0  4  5  6  0   │
│ 7  8  9 │                         │ 0  7  8  9  0   │
└─────────┘                         │ 0  0  0  0  0   │
                                    └─────────────────┘
```

**What we're testing**: Zero-padding adds correct borders to spatial dims
**Why it matters**: Padding controls whether convolution preserves spatial size
**Expected**: Padded array has correct shape and zero borders

#### 🧪 Unit Test: Conv2d Convolution Loops

This test validates the core sliding window computation in `_convolve_loops`.

```
Convolution = Sliding Window Dot Products:
┌────────────────────────────────────┐
│ For EACH output position:          │
│  Input patch      Kernel           │
│  ┌───┬───┬───┐   ┌────┬────┬────┐  │
│  │ a │ b │ c │   │ w₁ │ w₂ │ w₃ │  │
│  ├───┼───┼───┤   ├────┼────┼────┤  │
│  │ d │ e │ f │ × │ w₄ │ w₅ │ w₆ │  │
│  ├───┼───┼───┤   ├────┼────┼────┤  │
│  │ g │ h │ i │   │ w₇ │ w₈ │ w₉ │  │
│  └───┴───┴───┘   └────┴────┴────┘  │
│ output = a·w₁ + b·w₂ + ... + i·w₉  │
└────────────────────────────────────┘
```

**What we're testing**: The 7-nested-loop convolution produces correct values
**Why it matters**: This is THE core operation of computer vision
**Expected**: Output matches hand-computed convolution results

#### 🧪 Unit Test: Conv2d Forward (Composition)

This test validates the complete `forward` method that composes all helpers
together: validation, shape computation, padding, convolution loops, bias,
and gradient tracking.

**What we're testing**: End-to-end Conv2d with shape preservation, padding, stride
**Why it matters**: Convolution is the foundation of computer vision
**Expected**: Correct output shapes and reasonable value ranges

## 🏗️ Pooling Operations - Spatial Dimension Reduction

Pooling operations compress spatial information while keeping the most important features. Think of them as creating "thumbnail summaries" of local regions.

### MaxPool2d: Keeping the Strongest Signals

Max pooling finds the strongest activation in each window, preserving sharp features like edges and corners.

```
MaxPool2d Example (2×2 kernel, stride=2):
Input (4×4):              Windows:               Output (2×2):
┌─────────────┐          ┌─────┬─────┐          ┌───────┐
│ 1  3 │ 2  8 │          │ 1 3 │ 2 8 │          │ 6   8 │
│ 5  6 │ 7  4 │    →     │ 5 6 │ 7 4 │    →     │ 9   7 │
├──────┼──────┤          ├─────┼─────┤          └───────┘
│ 2  9 │ 1  7 │          │ 2 9 │ 1 7 │
│ 0  1 │ 3  6 │          │ 0 1 │ 3 6 │
└─────────────┘          └─────┴─────┘

Window Computations:
Top-left: max(1,3,5,6) = 6     Top-right: max(2,8,7,4) = 8
Bottom-left: max(2,9,0,1) = 9  Bottom-right: max(1,7,3,6) = 7
```

### AvgPool2d: Smoothing Local Features

Average pooling computes the mean of each window, creating smoother, more general features.

```
AvgPool2d Example (same 2×2 kernel, stride=2):
Input (4×4):              Output (2×2):
┌─────────────┐          ┌─────────────┐
│ 1  3 │ 2  8 │          │ 3.75   5.25 │
│ 5  6 │ 7  4 │    →     │ 3.0    4.25 │
├──────┼──────┤          └─────────────┘
│ 2  9 │ 1  7 │
│ 0  1 │ 3  6 │
└─────────────┘

Window Computations:
Top-left: (1+3+5+6)/4 = 3.75    Top-right: (2+8+7+4)/4 = 5.25
Bottom-left: (2+9+0+1)/4 = 3.0  Bottom-right: (1+7+3+6)/4 = 4.25
```

#### Why Pooling Matters for Computer Vision

```
Memory Impact:
Input: 224×224×64 = 3.2M values    After 2×2 pooling: 112×112×64 = 0.8M values
Memory reduction: 4× less!         Computation reduction: 4× less!

Information Trade-off:
✅ Preserves important features     ⚠️ Loses fine spatial detail
✅ Provides translation invariance  ⚠️ Reduces localization precision
✅ Reduces overfitting             ⚠️ May lose small objects
```

#### Sliding Window Pattern

Both pooling operations follow the same sliding window pattern:

```
Sliding 2×2 window with stride=2 on a 4×4 matrix:
Step 1:       Step 2:       Step 3:       Step 4:
┌──┬──┐       ┌──┬──┐       ┌──┬──┐       ┌──┬──┐
│▓▓│  │       │  │▓▓│       │  │  │       │  │  │
├──┼──┤       ├──┼──┤       ├──┼──┤       ├──┼──┤
│  │  │       │  │  │       │▓▓│  │       │  │▓▓│
└──┴──┘       └──┴──┘       └──┴──┘       └──┴──┘

Non-overlapping windows → Each input pixel used exactly once
Stride=2 → Output dimensions halved in each direction
```

The key difference: MaxPool takes max(window), AvgPool takes mean(window).

### MaxPool2d Implementation - Preserving Strong Features

MaxPool2d finds the strongest activation in each spatial window, creating a compressed representation that keeps the most important information.

#### Why Max Pooling Works for Computer Vision

```
Edge Detection Example:
Input Window (2×2):         Max Pooling Result:
┌─────┬─────┐
│ 0.1 │ 0.8 │ ←  Strong edge signal
├─────┼─────┤
│ 0.2 │ 0.1 │              Output: 0.8 (preserves edge)
└─────┴─────┘

Noise Reduction Example:
Input Window (2×2):
┌─────┬─────┐
│ 0.9 │ 0.1 │ ←  Feature + noise
├─────┼─────┤
│ 0.2 │ 0.1 │              Output: 0.9 (removes noise)
└─────┴─────┘
```

#### The Sliding Window Pattern

```
MaxPool with 2×2 kernel, stride=2:

Input (4×4):                Output (2×2):
┌───┬───┬───┬───┐          ┌───────┬───────┐
│ a │ b │ c │ d │          │max(a,b│max(c,d│
├───┼───┼───┼───┤     →    │   e,f)│   g,h)│
│ e │ f │ g │ h │          ├───────┼───────┤
├───┼───┼───┼───┤          │max(i,j│max(k,l│
│ i │ j │ k │ l │          │   m,n)│   o,p)│
├───┼───┼───┼───┤          └───────┴───────┘
│ m │ n │ o │ p │
└───┴───┴───┴───┘

Benefits:
✓ Translation invariance (cat moved 1 pixel still detected)
✓ Computational efficiency (4× fewer values to process)
✓ Hierarchical feature building (next layer sees larger receptive field)
```

#### Memory and Computation Impact

For input (1, 64, 224, 224) with 2×2 pooling:
- **Input memory**: 64 × 224 × 224 × 4 bytes = 12.8 MB
- **Output memory**: 64 × 112 × 112 × 4 bytes = 3.2 MB
- **Memory reduction**: 4× less memory needed
- **Computation**: No parameters, minimal compute cost

#### Unit Test: MaxPool2d Output Shape

This test validates that `_compute_pool_output_shape` correctly computes
the spatial dimensions after max pooling.

```
Output size formula (same as convolution):
  out = (in + 2 * padding - kernel) // stride + 1

Common case: kernel=2, stride=2, padding=0
  (8 + 0 - 2) // 2 + 1 = 4  (spatial dimensions halved)
```

**What we're testing**: Pooling output dimension calculation
**Why it matters**: Wrong dimensions break the CNN dimension chain
**Expected**: Matches hand-calculated values for various configs

#### Unit Test: MaxPool2d Loops

This test validates that `_maxpool_loops` correctly finds the maximum
value in each pooling window.

```
MaxPool2d sliding window (2x2, stride 2):
┌─────┬─────┐    ┌─────┐
│ 1 3 │ 2 8 │    │ 6 8 │
│ 5 6 │ 7 4 │ -> │ 9 7 │
├─────┼─────┤    └─────┘
│ 2 9 │ 1 7 │
│ 0 1 │ 3 6 │
└─────┴─────┘
```

**What we're testing**: The max-finding loops produce correct values
**Why it matters**: Max pooling preserves the strongest activations
**Expected**: Output matches hand-computed max values per window

### AvgPool2d Implementation - Smoothing and Generalizing Features

AvgPool2d computes the average of each spatial window, creating smoother features that are less sensitive to noise and exact pixel positions.

#### MaxPool vs AvgPool: Different Philosophies

```
Same Input Window (2×2):    MaxPool Output:    AvgPool Output:
┌─────┬─────┐
│ 0.1 │ 0.9 │               0.9              0.425
├─────┼─────┤              (max)             (mean)
│ 0.3 │ 0.3 │
└─────┴─────┘

Interpretation:
MaxPool: "What's the strongest feature here?"
AvgPool: "What's the general feature level here?"
```

#### When to Use Average Pooling

```
Use Cases:
✓ Global Average Pooling (GAP) for classification
✓ When you want smoother, less noisy features
✓ When exact feature location doesn't matter
✓ In shallower networks where sharp features aren't critical

Typical Pattern:
Feature Maps → Global Average Pool → Dense → Classification
(256×7×7)   →        (256×1×1)      → FC   →    (10)
              Replaces flatten+dense with parameter reduction
```

#### Mathematical Implementation

```
Average Pooling Computation:
Window: [a, b]    Result = (a + b + c + d) / 4
        [c, d]

For efficiency, we:
1. Sum all values in window: window_sum = a + b + c + d
2. Divide by window area: result = window_sum / (kernel_h × kernel_w)
3. Store result at output position

Memory access pattern identical to MaxPool, just different aggregation!
```

#### Practical Considerations

- **Memory**: Same 4× reduction as MaxPool
- **Computation**: Slightly more expensive (sum + divide vs max)
- **Features**: Smoother, more generalized than MaxPool
- **Use**: Often in final layers (Global Average Pooling) to reduce parameters

#### Unit Test: AvgPool2d Output Shape

This test validates that `_compute_pool_output_shape` correctly computes
the spatial dimensions after average pooling.

**What we're testing**: Pooling output dimension calculation
**Why it matters**: Must match MaxPool2d formula for interchangeability
**Expected**: Same results as MaxPool2d for identical configurations

#### Unit Test: AvgPool2d Loops

This test validates that `_avgpool_loops` correctly computes the mean of
each pooling window.

```
AvgPool2d sliding window (2x2, stride 2):
┌─────┬─────┐    ┌───────────┐
│ 1 2 │ 3 4 │    │  3.5  5.5 │
│ 5 6 │ 7 8 │ -> │ 11.5 13.5 │
├─────┼─────┤    └───────────┘
│ 9 10│11 12│
│13 14│15 16│
└─────┴─────┘

Top-left: (1+2+5+6)/4 = 3.5
```

**What we're testing**: The sum-and-divide loops produce correct averages
**Why it matters**: Average pooling creates smoother features than max pooling
**Expected**: Output matches hand-computed averages per window

## 🏗️ Batch Normalization - Stabilizing Deep Network Training

Batch Normalization (BatchNorm) is one of the most important techniques for training deep networks. It normalizes activations across the batch dimension, dramatically improving training stability and speed.

### Why BatchNorm Matters

```
Without BatchNorm:                  With BatchNorm:
Layer outputs can have              Layer outputs are normalized
wildly varying scales:              to consistent scale:

Layer 1: mean=0.5, std=0.3         Layer 1: mean≈0, std≈1
Layer 5: mean=12.7, std=8.4   →    Layer 5: mean≈0, std≈1
Layer 10: mean=0.001, std=0.0003   Layer 10: mean≈0, std≈1

Result: Unstable gradients         Result: Stable training
        Slow convergence                   Fast convergence
        Careful learning rate              Robust to hyperparameters
```

### The BatchNorm Computation

For each channel c, BatchNorm computes:
```
1. Batch Statistics (during training):
   μ_c = mean(x[:, c, :, :])     # Mean over batch and spatial dims
   σ²_c = var(x[:, c, :, :])     # Variance over batch and spatial dims

2. Normalize:
   x̂_c = (x[:, c, :, :] - μ_c) / sqrt(σ²_c + ε)

3. Scale and Shift (learnable parameters):
   y_c = γ_c * x̂_c + β_c       # γ (gamma) and β (beta) are learned
```

### Train vs Eval Mode

This is a critical systems concept:

```
Training Mode:                      Eval Mode:
┌────────────────────┐             ┌────────────────────┐
│ Use batch stats    │             │ Use running stats  │
│ Update running     │             │ (accumulated from  │
│ mean/variance      │             │  training)         │
└────────────────────┘             └────────────────────┘
   ↓                                  ↓
Computes μ, σ² from                Uses frozen μ, σ² for
current batch                      consistent inference
```

**Why this matters**: During inference, you might process just 1 image. Batch statistics from 1 sample would be meaningless. Running statistics provide stable normalization.

### 🧪 Unit Test: BatchNorm2d._validate_input

**What we're testing**: Input shape validation catches common mistakes
**Why it matters**: Clear errors save hours of debugging wrong tensor shapes
**Expected**: Accepts valid 4D input, rejects 2D/3D/wrong channels with helpful messages

### 🧪 Unit Test: BatchNorm2d._get_stats

**What we're testing**: Statistics computation in training vs eval mode
**Why it matters**: Wrong statistics = wrong normalization = broken model
**Expected**: Training mode computes batch stats and updates running stats; eval mode uses frozen stats

### 🧪 Unit Test: BatchNorm2d

This test validates batch normalization implementation.

**What we're testing**: Normalization behavior, train/eval mode, running statistics
**Why it matters**: BatchNorm is essential for training deep CNNs effectively
**Expected**: Normalized outputs with proper mean/variance characteristics

### 🧪 Unit Test: Pooling Operations

This test validates both max and average pooling implementations.

**What we're testing**: Dimension reduction, aggregation correctness
**Why it matters**: Pooling is essential for computational efficiency in CNNs
**Expected**: Correct output shapes and proper value aggregation

## 📊 Systems Analysis: Spatial Operation Performance

Let's understand ONE key systems concept: **computational complexity and memory trade-offs in spatial operations**.

This single analysis reveals why certain design choices matter for real-world performance, and why modern CNNs use specific architectural patterns.

### SimpleCNN Implementation - Putting It All Together

Now we'll build a complete CNN that demonstrates how convolution and pooling work together. This is your first step from processing individual tensors to understanding complete images!

#### The CNN Architecture Pattern

```
SimpleCNN Architecture Visualization:

Input: (batch, 3, 32, 32)     ← RGB images
         ↓
┌─────────────────────────┐
│ Conv2d(3→16, 3×3, p=1)  │    ← Detect edges, textures
│ ReLU()                  │    ← Remove negative values
│ MaxPool(2×2)            │    ← Reduce to (batch, 16, 16, 16)
└─────────────────────────┘
         ↓
┌─────────────────────────┐
│ Conv2d(16→32, 3×3, p=1) │   ← Detect shapes, patterns
│ ReLU()                  │   ← Remove negative values
│ MaxPool(2×2)            │   ← Reduce to (batch, 32, 8, 8)
└─────────────────────────┘
         ↓
┌─────────────────────────┐
│ Flatten()               │   ← Reshape to (batch, 2048)
│ Linear(2048→10)         │   ← Final classification
└─────────────────────────┘
         ↓
Output: (batch, 10)           ← Class probabilities
```

#### Why This Architecture Works

```
Feature Hierarchy Development:

Raw RGB images → simple features → complex combinations → class prediction

Spatial Dimension Reduction:
32×32 → 16×16 → 8×8
1024     256    64  (per channel)

Channel Expansion:
3 → 16 → 32
More feature types at each level
```

#### Parameter Efficiency Demonstration

```
CNN vs Dense Comparison for 32×32×3 → 10 classes:

CNN Approach:                    Dense Approach:
┌────────────────────┐          ┌────────────────────┐
│ Conv1: 3→16, 3×3   │          │ Input: 3072 values │
│ Params: 448        │          │        ↓           │
├────────────────────┤          │ Dense: 3072→512    │
│ Conv2: 16→32, 3×3  │          │ Params: 1.57M      │
│ Params: 4,640      │          ├────────────────────┤
├────────────────────┤          │ Dense: 512→10      │
│ Dense: 2048→10     │          │ Params: 5,120      │
│ Params: 20,490     │          └────────────────────┘
└────────────────────┘          Total: 1.58M params
Total: 25,578 params

CNN has 62× fewer parameters while preserving spatial structure!
```

#### Receptive Field Growth

```
How each layer sees progressively larger input regions:

Layer 1 Conv (3×3):           Layer 2 Conv (3×3):
Each output pixel sees        Each output pixel sees
3×3 = 9 input pixels          8×8 = 64 input pixels
                              (after conv, pooling, and conv)

Final Result: Layer 2 can detect complex patterns
spanning 8×8 regions of original image!
```

#### 🧪 Unit Test: SimpleCNN Integration

This test validates that spatial operations work together in a complete CNN architecture.

**What we're testing**: End-to-end spatial processing pipeline
**Why it matters**: Spatial operations must compose correctly for real CNNs
**Expected**: Proper dimension reduction and feature extraction

## 🧪 Module Integration Test

Final validation that everything works together correctly.

## 🤔 ML Systems Reflection Questions

Answer these to deepen your understanding of spatial operations and their systems implications:

### 1. Conv2d Memory Footprint
A Conv2d layer with 64 filters (3×3) processes a (224×224×3) image.
- Calculate the memory footprint during the forward pass
- Consider: input activations, output activations, filter weights, and biases
- What happens when batch size increases from 1 to 32?

**Think about**: Why do modern vision models use techniques like gradient checkpointing?

---

### 2. Spatial Locality and CPU Performance
Why are CNNs faster on CPUs than fully-connected networks of similar parameter count?

**Consider**:
- Cache locality in convolution operations
- Data reuse patterns in sliding windows
- Memory access patterns (sequential vs random)

**Hint**: Think about what happens when the same filter is applied across the image.

---

### 3. Im2col Trade-off
The im2col algorithm transforms convolution into matrix multiplication, using more memory but speeding up computation.

**When is this trade-off worthwhile?**
- Small vs large batch sizes
- Small vs large images
- Training vs inference
- Mobile vs server deployment

**Think about**: Why don't mobile devices always use im2col?

---

### 4. Pooling's Systems Benefits
MaxPool2d reduces spatial dimensions (e.g., 224×224 → 112×112).

**What's the systems benefit beyond reducing parameters?**
- Memory bandwidth requirements
- Computation in subsequent layers
- Gradient memory during backpropagation
- Cache efficiency in deeper layers

**Calculate**: If 5 layers each use 2x2 pooling, what's the total memory reduction?

---

### 5. Mobile ML Deployment
Why do mobile ML models prefer depthwise-separable convolutions over standard Conv2d?

**Analyze the FLOPs**:
- Standard 3×3 conv: C_in × C_out × H × W × 9
- Depthwise + Pointwise: (C_in × H × W × 9) + (C_in × C_out × H × W)

**When does the trade-off favor depthwise separable?**
- As number of channels increases
- As spatial dimensions change
- Energy consumption vs accuracy

**Real-world context**: This is why MobileNet and EfficientNet architectures exist.

## ⭐ Aha Moment: Convolution Extracts Features

**What you built:** Convolutional layers that process spatial data like images.

**Why it matters:** Conv2d looks at local neighborhoods, detecting edges, textures, and patterns.
Unlike Linear layers that see pixels independently, Conv2d understands that nearby pixels are
related. This is why CNNs revolutionized computer vision!

In the milestones, you'll use these spatial operations to build a CNN that recognizes digits.

## 🚀 MODULE SUMMARY: Spatial Operations

Congratulations! You've built the spatial processing foundation that powers computer vision!

### Key Accomplishments
- **Built Conv2d** with explicit loops showing O(N^2 M^2 K^2) complexity
- **Implemented BatchNorm2d** with train/eval mode and running statistics
- **Implemented MaxPool2d and AvgPool2d** for spatial dimension reduction
- **Created SimpleCNN** demonstrating spatial operation integration
- **Analyzed computational complexity** and memory trade-offs in spatial processing
- **All tests pass** (validated by `test_module()`)

### Systems Insights Discovered
- **Convolution Complexity**: Quadratic scaling with spatial size; kernel size significantly impacts cost
- **Batch Normalization**: Train vs eval mode is critical (batch stats during training, running stats during inference)
- **Memory Patterns**: Pooling provides 4× memory reduction while preserving important features
- **Architecture Design**: Strategic spatial reduction enables parameter-efficient feature extraction
- **Cache Performance**: Spatial locality in convolution benefits from optimal memory access patterns

### Ready for Next Steps
Your spatial operations enable building complete CNNs for computer vision tasks!

**Next**: Milestone 03 will combine your spatial operations with training pipeline to build a CNN for CIFAR-10!

Export with: `tito module complete 09`
