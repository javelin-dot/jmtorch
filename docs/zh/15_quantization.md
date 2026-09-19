> **文档来源**：`src/15_quantization/15_quantization.py` 中的 Markdown 教学说明（英文原版，本模块尚无 `*_zh.py`）。  
> 下文保留原模块一级标题；组件主题下的从属小节已下调一级，避免同级标题重复。

# Module 15: Quantization - Reduced Precision for Efficiency

Welcome to Module 15! You're about to build a complete INT8 quantization system that can reduce model size by 4x with minimal accuracy loss.

## 🔗 Prerequisites & Progress
**You've Built**: Complete ML pipeline with profiling (Module 14)
**You'll Build**: INT8 quantization system with calibration and memory savings
**You'll Enable**: 4x memory reduction and 2-4x speedup for production deployment

**Connection Map**:
```
Profiling (14) → Quantization (15)
(measure memory)   (reduce precision)
```

## 🎯 Learning Objectives
By the end of this module, you will:
1. Implement INT8 quantization with proper scaling
2. Build quantization-aware training for minimal accuracy loss
3. Apply post-training quantization to existing models
4. Measure actual memory and compute savings
5. Understand quantization error and mitigation strategies

Let's make models 4x smaller!

## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in modules/15_quantization/quantization_dev.py
**Building Side:** Code exports to tinytorch.perf.quantization

```python
# Final package structure:
from tinytorch.perf.quantization import quantize_int8, QuantizedLinear, quantize_model
```

**Why this matters:**
- **Learning:** Complete quantization system in one focused module for deep understanding
- **Production:** Proper organization like PyTorch's torch.quantization with all optimization components together
- **Consistency:** All quantization operations and calibration tools in perf.quantization
- **Integration:** Works seamlessly with existing models for complete optimization pipeline

## 📋 Module Dependencies

**Prerequisites**: Module 14 (Profiling) must be complete

**External Dependencies**:
- `numpy` (for array operations and numerical computing)
- `time` (for performance measurements)
- `typing` (for type annotations)

**TinyTorch Dependencies**:
- `tinytorch.core.tensor` (Tensor class from Module 01)
- `tinytorch.core.layers` (Linear, Sequential from Module 03)
- `tinytorch.core.activations` (ReLU from Module 02)
- `tinytorch.perf.profiling` (Profiler from Module 14)

**Dependency Flow**:
```
Module 01 (Tensor) → Module 02 (Activations) → Module 03 (Layers)
     ↓                                              ↓
Module 14 (Profiling) ─────────────────────→ Module 15 (Quantization)
```

Students completing this module will have built a complete
quantization system that achieves 4x memory reduction.

## 💡 Motivation: Why Quantization Matters

Before we learn quantization, let's profile a model to see how much memory
FP32 weights actually consume. This will show us why reduced precision matters.

## 💡 Introduction: The Memory Wall Problem

Imagine trying to fit a library in your backpack. Neural networks face the same challenge - models are getting huge, but devices have limited memory!

### The Precision Paradox

Modern neural networks use 32-bit floating point numbers with incredible precision:

```
FP32 Number: 3.14159265359...
             ^^^^^^^^^^^^^^^^
             32 bits = 4 bytes per weight
```

But here's the surprising truth: **we don't need all that precision for most AI tasks!**

### The Growing Memory Crisis

```
Model Memory Requirements (FP32):
┌─────────────────────────────────────────────────────────────┐
│ BERT-Base:   110M params ×  4 bytes = 440MB                 │
│ GPT-2:       1.5B params ×  4 bytes = 6GB                   │
│ GPT-3:       175B params × 4 bytes = 700GB                  │
│ Your Phone:  Available RAM = 4-8GB                          │
└─────────────────────────────────────────────────────────────┘
                        ↑
                    Problem!
```

### The Quantization Solution

What if we could represent each weight with just 8 bits instead of 32?

```
Before Quantization (FP32):
┌───────────────────────────────┐
│  3.14159265   │  2.71828183   │  32 bits each
└───────────────────────────────┘

After Quantization (INT8):
┌────────┬────────┬────────┬────────┐
│   98   │   85   │   72   │   45   │  8 bits each
└────────┴────────┴────────┴────────┘
         ↑
    4× less memory!
```

### Real-World Impact You'll Achieve

**Memory Reduction:**
- BERT-Base: 440MB → 110MB (4× smaller)
- Fits on mobile devices!
- Faster loading from disk
- More models in GPU memory

**Speed Improvements:**
- 2-4× faster inference (hardware dependent)
- Lower power consumption
- Better user experience

**Accuracy Preservation:**
- <1% accuracy loss with proper techniques
- Sometimes even improves generalization!

**Why This Matters:**
- **Mobile AI:** Deploy powerful models on phones
- **Edge Computing:** Run AI without cloud connectivity
- **Data Centers:** Serve more users with same hardware
- **Environmental:** Reduce energy consumption by 2-4×

Today you'll build the production-quality quantization system that makes all this possible!

## 📐 Foundations: The Mathematics of Compression

### Understanding the Core Challenge

Think of quantization like converting a smooth analog signal to digital steps. We need to map infinite precision (FP32) to just 256 possible values (INT8).

### The Quantization Mapping

```
The Fundamental Problem:

FP32 Numbers (Continuous):        INT8 Numbers (Discrete):
    ∞ possible values         →      256 possible values

  ...  -1.7  -1.2  -0.3  0.0  0.8  1.5  2.1  ...
         ↓     ↓     ↓    ↓    ↓    ↓    ↓
      -128  -95   -38    0   25   48   67   127
```

### The Magic Formula

Every quantization system uses this fundamental relationship:

```
Quantization (FP32 → INT8):
┌─────────────────────────────────────────────────────────┐
│  quantized = round(float_value / scale + zero_point)     │
└─────────────────────────────────────────────────────────┘

Dequantization (INT8 → FP32):
┌─────────────────────────────────────────────────────────┐
│  float_value = (quantized - zero_point) × scale         │
└─────────────────────────────────────────────────────────┘
```

### The Two Critical Parameters

**1. Scale (s)** - How big each INT8 step is in FP32 space:
```
Small Scale (high precision):       Large Scale (low precision):
 FP32: [0.0, 0.255]                 FP32: [0.0, 25.5]
   ↓     ↓     ↓                       ↓     ↓     ↓
 INT8:  0    128   255              INT8:  0    128   255
        │     │     │                      │     │     │
      0.0   0.127  0.255                 0.0   12.75  25.5

 Scale = 0.001 (very precise)        Scale = 0.1 (less precise)
```

**2. Zero Point (z)** - Which INT8 value represents FP32 zero:
```
Symmetric Range:                    Asymmetric Range:
 FP32: [-2.0, 2.0]                  FP32: [-1.0, 3.0]
   ↓     ↓     ↓                       ↓     ↓     ↓
 INT8: -128    0   127              INT8: -128  -64   127
        │     │     │                      │     │     │
     -2.0    0.0   2.0                  -1.0   0.0   3.0

 Zero Point = 0                     Zero Point = -64
```

### Visual Example: Weight Quantization

```
Original FP32 Weights:           Quantized INT8 Mapping:
┌─────────────────────────┐      ┌─────────────────────────┐
│ -0.8  -0.3   0.0   0.5  │  →   │ -128  -64  -26   38     │
│  0.9   1.2  -0.1   0.7  │      │   89  127  -39   63     │
└─────────────────────────┘      └─────────────────────────┘
     4 bytes each                      1 byte each
   Total: 32 bytes                   Total: 8 bytes
                                    ↑
                              4× compression!
```

### Quantization Error Analysis

```
Perfect Reconstruction (Impossible):  Quantized Reconstruction (Reality):

Original: 0.73                       Original: 0.73
    ↓                                     ↓
INT8: ? (can't represent exactly)     INT8: 93 (closest)
    ↓                                     ↓
Restored: 0.73                        Restored: 0.728
                                           ↑
                                    Error: 0.002
```

**The Quantization Trade-off:**
- **More bits** = Higher precision, larger memory
- **Fewer bits** = Lower precision, smaller memory
- **Goal:** Find the sweet spot where error is acceptable

### Why INT8 is the Sweet Spot

```
Precision vs Memory Trade-offs:

FP32: ████████████████████████████████ (32 bits) - Overkill precision
FP16: ████████████████ (16 bits)                  - Good precision
INT8: ████████ (8 bits)                           - Sufficient precision ← Sweet spot!
INT4: ████ (4 bits)                               - Often too little

Memory:    100%    50%    25%    12.5%
Accuracy:  100%   99.9%  99.5%   95%
```

INT8 gives us 4× memory reduction with <1% accuracy loss - the perfect balance for production systems!

## 🏗️ Implementation: Building the Quantization Engine

### Our Implementation Strategy

We'll build quantization in logical layers, each building on the previous:

```
Quantization System Architecture:

┌─────────────────────────────────────────────────────────────┐
│                    Layer 4: Model Quantization              │
│  quantize_model() - Convert entire neural networks          │
├─────────────────────────────────────────────────────────────┤
│                    Layer 3: Layer Quantization              │
│  QuantizedLinear - Quantized linear transformations         │
├─────────────────────────────────────────────────────────────┤
│                    Layer 2: Tensor Operations               │
│  quantize_int8() - Core quantization algorithm              │
│  dequantize_int8() - Restore to floating point              │
├─────────────────────────────────────────────────────────────┤
│                    Layer 1: Foundation                      │
│  Scale & Zero Point Calculation - Parameter optimization    │
└─────────────────────────────────────────────────────────────┘
```

### What We're About to Build

**Core Functions:**
- `quantize_int8()` - Convert FP32 tensors to INT8
- `dequantize_int8()` - Convert INT8 back to FP32
- `QuantizedLinear` - Quantized version of Linear layers
- `quantize_model()` - Quantize entire neural networks

**Key Features:**
- **Automatic calibration** - Find optimal quantization parameters
- **Error minimization** - Preserve accuracy during compression
- **Memory tracking** - Measure actual savings achieved
- **Production patterns** - Industry-standard algorithms

Let's start with the fundamental building block!

## 🏗️ INT8 Quantization - The Foundation

This is the core function that converts any FP32 tensor to INT8. Think of it as a smart compression algorithm that preserves the most important information.

```
Quantization Process Visualization:

Step 1: Analyze Range              Step 2: Calculate Parameters       Step 3: Apply Formula
┌─────────────────────────┐    ┌─────────────────────────┐  ┌─────────────────────────┐
│ Input: [-1.5, 0.2, 2.8] │    │ Min: -1.5               │  │ quantized = round(      │
│                         │    │ Max: 2.8                │  │   value / scale + zp)   │
│ Find min/max values     │ →  │ Range: 4.3              │ →│                         │
│                         │    │ Scale: 4.3/255 = 0.017  │  │                         │
│                         │    │ Zero Point: -39         │  │ Result: [-128,-27, 127] │
└─────────────────────────┘    └─────────────────────────┘  └─────────────────────────┘
```

**Key Challenges This Function Solves:**
- **Dynamic Range:** Each tensor has different min/max values
- **Precision Loss:** Map 4 billion FP32 values to just 256 INT8 values
- **Zero Preservation:** Ensure FP32 zero maps exactly to an INT8 value
- **Asymmetric Mapping:** Distribute quantization levels efficiently

**Why This Algorithm:**
- **Linear mapping** preserves relative relationships between values
- **Affine (asymmetric) quantization** works well for most neural network weights
- **Clipping to [-128, 127]** ensures valid INT8 range
- **Round-to-nearest** minimizes quantization error

### 🧪 Unit Test: INT8 Quantization

This test validates our INT8 quantization function works correctly with various data types and edge cases.

**What we're testing**: Basic quantization and dequantization roundtrip
**Why it matters**: Foundation for all memory reduction - if quantization fails, nothing works
**Expected**: Quantized values in INT8 range with acceptable reconstruction error

## 🏗️ INT8 Dequantization - Restoring Precision

Dequantization is the inverse process - converting compressed INT8 values back to usable FP32. This is where we "decompress" our quantized data.

```
Dequantization Process:

INT8 Values + Parameters → FP32 Reconstruction

┌───────────────────────────────────┐
│ Quantized: [-128, -27, 127]       │
│ Scale: 0.017                      │
│ Zero Point: -39                   │
└───────────────────────────────────┘
                 │
                 ▼ Apply Formula
┌───────────────────────────────────┐
│ FP32 = (quantized - zero_point)   │
│        × scale                    │
└───────────────────────────────────┘
                 │
                 ▼
┌───────────────────────────────────┐
│ Result: [-1.501, 0.202, 2.799]    │
│ Original: [-1.5, 0.2, 2.8]        │
│ Error: [0.001, 0.002, 0.001]      │
└───────────────────────────────────┘
       ↑
  Excellent approximation!
```

**Why This Step Is Critical:**
- **Neural networks expect FP32** - INT8 values would confuse computations
- **Preserves computation compatibility** - works with existing matrix operations
- **Controlled precision loss** - error is bounded and predictable
- **Hardware flexibility** - can use FP32 or specialized INT8 operations

**When Dequantization Happens:**
- **During forward pass** - before matrix multiplications
- **For gradient computation** - during backward pass
- **Educational approach** - production uses INT8 GEMM directly

### 🧪 Unit Test: INT8 Dequantization

This test validates our dequantization function correctly restores FP32 values from INT8.

**What we're testing**: Roundtrip quantize -> dequantize preserves values
**Why it matters**: Neural networks need FP32 values for computation
**Expected**: Small reconstruction error after roundtrip

## 🏗️ QuantizedLinear - The Heart of Efficient Networks

### Why We Need Quantized Layers

A quantized model isn't just about storing weights in INT8 - we need layers that can work efficiently with quantized data.

```
Regular Linear Layer:              QuantizedLinear Layer:

┌─────────────────────┐            ┌─────────────────────┐
│ Input: FP32         │            │ Input: FP32         │
│ Weights: FP32       │            │ Weights: INT8       │
│ Computation: FP32   │    VS      │ Computation: Mixed  │
│ Output: FP32        │            │ Output: FP32        │
│ Memory: 4× more     │            │ Memory: 4× less     │
└─────────────────────┘            └─────────────────────┘
```

### The Quantized Forward Pass

```
Quantized Linear Layer Forward Pass:

    Input (FP32)                  Quantized Weights (INT8)
         │                               │
         ▼                               ▼
┌─────────────────┐              ┌─────────────────┐
│    Calibrate    │              │   Dequantize    │
│   (optional)    │              │   Weights       │
└─────────────────┘              └─────────────────┘
         │                               │
         ▼                               ▼
    Input (FP32)                  Weights (FP32)
         │                               │
         └───────────────┬───────────────┘
                         ▼
                ┌─────────────────┐
                │ Matrix Multiply │
                │   (FP32 GEMM)   │
                └─────────────────┘
                         │
                         ▼
                   Output (FP32)

Memory Saved: 4× for weights storage!
Speed: Depends on dequantization overhead vs INT8 GEMM support
```

### Calibration - Finding Optimal Input Quantization

```
Calibration Process:

 Step 1: Collect Sample Inputs    Step 2: Analyze Distribution    Step 3: Optimize Parameters
 ┌─────────────────────────┐      ┌─────────────────────────┐    ┌─────────────────────────┐
 │ input_1: [-0.5, 0.2, ..]│      │   Min: -0.8             │    │ Scale: 0.00627          │
 │ input_2: [-0.3, 0.8, ..]│  →   │   Max: +0.8             │ →  │ Zero Point: 0           │
 │ input_3: [-0.1, 0.5, ..]│      │   Range: 1.6            │    │ Optimal for this data   │
 │ ...                     │      │   Distribution: Normal  │    │ range and distribution  │
 └─────────────────────────┘      └─────────────────────────┘    └─────────────────────────┘
```

**Why Calibration Matters:**
- **Without calibration:** Generic quantization parameters may waste precision
- **With calibration:** Parameters optimized for actual data distribution
- **Result:** Better accuracy preservation with same memory savings

## 🏗️ QuantizedLinear Class - Efficient Neural Network Layer

This class replaces regular Linear layers with quantized versions that use 4× less memory while preserving functionality.

```
QuantizedLinear Architecture:

Creation Time:                       Runtime:
┌───────────────────────────────┐    ┌───────────────────────────────┐
│ Regular Linear Layer          │    │ Input (FP32)                  │
│ ↓                             │    │ ↓                             │
│ Quantize weights → INT8       │    │ Optional: quantize input      │
│ Quantize bias → INT8          │ →  │ ↓                             │
│ Store quantization params     │    │ Dequantize weights            │
│ Ready for deployment!         │    │ ↓                             │
└───────────────────────────────┘    │ Matrix multiply (FP32)        │
      One-time cost                  │ ↓                             │
                                     │ Output (FP32)                 │
                                     └───────────────────────────────┘
                                        Per-inference cost
```

**Key Design Decisions:**

1. **Store original layer reference** - for debugging and comparison
2. **Separate quantization parameters** - weights and bias may need different scales
3. **Calibration support** - optimize input quantization using real data
4. **FP32 computation** - educational approach, production uses INT8 GEMM
5. **Memory tracking** - measure actual compression achieved

**Memory Layout:**

Regular Linear layers store weights in FP32 (4 bytes each), while QuantizedLinear stores them in INT8 (1 byte each) plus a small overhead for quantization parameters (scales and zero points). This achieves approximately 4× memory reduction with minimal overhead.

**Production vs Educational Trade-off:**
- **Our approach:** Dequantize → FP32 computation (easier to understand)
- **Production:** INT8 GEMM operations (faster, more complex)
- **Both achieve:** Same memory savings, similar accuracy

### 🧪 Unit Test: QuantizedLinear

This test validates our QuantizedLinear layer works correctly and achieves memory savings.

**What we're testing**: Quantized layer forward pass and compression ratio
**Why it matters**: This is the core component that replaces Linear layers in models
**Expected**: Forward pass produces similar output with ~4x compression

## 🔧 Integration: Scaling to Full Neural Networks

### The Model Quantization Challenge

Quantizing individual tensors is useful, but real applications need to quantize entire neural networks with multiple layers, activations, and complex data flows. The key is replacing standard layers (like Linear) with their quantized equivalents (QuantizedLinear) while keeping activation functions unchanged since they have no parameters.

### Smart Layer Selection

Not all layers benefit equally from quantization. Linear and convolutional layers with many parameters see the largest benefits, while activation functions (which have no parameters) cannot be quantized. Some layers like input/output projections may be sensitive to quantization and should be kept in higher precision for critical applications.

### Calibration Data Flow

Calibration runs sample data through the model layer-by-layer, collecting activation statistics at each layer. These statistics (min/max values, distributions) determine optimal quantization parameters for each layer, ensuring minimal accuracy loss during quantization.

### Memory Impact

Quantization provides consistent 4× memory reduction across all model sizes. The actual impact depends on model architecture, but the compression ratio remains constant since we're reducing precision from 32 bits to 8 bits per parameter.

Now let's implement the functions that make this transformation possible!

## 🔧 Model Quantization - Scaling to Full Networks

Quantizing individual layers is useful, but real applications need to quantize entire neural
networks. We'll build this capability in two steps:

1. **Collect layer inputs** - Forward calibration data through preceding layers to get
   the activation distribution at each layer's input
2. **Quantize a single layer** - Replace one Linear layer with its QuantizedLinear equivalent

Then the composition function `quantize_model()` ties them together to transform a full model.

```
Model Transformation Process:

Input Model:                    Quantized Model:
┌─────────────────────────────┐    ┌─────────────────────────────┐
│ layers[0]: Linear(784, 128) │    │ layers[0]: QuantizedLinear  │
│ layers[1]: ReLU()           │    │ layers[1]: ReLU()           │
│ layers[2]: Linear(128, 64)  │ →  │ layers[2]: QuantizedLinear  │
│ layers[3]: ReLU()           │    │ layers[3]: ReLU()           │
│ layers[4]: Linear(64, 10)   │    │ layers[4]: QuantizedLinear  │
└─────────────────────────────┘    └─────────────────────────────┘
   Memory: 100%                      Memory: ~25%
   Interface: Same                   Interface: Identical
```

## 🏗️ Collecting Layer Inputs - Calibration Data Flow

Before we can calibrate a quantized layer, we need to know what its inputs look like
at runtime. This helper forwards calibration samples through all preceding layers
to collect the activation tensors that arrive at a given layer index.

```
Calibration Data Flow for Layer at Index i:

  Sample Data          Layers 0..i-1          Activations at Layer i
  ┌──────────┐      ┌──────────────────┐      ┌──────────────────┐
  │ sample_0  │ ──→ │ forward through  │ ──→  │ activation_0     │
  │ sample_1  │ ──→ │ preceding layers │ ──→  │ activation_1     │
  │ ...       │     │ (0, 1, ..., i-1) │      │ ...              │
  │ sample_N  │ ──→ │                  │ ──→  │ activation_N     │
  └──────────┘      └──────────────────┘      └──────────────────┘
```

### 🧪 Unit Test: Collect Layer Inputs

This test validates that we correctly forward calibration data through preceding layers.

**What we're testing**: Intermediate activation collection for calibration
**Why it matters**: Accurate calibration requires knowing the true input distribution at each layer
**Expected**: Correct number of samples with correct shape after forwarding through preceding layers

## 🏗️ Quantizing a Single Layer - The Replacement Step

This helper takes one Linear layer, wraps it in a QuantizedLinear, and optionally
calibrates it using pre-collected activation samples. This is the atomic operation
that `quantize_model()` applies to each eligible layer.

```
Single Layer Quantization:

  Linear Layer          QuantizedLinear
  ┌──────────────┐      ┌──────────────────────────┐
  │ weight: FP32 │  →   │ q_weight: INT8           │
  │ bias: FP32   │      │ q_bias: INT8             │
  │              │      │ weight_scale, zero_point  │
  └──────────────┘      │ calibrated: Yes/No       │
                        └──────────────────────────┘
       4 bytes/param          1 byte/param + overhead
```

### 🧪 Unit Test: Quantize Single Layer

This test validates that we correctly quantize one Linear layer with optional calibration.

**What we're testing**: Single-layer quantization and calibration
**Why it matters**: This is the atomic building block for full model quantization
**Expected**: INT8 weights, optional calibration parameters set

## 🔧 Model Quantization - The Composition Function

Now we compose the helpers into the full model quantization function. For each Linear
layer in the model, we collect its calibration inputs and replace it with a quantized version.

```
quantize_model() orchestrates the full pipeline:

  For each layer in model.layers:
      │
      ├── isinstance(layer, Linear)?
      │   ├── YES → _collect_layer_inputs()  → calibration activations
      │   │         _quantize_single_layer()  → QuantizedLinear
      │   │         Replace model.layers[i]
      │   │
      │   └── NO  → Keep unchanged (ReLU, etc.)
```

### 🧪 Unit Test: Model Quantization

This test validates our model quantization function correctly replaces Linear layers with QuantizedLinear.

**What we're testing**: Full model quantization and layer replacement
**Why it matters**: Real applications need to quantize entire neural networks
**Expected**: Linear layers replaced, ReLU unchanged, output shape preserved

## 🔧 Model Size Comparison - Measuring the Impact

To compare memory usage between original and quantized models, we need to measure
bytes at the individual layer level first, then aggregate. We'll build this in two steps:

1. **Measure one layer** - Count parameters and bytes for a single layer, handling both
   FP32 (Linear) and INT8 (QuantizedLinear) layers correctly
2. **Aggregate and compare** - Sum across all layers and compute compression metrics

```
Per-Layer Measurement:

  Layer Type          Measurement Strategy
  ┌──────────────┐    ┌───────────────────────────────────┐
  │ Linear       │ →  │ params × 4 bytes (FP32)           │
  │ QuantizedLin │ →  │ memory_usage() dict (INT8 + ovhd) │
  │ ReLU/other   │ →  │ 0 params, 0 bytes (no weights)    │
  └──────────────┘    └───────────────────────────────────┘
```

## 🏗️ Measuring a Single Layer - Per-Layer Byte Accounting

This helper measures the parameter count and byte usage for one layer. It handles
the key distinction: FP32 layers store parameters at 4 bytes each, while QuantizedLinear
layers use INT8 storage with a small overhead for scale/zero_point metadata.

```
Byte Accounting per Layer Type:

  FP32 Linear:                     QuantizedLinear:
  ┌─────────────────────────┐      ┌─────────────────────────────────┐
  │ weight: N × 4 bytes     │      │ q_weight: N × 1 byte            │
  │ bias:   M × 4 bytes     │      │ q_bias:   M × 1 byte            │
  │                         │      │ overhead: ~8 bytes (scale+zp)    │
  │ Total: (N+M) × 4       │      │ Total: (N+M) × 1 + overhead     │
  └─────────────────────────┘      └─────────────────────────────────┘
```

### 🧪 Unit Test: Measure Layer Bytes

This test validates that we correctly measure bytes for both FP32 and quantized layers.

**What we're testing**: Per-layer byte accounting for different layer types
**Why it matters**: Accurate per-layer measurement is needed for reliable compression metrics
**Expected**: FP32 layers use 4 bytes/param, quantized layers use ~1 byte/param + overhead

## 🔧 Model Size Analysis - The Composition Function

Now we aggregate per-layer measurements across the full model to produce a comprehensive
comparison between original and quantized versions.

```
Aggregation Flow:

  Original Model                    Quantized Model
  ┌──────────────────────────┐      ┌──────────────────────────┐
  │ Layer 0: _measure(FP32)  │      │ Layer 0: _measure(INT8)  │
  │ Layer 1: _measure(skip)  │      │ Layer 1: _measure(skip)  │
  │ Layer 2: _measure(FP32)  │      │ Layer 2: _measure(INT8)  │
  └──────────────────────────┘      └──────────────────────────┘
           │                                  │
           ▼                                  ▼
     Sum params, bytes                  Sum params, bytes
           │                                  │
           └─────────────┬────────────────────┘
                         ▼
               Compression metrics
```

### 🧪 Unit Test: Model Size Analysis

This test validates our model size analysis function correctly measures compression.

**What we're testing**: Memory comparison between original and quantized models
**Why it matters**: Need accurate metrics to verify quantization benefits
**Expected**: Compression ratio > 2x and significant memory savings

## 🔧 Consolidated Quantization Classes for Export

Now that we've implemented all quantization components, let's create consolidated classes
for export to the tinytorch package. This allows milestones to use the complete quantization system.

## 📊 Systems Analysis: Quantization in Production

Now let's measure the real-world impact of quantization through systematic analysis.

## 📊 Advanced Quantization Strategies - Production Techniques

This analysis compares different quantization approaches used in production systems, revealing the trade-offs between accuracy, complexity, and performance.

```
Strategy Comparison Framework:

┌──────────────────────────────────────────────────────────────────────────────────┐
│                          Three Advanced Strategies                             │
├──────────────────────────┬──────────────────────────┬──────────────────────────┤
│       Strategy 1         │       Strategy 2         │       Strategy 3         │
│    Per-Tensor (Ours)     │    Per-Channel Scale     │    Mixed Precision       │
├──────────────────────────┼──────────────────────────┼──────────────────────────┤
│                          │                          │                          │
│ ┌──────────────────────┐ │ ┌──────────────────────┐ │ ┌──────────────────────┐ │
│ │ Weights:             │ │ │ Channel 1: scale₁   │ │ │ Sensitive: FP32      │ │
│ │ [W₁₁ W₁₂ W₁₃]        │ │ │ Channel 2: scale₂   │ │ │ Regular: INT8        │ │
│ │ [W₂₁ W₂₂ W₂₃] scale  │ │ │ Channel 3: scale₃   │ │ │                      │ │
│ │ [W₃₁ W₃₂ W₃₃]        │ │ │                      │ │ │ Input: FP32          │ │
│ └──────────────────────┘ │ │ Better precision     │ │ │ Output: FP32         │ │
│                          │ │ per channel          │ │ │ Hidden: INT8         │ │
│ Simple, fast             │ └──────────────────────┘ │ └──────────────────────┘ │
│ Good baseline            │                          │                          │
│                          │ More complex             │ Optimal accuracy         │
│                          │ Better accuracy          │ Selective compression    │
└──────────────────────────┴──────────────────────────┴──────────────────────────┘
```

**Strategy 1: Per-Tensor Quantization (Our Implementation)**
```
Weight Matrix:                Scale Calculation:
┌─────────────────────────┐     ┌─────────────────────────┐
│ 0.1 -0.3  0.8  0.2      │     │ Global min: -0.5        │
│-0.2  0.5 -0.1  0.7      │ →   │ Global max: +0.8        │
│ 0.4 -0.5  0.3 -0.4      │     │ Scale: 1.3/255 = 0.0051 │
└─────────────────────────┘     └─────────────────────────┘

Pros: Simple, fast           Cons: May waste precision
```

**Strategy 2: Per-Channel Quantization (Advanced)**
```
Weight Matrix:                Scale Calculation:
┌─────────────────────────┐     ┌─────────────────────────┐
│ 0.1 -0.3  0.8  0.2      │     │ Col 1: [-0.2,0.4] → s₁  │
│-0.2  0.5 -0.1  0.7      │ →   │ Col 2: [-0.5,0.5] → s₂  │
│ 0.4 -0.5  0.3 -0.4      │     │ Col 3: [-0.1,0.8] → s₃  │
└─────────────────────────┘     │ Col 4: [-0.4,0.7] → s₄  │
                             └─────────────────────────┘

Pros: Better precision       Cons: More complex
```

**Strategy 3: Mixed Precision (Production)**
```
Model Architecture:            Precision Assignment:
┌─────────────────────────┐     ┌─────────────────────────┐
│ Input Layer  (sensitive) │     │ Keep in FP32 (precision) │
│ Hidden 1     (bulk)     │ →   │ Quantize to INT8        │
│ Hidden 2     (bulk)     │     │ Quantize to INT8        │
│ Output Layer (sensitive)│     │ Keep in FP32 (quality)   │
└─────────────────────────┘     └─────────────────────────┘

Pros: Optimal trade-off      Cons: Requires expertise
```

**Experimental Design:**
```
Comparative Testing Protocol:

1. Create identical test model   →  2. Apply each strategy        →  3. Measure results
   ┌───────────────────────┐     ┌───────────────────────┐     ┌───────────────────────┐
   │ 128 → 64 → 10 MLP      │     │ Per-tensor quantization │     │ MSE error calculation  │
   │ Identical weights       │     │ Per-channel simulation  │     │ Compression measurement│
   │ Same test input         │     │ Mixed precision setup   │     │ Speed comparison       │
   └───────────────────────┘     └───────────────────────┘     └───────────────────────┘
```

**Expected Strategy Rankings:**
1. **Mixed Precision** - Best accuracy, moderate complexity
2. **Per-Channel** - Good accuracy, higher complexity
3. **Per-Tensor** - Baseline accuracy, simplest implementation

This analysis reveals which strategies work best for different deployment scenarios and accuracy requirements.

## 📊 Measuring Quantization Savings with Profiler

Now let's use the Profiler tool from Module 14 to measure the actual memory savings from quantization. This demonstrates end-to-end workflow: profile baseline (M14) -> apply quantization (M15) -> measure savings (M14+M15).

This is the production workflow: measure -> compress -> validate -> deploy.

## 🔧 Verification: Prove Quantization Works

Before running the full integration test, let's create a verification function that
proves quantization actually reduces memory using real `.nbytes` measurements.

## 🧪 Module Integration Test

Final validation that everything works together correctly before module completion.

## 🤔 ML Systems Reflection Questions

Answer these to deepen your understanding of quantization and its systems implications:

### Question 1: Memory Architecture Impact
You implemented INT8 quantization that reduces each parameter from 4 bytes to 1 byte.
For a model with 100M parameters:
- Original memory usage: _____ GB
- Quantized memory usage: _____ GB
- Memory bandwidth reduction when loading from disk: _____ ×

#### 参考解答
**Answer 1: Memory Architecture Impact**
- Original memory usage: **0.4 GB** (100M parameters × 4 bytes = 400MB = 0.4 GB)
- Quantized memory usage: **0.1 GB** (100M parameters × 1 byte = 100MB = 0.1 GB)
- Memory bandwidth reduction: **4×** (loading 100MB instead of 400MB from disk)

**Key Insight**: Quantization reduces not just RAM usage, but also disk I/O, network transfer time, and memory bandwidth pressure. A 4× reduction in bandwidth means 4× faster model loading and 4× less network traffic when deploying models.

### Question 2: Quantization Error Analysis
Your quantization maps a continuous range to 256 discrete values (INT8).
For weights uniformly distributed in [-0.1, 0.1]:
- Quantization scale: _____
- Maximum quantization error: _____
- Signal-to-noise ratio approximately: _____ dB

#### 参考解答
**Answer 2: Quantization Error Analysis**
- Quantization scale: **0.0007843** (range 0.2 / 255 steps = 0.0007843)
- Maximum quantization error: **±0.000392** (scale / 2 = ±0.0003922)
- Signal-to-noise ratio: **~48 dB** (20 × log10(signal_range / quantization_step) ≈ 20 × log10(255) ≈ 48 dB)

**Key Insight**: For 8-bit quantization, theoretical SNR is approximately 6 dB per bit × 8 bits = 48 dB. This is sufficient for neural networks because weights typically have bounded ranges and networks are robust to small perturbations.

### Question 3: Hardware Efficiency
Modern processors have specialized INT8 instructions (like AVX-512 VNNI).
Compared to FP32 operations:
- How many INT8 operations fit in one SIMD instruction vs FP32? _____ × more
- Why might actual speedup be less than this theoretical maximum? _____
- What determines whether quantization improves or hurts performance? _____

#### 参考解答
**Answer 3: Hardware Efficiency**
- INT8 operations per SIMD: **4× more** (512-bit register can hold 64 INT8 values vs 16 FP32 values)
- Why actual speedup is less: **Dequantization overhead, memory bandwidth bottlenecks, and non-compute operations** (data movement, activation functions, etc. remain in FP32)
- Performance determinant: **Hardware INT8 support availability** (modern CPUs with VNNI, GPUs with Tensor Cores, mobile chips with Neural Engine) and **compute vs memory-bound workload** (compute-bound benefits more from INT8 ops, memory-bound benefits from reduced bandwidth)

**Key Insight**: Theoretical 4× speedup requires: (1) Hardware with native INT8 instructions, (2) Large matrix multiplications where compute dominates, (3) Minimal dequantization overhead. Real-world speedups are typically 2-3× due to mixed precision operations and data movement costs.

### Question 4: Calibration Strategy Trade-offs
Your calibration process finds optimal scales using sample data.
- Too little calibration data: Risk of _____
- Too much calibration data: Cost of _____
- Per-channel vs per-tensor quantization trades _____ for _____

#### 参考解答
**Answer 4: Calibration Strategy Trade-offs**
- Too little calibration data: Risk of **suboptimal quantization parameters that don't represent the true activation distribution**, leading to **clipping of outliers and accuracy degradation**
- Too much calibration data: Cost of **increased calibration time** and **diminishing returns** (accuracy stops improving after ~100-1000 samples typically)
- Per-channel vs per-tensor trades: **Complexity and overhead** (more scales to store/compute) for **better precision** (each channel optimized independently, preserving more information)

**Key Insight**: Calibration is about finding representative data statistics. The rule of thumb: 100-1000 diverse samples usually suffice. Per-channel quantization is worth the complexity for sensitive layers (first/last layers, attention) but overkill for bulk middle layers.

### Question 5: Production Deployment
In mobile/edge deployment scenarios:
- When is 4× memory reduction worth <1% accuracy loss? _____
- Why might you keep certain layers in FP32? _____
- How does quantization affect battery life? _____

#### 参考解答
**Answer 5: Production Deployment**
- When 4× reduction worth <1% loss: **Always in memory-constrained environments** (mobile devices with <4GB RAM, edge devices with <512MB, embedded systems). Also when **serving cost matters** (4× smaller = 4× more users per server) or **latency critical** (4× faster loading from disk/network).

- Keep layers in FP32: **First layer** (input quantization loses information), **last layer** (output precision matters for final predictions), **attention layers** (sensitive to precision for softmax stability), and **layers with extreme activation ranges** (quantization error amplifies).

- Battery life impact: **2-4× improvement** due to (1) **less memory access** = lower DRAM power, (2) **INT8 operations use less energy** than FP32 ALUs, (3) **faster inference** = shorter active time. Typical mobile inference: 60% energy from memory, 30% from compute, 10% other.

**Key Insight**: Quantization is essential for edge AI. The 1% accuracy loss is usually imperceptible to users, but 4× memory savings and 2-3× speedup enable entirely new applications (real-time on-device AI, offline functionality, privacy-preserving local inference).

## ⭐ Aha Moment: Quantization Shrinks Models

**What you built:** A complete INT8 quantization system with calibration and memory tracking.

**Why it matters:** A 400MB model becomes 100MB, small enough to run on a phone! Quantization
is how production ML deploys large models to edge devices, achieving 4x memory reduction with
minimal accuracy loss.

Your quantization system is ready for production deployment!

## 🚀 MODULE SUMMARY: Quantization

Congratulations! You've built a complete INT8 quantization system that can reduce model size by 4x with minimal accuracy loss!

### Key Accomplishments
- Built INT8 quantization with proper scaling and zero-point calculation
- Implemented QuantizedLinear layer with calibration support
- Created model-level quantization for complete neural networks
- Analyzed quantization trade-offs across different distributions and strategies
- Measured real memory savings and performance improvements
- All tests pass (validated by `test_module()`)

### Systems Insights Discovered
- Memory scaling: INT8 reduces storage by 4x (32 bits to 8 bits per parameter)
- Calibration trade-offs: Sample data quality affects quantization accuracy
- Hardware efficiency: Specialized INT8 instructions provide 2-4x speedup
- Deployment benefits: Smaller models fit on mobile and edge devices

Export with: `tito module complete 15`

Quantization is one of the most impactful optimization techniques — reducing precision to INT8 delivers 4x memory savings with minimal accuracy loss.
