> **文档来源**：`src/14_profiling/14_profiling.py` 中的 Markdown 教学说明（英文原版，本模块尚无 `*_zh.py`）。  
> 下文保留原模块一级标题；组件主题下的从属小节已下调一级，避免同级标题重复。

# Module 14: Profiling - Measuring What Matters in ML Systems

Welcome to Module 14! You'll build professional profiling tools to measure model performance and uncover optimization opportunities.

## 🔗 Prerequisites & Progress
**You've Built**: Complete ML stack from tensors to transformers (Modules 01-13)
**You'll Build**: Comprehensive profiling system for parameters, FLOPs, memory, and latency
**You'll Enable**: Data-driven optimization decisions and performance analysis

**Connection Map**:
```
All Modules (01-13) → Profiling (14)
(implementations)     (measurement)
```

## 🎯 Learning Objectives
By the end of this module, you will:
1. Implement a complete Profiler class for model analysis
2. Count parameters and FLOPs accurately for different architectures
3. Measure memory usage and latency with statistical rigor
4. Create production-quality performance analysis tools

Let's build the measurement foundation for ML systems optimization!

## 📦 Where This Code Lives in the Final Package

**Learning Side:** You work in modules/14_profiling/profiling_dev.py
**Building Side:** Code exports to tinytorch.perf.profiling

```python
# Final package structure:
from tinytorch.perf.profiling import Profiler, profile_forward_pass, profile_backward_pass
```

**Why this matters:**
- **Learning:** Complete profiling system for understanding model performance characteristics
- **Production:** Professional measurement tools like those used in PyTorch, TensorFlow
- **Consistency:** All profiling and measurement tools in perf.profiling
- **Integration:** Works with any model built using TinyTorch components

## 📋 Module Dependencies

**Prerequisites**: Modules 01-13 (Complete ML stack)

**External Dependencies**:
- `numpy` (for array operations and numerical computing)
- `time` (for latency measurement)
- `tracemalloc` (for memory tracking)
- `gc` (for garbage collection control)

**TinyTorch Dependencies**:
- `tinytorch.core.tensor` (Tensor class from Module 01)
- `tinytorch.core.layers` (Linear layer from Module 03)
- `tinytorch.core.spatial` (Conv2d from Module 09)

**Dependency Flow**:
```
Modules 01-13 → Module 14 (Profiling)
     ↓                   ↓
 Implementations    Measurement tools
```

Students completing this module will have built the measurement foundation
that enables data-driven optimization decisions.

## 💡 Introduction: Why Profiling Matters in ML Systems

Imagine you're a detective investigating a performance crime. Your model is running slowly, using too much memory, or burning through compute budgets. Without profiling, you're flying blind - making guesses about what to optimize. With profiling, you have evidence.

**The Performance Investigation Process:**
```
Suspect Model → Profile Evidence → Identify Bottleneck → Target Optimization
     ↓               ↓                    ↓                    ↓
   "Too slow"    "200 GFLOP/s"      "Memory bound"      "Reduce transfers"
```

**Questions Profiling Answers:**
- **How many parameters?** (Memory footprint, model size)
- **How many FLOPs?** (Computational cost, energy usage)
- **Where are bottlenecks?** (Memory vs compute bound)
- **What's actual latency?** (Real-world performance)

**Production Importance:**
In production ML systems, profiling isn't optional - it's survival. A model that's 10% more accurate but 100× slower often can't be deployed. Teams use profiling daily to make data-driven optimization decisions, not guesses.

### The Profiling Workflow Visualization
```
Model → Profiler → Measurements → Analysis → Optimization Decision
  ↓        ↓           ↓           ↓            ↓
 GPT   Parameter   125M params   Memory      Apply targeted
       Counter     2.5B FLOPs    bound       optimization
```

### From Implementation to Optimization: The Profiling Foundation

**In this module (14)**, you'll build the measurement tools to discover optimization opportunities.
Profiling insights guide targeted performance improvements — you can't optimize what you can't measure.

**The Real ML Engineering Workflow**:

```
┌────────────────────────────────────────────────────────────────┐
│ Step 1: Measure (This Module!)    Step 2: Analyze              │
│   ↓                                 ↓                          │
│ Profile baseline → Find bottleneck → Understand cause          │
│ 40 tok/s          80% in attention    O(n^2) recomputation     │
│                                       ↓                        │
│ Step 4: Validate                    Step 3: Optimize (Future)  │
│   ↓                                   ↓                        │
│ Profile optimized ← Verify speedup ← Implement optimization    │
│ 500 tok/s (12.5x)   Measure impact    Design solution          │
└────────────────────────────────────────────────────────────────┘
```

**Without profiling**: You'd never know WHERE to optimize!
**Without measurement**: You couldn't verify improvements!

This module teaches the measurement and analysis skills that enable optimization breakthroughs. You'll profile real models and discover bottlenecks just like production ML teams do.

## 📐 Foundations: Performance Measurement Principles

Before we build our profiler, let's understand what we're measuring and why each metric matters.

### Parameter Counting: Model Size Detective Work

Parameters determine your model's memory footprint and storage requirements. Every parameter is typically a 32-bit float (4 bytes), so counting them precisely predicts memory usage.

**Parameter Counting Formula:**
```
Linear Layer: (input_features × output_features) + output_features
               ↑              ↑                    ↑
            Weight matrix   Bias vector      Total parameters

Example: Linear(768, 3072) → (768 × 3072) + 3072 = 2,362,368 parameters
Memory: 2,362,368 × 4 bytes = 9.45 MB
```

### FLOP Counting: Computational Cost Analysis

FLOPs (Floating Point Operations) measure computational work. Unlike wall-clock time, FLOPs are hardware-independent and predict compute costs across different systems.

**FLOP Formulas for Key Operations:**

```
Matrix Multiplication (M,K) @ (K,N):
   FLOPs = M x N x K x 2
           ↑   ↑   ↑   ↑
        Rows Cols Inner Multiply+Add

Linear Layer Forward:
   FLOPs = input_features x output_features x 2  (per sample, batch-independent)
                ↑                  ↑              ↑
           Input dimension   Output dimension  Multiply+Add

Convolution (simplified):
   FLOPs = output_H x output_W x kernel_H x kernel_W x in_channels x out_channels x 2
```

### Memory Profiling: The Three Types of Memory

ML models use memory in three distinct ways, each with different optimization strategies:

**Memory Type Breakdown:**
```
Total Training Memory = Parameters + Activations + Gradients + Optimizer State
                           ↓            ↓           ↓            ↓
                        Model         Forward     Backward     Adam: 2×params
                        weights       pass cache  gradients    SGD: 0×params

Example for 125M parameter model:
Parameters:    500 MB (125M × 4 bytes)
Activations:   200 MB (depends on batch size)
Gradients:     500 MB (same as parameters)
Adam state:  1,000 MB (momentum + velocity)
Total:      2,200 MB (4.4× parameter memory!)
```

### Latency Measurement: Dealing with Reality

Latency measurement is tricky because systems have variance, warmup effects, and measurement overhead. Professional profiling requires statistical rigor.

**Latency Measurement Best Practices:**

```
Measurement Protocol:
┌────────────────────────────────────────────────────────────────┐
│ 1. Warmup runs (10+)  → CPU/GPU caches warm up                │
│ 2. Timed runs (100+)  → Statistical significance              │
│ 3. Outlier handling   → Use median, not mean                  │
│ 4. Memory cleanup     → Prevent contamination                 │
└────────────────────────────────────────────────────────────────┘

Timeline:
Warmup: [run][run][run]...[run]     <- Don't time these
Timing: [run][run]...[run]          <- Time these
Result: median(all_times)           <- Robust to outliers
```

## 🏗️ Implementation: Building the Profiler Class

Now let's implement our profiler step by step. We'll start with the foundation and build up to comprehensive analysis.

### The Profiler Architecture

```
Profiler Class Structure:
┌─────────────────────────────────────────────────────────────┐
│ Core Measurement Methods:                                   │
│ • count_parameters() → Model size analysis                 │
│ • count_flops() → Computational cost estimation            │
│ • measure_memory() → Memory usage tracking                 │
│ • measure_latency() → Performance timing                   │
├─────────────────────────────────────────────────────────────┤
│ Advanced Profiling Methods:                                 │
│ • profile_layer() → Layer-wise analysis                    │
│ • profile_forward_pass() → Complete forward analysis       │
│ • profile_backward_pass() → Training analysis              │
├─────────────────────────────────────────────────────────────┤
│ Integration:                                                 │
│ All methods work together for comprehensive insights        │
└─────────────────────────────────────────────────────────────┘
```

## 🏗️ Helper Functions: Quick Profiling Utilities

These helper functions provide simplified interfaces for common profiling tasks. They make it easy to quickly profile models and analyze characteristics without manually calling multiple profiler methods.

### Why Helper Functions Matter

In production ML engineering, you often need quick insights without setting up full profiling workflows. These utilities provide:
- **Quick profiling**: One-line model analysis with formatted output
- **Weight analysis**: Understanding parameter distributions for compression
- **Student-friendly output**: Clear, formatted results for learning

These functions wrap our core Profiler class with convenience interfaces used in real ML workflows for rapid iteration and debugging.

### 🧪 Unit Test: Helper Functions

This test validates our helper utilities work correctly and provide useful output.

**What we're testing**: Quick profiling and weight distribution analysis
**Why it matters**: These utilities are used daily in production ML workflows
**Expected**: Correct profiles with formatted output

## 🏗️ Parameter Counting: Model Size Analysis

Parameter counting is the foundation of model profiling. Every parameter contributes to memory usage, training time, and model complexity. Let's validate our implementation.

### Why Parameter Counting Matters

```
Model Deployment Pipeline:
Parameters → Memory → Hardware → Cost
    ↓         ↓         ↓        ↓
  125M    500MB     8GB GPU   $200/month

Parameter Growth Examples:
┌──────────────────────────────────────────────────┐
│ Small:   GPT-2 Small (124M parameters)  → 500MB  │
│ Medium:  GPT-2 Medium (350M parameters) → 1.4GB  │
│ Large:   GPT-2 Large (774M parameters)  → 3.1GB  │
│ XL:      GPT-2 XL (1.5B parameters)     → 6.0GB  │
└──────────────────────────────────────────────────┘
```

### 🧪 Unit Test: _count_layer_parameters

This test validates the helper that counts parameters from a single layer's weight and bias.

**What we're testing**: Single-layer parameter counting from weight/bias attributes
**Why it matters**: This is the atomic unit of parameter counting that count_parameters delegates to
**Expected**: Correct weight + bias element counts

### 🧪 Unit Test: Parameter Counting

This test validates our parameter counting works correctly for different model types.

**What we're testing**: Parameter counting accuracy for various architectures
**Why it matters**: Accurate parameter counts predict memory usage and model complexity
**Expected**: Correct counts for known model configurations

## 🏗️ FLOP Counting: Computational Cost Estimation

FLOPs measure the computational work required for model operations. Unlike latency, FLOPs are hardware-independent and help predict compute costs across different systems.

### FLOP Counting Visualization

```
Linear Layer FLOP Breakdown:
┌────────────────────────────────────────────────────────────────┐
│ Input (batch=32, features=768) × Weight (768, 3072) + Bias     │
│                         ↓                                       │
│ Matrix Multiplication: 32 × 768 × 3072 × 2 = 150,994,944 FLOPs │
│ Bias Addition:         32 × 3072 × 1      =      98,304 FLOPs  │
│                         ↓                                       │
│ Total FLOPs:                                 151,093,248 FLOPs │
└────────────────────────────────────────────────────────────────┘

Convolution FLOP Breakdown:
┌────────────────────────────────────────────────────────────────┐
│ Input (batch=1, channels=3, H=224, W=224)                      │
│ Kernel (out=64, in=3, kH=7, kW=7)                             │
│                         ↓                                       │
│ Output size: (224×224) → (112×112) with stride=2              │
│ FLOPs = 112 × 112 × 7 × 7 × 3 × 64 × 2 = 236,027,904 FLOPs    │
└────────────────────────────────────────────────────────────────┘
```

### FLOP Counting Strategy

Different operations require different FLOP calculations:
- **Matrix operations**: M x N x K x 2 (multiply + add)
- **Convolutions**: Output spatial x kernel spatial x channels
- **Activations**: Usually 1 FLOP per element

### 🧪 Unit Test: _count_linear_flops

This test validates the helper that computes FLOPs for a single Linear layer.

**What we're testing**: Linear layer FLOP formula: in_features x out_features x 2
**Why it matters**: Linear layers dominate FLOP counts in most ML models
**Expected**: Exact FLOP count matching the formula

### 🧪 Unit Test: _count_conv_flops

This test validates the helper that computes FLOPs for a Conv2d layer.

**What we're testing**: Conv2d FLOP formula: out_H x out_W x k^2 x in_C x out_C x 2
**Why it matters**: Convolutions are the most compute-intensive operations in vision models
**Expected**: Correct FLOPs accounting for kernel size and channel dimensions

### 🧪 Unit Test: _count_sequential_flops

This test validates the helper that sums FLOPs across layers in a sequential model.

**What we're testing**: Accumulation of per-layer FLOPs with shape propagation
**Why it matters**: Real models are sequences of layers; total FLOPs = sum of per-layer FLOPs
**Expected**: Sum of individual layer FLOPs with correct shape propagation

### 🧪 Unit Test: FLOP Counting

This test validates our FLOP counting for different operations and architectures.

**What we're testing**: FLOP calculation accuracy for various layer types
**Why it matters**: FLOPs predict computational cost and energy usage
**Expected**: Correct FLOP counts for known operation types

## 🏗️ Memory Profiling: Understanding Memory Usage Patterns

Memory profiling reveals how much RAM your model consumes during training and inference. This is critical for deployment planning and performance optimization.

### Memory Usage Breakdown

```
ML Model Memory Components:
┌───────────────────────────────────────────────────┐
│                 Total Memory                      │
├─────────────────┬─────────────────┬───────────────┤
│   Parameters    │   Activations   │  Gradients    │
│   (persistent)  │  (per forward)  │ (per backward)│
├─────────────────┼─────────────────┼───────────────┤
│ Linear weights  │ Hidden states   │ dL/dW         │
│ Conv filters    │ Attention maps  │ dL/db         │
│ Embeddings      │ Residual cache  │ Optimizer     │
└─────────────────┴─────────────────┴───────────────┘

Memory Scaling:
┌────────────────────────────────────────────────────┐
│ Batch Size      → Activation Memory (linear)       │
│ Model Size      → Parameter + Gradient (linear)    │
│ Sequence Length → Attention Memory (quadratic!)    │
└────────────────────────────────────────────────────┘
```

### Memory Measurement Strategy

We use Python's `tracemalloc` to track memory allocations during model execution. This gives us precise measurements of memory usage patterns.

### 🧪 Unit Test: _calculate_parameter_memory

This test validates the helper that converts parameter count to memory in MB.

**What we're testing**: Parameter count to megabytes conversion using FP32 (4 bytes each)
**Why it matters**: Memory budgets determine which hardware can run your model
**Expected**: Exact byte-level accuracy for known parameter counts

### 🧪 Unit Test: _calculate_memory_efficiency

This test validates the helper that computes useful-to-total memory ratio.

**What we're testing**: Efficiency = useful_memory / peak_memory, clamped to [0, 1]
**Why it matters**: Low efficiency means memory fragmentation or allocator overhead
**Expected**: Values between 0 and 1, with division-by-zero safety

### 🧪 Unit Test: Memory Measurement

This test validates our memory tracking works correctly and provides useful metrics.

**What we're testing**: Memory usage measurement and calculation accuracy
**Why it matters**: Memory constraints often limit model deployment
**Expected**: Reasonable memory measurements with proper components

## 🏗️ Latency Measurement: Accurate Performance Timing

Latency measurement is the most challenging part of profiling because it's affected by system state, caching, and measurement overhead. We need statistical rigor to get reliable results.

### Latency Measurement Challenges

```
Timing Challenges:
┌─────────────────────────────────────────────────┐
│                 Time Variance                   │
├─────────────────┬─────────────────┬─────────────┤
│  System Noise   │   Cache Effects │   Thermal   │
│                 │                 │  Throttling │
├─────────────────┼─────────────────┼─────────────┤
│ Background      │ Cold start vs   │ CPU slows   │
│ processes       │ warm caches     │ when hot    │
│ OS scheduling   │ Memory locality │ GPU thermal │
│ Network I/O     │ Branch predict  │ limits      │
└─────────────────┴─────────────────┴─────────────┘

Solution: Statistical Approach
Warmup → Multiple measurements → Robust statistics (median)
```

### Measurement Protocol

Our latency measurement follows professional benchmarking practices:
1. **Warmup runs** to stabilize system state
2. **Multiple measurements** for statistical significance
3. **Median calculation** to handle outliers
4. **Memory cleanup** to prevent contamination

### 🧪 Unit Test: Latency Measurement

This test validates our latency measurement provides consistent and reasonable results.

**What we're testing**: Timing accuracy and statistical robustness
**Why it matters**: Latency determines real-world deployment feasibility
**Expected**: Consistent timing measurements with proper statistical handling

## 🔧 Integration: Advanced Profiling Functions

Now let's validate our higher-level profiling functions that combine core measurements into comprehensive analysis tools.

### Advanced Profiling Architecture

```
Core Profiler Methods → Advanced Analysis Functions → Optimization Insights
        ↓                         ↓                         ↓
count_parameters()      profile_forward_pass()      "Memory-bound workload"
count_flops()          profile_backward_pass()      "Optimize data movement"
measure_memory()       profile_layer()              "Focus on bandwidth"
measure_latency()      benchmark_efficiency()       "Use quantization"
```

### Forward Pass Profiling: Complete Performance Picture

A forward pass profile combines all our measurements to understand model behavior comprehensively. This is essential for optimization decisions.

### Backward Pass Profiling: Training Analysis

Training requires both forward and backward passes. The backward pass typically uses 2x the compute and adds gradient memory. Understanding this is crucial for training performance.

### Training Memory Visualization

```
Training Memory Timeline:
┌────────────────────────────────────────────────────────────────┐
│ Forward Pass:   [Parameters] + [Activations]                   │
│                      ↓                                          │
│ Backward Pass:  [Parameters] + [Activations] + [Gradients]     │
│                      ↓                                          │
│ Optimizer:      [Parameters] + [Gradients] + [Optimizer State] │
└────────────────────────────────────────────────────────────────┘

Memory Examples:
Model: 125M parameters (500MB)
┌────────────────────────────────────────────────────────────────┐
│ Forward:  500MB params + 100MB activations = 600MB             │
│ Backward: 500MB params + 100MB acts + 500MB grads = 1,100MB    │
│ Adam:     500MB params + 500MB grads + 1,000MB state = 2,000MB │
└────────────────────────────────────────────────────────────────┘

Total Training Memory: 4x parameter memory!
```

### 🧪 Unit Test: _compute_derived_metrics

This test validates the helper that converts raw FLOPs and latency into throughput metrics.

**What we're testing**: GFLOP/s, memory bandwidth, and computational efficiency calculations
**Why it matters**: These derived metrics determine whether a workload is memory-bound or compute-bound
**Expected**: Correct throughput calculations from known FLOP counts and latencies

### 🧪 Unit Test: _analyze_bottleneck

This test validates the helper that identifies memory-bound vs compute-bound workloads.

**What we're testing**: Bottleneck classification based on bandwidth/compute ratio
**Why it matters**: Knowing the bottleneck determines the right optimization strategy
**Expected**: Correct classification of memory-bound and compute-bound workloads

### 🧪 Unit Test: _estimate_backward_costs

This test validates the helper that estimates backward pass FLOPs and latency from forward measurements.

**What we're testing**: Backward costs = 2x forward costs (standard ML heuristic)
**Why it matters**: Training cost = forward + backward; backward is typically 2x forward
**Expected**: Backward FLOPs and latency are exactly 2x the forward values

### 🧪 Unit Test: _estimate_optimizer_memory

This test validates the helper that estimates memory requirements for different optimizers.

**What we're testing**: Per-optimizer memory multipliers (SGD: 0x, Adam: 2x gradient memory)
**Why it matters**: Adam uses 2x extra memory vs SGD; this affects hardware requirements
**Expected**: SGD = 0 extra, Adam = 2x gradient memory, AdamW = 2x gradient memory

### 🧪 Unit Test: Advanced Profiling Functions

This test validates our advanced profiling functions provide comprehensive analysis.

**What we're testing**: Forward and backward pass profiling completeness
**Why it matters**: Training optimization requires understanding both passes
**Expected**: Complete profiles with all required metrics and relationships

## 📊 Systems Analysis: Understanding Performance Characteristics

Let's analyze how different model characteristics affect performance. This analysis guides optimization decisions and helps identify bottlenecks.

### Performance Analysis Workflow

```
Model Scaling Analysis:
┌─────────────────────────────────────────────────────────────────┐
│ Size → Memory → Latency → Throughput → Bottleneck Identification│
│  ↓      ↓        ↓         ↓            ↓                       │
│ 64    1MB     0.1ms    10K ops/s    Memory bound                │
│ 128   4MB     0.2ms    8K ops/s     Memory bound                │
│ 256   16MB    0.5ms    4K ops/s     Memory bound                │
│ 512   64MB    2.0ms    1K ops/s     Memory bound                │
└─────────────────────────────────────────────────────────────────┘

Insight: This workload is memory-bound -> Optimize data movement, not compute!
```

## 📊 Optimization Insights: Production Performance Patterns

Understanding profiling results helps guide optimization decisions. Let's analyze different operation types and measurement overhead.

### Operation Efficiency Analysis

```
Operation Types and Their Characteristics:
┌─────────────────┬──────────────────┬──────────────────┬─────────────────┐
│   Operation     │   Compute/Memory │   Optimization   │   Priority      │
├─────────────────┼──────────────────┼──────────────────┼─────────────────┤
│ Matrix Multiply │   Compute-bound  │   BLAS libraries │   High          │
│ Elementwise     │   Memory-bound   │   Data locality  │   Medium        │
│ Reductions      │   Memory-bound   │   Parallelization│   Medium        │
│ Attention       │   Memory-bound   │   FlashAttention │   High          │
└─────────────────┴──────────────────┴──────────────────┴─────────────────┘

Optimization Strategy:
┌────────────────────────────────────────────────────────────────┐
│ 1. Profile first      → Identify bottlenecks                  │
│ 2. Compute-bound ops  → Algorithmic improvements              │
│ 3. Memory-bound ops   → Data movement optimization            │
│ 4. Measure again      → Verify improvements                   │
└────────────────────────────────────────────────────────────────┘
```

## 🧪 Module Integration Test

Final validation that everything works together correctly.

## 🤔 ML Systems Reflection Questions

Answer these to deepen your understanding of profiling operations and their systems implications:

### 1. FLOP Analysis
**Question**: You implemented a profiler that counts FLOPs for different operations. For a Linear layer with 1000 input features and 500 output features:

**Consider**:
- How many FLOPs are required for one forward pass?
- If you process a batch of 32 samples, how does this change the per-sample FLOPs?
- How does the FLOP count help you predict compute costs across different hardware?

---

### 2. Memory Scaling
**Question**: Your profiler measures memory usage for models and activations. A transformer model has 125M parameters (500MB at FP32). During training with batch size 16:

**Calculate**:
- What's the minimum memory for gradients?
- With Adam optimizer, what's the total memory requirement?
- How would mixed precision (FP16) change these numbers?

---

### 3. Performance Bottlenecks
**Question**: You built tools to identify compute vs memory bottlenecks. A model achieves 10 GFLOP/s on hardware with 100 GFLOP/s peak.

**Think about**:
- What's the computational efficiency?
- If doubling batch size doesn't improve GFLOP/s, the bottleneck is likely...
- How would you use profiling data to guide optimization strategy?

---

### 4. Profiling Trade-offs
**Question**: Your profiler adds measurement overhead to understand performance. If profiling adds 5x overhead but reveals a 50% speedup opportunity:

**Consider**:
- Is the profiling cost justified for development?
- When should you disable profiling in production?
- How does the cost of profiling compare to the cost of optimizing the wrong thing?

## ⭐ Aha Moment: Know Your Model

**What you built:** A complete profiler that measures parameters, FLOPs, memory, and latency.

**Why it matters:** You can't optimize what you can't measure! Before making a model faster or smaller, you need to know where the time and memory go. Your profiler reveals these secrets, telling you exactly what your model costs in compute and memory.

Profiling data guides optimization decisions — quantization, compression, and acceleration all start with measurement.

## 🚀 MODULE SUMMARY: Profiling

Congratulations! You've built a comprehensive profiling system for ML performance analysis!

### Key Accomplishments
- **Built complete Profiler class** with parameter, FLOP, memory, and latency measurement
- **Implemented advanced profiling functions** for forward and backward pass analysis
- **Discovered performance characteristics** through scaling and efficiency analysis
- **Created production-quality measurement tools** for optimization guidance
- **All tests pass** (validated by `test_module()`)

### Systems Insights Discovered
- **FLOPs vs Reality**: Theoretical operations don't always predict actual performance
- **Memory Bottlenecks**: Many ML operations are limited by memory bandwidth, not compute
- **Batch Size Effects**: Larger batches improve throughput but increase memory requirements
- **Profiling Overhead**: Measurement tools have costs but enable data-driven optimization

### Ready for Next Steps
Your profiling implementation provides the measurement foundation for all optimization work.
Export with: `tito module complete 14`

You can't optimize what you can't measure — and now you can measure everything.
