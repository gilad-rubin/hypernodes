# HyperNodes vs Daft Benchmark Results

## Overview

This document presents comprehensive benchmarks comparing **HyperNodes** with **Daft** across multiple scenarios. We test three configurations:

1. **HyperNodes** - with various execution modes (sequential, threaded, async)
2. **Daft with UDFs** - custom functions using `@daft.func` and `@daft.cls`
3. **Daft with Built-ins** - using native Daft operations where available

## Benchmark Scenarios

### 1. Simple Text Processing
**Task**: Clean text, tokenize, and count tokens

**Results** (1000 items):
- HyperNodes (sequential): 0.1486s
- HyperNodes (threaded): 0.3283s
- Daft (UDFs): 0.6000s
- **Daft (Built-ins): 0.0715s** ⭐

**Key Insight**: Daft's built-in operations are significantly faster when available. However, Daft lacks `.str.strip()` and `.str.lower()` built-ins, so the built-in version skips text cleaning.

### 2. Stateful Processing with Expensive Initialization
**Task**: Encode text using a model with expensive initialization (simulated with 0.1s sleep)

**Results** (500 items):
- **HyperNodes (sequential): 0.0299s** ⭐
- Daft (UDF with @daft.cls): 0.1095s

**Key Insight**: HyperNodes initializes the encoder once upfront and reuses it. Daft's `@daft.cls` also initializes once per worker, but has some overhead. For single-machine execution with pre-initialized objects, HyperNodes is faster.

### 3. Batch Vectorized Operations
**Task**: Normalize numerical values using vectorized operations

**Results** (1000 items):
- HyperNodes (threaded, row-wise): 0.1393s
- Daft (Batch UDF): 0.1120s
- **Daft (Built-in ops): 0.0085s** ⭐

**Key Insight**: Daft's built-in vectorized operations are extremely fast. Batch UDFs are also efficient. HyperNodes processes row-wise by default, which is slower for vectorizable operations.

### 4. Nested Pipelines with Heavy Computation
**Task**: Encode documents, then search for each query (retrieval-like workflow)

**Results** (250 docs, 50 queries):
- HyperNodes (nested): 0.3706s (encode: 0.0912s, search: 0.2794s)
- **Daft (UDFs): 0.2398s** ⭐

**Key Insight**: Daft's automatic optimization and parallelization gives it an edge in complex workflows. HyperNodes provides more explicit control over the pipeline structure.

### 5. Generator Functions (One-to-Many)
**Task**: Tokenize sentences, expanding each sentence into multiple tokens

**Results** (1000 sentences):
- HyperNodes (manual flatten): 0.0432s
- Daft (Generator UDF): 0.0292s
- **Daft (Built-in explode): 0.0198s** ⭐

**Key Insight**: Daft's native generator support and `.explode()` method are more elegant and faster than manual flattening in HyperNodes.

## Summary

### When to Use HyperNodes
- ✅ **Explicit DAG control**: Need to visualize and control pipeline structure
- ✅ **Fine-grained caching**: Want node-level caching with explicit cache keys
- ✅ **Pre-initialized objects**: Have expensive objects initialized upfront
- ✅ **Modular components**: Building reusable pipeline components
- ✅ **Complex branching**: Need explicit control over execution flow
- ✅ **Multiple execution modes**: Want to switch between sequential/threaded/async

### When to Use Daft
- ✅ **Performance critical**: Need maximum speed for large datasets
- ✅ **Large datasets**: Processing >1GB of data
- ✅ **Automatic optimization**: Want the system to optimize for you
- ✅ **Distributed execution**: Need to scale across multiple machines
- ✅ **Vectorized operations**: Working with numerical data that can be batched
- ✅ **Built-in operations**: Can use Daft's native string/list/numeric operations

## Execution Modes Comparison

### HyperNodes Execution Modes
```python
# Sequential (default)
backend = LocalBackend(node_execution="sequential", map_execution="sequential")

# Threaded (I/O + CPU)
backend = LocalBackend(node_execution="threaded", map_execution="threaded")

# Async (I/O-bound)
backend = LocalBackend(node_execution="async", map_execution="async")

# Parallel (CPU-bound, requires picklable objects)
backend = LocalBackend(node_execution="parallel", map_execution="parallel")
```

### Daft Execution
Daft automatically parallelizes and optimizes execution. No manual configuration needed.

## Performance Characteristics

| Scenario | HyperNodes Best | Daft Best | Winner |
|----------|----------------|-----------|--------|
| Simple Text Processing | 0.1486s (seq) | 0.0715s (built-in) | **Daft** |
| Stateful Processing | 0.0299s (seq) | 0.1095s (UDF) | **HyperNodes** |
| Batch Operations | 0.1393s (threaded) | 0.0085s (built-in) | **Daft** |
| Nested Pipelines | 0.3706s (threaded) | 0.2398s (UDF) | **Daft** |
| Generators | 0.0432s (seq) | 0.0198s (built-in) | **Daft** |

## Scaling Characteristics

The benchmark script supports three scale factors:
- **Small**: 100 items
- **Medium**: 1000 items (default)
- **Large**: 5000 items

To test different scales, modify the `CURRENT_SCALE` variable in the script.

## Running the Benchmarks

```bash
cd /Users/giladrubin/python_workspace/hypernodes
uv run python scripts/benchmark_hypernodes_vs_daft.py
```

## Conclusion

Both frameworks have their strengths:

- **Daft** excels at performance, especially with built-in operations and automatic optimization
- **HyperNodes** excels at explicit control, modularity, and working with pre-initialized objects

The choice depends on your specific use case:
- For **data processing pipelines** with large datasets → **Daft**
- For **ML pipelines** with complex logic and caching → **HyperNodes**
- For **hybrid workflows** → Consider using both (HyperNodes for orchestration, Daft for data processing)
