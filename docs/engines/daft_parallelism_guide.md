# Daft Parallelism Guide: Achieving "For Free" Concurrency

## Overview

This guide helps you choose the right execution strategy to achieve maximum parallelism in HyperNodes. Based on comprehensive benchmarks, we've identified which strategies provide "for free" parallelism comparable to or better than DaskEngine's 7-8x speedup.

## Quick Decision Matrix

| Workload Type | Function Type | Best Engine | Config | Expected Speedup |
|---------------|--------------|-------------|--------|------------------|
| **I/O-bound** | Async | DaftEngine | default | **30-37x** ⚡⚡⚡ |
| **I/O-bound** | Sync | DaftEngine | `use_batch_udf=True` | **9-12x** ⚡⚡ |
| **I/O-bound** | Sync (simple) | DaskEngine | `scheduler="threads"` | **7-8x** ⚡ |
| **CPU-bound** | Sync | DaskEngine | `scheduler="processes"` | **4-6x** ⚡ |
| **CPU-bound** | Stateful | DaftEngine | `@daft.cls` approach | **1-2x** |
| **Mixed** | Sync | DaskEngine | `scheduler="threads"` | **3-5x** |

## Key Findings from Benchmarks

### 1. Async Functions: The Clear Winner (30-37x speedup)

**DaftEngine automatically detects async functions** and routes them to Daft's native async concurrency support.

```python
import asyncio
from hypernodes import Pipeline, node
from hypernodes.engines import DaftEngine

@node(output_name="result")
async def fetch_url(url: str) -> str:
    """Async I/O-bound operation."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

# DaftEngine automatically uses async concurrency!
pipeline = Pipeline(
    nodes=[fetch_url],
    engine=DaftEngine()  # That's it!
)

urls = ["https://api.example.com/1", "https://api.example.com/2", ...]
results = pipeline.map(inputs={"url": urls}, map_over="url")

# Result: 30-37x speedup for I/O-bound tasks! 🚀
```

**How it works:**
- DaftEngine detects `asyncio.iscoroutinefunction(node.func)`
- Routes to `@daft.func(async_func)` which Daft executes concurrently
- Zero configuration needed!

### 2. Sync Functions with Batch UDF (9-12x speedup)

**For sync I/O-bound functions**, DaftEngine uses ThreadPoolExecutor in batch mode:

```python
import time
from hypernodes import Pipeline, node
from hypernodes.engines import DaftEngine

@node(output_name="result")
def process_file(filepath: str) -> dict:
    """Sync I/O-bound operation."""
    time.sleep(0.01)  # Simulated I/O
    return {"processed": filepath}

# DaftEngine uses ThreadPoolExecutor for parallel execution
pipeline = Pipeline(
    nodes=[process_file],
    engine=DaftEngine(use_batch_udf=True)  # Enable batch mode
)

files = [f"file_{i}.txt" for i in range(100)]
results = pipeline.map(inputs={"filepath": files}, map_over="filepath")

# Result: 9-12x speedup! 🚀
```

**How it works:**
- Batches items together
- Uses `ThreadPoolExecutor` for parallel execution within batches
- Ideal for I/O-bound sync functions

**Configuration options:**
```python
engine = DaftEngine(
    use_batch_udf=True,
    default_daft_config={
        "max_workers": 16,  # ThreadPoolExecutor workers
        "batch_size": 1024,  # Items per batch
    }
)
```

### 3. DaskEngine for Simple Sync Functions (7-8x speedup)

**For straightforward sync functions**, DaskEngine is the simplest choice:

```python
from hypernodes import Pipeline, node
from hypernodes.engines import DaskEngine

@node(output_name="squared")
def compute(x: int) -> int:
    """Simple sync computation."""
    return x ** 2

# DaskEngine: simple and effective
pipeline = Pipeline(
    nodes=[compute],
    engine=DaskEngine(scheduler="threads")  # Or "processes" for CPU-bound
)

numbers = list(range(1000))
results = pipeline.map(inputs={"x": numbers}, map_over="x")

# Result: 7-8x speedup! 🚀
```

**When to use DaskEngine:**
- Simple sync functions
- Don't want to think about batch/async
- CPU-bound work (use `scheduler="processes"`)

## Detailed Performance Comparison

### I/O-Bound Workload (100 items, 10ms delay each)

| Strategy | Time (s) | Speedup | When to Use |
|----------|----------|---------|-------------|
| Sequential baseline | 1.245 | 1.00x | Never |
| **DaftEngine + async** | **0.033** | **37.24x** ⚡⚡⚡ | **Async functions (best!)** |
| **DaftEngine + sync batch** | **0.107** | **11.65x** ⚡⚡ | **Sync I/O functions** |
| DaskEngine (threads) | 0.171 | 7.29x ⚡ | Simple sync functions |
| Pure Daft sync @daft.func | 1.229 | 1.01x | Don't use (no parallelism) |

### CPU-Bound Workload

For CPU-bound workloads:
- **DaskEngine with `scheduler="processes"`** (bypasses GIL)
- For trivial tasks, overhead dominates - sequential may be faster
- For heavy computation (>10ms per item), processes provide 4-6x speedup

## Implementation Details

### How DaftEngine Routes Functions

```python
# In DaftEngine._apply_simple_node_transformation():

import asyncio

is_async = asyncio.iscoroutinefunction(node.func)

if is_async:
    # Route to Daft's async support (37x speedup)
    udf = daft.func(node.func)
    df = df.with_column(output_name, udf(*input_cols))
    
elif use_batch_udf:
    # Route to ThreadPoolExecutor batch mode (10x speedup)
    # Uses @daft.func.batch with concurrent.futures
    
else:
    # Row-wise sync (no parallelism - 1x)
    udf = daft.func(node.func)
```

### Batch UDF with ThreadPoolExecutor

The improved batch UDF implementation:

```python
@daft.func.batch(return_dtype=DataType.python())
def batch_udf(*series_args: Series) -> Series:
    import concurrent.futures
    
    # Convert Series to Python lists
    python_args = [s.to_pylist() for s in series_args]
    
    # Process items in parallel using ThreadPoolExecutor
    def process_item(idx):
        args = [arg[idx] if isinstance(arg, list) else arg 
                for arg in python_args]
        return node_func(*args)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_item, range(n_items)))
    
    return Series.from_pylist(results)
```

## Best Practices

### 1. Use Async for I/O-Bound Tasks

✅ **DO:**
```python
@node(output_name="data")
async def fetch_api(url: str) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()

pipeline = Pipeline(nodes=[fetch_api], engine=DaftEngine())
# Automatic 30-37x speedup!
```

❌ **DON'T:**
```python
@node(output_name="data")
def fetch_api(url: str) -> dict:  # Sync version
    response = requests.get(url)  # Blocks!
    return response.json()

pipeline = Pipeline(nodes=[fetch_api], engine=DaftEngine(use_batch_udf=False))
# Only 1x speedup (sequential)
```

### 2. Enable Batch UDF for Sync I/O Functions

✅ **DO:**
```python
pipeline = Pipeline(
    nodes=[sync_io_function],
    engine=DaftEngine(use_batch_udf=True)  # Enable ThreadPool
)
# 9-12x speedup
```

❌ **DON'T:**
```python
pipeline = Pipeline(
    nodes=[sync_io_function],
    engine=DaftEngine(use_batch_udf=False)  # Sequential!
)
# Only 1x speedup
```

### 3. Use DaskEngine for CPU-Bound Work

✅ **DO:**
```python
pipeline = Pipeline(
    nodes=[heavy_computation],
    engine=DaskEngine(
        scheduler="processes",  # Bypass GIL
        workload_type="cpu"
    )
)
# 4-6x speedup
```

### 4. Avoid Sync Row-wise for Multiple Items

❌ **NEVER DO:**
```python
# This gives NO parallelism (1x speedup)
@node(output_name="result")
def process(item: str) -> str:
    time.sleep(0.01)  # I/O simulation
    return item.upper()

pipeline = Pipeline(
    nodes=[process],
    engine=DaftEngine(use_batch_udf=False)  # DON'T!
)
```

## Advanced Configuration

### Custom ThreadPool Size

```python
engine = DaftEngine(
    use_batch_udf=True,
    default_daft_config={
        "max_workers": 32,  # More workers for high I/O concurrency
        "batch_size": 512,  # Smaller batches for better load balancing
    }
)
```

### Hybrid Approach

Mix engines for different pipeline stages:

```python
# Stage 1: I/O-bound with async
fetch_pipeline = Pipeline(nodes=[async_fetch], engine=DaftEngine())

# Stage 2: CPU-bound processing
process_pipeline = Pipeline(nodes=[heavy_compute], engine=DaskEngine(scheduler="processes"))

# Combine
results1 = fetch_pipeline.map(...)
results2 = process_pipeline.map(inputs={"data": results1}, ...)
```

## Common Pitfalls

### 1. Returning Lists/Dicts from Batch UDFs

Batch UDFs currently don't support returning complex types:

```python
# ❌ This will fail with batch UDF
@node(output_name="chunks")
def split_text(text: str) -> List[str]:
    return text.split()

# ✅ Solution: Disable batch for this node
# DaftEngine auto-detects list return type and uses row-wise
```

### 2. Overhead for Tiny Tasks

For very fast operations (<1ms per item), overhead may dominate:

```python
# Sequential may be faster!
@node(output_name="upper")
def simple_upper(text: str) -> str:
    return text.upper()  # Too fast for parallelism to help
```

**Rule of thumb:** Parallelism helps when per-item execution > 5ms

### 3. Not Using Async When Available

```python
# ❌ Sync version (7x speedup with Dask)
def fetch(url):
    return requests.get(url).text

# ✅ Async version (37x speedup with DaftEngine!)
async def fetch(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()
```

## Migration Guide

### From Sequential to Parallel

```python
# BEFORE (Sequential)
from hypernodes import Pipeline, node
from hypernodes.engines import SequentialEngine

pipeline = Pipeline(nodes=[...], engine=SequentialEngine())

# AFTER (Parallel)
# Option 1: Async functions (30-37x speedup)
from hypernodes.engines import DaftEngine
pipeline = Pipeline(nodes=[async_node], engine=DaftEngine())

# Option 2: Sync I/O (9-12x speedup)
pipeline = Pipeline(nodes=[sync_io_node], engine=DaftEngine(use_batch_udf=True))

# Option 3: Simple sync (7-8x speedup)
from hypernodes.engines import DaskEngine
pipeline = Pipeline(nodes=[sync_node], engine=DaskEngine())
```

## Summary: Which Engine Should I Use?

### Use DaftEngine when:
- ✅ You have **async functions** (best performance: 30-37x)
- ✅ You have **sync I/O functions** and enable batch mode (9-12x)
- ✅ You need **complex data transformations** (nested pipelines, explode patterns)
- ✅ You want **automatic async detection**

### Use DaskEngine when:
- ✅ You have **simple sync functions** (7-8x speedup)
- ✅ You have **CPU-bound work** (use `scheduler="processes"`)
- ✅ You want **zero configuration** (works out of the box)
- ✅ You don't need complex data transformations

### Use SequentialEngine when:
- ✅ **Debugging** (predictable, step-by-step execution)
- ✅ **Very small datasets** (<10 items where overhead > benefit)
- ✅ **Trivial operations** (<1ms per item)

## Benchmark Scripts

Full benchmark code is available in:
- `scripts/benchmark_daft_parallelism.py` - Comprehensive comparison
- `scripts/test_improved_daft_engine.py` - Validation tests
- `scripts/debug_async_daft.py` - Async performance debugging

## See Also

- [DaskEngine Documentation](dask_engine.md)
- [DaftEngine Advanced Features](../advanced/daft-engine.md)
- [Async Best Practices](../in-depth/async-patterns.md)

