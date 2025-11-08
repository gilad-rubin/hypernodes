# Execution Engines

HyperNodes supports multiple execution strategies to optimize pipeline performance based on your workload characteristics.

## Overview

The `HypernodesEngine` provides execution modes through different **executors**:

1. **Sequential**: Nodes/items execute one at a time (default, great for debugging)
2. **Async**: Concurrent execution using `asyncio` (best for I/O-bound work)
3. **Threaded**: Parallel execution using threads (good for I/O + some CPU)
4. **Parallel (map only)**: True multi-core parallelism across map items using processes (best for CPU-bound per-item work)

You can independently configure:
- **node_executor**: How nodes execute within a single pipeline run (sequential/async/threaded)
- **map_executor**: How items execute across `pipeline.map()` calls (sequential/async/threaded/parallel)

## Quick Start

```python
from hypernodes import Pipeline, HypernodesEngine

# Sequential execution (default)
pipeline = Pipeline(
    nodes=[...],
    engine=HypernodesEngine()
)

# Async execution for I/O-bound workloads
pipeline = Pipeline(
    nodes=[...],
    engine=HypernodesEngine(node_executor="async")
)

# Parallel map for CPU-bound workloads (node-level parallel is disabled)
pipeline = Pipeline(
    nodes=[...],
    engine=HypernodesEngine(
        node_executor="threaded",
        map_executor="parallel",
    )
)
```

## Executor Types

### Sequential (Default)

```python
engine = HypernodesEngine(node_executor="sequential")
```

**Behavior:**
- Nodes execute one at a time in topological order
- Items in `.map()` processed sequentially
- Predictable, easy to debug
- No concurrency overhead

**Best for:**
- Development and debugging
- Simple pipelines
- Small batches (<10 items)

### Async

```python
engine = HypernodesEngine(
    node_executor="async",
    map_executor="async"
)
```

**Behavior:**
- Uses asyncio for concurrent execution
- Auto-wraps sync functions (no need to write async code!)
- Independent nodes run concurrently
- High concurrency (default: 100 concurrent tasks)

**Best for:**
- I/O-bound pipelines (API calls, file I/O, database queries)
- High-latency operations
- Large batches (100s+ items) with I/O

**Example:**
```python
from hypernodes import Pipeline, node, HypernodesEngine

@node(output_name="data")
def fetch_api(url: str) -> dict:
    """Sync function - auto-wrapped for async execution"""
    import requests
    return requests.get(url).json()

@node(output_name="processed")
def process(data: dict) -> dict:
    return {"count": len(data)}

pipeline = Pipeline(
    nodes=[fetch_api, process],
    engine=HypernodesEngine(map_executor="async")
)

# Process 100 URLs concurrently (not sequentially!)
results = pipeline.map(
    inputs={"url": [f"https://api.example.com/item/{i}" for i in range(100)]},
    map_over="url"
)
```

### Threaded

```python
engine = HypernodesEngine(
    node_executor="threaded",
    map_executor="threaded"
)
```

**Behavior:**
- Uses `ThreadPoolExecutor` for parallelism
- Overcomes GIL for I/O operations
- Moderate concurrency (default: CPU count)

**Best for:**
- Mixed I/O and CPU workloads
- Operations that release the GIL (numpy, pandas, C extensions)
- Moderate batch sizes (10-100 items)

**Example:**
```python
import numpy as np
from hypernodes import Pipeline, node, HypernodesEngine

@node(output_name="matrix")
def compute(size: int) -> np.ndarray:
    """Numpy releases GIL - benefits from threading"""
    return np.random.rand(size, size) @ np.random.rand(size, size)

pipeline = Pipeline(
    nodes=[compute],
    engine=HypernodesEngine(map_executor="threaded")
)

# Process 20 matrix multiplications in parallel
results = pipeline.map(
    inputs={"size": [1000] * 20},
    map_over="size"
)
```

### Parallel Map (Multiprocessing)

```python
engine = HypernodesEngine(map_executor="parallel")
```

**Behavior:**
- Uses `loky` (or `ProcessPoolExecutor` as fallback)
- True multicore parallelism
- **Requires picklable functions and data**
- Higher startup overhead (~0.1-0.5s)

**Best for:**
- CPU-bound per-item work (pure Python computation)
- Large batches (100s+ items) with CPU-intensive per-item work
- Operations >1s per item

**Example:**
```python
from hypernodes import Pipeline, node, HypernodesEngine

@node(output_name="result")
def cpu_intensive(n: int) -> int:
    """Pure Python - benefits from multiprocessing"""
    def fib(x):
        return x if x < 2 else fib(x-1) + fib(x-2)
    return fib(n)

pipeline = Pipeline(nodes=[cpu_intensive], engine=HypernodesEngine(map_executor="parallel"))

# Compute 8 fibonacci numbers in parallel across CPU cores
results = pipeline.map(
    inputs={"n": [35, 36, 37, 38, 35, 36, 37, 38]},
    map_over="n"
)
```

## Node-Level Parallelism

The engine automatically detects independent nodes and runs them in parallel:

```python
from hypernodes import Pipeline, node, HypernodesEngine
import time

@node(output_name="data")
def load_data() -> list:
    time.sleep(0.1)
    return [1, 2, 3, 4, 5]

@node(output_name="stat_a")
def compute_a(data: list) -> int:
    """Independent from compute_b"""
    time.sleep(0.1)
    return sum(data)

@node(output_name="stat_b")
def compute_b(data: list) -> int:
    """Independent from compute_a - can run in parallel!"""
    time.sleep(0.1)
    return max(data)

@node(output_name="result")
def combine(stat_a: int, stat_b: int) -> dict:
    """Depends on both A and B"""
    return {"sum": stat_a, "max": stat_b}

# DAG: load_data → (compute_a || compute_b) → combine
pipeline = Pipeline(nodes=[load_data, compute_a, compute_b, combine], engine=HypernodesEngine(node_executor="threaded"))

result = pipeline.run(inputs={})
# Sequential: ~0.4s (0.1 + 0.1 + 0.1 + 0.1)
# Threaded: ~0.3s (0.1 + max(0.1, 0.1) + 0.1)  # A and B run in parallel!
```

The engine uses NetworkX's `topological_generations` to compute dependency levels and execute independent nodes concurrently.

## Independent Configuration

You can configure node and map execution independently:

### Pattern 1: I/O Pipeline, Small Batch

```python
# Async nodes for efficiency, sequential map to avoid overwhelming API
engine = HypernodesEngine(
    node_executor="async",     # Concurrent nodes within each item
    map_executor="sequential"  # One item at a time (rate limiting)
)
```

### Pattern 2: CPU Pipeline, Large Batch

```python
# Sequential nodes for simplicity, parallel map to distribute items
engine = HypernodesEngine(
    node_executor="sequential",  # Simple execution per item
    map_executor="parallel"      # Multiple items across cores
)
```

### Pattern 3: Maximum Concurrency

```python
# Both async for I/O-heavy workloads
engine = HypernodesEngine(
    node_executor="async",
    map_executor="async"
)
```

## Performance Tuning

### Choosing the Right Executor

| Workload Type | Node Executor | Map Executor | Why |
|---------------|---------------|--------------|-----|
| Pure I/O (API calls) | `async` | `async` | Maximum concurrency for I/O |
| Pure CPU (ML inference) | `sequential` | `parallel` | Avoid GIL, use multiple cores |
| Mixed I/O + CPU | `threaded` | `threaded` | Balance both |
| Development/Debug | `sequential` | `sequential` | Predictable, easy to trace |
| Small batches (<10) | `sequential` | `sequential` | Overhead not worth it |
| Large batches (100s+) | `sequential` | `parallel/async` | Amortize overhead |

### Always Measure!

```python
import time
from hypernodes import Pipeline, HypernodesEngine

executors = ["sequential", "async", "threaded", "parallel"]
items = [...]  # Your data

for executor in executors:
    pipeline = Pipeline(
        nodes=[...],
        engine=HypernodesEngine(map_executor=executor)
    )

    start = time.time()
    results = pipeline.map(inputs={"data": items}, map_over="data")
    duration = time.time() - start

    print(f"{executor:12s}: {duration:.2f}s")
```

### Limitations

**Node-level Parallel:**
- ⚠️ Disabled to avoid pickling/locking complexity with nested pipelines
- ✅ Use `node_executor="threaded"` for intra-item concurrency
- ✅ Use `map_executor="parallel"` to scale across items (data-parallel)

**Parallel Map Mode:**
- ⚠️ Functions and data must be picklable
- ⚠️ Cannot pickle lambda functions or local functions
- ⚠️ Startup overhead (~0.1-0.5s)
- ✅ Best for workloads >1s per item

**Async Mode:**
- ⚠️ Sync functions don't truly benefit from async I/O
- ⚠️ Won't speed up CPU-bound work
- ✅ Great for high-latency I/O operations

**Threaded Mode:**
- ⚠️ Subject to Python's GIL for CPU work
- ✅ Good for I/O and GIL-releasing operations
- ✅ Lower overhead than parallel

## DaftEngine (Distributed DataFrames)

For distributed execution using Daft DataFrames:

```python
from hypernodes.engines import DaftEngine

pipeline = Pipeline(
    nodes=[...],
    engine=DaftEngine(collect=True)
)

# Executes pipeline as a lazy Daft DataFrame computation
```

See [DaftEngine documentation](daft-backend.md) for details.

## See Also

- [Async Auto-Wrapping](async-autowrap.md) - How sync functions work with async executor
- [Caching](../in-depth/caching.md) - Cache results across runs
- [Callbacks](../in-depth/callbacks.md) - Monitor execution progress
