# Execution Engines

HyperNodes supports multiple execution strategies to optimize pipeline performance based on your workload characteristics.

## Overview

The `LocalBackend` provides four execution modes that can be independently configured for **node execution** (within a single pipeline run) and **map execution** (across multiple items):

### Execution Modes

1. **Sequential** (default): Nodes/items execute one at a time
2. **Async**: Concurrent execution using `asyncio` (best for I/O-bound work)
3. **Threaded**: Parallel execution using `ThreadPoolExecutor` (good for I/O + some CPU)
4. **Parallel**: True multi-core parallelism using `ProcessPoolExecutor` (best for CPU-bound work)

## Configuration

```python
from hypernodes import LocalBackend

# Configure node execution mode
backend = LocalBackend(
    node_execution="threaded",    # How nodes execute within pipeline.run()
    map_execution="async",         # How items execute in pipeline.map()
    max_workers=8                  # Max concurrent workers
)

pipeline.backend = backend
```

## Node Execution Modes

Controls how **independent nodes** within a single `pipeline.run()` execute:

### Sequential (Default)
```python
backend = LocalBackend(node_execution="sequential")
```
- Nodes execute one at a time in topological order
- Predictable, easy to debug
- Best for: Development, debugging, simple pipelines

### Async
```python
backend = LocalBackend(node_execution="async")
```
- Independent nodes execute concurrently using `asyncio`
- Nodes with no dependencies run at the same time
- Single-process, no true parallelism
- Best for: I/O-bound pipelines (API calls, file I/O, database queries)

**Example:**
```python
# Pipeline: A → (B, C) → D
# Sequential: A → B → C → D  (~0.4s if each takes 0.1s)
# Async: A → (B || C) → D     (~0.3s, B and C run concurrently)
```

### Threaded
```python
backend = LocalBackend(node_execution="threaded", max_workers=4)
```
- Independent nodes execute in parallel using threads
- Overcomes GIL for I/O operations
- Best for: Mixed I/O and CPU workloads

### Parallel
```python
backend = LocalBackend(node_execution="parallel", max_workers=4)
```
- Independent nodes execute in true parallel using processes
- Multiple CPU cores utilized
- **Requires picklable functions and data**
- Best for: CPU-bound work that can benefit from multiple cores

**Note:** Process-based parallelism has overhead. Test to ensure it provides speedup for your workload.

## Map Execution Modes

Controls how **multiple items** are processed in `pipeline.map()`:

### Sequential (Default)
```python
backend = LocalBackend(map_execution="sequential")
```
- Items processed one at a time
- Simple loop: `for item in items: pipeline.run(item)`
- Best for: Development, debugging, rate-limited APIs

### Async
```python
backend = LocalBackend(map_execution="async", max_workers=10)
```
- Multiple items processed concurrently using `asyncio`
- Items don't block each other during I/O
- Best for: I/O-bound per-item work (e.g., API calls for each item)

```python
# Process 100 items with API calls
backend = LocalBackend(map_execution="async", max_workers=20)
results = pipeline.map(
    inputs={"url": urls},  # 100 URLs
    map_over="url"
)
# Sequential: ~100s (1s per item)
# Async: ~5s (20 concurrent requests)
```

### Threaded
```python
backend = LocalBackend(map_execution="threaded", max_workers=8)
```
- Multiple items processed in parallel using threads
- Good for I/O + CPU mixed workloads
- Best for: Moderate CPU work per item with some I/O

### Parallel
```python
backend = LocalBackend(map_execution="parallel", max_workers=8)
```
- Multiple items processed in true parallel using processes
- Utilizes multiple CPU cores
- **Requires picklable functions and data**
- Best for: CPU-intensive per-item work, large batches

## Mixing Execution Modes

You can independently configure node and map execution:

```python
# Pattern 1: I/O-bound pipeline, small batch
# Async nodes for I/O efficiency, sequential map to avoid overwhelming API
backend = LocalBackend(
    node_execution="async",
    map_execution="sequential"
)

# Pattern 2: CPU-bound nodes, large batch
# Sequential nodes for simplicity, parallel map to distribute items
backend = LocalBackend(
    node_execution="sequential",
    map_execution="parallel",
    max_workers=8
)

# Pattern 3: Maximum concurrency
# Both async for I/O-heavy workloads
backend = LocalBackend(
    node_execution="async",
    map_execution="async",
    max_workers=20
)
```

## Intelligent Resource Management

HyperNodes automatically manages resources for **nested map operations** to prevent exponential explosion of concurrent tasks:

```python
# Outer: 100 items, each with inner: 50 sub-items = 5000 total operations
# Without management: 5000 parallel tasks (overwhelms system!)
# With management: Intelligently limits concurrency

backend = LocalBackend(map_execution="threaded", max_workers=8)

# Top-level map: Uses full 8 workers
# Nested map (depth=1): Reduces to sqrt(8) ≈ 3 workers
# Deeper nesting (depth=2+): Sequential to prevent explosion
```

**Algorithm:**
- **Depth 0** (top level): `min(max_workers, num_items)`
- **Depth 1** (first nested): `min(sqrt(max_workers), num_items)`
- **Depth 2+** (deeper nesting): `1` (sequential)

## Custom Executors

You can provide your own executor (e.g., from `concurrent.futures` or `joblib`):

```python
from concurrent.futures import ThreadPoolExecutor

# Create custom executor with specific configuration
custom_executor = ThreadPoolExecutor(
    max_workers=4,
    thread_name_prefix="HyperNodes-"
)

backend = LocalBackend(
    map_execution="threaded",
    executor=custom_executor  # Use your executor
)

# HyperNodes will use your executor for threaded/parallel operations
# You're responsible for shutting it down
custom_executor.shutdown(wait=True)
```

## Performance Considerations

### When to Use Each Mode

| Workload Type | Node Execution | Map Execution | Why |
|---------------|----------------|---------------|-----|
| Pure I/O (API calls) | `async` | `async` | Maximum concurrency for I/O |
| Pure CPU (ML inference) | `sequential` | `parallel` | Avoid GIL, use multiple cores |
| Mixed I/O + CPU | `threaded` | `threaded` | Balance between both |
| Development/Debug | `sequential` | `sequential` | Predictable, easy to trace |
| Small batches (<10) | `sequential` | `sequential` | Overhead not worth it |
| Large batches (100s+) | `sequential` | `parallel/async` | Amortize overhead |

### Measuring Performance

Always measure! Parallelism overhead varies:

```python
import time

modes = ["sequential", "async", "threaded", "parallel"]
for mode in modes:
    backend = LocalBackend(map_execution=mode, max_workers=8)
    pipeline.backend = backend
    
    start = time.time()
    results = pipeline.map(inputs={"data": items}, map_over="data")
    duration = time.time() - start
    
    print(f"{mode}: {duration:.2f}s")
```

### Limitations

**Parallel Mode (ProcessPoolExecutor):**
- Functions and data must be picklable
- Cannot pickle lambda functions, local functions, or certain objects
- Has startup overhead (~0.1-0.5s)
- Best for workloads >1s per item

**Async Mode:**
- Functions still run synchronously (not truly async I/O)
- Concurrency, not parallelism
- Won't speed up CPU-bound work
- Best combined with async-aware libraries

**Threaded Mode:**
- Subject to Python's GIL for CPU work
- Good for I/O, limited for pure CPU
- Lower overhead than processes

## Examples

### Example 1: Text Processing Pipeline

```python
from hypernodes import Pipeline, node, LocalBackend

@node(output_name="text")
def load_text(file_path):
    """I/O-bound: Read file"""
    with open(file_path) as f:
        return f.read()

@node(output_name="tokens")
def tokenize(text):
    """CPU-bound: Process text"""
    return text.split()

@node(output_name="count")
def count_words(tokens):
    """CPU-light: Count"""
    return len(tokens)

pipeline = Pipeline(nodes=[load_text, tokenize, count_words])

# Async map for I/O efficiency across many files
backend = LocalBackend(
    node_execution="sequential",  # Simple within each file
    map_execution="async",        # Process many files concurrently
    max_workers=20
)
pipeline.backend = backend

file_paths = [f"doc_{i}.txt" for i in range(100)]
results = pipeline.map(
    inputs={"file_path": file_paths},
    map_over="file_path"
)
```

### Example 2: Independent Nodes

```python
@node(output_name="data")
def load_data():
    time.sleep(0.1)
    return [1, 2, 3, 4, 5]

@node(output_name="stats_a")
def compute_stats_a(data):
    """Independent computation A"""
    time.sleep(0.1)
    return sum(data)

@node(output_name="stats_b")
def compute_stats_b(data):
    """Independent computation B (runs in parallel with A)"""
    time.sleep(0.1)
    return max(data)

@node(output_name="result")
def combine(stats_a, stats_b):
    """Depends on both A and B"""
    return {"sum": stats_a, "max": stats_b}

pipeline = Pipeline(nodes=[load_data, compute_stats_a, compute_stats_b, combine])

# Use threaded execution to run stats_a and stats_b in parallel
backend = LocalBackend(node_execution="threaded", max_workers=4)
pipeline.backend = backend

result = pipeline.run(inputs={})
# Sequential: ~0.4s (0.1 + 0.1 + 0.1 + 0.1)
# Threaded: ~0.3s (0.1 + max(0.1, 0.1) + 0.1)
```

## See Also

- [Modal Backend](modal-backend.md) - Remote GPU execution
- [Caching](../in-depth/caching.md) - Cache results across runs
- [Progress Tracking](../in-depth/callbacks.md) - Monitor execution
