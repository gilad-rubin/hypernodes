# DaskEngine Documentation

## Overview

`DaskEngine` is a parallel execution engine for HyperNodes that leverages Dask Bag to provide automatic parallelization of map operations. It's designed to be a drop-in replacement for `SequentialEngine` with zero configuration required.

## Key Features

- **Zero Configuration**: Works out of the box with sensible defaults
- **Automatic Optimization**: Calculates optimal `npartitions` based on dataset size and workload type
- **No Overhead for Single Runs**: Uses sequential execution for regular `.run()` calls
- **Smart Parallelism**: Only activates Dask for `.map()` operations where it provides benefit
- **Configurable**: Fine-tune scheduler and workload type when needed

## Quick Start

```python
from hypernodes import Pipeline, node
from hypernodes.engines import DaskEngine

@node(output_name="result")
def process(x: int) -> int:
    return x * 2

# Create pipeline with DaskEngine
engine = DaskEngine()
pipeline = Pipeline(nodes=[process], engine=engine)

# Regular run (sequential, no overhead)
result = pipeline.run(inputs={"x": 5})

# Map operation (parallel via Dask Bag)
results = pipeline.map(
    inputs={"x": list(range(100))},
    map_over="x"
)
```

## Configuration

### Default (Auto-Optimized)

```python
engine = DaskEngine()
# Uses: scheduler="threads", workload_type="mixed", auto npartitions
```

### CPU-Bound Workloads

For heavy computation (ML inference, complex math):

```python
engine = DaskEngine(
    scheduler="threads",  # or "processes" to bypass GIL
    workload_type="cpu"
)
```

### I/O-Bound Workloads

For API calls, file I/O, database queries:

```python
engine = DaskEngine(
    scheduler="threads",
    workload_type="io"
)
```

### Manual Control

Override automatic npartitions calculation:

```python
engine = DaskEngine(
    npartitions=16  # Fixed number of partitions
)
```

## How It Works

### Architecture

1. **Single Run (`pipeline.run()`)**: Uses sequential execution identical to `SequentialEngine` to avoid Dask overhead
2. **Map Operations (`pipeline.map()`)**: Uses Dask Bag with automatically calculated `npartitions`
3. **Nested Pipelines**: Parent engine configuration is inherited

### Npartitions Heuristic

The engine calculates optimal `npartitions` based on:

1. **Workload Type**:
   - `"io"`: 4x CPU count (higher parallelism for I/O wait)
   - `"cpu"`: 2x CPU count (match cores for compute)
   - `"mixed"`: 3x CPU count (balanced approach)

2. **Dataset Size**:
   - Target: 10-1000 items per partition
   - Avoids too few partitions (limits parallelism)
   - Avoids too many partitions (excessive overhead)

3. **Power of 2**:
   - Prefers powers of 2 for better distribution (when close)

### Example Calculations

```python
# Small dataset (10 items), 8 CPU cores, mixed workload
# -> 8 partitions (8 cores * 3 = 24, clamped to 8 due to granularity)

# Medium dataset (100 items), 8 CPU cores, mixed workload
# -> 8 partitions (optimal balance)

# Large dataset (500 items), 8 CPU cores, mixed workload
# -> 16 partitions (allows better parallelism)
```

## Schedulers

### Threads (Default)

```python
engine = DaskEngine(scheduler="threads")
```

**Best for:**
- Mixed I/O and CPU workloads
- Python code that releases GIL (NumPy, pandas, etc.)
- Most common use case

**Pros:**
- Low overhead
- Shared memory (no serialization cost)
- Fast for I/O-bound tasks

**Cons:**
- Subject to Python GIL for pure Python code

### Processes

```python
engine = DaskEngine(scheduler="processes")
```

**Best for:**
- Pure Python CPU-intensive code
- Workloads that hold GIL

**Pros:**
- Bypasses GIL
- True parallelism for Python code

**Cons:**
- Higher overhead (serialization cost)
- Slower for small tasks
- More memory usage

### Synchronous

```python
engine = DaskEngine(scheduler="synchronous")
```

**Best for:**
- Debugging
- Benchmarking overhead

**Pros:**
- Predictable, sequential execution
- Easy to debug

**Cons:**
- No parallelism

## Performance Benchmarks

Based on empirical testing (see `notebooks/map_benchmark_io_cpu.ipynb`):

| Dataset Size | HyperNodes Sequential | HyperNodes DaskEngine | Speedup |
|--------------|----------------------|----------------------|---------|
| 10 items     | ~20ms               | ~17ms                | 1.2x    |
| 100 items    | ~200ms              | ~27ms                | 7.4x    |
| 500 items    | ~1000ms             | ~130ms               | 7.7x    |

**Key Findings:**
- Small datasets: Minimal benefit due to overhead
- Medium to large datasets: Significant speedup (7-8x)
- Auto-optimization matches manually-tuned configurations

## When to Use DaskEngine

### ✅ Good Use Cases

1. **Map operations over 50+ items**
2. **CPU-intensive computations** (ML inference, complex math)
3. **I/O-bound pipelines** (API calls, file processing)
4. **Embarrassingly parallel workloads**
5. **Want parallelism without configuration complexity**

### ❌ Not Ideal For

1. **Very small datasets** (<10 items) - overhead outweighs benefit
2. **Already-optimized single runs** - use SequentialEngine
3. **Distributed clusters** - consider DaftEngine instead
4. **Heavy inter-task communication** - Dask Bag isn't optimized for this

## Comparison with Other Engines

| Feature | SequentialEngine | DaskEngine | DaftEngine |
|---------|-----------------|------------|------------|
| Parallelism | ❌ None | ✅ Local multi-core | ✅ Distributed cluster |
| Configuration | ✅ None needed | ✅ Auto-optimized | ⚠️ Manual setup |
| Overhead | ✅ Minimal | ⚠️ Small | ⚠️ Higher |
| Best for | Single runs, small data | Map operations, local | Large-scale distributed |
| Learning curve | ✅ None | ✅ Minimal | ⚠️ Requires Dask knowledge |

## Advanced Usage

### Nested Pipelines

The engine automatically handles nested pipelines:

```python
@node(output_name="inner_result")
def inner_node(x: int) -> int:
    return x * 2

@node(output_name="outer_result")
def outer_node(x: int) -> int:
    return x + 1

inner_pipeline = Pipeline(nodes=[inner_node])
pipeline_node = PipelineNode(
    pipeline=inner_pipeline,
    name="inner",
    inputs={"x": "data"}
)

# Engine configuration is inherited by nested pipeline
engine = DaskEngine()
outer_pipeline = Pipeline(
    nodes=[outer_node, pipeline_node],
    engine=engine
)
```

### With Callbacks

DaskEngine fully supports telemetry and callbacks:

```python
from hypernodes.telemetry import ProgressCallback

engine = DaskEngine()
pipeline = Pipeline(
    nodes=[...],
    engine=engine,
    callbacks=[ProgressCallback()]
)

# Progress bar shows parallel execution
results = pipeline.map(inputs={"x": range(100)}, map_over="x")
```

## Troubleshooting

### "Results are slow for small datasets"

**Solution**: This is expected. Dask has overhead. For <50 items, use `SequentialEngine`.

### "Getting serialization errors"

**Solution**: Switch to `scheduler="threads"` (shares memory) or ensure your objects are pickle-able.

### "Want more control over partitioning"

**Solution**: Use manual `npartitions`:

```python
engine = DaskEngine(npartitions=32)
```

### "CPU-bound code is slow"

**Solution**: Try `scheduler="processes"` to bypass GIL:

```python
engine = DaskEngine(scheduler="processes", workload_type="cpu")
```

## API Reference

### DaskEngine

```python
class DaskEngine:
    def __init__(
        self,
        scheduler: str = "threads",
        num_workers: Optional[int] = None,
        workload_type: str = "mixed",
        npartitions: Optional[int] = None,
    ):
        """
        Args:
            scheduler: "threads", "processes", or "synchronous"
            num_workers: Number of workers (None = CPU count)
            workload_type: "io", "cpu", or "mixed"
            npartitions: Manual override (None = auto-calculate)
        """
```

## See Also

- [Dask Documentation](https://docs.dask.org/)
- [Dask Bag API](https://docs.dask.org/en/stable/bag.html)
- [Benchmark Notebook](../notebooks/map_benchmark_io_cpu.ipynb)
- [SequentialEngine](sequential_engine.md)
- [DaftEngine](integrations/daft/README.md)
