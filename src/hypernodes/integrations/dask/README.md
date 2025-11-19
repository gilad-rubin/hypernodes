# DaskEngine - Parallel Execution for HyperNodes

A Dask-powered execution engine for HyperNodes that provides automatic parallelization with zero configuration required.

## ✨ Features

- **Zero Configuration**: Works out of the box with sensible defaults
- **Automatic Optimization**: Intelligently calculates optimal partitioning
- **No Single-Run Overhead**: Sequential execution for `.run()`, parallel only for `.map()`
- **Smart Heuristics**: Adapts to dataset size and workload type
- **Fully Integrated**: Works seamlessly with HyperNodes callbacks, caching, and nested pipelines

## 🚀 Quick Start

```python
from hypernodes import Pipeline, node
from hypernodes.engines import DaskEngine

@node(output_name="result")
def process(x: int) -> int:
    return x * 2

# Just add DaskEngine - that's it!
pipeline = Pipeline(
    nodes=[process],
    engine=DaskEngine()  # One line for parallelism
)

# Map operations are now parallel
results = pipeline.map(
    inputs={"x": list(range(100))},
    map_over="x"
)
```

## 📊 Performance

Based on benchmarks with mixed I/O/CPU workload:

| Dataset Size | Sequential | DaskEngine | Speedup |
|--------------|-----------|------------|---------|
| 10 items     | 20ms      | 17ms       | 1.2x    |
| 100 items    | 200ms     | 27ms       | **7.4x** |
| 500 items    | 1000ms    | 130ms      | **7.7x** |

**Key Finding**: Significant speedup for medium to large datasets (50+ items).

## 📖 Usage

### Default (Auto-Optimized)

```python
engine = DaskEngine()
# Scheduler: threads
# Workload: mixed (I/O + CPU)
# npartitions: auto-calculated
```

### CPU-Bound Workloads

```python
engine = DaskEngine(
    scheduler="threads",  # or "processes" to bypass GIL
    workload_type="cpu"
)
```

### I/O-Bound Workloads

```python
engine = DaskEngine(
    scheduler="threads",
    workload_type="io"
)
```

### Manual Control

```python
engine = DaskEngine(
    npartitions=16  # Override automatic calculation
)
```

## 🎯 When to Use

### ✅ Good For

- Map operations over 50+ items
- CPU-intensive computations (ML inference, math)
- I/O-bound pipelines (API calls, file processing)
- Embarrassingly parallel workloads
- Want parallelism without Dask expertise

### ❌ Not Ideal For

- Very small datasets (<10 items) - overhead outweighs benefit
- Single pipeline runs - use SeqEngine
- Distributed clusters - consider DaftEngine
- Heavy inter-task communication

## 🔧 How It Works

### Architecture

1. **`.run()` calls**: Sequential execution (no Dask overhead)
2. **`.map()` calls**: Parallel execution via Dask Bag
3. **Nested pipelines**: Inherit parent engine configuration

### Automatic Optimization

The engine calculates optimal `npartitions` based on:

**1. Workload Type:**
- `"io"`: 4x CPU count (more parallelism for I/O wait)
- `"cpu"`: 2x CPU count (match cores for compute)
- `"mixed"`: 3x CPU count (balanced approach)

**2. Dataset Size:**
- Target: 10-1000 items per partition
- Avoids too few partitions (limits parallelism)
- Avoids too many partitions (excessive overhead)

**3. Smart Clamping:**
- Minimum: 2 partitions
- Maximum: 10x CPU count
- Prefers powers of 2 for better distribution

### Example

```python
# 100 items, 8 CPU cores, mixed workload
# Calculation: 8 cores * 3 = 24 partitions
# Clamped to: 8 partitions (better granularity for 100 items)
# Result: ~12 items per partition
```

## 📚 Documentation

- [Full Documentation](../../docs/engines/dask_engine.md)
- [Benchmark Notebook](../../notebooks/map_benchmark_io_cpu.ipynb)
- [Examples](../../examples/dask_engine_example.py)

## 🔍 Comparison with Other Engines

| Feature | Sequential | DaskEngine | DaftEngine |
|---------|-----------|------------|------------|
| Parallelism | ❌ | ✅ Local multi-core | ✅ Distributed |
| Configuration | ✅ None | ✅ Auto | ⚠️ Manual |
| Overhead | ✅ Minimal | ⚠️ Small | ⚠️ Higher |
| Best for | Single runs | Map operations | Large-scale |
| Learning curve | ✅ None | ✅ Minimal | ⚠️ Steep |

## 💡 Examples

### Basic Usage

```python
from hypernodes import Pipeline, node
from hypernodes.engines import DaskEngine

@node(output_name="processed")
def expensive_computation(data: dict) -> float:
    # Some CPU-intensive work
    return sum(data.values())

engine = DaskEngine()
pipeline = Pipeline(nodes=[expensive_computation], engine=engine)

# Process 100 items in parallel
results = pipeline.map(
    inputs={"data": [{"a": i, "b": i*2} for i in range(100)]},
    map_over="data"
)
```

### With Progress Tracking

```python
from hypernodes.telemetry import ProgressCallback

pipeline = Pipeline(
    nodes=[expensive_computation],
    engine=DaskEngine(),
    callbacks=[ProgressCallback()]  # See progress bars
)

results = pipeline.map(inputs={"data": items}, map_over="data")
```

### With Caching

```python
from hypernodes import DiskCache

pipeline = Pipeline(
    nodes=[expensive_computation],
    engine=DaskEngine(),
    cache=DiskCache(path=".cache")  # Cache results
)

# First run: computes all
results1 = pipeline.map(inputs={"data": items}, map_over="data")

# Second run: instant (from cache)
results2 = pipeline.map(inputs={"data": items}, map_over="data")
```

## 🐛 Troubleshooting

### Slow for Small Datasets

**Problem**: DaskEngine is slower than SeqEngine for small datasets.

**Solution**: This is expected. Dask has overhead. For <50 items, use `SeqEngine`.

### Serialization Errors

**Problem**: Getting pickle errors.

**Solution**: Use `scheduler="threads"` (shares memory) or ensure objects are pickle-able.

### Want More Control

**Problem**: Need specific partition count.

**Solution**: Use manual `npartitions`:

```python
engine = DaskEngine(npartitions=32)
```

## 🔬 Benchmarking

See the comprehensive benchmark in [`notebooks/map_benchmark_io_cpu.ipynb`](../../notebooks/map_benchmark_io_cpu.ipynb) which compares:

- HyperNodes SeqEngine (baseline)
- HyperNodes DaskEngine (this engine)
- Daft
- Dask Bag (manual configuration)
- Dask Bag (grid-search optimized)

**Result**: DaskEngine matches or exceeds manually-tuned Dask configurations while requiring zero configuration.

## 📦 Installation

DaskEngine requires the `dask[bag]` extra:

```bash
pip install hypernodes[dask]
# or
uv add "hypernodes[dask]"
```

## 🤝 Contributing

Found a bug or have a suggestion? Please open an issue or submit a PR!

## 📄 License

Same as HyperNodes project.
