# HyperNodes vs Daft Quick Reference

## At a Glance

| Feature | HyperNodes | Daft |
|---------|-----------|------|
| **Philosophy** | Explicit DAG control | Automatic optimization |
| **Execution** | Configurable (seq/thread/async) | Auto-parallel |
| **Caching** | Node-level, explicit | Lazy evaluation |
| **Best For** | ML pipelines, complex logic | Large-scale data processing |
| **Learning Curve** | Moderate | Easy for DataFrame users |

## Performance Winners

| Scenario | Winner | Speedup |
|----------|--------|---------|
| Text Processing (built-ins) | **Daft** | 13x |
| Stateful Processing | **HyperNodes** | 3.7x |
| Batch Operations | **Daft** | 16x |
| Nested Pipelines | **Daft** | 1.35x |
| Generators | **Daft** | 2x |

## Code Comparison

### Simple Transformation

**HyperNodes:**
```python
@node(output_name="result")
def transform(x: int) -> int:
    return x * 2

pipeline = Pipeline(nodes=[transform])
results = pipeline.map(inputs={"x": [1, 2, 3]}, map_over="x")
```

**Daft:**
```python
@daft.func
def transform(x: int) -> int:
    return x * 2

df = daft.from_pydict({"x": [1, 2, 3]})
df = df.with_column("result", transform(df["x"]))
results = df.collect()
```

### Stateful Processing

**HyperNodes:**
```python
encoder = Encoder()  # Initialize once

@node(output_name="encoded")
def encode(text: str, encoder: Encoder) -> np.ndarray:
    return encoder.encode(text)

pipeline = Pipeline(nodes=[encode])
results = pipeline.map(
    inputs={"text": texts, "encoder": encoder},
    map_over="text"
)
```

**Daft:**
```python
@daft.cls
class Encoder:
    def __init__(self):
        # Lazy init per worker
        self.model = load_model()
    
    @daft.method(return_dtype=DataType.python())
    def encode(self, text: str) -> np.ndarray:
        return self.model.encode(text)

encoder = Encoder()
df = daft.from_pydict({"text": texts})
df = df.with_column("encoded", encoder.encode(df["text"]))
results = df.collect()
```

### Batch Operations

**HyperNodes:**
```python
# Row-wise by default
@node(output_name="normalized")
def normalize(value: float, mean: float, std: float) -> float:
    return (value - mean) / std

# Manual batching required for vectorization
```

**Daft:**
```python
# Built-in vectorization
df = df.with_column("normalized", (df["value"] - mean) / std)

# Or batch UDF
@daft.func.batch(return_dtype=DataType.float64())
def normalize(values: Series, mean: float, std: float) -> Series:
    arr = values.to_arrow().to_numpy()
    return Series.from_numpy((arr - mean) / std)
```

## Decision Tree

```
Start
  |
  ├─ Need distributed execution? ──> YES ──> Use Daft
  |
  ├─ Processing >1GB data? ──> YES ──> Use Daft
  |
  ├─ Need explicit DAG control? ──> YES ──> Use HyperNodes
  |
  ├─ Have pre-initialized objects? ──> YES ──> Use HyperNodes
  |
  ├─ Mostly vectorized ops? ──> YES ──> Use Daft
  |
  ├─ Complex branching logic? ──> YES ──> Use HyperNodes
  |
  └─ Default ──> Use Daft (better performance)
```

## Execution Modes

### HyperNodes

```python
# Sequential (default)
backend = LocalBackend(node_execution="sequential")

# Threaded (I/O + CPU)
backend = LocalBackend(node_execution="threaded", max_workers=8)

# Async (I/O-bound)
backend = LocalBackend(node_execution="async")

# Parallel (CPU-bound)
backend = LocalBackend(node_execution="parallel", max_workers=4)

pipeline = pipeline.with_backend(backend)
```

### Daft

```python
# Automatic - no configuration needed
df = df.collect()  # Triggers execution with auto-parallelization
```

## Common Patterns

### Map Over Collection

**HyperNodes:**
```python
results = pipeline.map(
    inputs={"item": items, "constant": value},
    map_over="item"
)
```

**Daft:**
```python
df = daft.from_pydict({"item": items})
df = df.with_column("result", func(df["item"], value))
results = df.collect()
```

### Generator (One-to-Many)

**HyperNodes:**
```python
# Manual flattening required
results = pipeline.map(inputs={"text": texts}, map_over="text")
all_tokens = [t for tokens in results["tokens"] for t in tokens]
```

**Daft:**
```python
@daft.func
def tokenize(text: str) -> Iterator[str]:
    for token in text.split():
        yield token

df = df.select(tokenize(df["text"]))
# Or use built-in
df = df.with_column("tokens", df["text"].str.split(" ")).explode("tokens")
```

### Nested Pipelines

**HyperNodes:**
```python
# Explicit nesting
inner_pipeline = Pipeline(nodes=[...], name="inner")
outer_pipeline = Pipeline(nodes=[inner_pipeline.as_node(), ...])
```

**Daft:**
```python
# Chained operations
df = df.with_column("step1", func1(df["input"]))
df = df.with_column("step2", func2(df["step1"]))
df = df.with_column("step3", func3(df["step2"]))
```

## Tips & Tricks

### HyperNodes

- ✅ Use caching for expensive computations
- ✅ Visualize pipelines with `.visualize()`
- ✅ Pre-initialize expensive objects
- ✅ Use `.as_node()` for nested pipelines
- ✅ Choose execution mode based on workload

### Daft

- ✅ Use built-in operations when available
- ✅ Leverage `@daft.func.batch` for vectorization
- ✅ Use `@daft.cls` for stateful processing
- ✅ Chain operations for lazy evaluation
- ✅ Call `.collect()` only when needed

## Benchmarking

Run the comprehensive benchmark:
```bash
cd /Users/giladrubin/python_workspace/hypernodes
uv run python scripts/benchmark_hypernodes_vs_daft.py
```

Test different scales by editing `CURRENT_SCALE` in the script.

## Resources

- **Benchmark Script**: `scripts/benchmark_hypernodes_vs_daft.py`
- **Detailed Results**: `scripts/BENCHMARK_RESULTS.md`
- **Full Guide**: `scripts/README_BENCHMARK.md`
- **Summary**: `scripts/BENCHMARK_SUMMARY.txt`
- **Translation Guide**: `notebooks/DAFT_TRANSLATION_GUIDE.md`
