# Hypernodes Engine Benchmarks

This document describes the benchmark suite comparing the **Sequential** and **DAFT** execution engines for Hypernodes pipelines.

## Running Benchmarks

### As a Script (Quick Results)
```bash
uv run scripts/benchmark_engines.py
```

### As Pytest Tests (Full Output)
```bash
pytest tests/test_benchmarks_engines.py -v -s
```

## Benchmark Scenarios

### 1. **Simple Execution** (Baseline)
- **What**: Sequential execution of a 2-node pipeline
- **Expected**: SeqEngine faster (no overhead)
- **DAFT Cost**: Graph construction, DataFrame creation

```python
@node(output_name="add_result")
def add_one(x: int) -> int:
    return x + 1

@node(output_name="result")
def multiply_by_two(add_result: int) -> int:
    return add_result * 2
```

### 2. **Map - Basic** (10 items)
- **What**: Map operation over 10 simple items
- **Expected**: SeqEngine faster (less overhead)
- **Use Case**: Shows when DAFT overhead outweighs benefits

```python
inputs = {"x": list(range(10))}
pipeline.map(inputs=inputs, map_over="x")
```

### 3. **Map - I/O Heavy** (0.05s per item)
- **What**: Map operation with sleep (I/O simulation)
- **Expected**: DAFT potential advantage (parallel I/O)
- **Reality**: Limited by GIL/Python threading
- **Use Case**: Real I/O operations (API calls, file reads)

```python
@node(output_name="result")
def io_operation(x: int) -> int:
    time.sleep(0.05)  # Simulate I/O
    return x * 2
```

### 4. **Map - CPU Heavy** (500k iterations)
- **What**: Map operation with CPU computation
- **Expected**: DAFT faster (multiprocessing/parallelization)
- **Use Case**: CPU-bound workloads

```python
@node(output_name="result")
def cpu_operation(x: int) -> int:
    result = x
    for _ in range(500000):
        result = (result * 7 + 11) % 1000000
    return result
```

### 5. **Nested Pipeline**
- **What**: Pipeline containing another pipeline as a node
- **Expected**: Similar performance
- **Use Case**: Modular pipeline composition

```python
inner_pipeline = Pipeline(nodes=[inner_add, inner_multiply])
pipeline_node = PipelineNode(pipeline=inner_pipeline)
outer_pipeline = Pipeline(nodes=[pipeline_node, outer_transform])
```

### 6. **Nested Pipeline + Map**
- **What**: Map over a nested pipeline
- **Expected**: DAFT may benefit (DAG complexity)
- **Use Case**: Batch processing with complex operations

```python
outer_pipeline.map(inputs=inputs, map_over="x")
```

### 7. **Complex DAG**
- **What**: Pipeline with multiple branches and joins
- **Expected**: DAFT potential advantage (lazy evaluation)
- **Use Case**: Complex data processing graphs

```python
compute_a(x) -> a
compute_b(x) -> b
compute_c(a, b) -> c
compute_result(a, b, c) -> result
```

### 8. **Map - Multiple Parameters**
- **What**: Map over multiple parameters in zip mode
- **Expected**: Similar performance
- **Use Case**: Element-wise operations

```python
inputs = {"x": list(range(10)), "y": list(range(10, 20))}
pipeline.map(inputs=inputs, map_over=["x", "y"], map_mode="zip")
```

## Results Interpretation

### Speedup Values
- **1.00x** or less: SeqEngine is faster (or comparable)
- **>1.00x**: DaftEngine is faster (speedup factor)

Example: **1.10x** means DAFT is 10% faster

### When Sequential Wins
- ✓ Simple pipelines with low overhead
- ✓ Small datasets
- ✓ Few map items
- ✗ I/O-bound operations (limited parallelization benefit)

### When DAFT Wins
- ✓ CPU-heavy operations
- ✓ Large datasets
- ✓ Many map items
- ✓ Complex DAGs
- ✓ Lazy evaluation benefits

## Key Metrics

| Metric | Sequential | DAFT |
|--------|-----------|------|
| Startup Cost | ~0ms | ~150-250ms |
| Overhead per Node | Minimal | Graph construction |
| Map Parallelization | No | Yes (but GIL-limited) |
| Lazy Evaluation | No | Yes |
| Memory Efficiency | Good | Better (columnar) |

## Performance Notes

### DAFT Graph Construction Overhead
- ~150-250ms per pipeline creation
- Fixed cost, not dependent on pipeline size
- Amortized over larger workloads

### GIL Limitations
- Python's Global Interpreter Lock limits thread parallelization
- CPU-bound operations: DAFT shows more benefit
- I/O-bound operations: Limited parallelization advantage

### Lazy Evaluation
- DAFT builds computation graph first (no immediate execution)
- Benefits from query optimization
- Reduced memory footprint for large datasets

## Running Custom Benchmarks

Add your own test to `tests/test_benchmarks_engines.py`:

```python
def test_benchmarks_custom(benchmark_suite):
    """Run custom benchmark."""
    test_name = "My Custom Test"
    
    # Define your nodes
    @node(output_name="result")
    def my_operation(x: int) -> int:
        # Your operation
        return x * 2
    
    # Create pipelines
    seq_engine = SeqEngine()
    seq_pipeline = Pipeline(nodes=[my_operation], engine=seq_engine)
    seq_time = benchmark_suite.run_benchmark(
        test_name, seq_engine, seq_pipeline, {"x": 5}
    )
    
    # Run with DAFT
    if DAFT_AVAILABLE:
        daft_engine = DaftEngine()
        daft_pipeline = Pipeline(nodes=[my_operation], engine=daft_engine)
        daft_time = benchmark_suite.run_benchmark(
            test_name, daft_engine, daft_pipeline, {"x": 5}
        )
```

## Machine Specs for Reference

Results can vary based on:
- CPU cores/threads
- Available memory
- System load
- Python version
- Daft version

Always run on representative hardware for your use case.

## Troubleshooting

### DAFT not available
```bash
pip install getdaft
```

### Slow sequential times
- Check system load
- Run multiple times (warm up)
- Increase workload size

### DAFT slower than expected
- Check for GIL contention
- Verify CPU/I/O bound nature of operations
- Consider dataset size

## References

- [Hypernodes Documentation](./docs/README.md)
- [Daft Documentation](https://www.getdaft.io/)
- [Sequential Engine](./src/hypernodes/sequential_engine.py)
- [DAFT Engine](./src/hypernodes/integrations/daft/engine.py)

