# DaftBackend - Automatic Conversion to Daft

The `DaftBackend` automatically converts HyperNodes pipelines into [Daft](https://www.getdaft.io/) DataFrames, providing lazy evaluation, automatic optimization, and high-performance execution.

## Overview

DaftBackend translates HyperNodes pipelines into Daft operations:
- **Nodes** → Daft UDFs (`@daft.func`)
- **Map operations** → DataFrame operations
- **Pipelines** → Lazy DataFrame transformations

### Key Benefits

1. **Lazy Evaluation**: Operations are optimized before execution
2. **Automatic Parallelization**: No manual configuration needed
3. **Performance**: Optimized execution with vectorization
4. **Scalability**: Designed for distributed execution
5. **Zero Code Changes**: Drop-in replacement for LocalBackend

## Installation

```bash
pip install daft
# or
uv add daft
```

## Basic Usage

### Simple Pipeline

```python
from hypernodes import node, Pipeline
from hypernodes.daft_backend import DaftBackend

@node(output_name="result")
def add_one(x: int) -> int:
    return x + 1

# Use DaftBackend
pipeline = Pipeline(nodes=[add_one], backend=DaftBackend())
result = pipeline.run(inputs={"x": 5})
# result == {"result": 6}
```

### Map Operations

```python
@node(output_name="doubled")
def double(x: int) -> int:
    return x * 2

pipeline = Pipeline(nodes=[double], backend=DaftBackend())

# Process multiple items
results = pipeline.map(inputs={"x": [1, 2, 3, 4, 5]}, map_over="x")
# results == {"doubled": [2, 4, 6, 8, 10]}
```

## Advanced Features

### Nested Pipelines

DaftBackend automatically handles nested pipelines:

```python
@node(output_name="cleaned")
def clean_text(text: str) -> str:
    return text.strip().lower()

@node(output_name="word_count")
def count_words(cleaned: str) -> int:
    return len(cleaned.split())

# Inner pipeline
preprocess = Pipeline(nodes=[clean_text, count_words])

@node(output_name="is_long")
def classify(word_count: int) -> bool:
    return word_count > 3

# Outer pipeline with DaftBackend
full_pipeline = Pipeline(
    nodes=[preprocess, classify],
    backend=DaftBackend()
)

result = full_pipeline.run(inputs={"text": "  Hello World  "})
# result == {"cleaned": "hello world", "word_count": 2, "is_long": False}
```

### Execution Plan Visualization

View Daft's execution plan before running:

```python
backend = DaftBackend(show_plan=True)
pipeline = Pipeline(nodes=[...], backend=backend)
result = pipeline.run(inputs={...})
# Prints the optimized execution plan
```

### Selective Output

Request only specific outputs:

```python
@node(output_name="step1")
def step_one(x: int) -> int:
    return x * 2

@node(output_name="step2")
def step_two(step1: int) -> int:
    return step1 + 10

@node(output_name="final")
def step_three(step2: int) -> int:
    return step2 ** 2

pipeline = Pipeline(nodes=[step_one, step_two, step_three], backend=DaftBackend())

# Get only final result
result = pipeline.run(inputs={"x": 5}, output_name="final")
# result == {"final": 400}

# Get multiple specific outputs
result = pipeline.run(inputs={"x": 5}, output_name=["step1", "final"])
# result == {"step1": 10, "final": 400}
```

## Configuration Options

### DaftBackend Parameters

```python
DaftBackend(
    collect=True,      # Auto-collect results (default: True)
    show_plan=False    # Print execution plan (default: False)
)
```

- **collect**: If `True`, automatically materializes the DataFrame. Set to `False` to get a lazy DataFrame for further processing.
- **show_plan**: If `True`, prints Daft's optimized execution plan before running.

## Performance Considerations

### When to Use DaftBackend

✅ **Use DaftBackend when:**
- Processing large datasets (>1GB)
- Performance is critical
- You want automatic optimization
- You need distributed execution
- Operations can be vectorized

❌ **Use LocalBackend when:**
- You need explicit DAG visualization
- Fine-grained caching at node level is important
- Complex branching logic with inspection
- Small datasets (<1MB)
- Debugging intermediate results

### Optimization Tips

1. **Batch Operations**: Daft automatically optimizes operations across rows
2. **Lazy Evaluation**: Build complex pipelines without immediate execution
3. **Selective Outputs**: Request only needed outputs to skip unnecessary computation
4. **Type Hints**: Provide type hints for better optimization

## Comparison with LocalBackend

| Feature | LocalBackend | DaftBackend |
|---------|--------------|-------------|
| Execution | Sequential/Async/Threaded/Parallel | Lazy + Optimized |
| Parallelization | Manual configuration | Automatic |
| Optimization | None | Query optimization |
| Caching | Node-level | DataFrame-level |
| Visualization | DAG graphs | Execution plans |
| Distributed | Via ModalBackend | Native support |
| Best for | Development, debugging | Production, performance |

## Translation Patterns

### Simple Transformations

**HyperNodes:**
```python
@node(output_name="result")
def transform(x: int) -> int:
    return x * 2

pipeline = Pipeline(nodes=[transform])
result = pipeline.run(inputs={"x": 5})
```

**Daft (internal):**
```python
@daft.func
def transform(x: int) -> int:
    return x * 2

df = daft.from_pydict({"x": [5]})
df = df.with_column("result", transform(df["x"]))
```

### Map Operations

**HyperNodes:**
```python
pipeline.map(inputs={"x": [1, 2, 3]}, map_over="x")
```

**Daft (internal):**
```python
df = daft.from_pydict({"x": [1, 2, 3]})
df = df.with_column("result", transform(df["x"]))
```

## Limitations

### Current Limitations

1. **Callbacks**: Callback context is not used (progress bars, tracing)
2. **Caching**: Uses Daft's caching, not HyperNodes' node-level cache
3. **Stateful Classes**: Not yet supported (planned for future)
4. **Custom Executors**: Uses Daft's execution engine

### Planned Features

- [ ] Support for `@daft.cls` for stateful processing
- [ ] Support for `@daft.func.batch` for vectorized operations
- [ ] Generator functions with `Iterator[T]` type hints
- [ ] Async functions for I/O-bound operations
- [ ] Integration with HyperNodes callbacks
- [ ] Custom resource requirements (GPUs, memory)

## Examples

See the following examples:
- `examples/daft_backend_example.py` - Comprehensive examples
- `notebooks/hypernodes_to_daft.ipynb` - Interactive tutorial
- `notebooks/DAFT_TRANSLATION_GUIDE.md` - Translation patterns

## Troubleshooting

### Import Error

```
ImportError: Daft is not installed
```

**Solution**: Install Daft with `pip install daft` or `uv add daft`

### Type Inference Issues

If Daft cannot infer types, provide explicit type hints:

```python
@node(output_name="result")
def process(x: int) -> list[str]:  # Explicit return type
    return [str(x)]
```

### Performance Not Improved

- Check dataset size (Daft shines with larger datasets)
- Use `show_plan=True` to inspect optimization
- Consider using `@daft.func.batch` for vectorized operations (future)

## References

- [Daft Documentation](https://www.getdaft.io/)
- [Daft UDF Guide](https://www.getdaft.io/projects/docs/en/stable/user_guide/udf.html)
- [HyperNodes Documentation](../../README.md)
- [Translation Guide](../../notebooks/DAFT_TRANSLATION_GUIDE.md)
