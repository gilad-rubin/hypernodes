# HyperNodes to Daft Translation Guide

This guide demonstrates how to translate HyperNodes pipelines into Daft for performance gains.

## Notebook: `hypernodes_to_daft.ipynb`

The notebook contains 6 progressive examples showing the translation patterns:

### Example 1: Simple Text Processing
**What it shows:** Basic data transformations (cleaning, tokenizing, counting)
- HyperNodes: Explicit nodes and `.map()` for collections
- Daft: DataFrame operations with `@daft.func`

**Key learning:** Daft automatically applies operations to all rows, no need for explicit mapping.

### Example 2: Generator Functions - Text Tokenization
**What it shows:** One-to-many transformations where each input produces multiple outputs
- HyperNodes: Manual flattening required
- Daft: Built-in generator support with `Iterator[T]` type hints

**Key learning:** Daft generators automatically expand rows and broadcast other columns.

### Example 3: Stateful Processing with Classes
**What it shows:** Operations requiring expensive initialization (e.g., model loading)
- HyperNodes: Create instance upfront, pass to `.map()`
- Daft: Use `@daft.cls` for lazy initialization per worker

**Key learning:** Daft's lazy initialization is powerful for distributed execution.

### Example 4: Batch Processing with NumPy
**What it shows:** Vectorized operations for performance
- HyperNodes: Row-wise processing by default
- Daft: `@daft.func.batch` for vectorized operations with `Series`

**Key learning:** Batch processing can be significantly faster for vectorizable operations.

### Example 5: Complex Pipeline - Document Encoding
**What it shows:** Multi-stage realistic pipeline (load → clean → encode → index)
- HyperNodes: Explicit pipeline composition
- Daft: Chained DataFrame operations

**Key learning:** Daft's lazy evaluation optimizes the entire pipeline before execution.

### Example 6: Nested Structure Handling
**What it shows:** Working with complex data structures
- HyperNodes: Separate nodes for each field
- Daft: `unnest=True` to automatically expand struct fields into columns

**Key learning:** Daft provides elegant handling of nested data with struct unnesting.

## Translation Patterns

### 1. Simple Transformations
```python
# HyperNodes
@node(output_name="result")
def transform(x: int) -> int:
    return x * 2

# Daft
@daft.func
def transform(x: int) -> int:
    return x * 2
df = df.with_column("result", transform(df["x"]))
```

### 2. Map Operations
```python
# HyperNodes
pipeline.map(inputs={"items": data}, map_over="items")

# Daft
df = daft.from_pydict({"items": data})
df = df.with_column("result", func(df["items"]))
```

### 3. Stateful Processing
```python
# HyperNodes
encoder = Encoder()  # Initialize once
pipeline.map(inputs={"text": texts, "encoder": encoder}, map_over="text")

# Daft
@daft.cls
class Encoder:
    def __init__(self): ...
    @daft.method(...)
    def encode(self, text): ...

encoder = Encoder()  # Lazy init
df = df.with_column("encoded", encoder.encode(df["text"]))
```

### 4. Batch Operations
```python
# HyperNodes: Manual batching or row-wise

# Daft
@daft.func.batch(return_dtype=...)
def process_batch(series: Series) -> Series:
    # Vectorized operations
    return result_series
```

## Key Advantages of Daft

1. **Lazy Evaluation**: Optimizes the entire pipeline before execution
2. **Automatic Parallelization**: No manual configuration needed
3. **Batch Processing**: Easy vectorized operations
4. **Generator Support**: Built-in one-to-many transformations
5. **Struct Unnesting**: Elegant nested data handling
6. **Scalability**: Designed for distributed execution

## When to Use Each

### Use HyperNodes when:
- You need explicit DAG visualization and control
- You want fine-grained caching at the node level
- Your pipeline has complex branching logic
- You need to inspect intermediate results easily
- You're building modular, reusable components

### Use Daft when:
- Performance is critical
- You're processing large datasets (>1GB)
- You want automatic optimization
- You need distributed execution
- Your operations can be vectorized
- You're doing heavy document processing or ML inference

## Performance Considerations

From the notebook examples:

1. **Batch Processing**: Daft's vectorized operations can be faster for large datasets
2. **Lazy Evaluation**: Daft optimizes the entire pipeline, potentially skipping unnecessary work
3. **Parallelization**: Daft automatically parallelizes without configuration
4. **Memory Efficiency**: Daft streams data and doesn't materialize until `.collect()`

## Running the Notebook

```bash
cd /Users/giladrubin/python_workspace/hypernodes/notebooks
uv run jupyter notebook hypernodes_to_daft.ipynb
```

Or in VS Code, simply open the notebook and run the cells sequentially.

## Next Steps

1. Try adapting the `retrieval.ipynb` pipeline to Daft
2. Benchmark performance on your own data
3. Experiment with Daft's distributed execution for large datasets
4. Explore Daft's integration with Ray for even more scalability

## References

- [Daft Documentation](https://www.getdaft.io/)
- [Daft UDF Guide](https://www.getdaft.io/projects/docs/en/stable/user_guide/udf.html)
- [HyperNodes Documentation](../docs/)
