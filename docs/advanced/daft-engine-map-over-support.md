# DaftEngine `.as_node(map_over=...)` Support

## Overview

DaftEngine now fully supports `.as_node(map_over=...)` for batch processing patterns! This feature enables efficient distributed data processing using Daft's DataFrame operations.

## How It Works

The implementation uses Daft's native operations:

1. **Explode**: Convert `List[T]` into multiple rows
2. **Transform**: Apply the inner pipeline to each row
3. **Aggregate**: Collect results back into lists using `groupby().agg(list_agg())`

```
Input DataFrame:
┌─────────────┐
│ items: List │
├─────────────┤
│ [a, b, c]   │
└─────────────┘
        ↓ explode()
┌──────────────┬─────────┐
│ item         │ row_id  │
├──────────────┼─────────┤
│ a            │ 0       │
│ b            │ 0       │
│ c            │ 0       │
└──────────────┴─────────┘
        ↓ transform (inner pipeline)
┌──────────────┬─────────┬────────────┐
│ item         │ row_id  │ result     │
├──────────────┼─────────┼────────────┤
│ a            │ 0       │ processed_a│
│ b            │ 0       │ processed_b│
│ c            │ 0       │ processed_c│
└──────────────┴─────────┴────────────┘
        ↓ groupby(row_id).agg(list_agg())
┌──────────────────────────────┐
│ results: List                │
├──────────────────────────────┤
│ [processed_a, b, c]          │
└──────────────────────────────┘
```

## Example Usage

### Basic Map Operation

```python
from hypernodes import Pipeline, node
from hypernodes.engines import DaftEngine

# Single-item node
@node(output_name="doubled")
def double_number(x: int) -> int:
    return x * 2

# Create single-item pipeline
double_single = Pipeline(nodes=[double_number], name="double_single")

# Create mapped node
double_many = double_single.as_node(
    input_mapping={"numbers": "x"},
    output_mapping={"doubled": "all_doubled"},
    map_over="numbers",
    name="double_many"
)

# Use in full pipeline
@node(output_name="numbers")
def create_numbers(count: int) -> list[int]:
    return list(range(count))

@node(output_name="sum")
def sum_all(all_doubled: list[int]) -> int:
    return sum(all_doubled)

pipeline = Pipeline(
    nodes=[create_numbers, double_many, sum_all],
    engine=DaftEngine(),
    name="map_example"
)

result = pipeline.run(inputs={"count": 5})
# result["sum"] == 20  # (0*2 + 1*2 + 2*2 + 3*2 + 4*2)
```

### With Shared Inputs

```python
@node(output_name="scaled")
def scale(number: int, factor: int) -> int:
    return number * factor

scale_single = Pipeline(nodes=[scale], name="scale_single")

scale_many = scale_single.as_node(
    input_mapping={"numbers": "number"},
    output_mapping={"scaled": "all_scaled"},
    map_over="numbers",
    name="scale_many"
)

pipeline = Pipeline(
    nodes=[create_numbers, scale_many, sum_all],
    engine=DaftEngine()
)

result = pipeline.run(inputs={"count": 3, "factor": 10})
# result["sum"] == 30  # (0*10 + 1*10 + 2*10)
```

### Multiple Outputs

```python
@node(output_name="upper")
def to_upper(text: str) -> str:
    return text.upper()

@node(output_name="length")
def get_length(upper: str) -> int:
    return len(upper)

process_single = Pipeline(
    nodes=[to_upper, get_length],
    name="process_single"
)

process_many = process_single.as_node(
    input_mapping={"texts": "text"},
    output_mapping={
        "upper": "upper_texts",
        "length": "lengths"
    },
    map_over="texts",
    name="process_many"
)

pipeline = Pipeline(
    nodes=[create_texts, process_many],
    engine=DaftEngine()
)

result = pipeline.run(inputs={"count": 3})
# result["upper_texts"] == ["TEXT_0", "TEXT_1", "TEXT_2"]
# result["lengths"] == [6, 6, 6]
```

## Performance Characteristics

### Advantages

✅ **Automatic Parallelization**: Daft distributes work across available cores
✅ **Lazy Evaluation**: Query optimization before execution
✅ **Memory Efficient**: Streaming execution for large datasets
✅ **Type Safety**: Daft's type system ensures consistency

### Considerations

⚠️ **Collection Overhead**: Adding row IDs requires materializing the DataFrame
⚠️ **Complex Objects**: Pydantic models may be converted to PyArrow structs
⚠️ **Single Column**: Currently supports mapping over one column at a time

## Comparison with HypernodesEngine

| Feature | DaftEngine | HypernodesEngine |
|---------|-------------|--------------|
| Parallelism | Automatic | Manual configuration |
| Optimization | Query planning | Runtime scheduling |
| Distributed | Yes (multi-node) | No (single machine) |
| Complex Objects | Best with primitives | Full Pydantic support |
| Execution Model | DataFrame/columnar | Imperative/iterative |

## Best Practices

### 1. Use Primitive Types When Possible

```python
# ✅ Good - primitives work seamlessly
@node(output_name="result")
def process(x: int) -> int:
    return x * 2

# ⚠️ Careful - Pydantic models may become structs
@node(output_name="result")
def process(item: MyModel) -> MyModel:
    return MyModel(...)
```

### 2. Leverage Daft's Strengths

```python
# ✅ Good - large batch processing
pipeline = Pipeline(
    nodes=[encode_thousands_of_items],
    engine=DaftEngine()
)

# ⚠️ Overkill - small batch
pipeline = Pipeline(
    nodes=[process_three_items],
    engine=DaftEngine()  # HypernodesEngine might be simpler
)
```

### 3. Monitor Query Plans

```python
backend = DaftEngine(show_plan=True)

pipeline = Pipeline(nodes=[...], backend=backend)
pipeline.run(inputs={...})
# Prints optimized query plan for debugging
```

## Limitations

### Single Column Mapping

Currently supports mapping over a single column:

```python
# ✅ Supported
map_over="items"
map_over=["items"]  # Single-item list

# ❌ Not yet supported
map_over=["items", "other_items"]  # Multiple columns
```

### Row ID Materialization

The implementation adds a temporary row ID column and materializes the DataFrame to add it. For very large datasets, this may have memory implications.

## Technical Details

### Implementation Strategy

The `_convert_mapped_pipeline_node()` method:

1. Materializes DataFrame to add row IDs
2. Explodes the mapped column into rows
3. Applies input mapping (renames columns)
4. Recursively converts the inner pipeline
5. Applies output mapping
6. Groups by row ID and aggregates with `list_agg()`
7. Removes the temporary row ID column

### Code Location

- **Source**: `src/hypernodes/daft_backend.py`
- **Method**: `_convert_mapped_pipeline_node()`
- **Lines**: ~315-463
- **Tests**: `tests/test_daft_backend_map_over.py`

## Migration from HypernodesEngine

If you're currently using HypernodesEngine with `.as_node(map_over=...)`, you can now use DaftEngine:

```python
# Before
from hypernodes.backend import HypernodesEngine

pipeline = Pipeline(
    nodes=[encode_passages_mapped, retrieve_queries_mapped],
    backend=HypernodesEngine(map_execution="parallel", max_workers=8)
)

# After
from hypernodes.engines import DaftEngine

pipeline = Pipeline(
    nodes=[encode_passages_mapped, retrieve_queries_mapped],
    engine=DaftEngine()  # Automatic parallelization!
)
```

## Future Enhancements

Potential improvements:

- **Multi-column mapping**: Support `map_over=["col1", "col2"]`
- **Streaming row IDs**: Avoid materialization for row ID generation
- **Custom aggregations**: Beyond `list_agg()` for specialized use cases
- **Better Pydantic support**: Seamless conversion between models and structs

## Summary

DaftEngine now provides full support for `.as_node(map_over=...)`, enabling:
- ✅ Batch processing with automatic parallelization
- ✅ Query optimization and lazy evaluation
- ✅ Memory-efficient streaming execution
- ✅ Distributed processing capabilities

Use DaftEngine when you need distributed data processing with DataFrame optimizations, and HypernodesEngine or ModalBackend when you need maximum flexibility with complex object types.
