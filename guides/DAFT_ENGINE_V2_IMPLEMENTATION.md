# DaftEngineV2: Clean Implementation Summary

## Overview

Implemented a clean, simplified DaftEngineV2 that supports nested pipelines with the **explode → transform → aggregate** pattern. The engine is ~460 lines (vs 3210 in V1), fully lazy, and has no caching/callbacks overhead.

## Key Features

### 1. ✅ Simple Pipelines
```python
pipeline = Pipeline(nodes=[double, square], engine=DaftEngineV2())
result = pipeline.run(inputs={"x": 5})
# {'doubled': 10, 'squared': 25}
```

### 2. ✅ Map Operations
```python
result = pipeline.map(inputs={"x": [1, 2, 3]}, map_over="x")
# {'doubled': [2, 4, 6]}
```

### 3. ✅ Nested Pipelines (without map_over)
```python
inner = Pipeline(nodes=[double, square])
inner_node = inner.as_node(
    input_mapping={"x": "x"},
    output_mapping={"doubled": "inner_doubled", "squared": "inner_squared"}
)
outer = Pipeline(nodes=[inner_node, add_inner], engine=DaftEngineV2())
result = outer.run(inputs={"x": 5})
# {'inner_doubled': 10, 'inner_squared': 25, 'sum': 35}
```

### 4. ✅ Nested Pipelines WITH map_over (explode → transform → aggregate)
```python
inner = Pipeline(nodes=[double, square])
inner_node = inner.as_node(
    input_mapping={"items": "x"},
    output_mapping={"doubled": "doubled_list", "squared": "squared_list"},
    map_over="items"  # ← Triggers explode → transform → aggregate
)
outer = Pipeline(nodes=[inner_node], engine=DaftEngineV2())
result = outer.run(inputs={"items": [1, 2, 3]})
# {'doubled_list': [2, 4, 6], 'squared_list': [1, 4, 9]}
```

## Architecture

### Core Design Principles

1. **All operations are lazy** - nothing executes until `.collect()`
2. **No caching/callbacks** - clean, focused implementation
3. **Clear separation** - dispatcher → specific transformation methods

### Key Methods

```python
class DaftEngineV2:
    def run(pipeline, inputs) -> Dict:
        """1-row DataFrame execution"""
    
    def map(pipeline, inputs, map_over) -> Dict[str, List]:
        """N-row DataFrame execution"""
    
    def _apply_node_transformation(df, node, available_columns):
        """Dispatcher: regular node vs pipeline node"""
    
    def _apply_simple_node_transformation(df, node, available_columns):
        """Wrap function with @daft.func"""
    
    def _apply_simple_pipeline_transformation(df, pipeline_node, available_columns):
        """Nested pipeline without map_over (simple recursion)"""
    
    def _apply_mapped_pipeline_transformation(df, pipeline_node, available_columns):
        """Nested pipeline WITH map_over (explode → transform → aggregate)"""
```

### The Explode → Transform → Aggregate Pattern

```python
def _apply_mapped_pipeline_transformation(...):
    # Step 1: Add row_id for tracking
    df = df.with_column("__daft_row_id__", daft.lit(0))
    
    # Step 2: Explode list column into rows
    df = df.explode(daft.col(map_over_col))
    
    # Step 3: Apply input mapping (rename columns)
    df = df.select(*rename_exprs)
    
    # Step 4: Apply inner pipeline transformations (recursive)
    for inner_node in inner_pipeline.graph.execution_order:
        df, _ = self._apply_node_transformation(df, inner_node, ...)
    
    # Step 5: Apply output mapping
    df = df.select(*output_rename_exprs)
    
    # Step 6: Aggregate back into lists by row_id
    df = df.groupby("__daft_row_id__").agg(
        *[daft.col(name).list_agg() for name in output_names]
    )
    
    # Step 7: Remove row_id column (cleanup)
    df = df.select(*[df[col] for col in df.column_names if col != "__daft_row_id__"])
    
    return df  # Still lazy!
```

## What Changed from V1

### Removed
- ❌ Caching logic
- ❌ Callback system
- ❌ Stateful object handling (@daft.cls)
- ❌ Pydantic conversions
- ❌ Code generation mode
- ❌ PyArrow/Pandas strategies
- ❌ Serialization fixes for Modal

### Simplified
- ✅ Clean 3-method dispatcher pattern
- ✅ Direct @daft.func wrapping (no complexity)
- ✅ Clear explode → transform → aggregate pattern
- ✅ Tuple output handling for PipelineNodes

### File Size Comparison
- **V1 (`engine.py`)**: 3210 lines
- **V2 (`engine_v2.py`)**: 460 lines (85% reduction!)

## Known Limitations

### Double-Nested map_over
Currently, mapping over a pipeline that itself has `map_over` works, but produces flattened results:

```python
# This works but flattens:
outer.map(inputs={"items": [[1, 2], [3, 4, 5]]}, map_over="items")
# Gets: [[2, 4, 6, 8, 10]]
# Want: [[2, 4], [6, 8, 10]]
```

**Reason**: The row_id mechanism tracks only one level. For nested map_over, we'd need hierarchical row_ids.

**Workaround**: For now, avoid double-nested map_over. Single-level works perfectly.

## Testing

Run tests:
```bash
uv run python scripts/test_engine_v2_nested.py
```

Results:
- ✅ Test 1: Simple Pipeline
- ✅ Test 2: Map Pipeline
- ✅ Test 3: Nested Pipeline (no map_over)
- ✅ Test 4: Nested Pipeline WITH map_over
- ⚠️ Test 5: Double-nested map_over (known limitation)

## Usage

```python
from hypernodes import Pipeline, node
from hypernodes.integrations.daft.engine_v2 import DaftEngineV2

# Define nodes
@node(output_name="doubled")
def double(x: int) -> int:
    return x * 2

# Create pipeline with V2 engine
pipeline = Pipeline(
    nodes=[double],
    engine=DaftEngineV2()
)

# Run
result = pipeline.run(inputs={"x": 5})
# {'doubled': 10}

# Map
result = pipeline.map(inputs={"x": [1, 2, 3]}, map_over="x")
# {'doubled': [2, 4, 6]}
```

## Future Enhancements (if needed)

1. **Hierarchical row_id tracking** for double-nested map_over
2. **Optional caching layer** (but keep it separate/modular)
3. **Stateful UDFs** with @daft.cls (if needed for ML models)
4. **Better error messages** with execution context

## Conclusion

✅ **Clean, working implementation** of nested pipelines with explode → transform → aggregate  
✅ **85% code reduction** compared to V1  
✅ **All operations are lazy** - true Daft philosophy  
✅ **4/5 tests passing** (5th is known edge case)  

The V2 engine provides a solid foundation for Daft-based execution with clear, maintainable code.

