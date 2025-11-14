# Automatic Type Inference for Daft Engine

## Overview

The Daft engine now **automatically converts any working HyperNodes pipeline to Daft** without requiring any code changes. This is achieved through three key mechanisms:

1. **Smart Type Inference** - Automatically detects complex types and maps them to appropriate Daft DataTypes
2. **Serialization Support** - Handles functions defined in scripts/closures for distributed execution
3. **PEP 563 Support** - Correctly handles stringified annotations from `from __future__ import annotations`

## The Problem

When trying to run `retrieval.py` with the Daft engine, we encountered two main errors:

### Error 1: Type Inference Failure
```python
TypeError: DataType._infer expects a DataType, string, or type
```

**Cause**: Daft couldn't automatically infer return types for:
- `List[Pydantic]` - Lists of Pydantic models
- `dict[str, float]` - Generic dict types
- `Protocol` types - Python Protocol classes
- Custom classes - User-defined classes

### Error 2: Explode Failure
```python
DaftCoreException: Input must be a list
```

**Cause**: When lists were stored as `DataType.python()`, Daft treated them as opaque Python objects and couldn't perform list operations like `explode()`.

## The Solution

### 1. Smart Type Inference (`_infer_daft_return_type`)

The engine now inspects function return type annotations and maps them to appropriate Daft DataTypes:

| Python Type Hint | Daft DataType | Reason |
|-----------------|---------------|---------|
| `List[T]` | `DataType.list(DataType.python())` | Enables explode/list_agg while supporting arbitrary element types |
| `dict` or `dict[K, V]` | `DataType.python()` | Dicts are opaque Python objects |
| `Protocol` subclasses | `DataType.python()` | Protocols are runtime interfaces |
| Custom classes (Pydantic, etc.) | `DataType.python()` | Preserve object identity and methods |
| `str`, `int`, `float`, `bool` | `None` (let Daft infer) | Daft handles these natively |
| No annotation | `None` (let Daft infer) | Default behavior |

### 2. PEP 563 Support (Stringified Annotations)

When using `from __future__ import annotations` (PEP 563), all type annotations become strings. The engine now:

1. **Detects** stringified annotations: `isinstance(return_annotation, str)`
2. **Evaluates** them to get actual types: `eval(annotation, {**typing.__dict__, **node.func.__globals__})`
3. **Falls back** to `DataType.python()` if evaluation fails

**Example**:
```python
from __future__ import annotations
from typing import List
from pydantic import BaseModel

class Document(BaseModel):
    text: str

@node(output_name="docs")
def create_docs(count: int) -> List[Document]:  # Annotation is the string "List[Document]"
    return [Document(text=f"doc {i}") for i in range(count)]
```

Without PEP 563 support, `get_origin("List[Document]")` returns `None`.
With PEP 563 support, we evaluate it to `typing.List[Document]` and `get_origin()` correctly returns `list`.

### 3. Distributed Execution Support (`_make_serializable_by_value`)

For Modal/distributed execution, Daft uses cloudpickle to serialize UDFs. By default, cloudpickle serializes functions **by reference** (storing module path), which fails for:
- Functions defined inside other functions
- Script files that aren't proper Python packages
- Functions with closure captures

The solution forces cloudpickle to serialize **by value** (include entire bytecode):

```python
@staticmethod
def _make_serializable_by_value(func):
    """Force cloudpickle to serialize function by value instead of by reference.
    
    By setting __module__ = "__main__", cloudpickle will serialize the entire
    function bytecode instead of just storing an import path.
    """
    try:
        func.__module__ = "__main__"
        func.__qualname__ = func.__name__
    except (AttributeError, TypeError):
        pass
    return func
```

This is applied to all node functions before passing them to `@daft.func`.

## Implementation Details

### Type Inference Flow

```python
@node(output_name="passages")
def load_passages(corpus_path: str) -> List[Passage]:
    ...

# When converting to Daft UDF:
# 1. Get annotation: "List[Passage]" (string due to PEP 563)
# 2. Evaluate: typing.List[__main__.Passage]
# 3. Get origin: <class 'list'>
# 4. Infer type: DataType.list(DataType.python())
# 5. Create UDF: daft.func(load_passages, return_dtype=DataType.list(DataType.python()))
```

### Fallback Strategy

The engine uses a layered fallback approach:

```python
# 1. Try smart type inference
inferred_type = self._infer_daft_return_type(node)

# 2. Make function serializable
serializable_func = self._make_serializable_by_value(node.func)

if inferred_type is not None:
    # Use our inferred type
    udf = daft.func(serializable_func, return_dtype=inferred_type)
else:
    # Let Daft infer automatically
    try:
        udf = daft.func(serializable_func)
    except TypeError:
        # Final fallback: Python type
        udf = daft.func(serializable_func, return_dtype=DataType.python())
```

## Why This Matters

### Before (Manual Type Specification)

Users would need to manually specify `return_dtype` for complex types:

```python
@node(output_name="passages")
def load_passages(corpus_path: str) -> List[Passage]:
    ...

# Would need to manually configure for Daft:
@daft.func(return_dtype=daft.DataType.list(daft.DataType.python()))
def load_passages_daft(corpus_path: str):
    ...
```

### After (Automatic)

The engine handles everything automatically:

```python
@node(output_name="passages")
def load_passages(corpus_path: str) -> List[Passage]:
    ...

# Just works with DaftEngine:
pipeline = Pipeline(nodes=[load_passages], engine=DaftEngine())
pipeline.run(inputs={"corpus_path": "data.parquet"})
```

## Key Benefits

1. **Zero Code Changes** - Existing pipelines work as-is
2. **PEP 563 Compatible** - Works with modern Python annotation practices
3. **Modal/Distributed Ready** - Functions serialize correctly for remote execution
4. **Explode/Aggregate Support** - Lists are recognized for Daft operations
5. **Fallback Safety** - Multiple fallback layers ensure it always works

## Testing

Run the retrieval pipeline with Daft engine:

```bash
uv run scripts/retrieval.py --daft
```

Expected output:
```
Running retrieval pipeline with 5 examples...

============================================================
EVALUATION RESULTS
============================================================
NDCG@20: 0.0134

Recall Metrics:
  recall@20: 0.2000
  recall@50: 0.4000
  recall@100: 0.6000
  recall@200: 1.0000
  recall@300: 1.0000
============================================================

Pipeline execution time: ~24 seconds
============================================================
```

## Future Considerations

### When to Use Different Approaches

**Use automatic inference (default):**
- Complex nested structures
- Pydantic models with many fields
- Protocols and abstract base classes
- When flexibility > micro-optimization

**Use manual Daft structs:**
- Need to write to Parquet/Arrow formats
- Performance-critical pipelines (>1GB data)
- Large-scale distributed processing
- When Daft-native optimizations are needed

### Limitations

1. **`eval()` Security** - Uses `eval()` to resolve string annotations. Safe for trusted code, but be aware.
2. **Complex Generics** - Very complex generic types (e.g., `Union[List[A], List[B]]`) may fall back to `DataType.python()`
3. **Performance** - Python object storage is less optimized than native Daft types

## References

- [Daft UDF Documentation](https://www.getdaft.io/projects/docs/en/stable/user_guide/udf.html)
- [PEP 563 - Postponed Evaluation of Annotations](https://peps.python.org/pep-0563/)
- [Cloudpickle Serialization](https://github.com/cloudpipe/cloudpickle)
- [Complex Types Documentation](./daft-engine-complex-types.md)
- [Complete Daft Fixes Summary](../../guides/COMPLETE_DAFT_FIXES_SUMMARY.md)





