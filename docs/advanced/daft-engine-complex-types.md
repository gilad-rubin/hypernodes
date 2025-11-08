# DaftEngine: Handling Complex Types

## Overview

The DaftEngine automatically handles complex Python types (Pydantic models, Lists, Dicts, nested structures) by using Daft's Python object storage feature. This makes it work seamlessly with any HyperNodes pipeline without requiring code changes.

## The Problem

Daft's automatic type inference works well for simple types (`int`, `str`, `float`, `bool`), but fails for complex types like:

- `List[Pydantic]` - Lists of Pydantic models
- `Dict[str, Any]` - Dictionaries with mixed types
- Nested structures like `List[List[int]]`
- Custom classes and Pydantic models

When Daft encounters these types, it raises a `TypeError`:
```
TypeError: DataType._infer expects a DataType, string, or type
```

## The Solution

DaftEngine automatically detects complex types and uses `daft.DataType.python()` to store them as Python objects. This approach:

1. **Works for all types** - No special handling needed
2. **Zero code changes** - Existing HyperNodes code works as-is
3. **Preserves objects** - Pydantic models, custom classes remain intact
4. **Automatic detection** - Inspects type hints to decide storage strategy

## How It Works

### Type Detection Logic

```python
from typing import get_type_hints

# Get return type from function signature
type_hints = get_type_hints(func)
return_type = type_hints.get('return', None)

# Check if it's complex
origin = getattr(return_type, '__origin__', None)

if origin is not None:  # Generic types: List, Dict, etc.
    use_python_storage = True
elif isinstance(return_type, type) and return_type.__module__ != 'builtins':
    use_python_storage = True  # Custom classes, Pydantic models
else:
    use_python_storage = False  # Simple types: int, str, float
```

### UDF Creation

```python
if use_python_storage:
    daft_func = daft.func(func, return_dtype=daft.DataType.python())
else:
    daft_func = daft.func(func)  # Let Daft infer
```

## Examples

### Example 1: List of Pydantic Models

```python
from pydantic import BaseModel
from typing import List
from hypernodes import node, Pipeline
from hypernodes.engines import DaftEngine

class Document(BaseModel):
    id: str
    text: str

@node(output_name="documents")
def create_documents(count: int) -> List[Document]:
    return [Document(id=f"doc_{i}", text=f"Text {i}") for i in range(count)]

pipeline = Pipeline(nodes=[create_documents], engine=DaftEngine())
result = pipeline.run(inputs={"count": 3})
# result["documents"] = [Document(...), Document(...), Document(...)]
```

### Example 2: Pydantic to Pydantic Transformation

```python
class EncodedDocument(BaseModel):
    id: str
    text: str
    embedding: List[float]

@node(output_name="encoded")
def encode_document(document: Document) -> EncodedDocument:
    return EncodedDocument(
        id=document.id,
        text=document.text,
        embedding=[1.0, 2.0, 3.0]
    )

pipeline = Pipeline(
    nodes=[create_documents, encode_document],
    engine=DaftEngine()
)
# Works seamlessly!
```

### Example 3: Dict Return Types

```python
from typing import Dict, Any

@node(output_name="config")
def create_config(name: str) -> Dict[str, Any]:
    return {"name": name, "value": 42, "nested": {"key": "value"}}

pipeline = Pipeline(nodes=[create_config], engine=DaftEngine())
result = pipeline.run(inputs={"name": "test"})
# result["config"] = {"name": "test", "value": 42, ...}
```

### Example 4: Map with Pydantic Models

```python
@node(output_name="document")
def create_single_document(text: str, idx: int) -> Document:
    return Document(id=f"doc_{idx}", text=text)

pipeline = Pipeline(nodes=[create_single_document], engine=DaftEngine())

results = pipeline.map(
    inputs={
        "text": ["Hello", "World", "Test"],
        "idx": [0, 1, 2]
    },
    map_over=["text", "idx"]
)
# results["document"] = [Document(...), Document(...), Document(...)]
```

## Alternative: Native Struct Support

Daft **does** support Pydantic models natively through struct types, but this requires:

1. Converting Pydantic models to PyArrow schemas
2. Specifying `return_dtype` as a struct
3. Calling `.model_dump()` on returned Pydantic objects

### Example with Native Structs

```python
from pydantic_to_pyarrow import get_pyarrow_schema
import pyarrow as pa

def daft_pyarrow_datatype(pydantic_class):
    schema = get_pyarrow_schema(pydantic_class)
    arrow_type = pa.struct([(f, schema.field(f).type) for f in schema.names])
    return daft.DataType.from_arrow_type(arrow_type)

@daft.udf(return_dtype=daft_pyarrow_datatype(Document))
class CreateDocuments:
    def __call__(self, count_col):
        results = []
        for count in count_col:
            docs = [Document(id=f"doc_{i}", text=f"Text {i}") for i in range(count)]
            # MUST call .model_dump() to convert to dict
            results.append([doc.model_dump() for doc in docs])
        return results
```

### Why We Use Python Storage Instead

1. **Simpler** - No schema conversion needed
2. **More flexible** - Works with any type
3. **Less boilerplate** - No `.model_dump()` calls
4. **Automatic** - Zero user intervention
5. **Robust** - Handles edge cases gracefully

The trade-off is that Daft can't optimize these columns as efficiently, but for HyperNodes use cases (where we're orchestrating complex ML pipelines), the flexibility is more valuable than micro-optimizations.

## Performance Considerations

### Python Object Storage

**Pros:**
- Works with any Python type
- No serialization overhead for complex objects
- Preserves object identity and methods

**Cons:**
- Cannot be optimized by Daft's query engine
- Larger memory footprint
- Cannot be pushed down to storage layer

### When to Use Each Approach

**Use Python Storage (DaftEngine default):**
- Complex nested structures
- Pydantic models with many fields
- Custom classes with methods
- When flexibility > performance

**Use Native Structs (manual UDF):**
- Simple Pydantic models
- Need to write to Parquet/Arrow formats
- Performance-critical pipelines
- Large-scale data processing (>1GB)

## Testing

The DaftEngine includes comprehensive tests for complex types:

```bash
# Run complex type tests
uv run pytest tests/test_daft_backend_complex_types.py -v

# Run all DaftEngine tests
uv run pytest tests/test_daft_backend*.py -v
```

Test coverage includes:
- ✅ List of Pydantic models
- ✅ Pydantic to Pydantic transformations
- ✅ Map operations with Pydantic
- ✅ Dict return types
- ✅ Nested lists
- ✅ Any/dynamic types
- ✅ Mixed simple and complex types

## References

- [Daft Documentation](https://www.getdaft.io/)
- [Daft UDF Guide](https://www.getdaft.io/projects/docs/en/stable/user_guide/udf.html)
- [Pydantic to PyArrow](https://github.com/0x26res/pydantic-to-pyarrow)
- [PDF Processing Example](../../guides/daft-pdf.md) - Shows native struct approach
- [Complex Types Example](../../examples/daft_backend_complex_types_example.py)
