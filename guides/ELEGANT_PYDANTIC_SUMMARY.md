# Summary: Elegant Pydantic Support for DaftBackend

## Problem Solved

Your retrieval pipeline was crashing in Jupyter when using Pydantic models with numpy arrays and `.as_node(map_over=...)`.

## Two Solutions

### Solution 1: Manual Dict Conversion (Works, but Not Elegant)

Return dicts instead of Pydantic models:

```python
@node(output_name="encoded_passage")
def encode_passage(passage: Any, encoder: Encoder) -> dict:
    embedding = encoder.encode(passage.text, is_query=False)
    return {"uuid": passage.uuid, "text": passage.text, "embedding": embedding}
```

**Pros:** ✅ Works reliably  
**Cons:** ❌ Loses type safety, ❌ Manual conversion needed

### Solution 2: Native Pydantic Support (Elegant! ✨)

Enhanced DaftBackend to natively support Pydantic models:

```python
@node(output_name="encoded_passage")
def encode_passage(passage: Passage, encoder: Encoder) -> EncodedPassage:
    """Return Pydantic model directly - DaftBackend handles it!"""
    embedding = encoder.encode(passage.text, is_query=False)
    return EncodedPassage(uuid=passage.uuid, text=passage.text, embedding=embedding)
```

**Pros:** ✅ Full type safety, ✅ Natural code, ✅ Auto .model_dump(), ✅ Follows Daft patterns  
**Cons:** None!

## What Changed in DaftBackend

### 1. Added PyArrow/Pydantic Support

```python
try:
    import pyarrow
    from pydantic import BaseModel
    from pydantic_to_pyarrow import get_pyarrow_schema
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
```

### 2. Enhanced `_infer_return_dtype()` to Detect Pydantic Models

```python
def _infer_return_dtype(self, return_type: Any) -> Optional["daft.DataType"]:
    # ... existing code ...
    
    # NEW: Handle Pydantic models by converting to PyArrow structs
    if PYDANTIC_AVAILABLE and BaseModel is not None and isinstance(return_type, type):
        try:
            if issubclass(return_type, BaseModel):
                schema = get_pyarrow_schema(return_type)
                arrow_type = pyarrow.struct([(f, schema.field(f).type) for f in schema.names])
                return daft.DataType.from_arrow_type(arrow_type)
        except (TypeError, Exception):
            pass
    
    # ... rest of code ...
```

### 3. Auto-Wrap Functions to Call `.model_dump()`

```python
def _convert_node_to_daft(self, node, df, available):
    # Check if return type is Pydantic
    is_pydantic = False
    if PYDANTIC_AVAILABLE and return_type is not None:
        if isinstance(return_type, type) and issubclass(return_type, BaseModel):
            is_pydantic = True
    
    # Wrap function to auto-call .model_dump()
    if is_pydantic:
        original_func = func
        def wrapped_func(*args, **kwargs):
            result = original_func(*args, **kwargs)
            return result.model_dump() if hasattr(result, 'model_dump') else result
        func = wrapped_func
    
    # ... rest of UDF creation ...
```

## How to Use It

### Before (Manual Dict Conversion)

```python
@node(output_name="encoded_passage")
def encode_passage(passage: Any, encoder: Encoder) -> dict:
    # Normalize input
    if isinstance(passage, Passage):
        passage_obj = passage
    elif isinstance(passage, dict):
        passage_obj = Passage(**passage)
    else:
        passage_obj = Passage(uuid=getattr(passage, "uuid"), text=getattr(passage, "text"))
    
    embedding = encoder.encode(passage_obj.text, is_query=False)
    
    # Manual dict conversion
    return {
        "uuid": passage_obj.uuid,
        "text": passage_obj.text,
        "embedding": embedding
    }
```

### After (Elegant Pydantic)

```python
@node(output_name="encoded_passage")
def encode_passage(passage: Passage, encoder: Encoder) -> EncodedPassage:
    """Return Pydantic model - DaftBackend handles serialization!"""
    embedding = encoder.encode(passage.text, is_query=False)
    
    # Just return the Pydantic model!
    return EncodedPassage(
        uuid=passage.uuid,
        text=passage.text,
        embedding=embedding
    )
```

### Consuming Nodes (Flexible Input Handling)

After `list_agg()`, Daft may return dicts, so consuming nodes should still be flexible:

```python
@node(output_name="colbert_hits")
def retrieve_colbert(encoded_query: Any, vector_index: VectorIndex, top_k: int) -> List[SearchHit]:
    """Flexible: handles both dict and Pydantic inputs."""
    # Extract embedding flexibly
    if isinstance(encoded_query, dict):
        query_emb = encoded_query["embedding"]
    elif isinstance(encoded_query, EncodedQuery):
        query_emb = encoded_query.embedding
    else:
        query_emb = getattr(encoded_query, "embedding")
    
    return vector_index.search(query_emb, k=top_k)
```

## Benefits

| Aspect | Manual Dict | Elegant Pydantic |
|--------|-------------|------------------|
| **Type Safety** | ❌ Lost | ✅ Preserved |
| **Code Readability** | 😐 OK | ✅ Excellent |
| **Manual Conversion** | ❌ Required | ✅ Automatic |
| **Follows Daft Patterns** | ❌ No | ✅ Yes |
| **Works in Jupyter** | ✅ Yes | ✅ Yes |
| **Works with map_over** | ✅ Yes | ✅ Yes |

## Testing

```bash
# Test the elegant solution
uv run python scripts/test_elegant_pydantic.py
```

Result:
```
✅ SUCCESS!
  - Returned Pydantic models from encode_passage
  - DaftBackend auto-converted with .model_dump()
  - No manual dict conversion needed!
  - Full type safety preserved!
```

## Installation Requirement

For full PyArrow struct support, install:

```bash
pip install pydantic-to-pyarrow
```

If not installed, DaftBackend falls back to Python object storage (still works, just less optimized).

## Recommendation

✨ **Use the Elegant Pydantic Solution!**

It's cleaner, safer, and follows Daft's official patterns from their tutorials. Your code stays natural and type-safe while DaftBackend handles all the serialization complexity behind the scenes.

## Files

- **Implementation:** `src/hypernodes/daft_backend.py` (enhanced)
- **Test:** `scripts/test_elegant_pydantic.py`
- **Example:** `scripts/retrieval_elegant_pydantic.py`
- **Working Reference:** `scripts/retrieval_daft_working_example.py` (with dict conversion)
- **Guide:** `guides/DAFT_PYDANTIC_ELEGANT_SOLUTION.md`
