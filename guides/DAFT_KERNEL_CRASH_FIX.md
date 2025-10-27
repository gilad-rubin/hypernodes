# Fixing Kernel Crashes with DaftBackend and Pydantic Models

## Problem

When running a retrieval pipeline in Jupyter with `DaftBackend` and `.as_node(map_over=...)`, the kernel crashes during execution. This happens specifically when:

1. Using Pydantic models with numpy arrays (like `EncodedPassage` with `embedding: np.ndarray`)
2. Using `.as_node(map_over=...)` to process lists
3. Running in Jupyter notebooks (works fine in scripts)

## Root Cause

The issue occurs during Daft's `list_agg()` operation when aggregating results after exploding lists. When Daft tries to serialize/deserialize complex Python objects (Pydantic models containing numpy arrays) during the groupby aggregation, it can trigger serialization failures that crash the kernel.

### Why It Works in Scripts But Not Jupyter

- **Scripts**: Daft runs in the same process, fewer serialization boundaries
- **Jupyter**: Additional kernel communication layer, stricter serialization requirements

## Solution

**Return dicts instead of Pydantic models from nodes that are mapped over.**

### Before (Causes Crashes)

```python
@node(output_name="encoded_passage")
def encode_passage(passage: Passage, encoder: Encoder) -> EncodedPassage:
    """Returns Pydantic model - problematic with Daft."""
    embedding = encoder.encode(passage.text, is_query=False)
    return EncodedPassage(
        uuid=passage.uuid,
        text=passage.text,
        embedding=embedding  # numpy array inside Pydantic model
    )
```

### After (Works Reliably)

```python
@node(output_name="encoded_passage")
def encode_passage(passage: Any, encoder: Encoder) -> dict:
    """Returns dict - works with Daft's list_agg()."""
    # Normalize input (might be dict, Pydantic, or struct from Daft)
    if isinstance(passage, Passage):
        passage_obj = passage
    elif isinstance(passage, dict):
        passage_obj = Passage(**passage)
    else:
        passage_obj = Passage(uuid=getattr(passage, "uuid"), text=getattr(passage, "text"))
    
    # Encode
    embedding = encoder.encode(passage_obj.text, is_query=False)
    
    # Return as dict (Daft handles this better)
    return {
        "uuid": passage_obj.uuid,
        "text": passage_obj.text,
        "embedding": embedding
    }
```

## Key Changes

### 1. Encoding Nodes Return Dicts

Change return type from `EncodedPassage` to `dict`:

```python
# ❌ Before
@node(output_name="encoded_passage")
def encode_passage(passage: Passage, encoder: Encoder) -> EncodedPassage:
    ...
    return EncodedPassage(uuid=..., text=..., embedding=...)

# ✅ After
@node(output_name="encoded_passage")
def encode_passage(passage: Any, encoder: Encoder) -> dict:
    ...
    return {"uuid": ..., "text": ..., "embedding": ...}
```

### 2. Downstream Nodes Handle Dict Inputs

Nodes that consume encoded data must handle dicts:

```python
@node(output_name="colbert_hits")
def retrieve_colbert(
    encoded_query: Any,  # Changed from EncodedQuery
    vector_index: VectorIndex,
    top_k: int
) -> List[SearchHit]:
    """Retrieve from ColBERT index."""
    # Extract embedding from dict
    if isinstance(encoded_query, dict):
        query_emb = encoded_query["embedding"]
    else:
        query_emb = getattr(encoded_query, "embedding")
    
    return vector_index.search(query_emb, k=top_k)
```

### 3. Index Builders Accept Any Type

Index builders must handle both dicts and Pydantic models:

```python
@node(output_name="vector_index")
def build_vector_index(
    encoded_passages: List[Any],  # Changed from List[EncodedPassage]
    ...
) -> VectorIndex:
    """Build vector index."""
    # Convert to EncodedPassage if needed
    passages = []
    for p in encoded_passages:
        if isinstance(p, dict):
            passages.append(EncodedPassage(**p))
        elif isinstance(p, EncodedPassage):
            passages.append(p)
    
    return PLAIDIndex(passages, ...)
```

## Working Example

See `scripts/retrieval_daft_fixed.py` for the complete fixed implementation.

## Testing

To verify the fix works:

```python
from hypernodes import Pipeline, node
from hypernodes.daft_backend import DaftBackend

# This will crash without the fix
pipeline = build_pipeline()
results = pipeline.run(inputs=inputs)  # ✅ Works now!
```

## Summary

| Component | Issue | Fix |
|-----------|-------|-----|
| `encode_passage` | Returns `EncodedPassage` (Pydantic) | Return `dict` instead |
| `encode_query` | Returns `EncodedQuery` (Pydantic) | Return `dict` instead |
| `retrieve_colbert` | Expects `EncodedQuery` | Accept `Any`, extract from dict |
| `build_vector_index` | Expects `List[EncodedPassage]` | Accept `List[Any]`, convert internally |
| `create_reranker` | Expects `List[EncodedPassage]` | Accept `List[Any]`, convert internally |

## Why This Works

1. **Dicts are simpler**: Daft serializes/deserializes dicts more reliably than Pydantic models
2. **No nested object complexity**: Avoids PyArrow struct conversion issues
3. **Direct numpy array support**: Daft handles numpy arrays in dicts natively
4. **Less serialization overhead**: Fewer transformation layers

## Alternative (Not Recommended)

You could try forcing Daft to use Python object storage for everything:

```python
@daft.func(return_dtype=daft.DataType.python())
def encode_passage(...):
    ...
```

But this is slower and still may have serialization issues in Jupyter.

## Conclusion

**Always return dicts (not Pydantic models) from nodes that are mapped over with DaftBackend**, especially when those objects contain numpy arrays or other complex types. This ensures reliable serialization through Daft's groupby/aggregate operations.
