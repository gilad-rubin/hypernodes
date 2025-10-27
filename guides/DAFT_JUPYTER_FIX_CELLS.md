# Daft Kernel Crash Fix - Quick Test

## Test cells to verify the fix works in Jupyter

### Cell 1: Setup and imports

```python
from typing import Any, List
import numpy as np
from hypernodes import Pipeline, node
from hypernodes.daft_backend import DaftBackend
from pydantic import BaseModel


class Passage(BaseModel):
    uuid: str
    text: str
    model_config = {"frozen": True}


@node(output_name="passages")
def load_passages(num_passages: int) -> List[Passage]:
    return [Passage(uuid=f"p{i}", text=f"Passage {i}") for i in range(num_passages)]


@node(output_name="encoder")
def create_encoder(model_name: str) -> object:
    return {"model": model_name}  # Mock encoder
```

### Cell 2: BROKEN version (will crash)

```python
# ❌ This will likely crash the kernel

@node(output_name="encoded_passage")
def encode_passage_broken(passage: Passage, encoder: object) -> dict:
    """This mimics returning a Pydantic model with numpy."""
    # Create a structure similar to what causes the crash
    from pydantic import BaseModel
    
    class EncodedPassage(BaseModel):
        uuid: str
        text: str
        embedding: Any
        model_config = {"arbitrary_types_allowed": True}
    
    embedding = np.array([len(passage.text), 1.0, 2.0], dtype=np.float32)
    result = EncodedPassage(uuid=passage.uuid, text=passage.text, embedding=embedding)
    return result  # Returning Pydantic model

encode_single_broken = Pipeline(nodes=[encode_passage_broken], name="encode_single_broken")

encode_many_broken = encode_single_broken.as_node(
    input_mapping={"passages": "passage"},
    output_mapping={"encoded_passage": "encoded_passages"},
    map_over="passages",
    name="encode_many_broken"
)

pipeline_broken = Pipeline(
    nodes=[load_passages, create_encoder, encode_many_broken],
    backend=DaftBackend(show_plan=False),
    name="broken_pipeline"
)

# This will crash:
try:
    result = pipeline_broken.run(inputs={"num_passages": 3, "model_name": "test"})
    print("✅ Surprisingly, it worked!")
except Exception as e:
    print(f"❌ Crashed with: {type(e).__name__}: {e}")
```

### Cell 3: FIXED version (works reliably)

```python
# ✅ This works reliably

@node(output_name="encoded_passage")
def encode_passage_fixed(passage: Any, encoder: object) -> dict:
    """KEY FIX: Return dict instead of Pydantic model."""
    # Normalize input
    if isinstance(passage, Passage):
        passage_obj = passage
    elif isinstance(passage, dict):
        passage_obj = Passage(**passage)
    else:
        passage_obj = Passage(uuid=getattr(passage, "uuid"), text=getattr(passage, "text"))
    
    # Create embedding
    embedding = np.array([len(passage_obj.text), 1.0, 2.0], dtype=np.float32)
    
    # Return as dict (not Pydantic model!)
    return {
        "uuid": passage_obj.uuid,
        "text": passage_obj.text,
        "embedding": embedding
    }

encode_single_fixed = Pipeline(nodes=[encode_passage_fixed], name="encode_single_fixed")

encode_many_fixed = encode_single_fixed.as_node(
    input_mapping={"passages": "passage"},
    output_mapping={"encoded_passage": "encoded_passages"},
    map_over="passages",
    name="encode_many_fixed"
)

@node(output_name="result")
def check_results(encoded_passages: List[dict]) -> dict:
    return {
        "count": len(encoded_passages),
        "first_uuid": encoded_passages[0]["uuid"],
        "has_embedding": "embedding" in encoded_passages[0],
        "embedding_shape": encoded_passages[0]["embedding"].shape
    }

pipeline_fixed = Pipeline(
    nodes=[load_passages, create_encoder, encode_many_fixed, check_results],
    backend=DaftBackend(show_plan=False),
    name="fixed_pipeline"
)

# This should work:
result = pipeline_fixed.run(inputs={"num_passages": 3, "model_name": "test"})
print("✅ Success!")
print(f"Result: {result['result']}")
```

### Cell 4: Apply fix to your retrieval code

```python
# In your retrieval code, change these nodes:

# ❌ BEFORE:
# @node(output_name="encoded_passage")
# def encode_passage(passage: Passage, encoder: Encoder) -> EncodedPassage:
#     embedding = encoder.encode(passage.text, is_query=False)
#     return EncodedPassage(uuid=passage.uuid, text=passage.text, embedding=embedding)

# ✅ AFTER:
# @node(output_name="encoded_passage")
# def encode_passage(passage: Any, encoder: Encoder) -> dict:
#     # Normalize input
#     if isinstance(passage, Passage):
#         passage_obj = passage
#     elif isinstance(passage, dict):
#         passage_obj = Passage(**passage)
#     else:
#         passage_obj = Passage(uuid=getattr(passage, "uuid"), text=getattr(passage, "text"))
#     
#     # Encode
#     embedding = encoder.encode(passage_obj.text, is_query=False)
#     
#     # Return as dict
#     return {
#         "uuid": passage_obj.uuid,
#         "text": passage_obj.text,
#         "embedding": embedding
#     }

# Do the same for encode_query!

print("Apply this pattern to all nodes that are mapped over")
```

### Cell 5: Update downstream nodes to handle dicts

```python
# Also update nodes that consume encoded data:

# ❌ BEFORE:
# @node(output_name="colbert_hits")
# def retrieve_colbert(
#     encoded_query: EncodedQuery,  # Expects Pydantic model
#     vector_index: VectorIndex,
#     top_k: int
# ) -> List[SearchHit]:
#     return vector_index.search(encoded_query.embedding, k=top_k)

# ✅ AFTER:
# @node(output_name="colbert_hits")
# def retrieve_colbert(
#     encoded_query: Any,  # Accepts dict or Pydantic
#     vector_index: VectorIndex,
#     top_k: int
# ) -> List[SearchHit]:
#     # Extract embedding from dict
#     if isinstance(encoded_query, dict):
#         query_emb = encoded_query["embedding"]
#     else:
#         query_emb = getattr(encoded_query, "embedding")
#     
#     return vector_index.search(query_emb, k=top_k)

print("Make sure all consuming nodes can handle dict inputs")
```

## Summary

The fix is simple:

1. ✅ **Mapped nodes return `dict`** (not Pydantic models)
2. ✅ **Consuming nodes accept `Any`** and handle both dict and Pydantic
3. ✅ **Keep Pydantic models for type safety** in your function implementations

This avoids Daft's serialization issues during `list_agg()` operations.
