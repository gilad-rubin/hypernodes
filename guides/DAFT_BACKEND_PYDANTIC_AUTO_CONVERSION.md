# DaftBackend Pydantic Auto-Conversion

## Summary

The DaftBackend now **automatically handles Pydantic model conversions** for both inputs and outputs, making all nodes fully compatible with Daft without requiring manual dict conversion.

## What Was Implemented

### 1. Automatic Input Conversion (Dict → Pydantic)

When using `map_over` with DaftBackend, Pydantic models in lists are exploded into dicts. The backend now automatically converts these dicts back to Pydantic models before passing them to node functions.

**Location:** `src/hypernodes/daft_backend.py` - `_convert_node_to_daft()` method

**Key Features:**
- Inspects function parameter type hints
- Detects Pydantic model parameters (single and `List[PydanticModel]`)
- Wraps functions to auto-convert dict inputs to Pydantic models
- Handles multiple input formats (dict, tuple, PyArrow struct)
- Converts lists of dicts to lists of Pydantic models
- Gracefully falls back if conversion fails

**Example:**
```python
@node(output_name="encoded_passage")
def encode_passage(passage: Passage, encoder: Encoder) -> EncodedPassage:
    """Accepts Passage (may receive dict, auto-converts to Pydantic)."""
    embedding = encoder.encode(passage.text)  # passage is guaranteed to be Pydantic
    return EncodedPassage(uuid=passage.uuid, text=passage.text, embedding=embedding)
```

### 2. Smart Pydantic Model Storage

The backend now intelligently chooses storage strategy based on Pydantic model configuration.

**Location:** `src/hypernodes/daft_backend.py` - `_infer_return_dtype()` method

**Storage Rules:**
1. **Models with `arbitrary_types_allowed`** → Use `daft.DataType.python()`
   - This includes models with numpy arrays, torch tensors, etc.
   - Avoids PyArrow serialization issues
   - Preserves object integrity

2. **Simple Pydantic models** → Try PyArrow struct conversion
   - More optimized when possible
   - Falls back to Python object storage if conversion fails

**Example:**
```python
class EncodedPassage(BaseModel):
    uuid: str
    text: str
    embedding: np.ndarray  # numpy array!
    model_config = {"frozen": True, "arbitrary_types_allowed": True}
    # → Stored as Python object automatically
```

### 3. Automatic Output Conversion (Pydantic → Dict)

This was already implemented, but now works seamlessly with the new input conversion.

**Feature:**
- When functions return Pydantic models, `.model_dump()` is called automatically
- Works with both Python object storage and PyArrow struct storage
- Maintains compatibility with downstream nodes

## How It Works

### The Complete Flow

```
1. User writes node: (Pydantic) → (Pydantic)
                     ↓
2. DaftBackend wraps function:
   - Detects Pydantic input parameters
   - Detects Pydantic return type
                     ↓
3. When map_over explodes list:
   [Pydantic, Pydantic] → explode → Row(dict), Row(dict)
                     ↓
4. Wrapper converts input:
   dict → Pydantic (auto)
                     ↓
5. User's function executes:
   Pydantic → process → Pydantic
                     ↓
6. Wrapper converts output:
   Pydantic → dict (via .model_dump())
                     ↓
7. Daft aggregates:
   Row(dict), Row(dict) → list_agg → [dict, dict]
```

### Storage Strategy Decision Tree

```
Is return type Pydantic?
  ├─ No → Use standard type inference
  └─ Yes
      ├─ Has arbitrary_types_allowed?
      │   ├─ Yes → Use daft.DataType.python()
      │   └─ No
      │       ├─ pydantic_to_pyarrow available?
      │       │   ├─ Yes → Try PyArrow struct
      │       │   │   ├─ Success → Use PyArrow struct
      │       │   │   └─ Fail → Use Python object
      │       │   └─ No → Use Python object
```

## Benefits

### ✅ For Users

1. **Write natural code** - accept and return Pydantic models directly
2. **Full type safety** - type hints work correctly
3. **No manual conversion** - DaftBackend handles everything
4. **Works with map_over** - nested pipelines just work
5. **Handles complex types** - numpy arrays, torch tensors, etc.

### ✅ For Code Quality

1. **Single source of truth** - Pydantic models define data structure
2. **Type checking** - IDEs and type checkers work correctly
3. **Validation** - Pydantic validation runs automatically
4. **Documentation** - Type hints serve as documentation
5. **Testability** - Easy to test with real Pydantic models

### ✅ For Compatibility

1. **Backend-agnostic nodes** - same nodes work with LocalBackend, ModalBackend, DaftBackend
2. **Backward compatible** - existing code still works
3. **Graceful degradation** - falls back if conversion fails
4. **Flexible consumption** - downstream nodes can handle both dicts and Pydantic

## Usage Examples

### Example 1: Simple Encoding Pipeline

```python
from pydantic import BaseModel
from hypernodes import node, Pipeline
from hypernodes.daft_backend import DaftBackend

class Passage(BaseModel):
    uuid: str
    text: str
    model_config = {"frozen": True}

class EncodedPassage(BaseModel):
    uuid: str
    text: str
    embedding: np.ndarray
    model_config = {"frozen": True, "arbitrary_types_allowed": True}

@node(output_name="encoded_passage")
def encode_passage(passage: Passage, encoder: Encoder) -> EncodedPassage:
    """Clean, natural code - no manual conversion!"""
    embedding = encoder.encode(passage.text)
    return EncodedPassage(uuid=passage.uuid, text=passage.text, embedding=embedding)

# Use with map_over - just works!
encode_single = Pipeline(nodes=[encode_passage], name="encode_single")
encode_many = encode_single.as_node(
    input_mapping={"passages": "passage"},
    output_mapping={"encoded_passage": "encoded_passages"},
    map_over="passages"
)

pipeline = Pipeline(
    nodes=[load_passages, create_encoder, encode_many],
    backend=DaftBackend()
)
```

### Example 2: Flexible Consumption

For nodes that consume aggregated results (after `list_agg()`), you can write flexible code:

```python
@node(output_name="vector_index")
def build_vector_index(encoded_passages: List[Any]) -> VectorIndex:
    """Flexible: handles both dict and Pydantic inputs."""
    passages = []
    for p in encoded_passages:
        # Handle both representations
        if isinstance(p, dict):
            passages.append(EncodedPassage(**p))
        else:
            passages.append(p)
    
    return VectorIndex(passages)
```

Or even simpler - rely on DaftBackend's conversion:

```python
@node(output_name="vector_index")
def build_vector_index(encoded_passages: List[EncodedPassage]) -> VectorIndex:
    """DaftBackend auto-converts dicts to EncodedPassage!"""
    return VectorIndex(encoded_passages)
```

## Testing

All tests pass:

1. ✅ **Elegant Pydantic Support Test** - `scripts/test_elegant_pydantic.py`
   - Basic Pydantic model handling
   - map_over with Pydantic models

2. ✅ **Working Example Test** - `scripts/retrieval_daft_working_example.py`
   - Backward compatibility verified
   - Manual dict conversion still works

3. ✅ **Original Pattern Test** - `scripts/test_original_script_pattern.py`
   - Original script pattern now works
   - No manual conversion needed

4. ✅ **Hebrew Retrieval Pattern Test** - `scripts/test_hebrew_retrieval_pattern.py`
   - Complete retrieval pipeline
   - Multiple nested map operations
   - Complex dependencies

## Performance Considerations

### Python Object Storage vs PyArrow Structs

**Python Object Storage:**
- ✅ Works with any Python object (numpy, torch, etc.)
- ✅ No serialization/deserialization overhead for complex types
- ✅ Preserves object identity and methods
- ⚠️ Less optimized for large-scale data processing
- ⚠️ No columnar access optimization

**PyArrow Structs:**
- ✅ Highly optimized for columnar operations
- ✅ Efficient memory layout
- ✅ Fast serialization/deserialization
- ⚠️ Limited to PyArrow-compatible types
- ⚠️ Can't handle numpy arrays, torch tensors, etc.

**Our Strategy:**
- Use PyArrow structs for simple Pydantic models (when possible)
- Use Python object storage for models with `arbitrary_types_allowed`
- This balances performance and compatibility

For typical retrieval pipelines (10K-1M passages), the performance difference is negligible compared to the actual ML operations (encoding, retrieval, reranking).

## Upgrade Guide

### If You Have Existing Code with Manual Dict Conversion

**Before:**
```python
@node(output_name="encoded_passage")
def encode_passage(passage: Any, encoder: Encoder) -> dict:
    # Manual conversion
    if isinstance(passage, dict):
        passage = Passage(**passage)
    elif isinstance(passage, (list, tuple)):
        passage = Passage(uuid=passage[0], text=passage[1])
    
    embedding = encoder.encode(passage.text)
    
    # Manual dict return
    return {
        "uuid": passage.uuid,
        "text": passage.text,
        "embedding": embedding
    }
```

**After (cleaner!):**
```python
@node(output_name="encoded_passage")
def encode_passage(passage: Passage, encoder: Encoder) -> EncodedPassage:
    """Clean, simple, type-safe!"""
    embedding = encoder.encode(passage.text)
    return EncodedPassage(uuid=passage.uuid, text=passage.text, embedding=embedding)
```

### Both Versions Work!

The old version with manual conversion still works - you can upgrade incrementally.

## Requirements

No additional dependencies required! Works with:
- ✅ `daft` - Required for DaftBackend
- ✅ `pydantic` - Required for Pydantic models
- ⚠️ `pydantic-to-pyarrow` - Optional, for PyArrow struct optimization
  - If not installed, falls back to Python object storage (still works!)

## Conclusion

Your Hebrew Retrieval Pipeline nodes can now be written naturally:
- Accept Pydantic models as inputs
- Return Pydantic models as outputs
- Use `map_over` without manual conversion
- Keep full type safety

**The DaftBackend handles all conversions automatically!** 🎉

