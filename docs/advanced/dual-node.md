# DualNode: Batch-Optimized Execution

## Overview

`DualNode` enables you to define nodes with **two execution modes**:
1. **Singular**: Processes one item at a time (scalar types)
2. **Batch**: Processes multiple items at once (Daft Series)

This design philosophy allows you to:
- ✅ Keep your pipeline definitions **simple and type-safe** (using scalar types)
- ✅ Maintain **easy debugging and testing** (with singular functions)
- ✅ Get **automatic batch optimization** during `.map()` operations (with DaftEngine)

## Key Benefits

### 1. Type Safety & Clarity
Pipeline visualization and type hints use the **singular** signature:
```python
def encode_singular(text: str, encoder: Encoder) -> list[float]
```
No confusing `Series[str]` or `list[list[float]]` in your main pipeline logic!

### 2. Testability
Test your nodes with simple, scalar values:
```python
result = pipeline.run(inputs={"text": "Hello", "encoder": encoder})
assert result["encoded"] == [0.1, 0.2, 0.3]
```

### 3. Performance
Automatically use batch operations during `.map()`:
- **SeqEngine**: Uses singular function (3 calls for 3 items)
- **DaftEngine**: Uses batch function (1 call for 3 items) ⚡

## Basic Usage

### Stateless DualNode

```python
from hypernodes import DualNode, Pipeline
from daft import Series

# Define singular function (canonical signature)
def encode_singular(text: str, encoder: Encoder) -> list[float]:
    return encoder.encode(text)

# Define batch function (optimized)
def encode_batch(texts: Series, encoder: Encoder) -> Series:
    # texts is a Daft Series of strings
    # encoder is automatically unwrapped (constant parameter)
    return encoder.encode_batch(texts)

# Create DualNode
encode_node = DualNode(
    output_name="encoded_text",
    singular=encode_singular,
    batch=encode_batch,
)

pipeline = Pipeline(nodes=[encode_node])
```

### Execution Behavior

```python
# .run() - uses singular function
result = pipeline.run(inputs={"text": "Hello", "encoder": encoder})
# Calls: encode_singular("Hello", encoder)

# .map() with SeqEngine - uses singular function
results = pipeline.map(
    inputs={"text": ["A", "B", "C"], "encoder": encoder},
    map_over="text"
)
# Calls: encode_singular("A", encoder), encode_singular("B", encoder), ...

# .map() with DaftEngine - uses batch function
from hypernodes.engines import DaftEngine
pipeline_fast = Pipeline(nodes=[encode_node], engine=DaftEngine())
results = pipeline_fast.map(
    inputs={"text": ["A", "B", "C"], "encoder": encoder},
    map_over="text"
)
# Calls: encode_batch(Series(["A", "B", "C"]), encoder)  ⚡
```

## Stateful DualNode

For nodes with expensive initialization (models, DB connections):

```python
class TextProcessor:
    def __init__(self, model_name: str, config: dict):
        # Expensive initialization - happens once
        self.model = load_model(model_name)
        self.config = config
    
    def process_singular(self, text: str) -> str:
        return self.model.process(text)
    
    def process_batch(self, texts: Series) -> Series:
        # Batch processing with shared model instance
        return self.model.process_batch(texts)

# Create instance (lazy - __init__ called on first use)
processor = TextProcessor(model_name="gpt-4", config={...})

# Create stateful DualNode
process_node = DualNode(
    output_name="processed",
    singular=processor.process_singular,
    batch=processor.process_batch,
)
```

**Benefits:**
- Model loaded once per worker (not per item)
- Same instance reused across all batch calls
- Follows Daft's `@daft.cls` pattern

## Parameter Handling

### Constant vs. Varying Parameters

DaftEngine automatically handles parameter types:

```python
def encode_batch(texts: Series, encoder: Encoder, prefix: str) -> Series:
    # texts: Series (varying - mapped over)
    # encoder: Encoder (constant - automatically unwrapped)
    # prefix: str (constant - automatically unwrapped)
    ...
```

**Automatic unwrapping:**
- Parameters being mapped over (`map_over="text"`) → stay as `Series`
- Constant parameters (same for all items) → unwrapped to scalars

You don't need to manually check or unwrap!

## Design Patterns

### Pattern 1: Encoder with Batch Methods

```python
class Encoder:
    def encode(self, text: str) -> list[float]:
        return self.encode_batch(Series([text])).to_pylist()[0]
    
    def encode_batch(self, texts: Series) -> Series:
        # Real implementation
        return Series.from_pylist([self._vectorize(t) for t in texts.to_pylist()])
```

**Best practice:** Make `encode()` delegate to `encode_batch()` for single source of truth.

### Pattern 2: Multiple Operations in One Class

```python
class TextOps:
    def __init__(self, model_name: str):
        self.model = load_model(model_name)
    
    # Operation 1: Encoding
    def encode_singular(self, text: str) -> list[float]: ...
    def encode_batch(self, texts: Series) -> Series: ...
    
    # Operation 2: Classification
    def classify_singular(self, text: str) -> str: ...
    def classify_batch(self, texts: Series) -> Series: ...

ops = TextOps(model_name="bert")

encode_node = DualNode(
    output_name="encoded",
    singular=ops.encode_singular,
    batch=ops.encode_batch,
)

classify_node = DualNode(
    output_name="label",
    singular=ops.classify_singular,
    batch=ops.classify_batch,
)
```

**Both nodes share the same lazily-initialized `ops` instance!**

## Technical Details

### Code Hash
DualNode computes a combined code hash from **both** implementations:
```python
hash(singular_code + batch_code)
```
This ensures cache invalidation when either implementation changes.

### Type Inference
The **singular** function's type hints are used for:
- Pipeline visualization
- Daft type inference
- Documentation

The batch function's types are only used internally by DaftEngine.

### Engine Support
- ✅ **SeqEngine**: Always uses singular function
- ✅ **DaftEngine**: Uses batch function during `.map()`, singular during `.run()`
- ⚠️ **DaskEngine**: Not yet supported (uses singular function)

## Migration from Regular Nodes

### Before (Regular Node)
```python
@node(output_name="encoded")
def encode(text: str, encoder: Encoder) -> list[float]:
    return encoder.encode(text)
```

**Problem:** No batch optimization during `.map()`

### After (DualNode)
```python
def encode_singular(text: str, encoder: Encoder) -> list[float]:
    return encoder.encode(text)

def encode_batch(texts: Series, encoder: Encoder) -> Series:
    return encoder.encode_batch(texts)

encode_node = DualNode(
    output_name="encoded",
    singular=encode_singular,
    batch=encode_batch,
)
```

**Benefit:** Automatic 10-100x speedup for batch operations! ⚡

## Common Pitfalls

### ❌ Forgetting Series.from_pylist()
```python
def encode_batch(texts: Series, encoder: Encoder) -> Series:
    # Wrong: Returns list, not Series
    return [encoder.encode(t) for t in texts.to_pylist()]
```

✅ **Fix:**
```python
def encode_batch(texts: Series, encoder: Encoder) -> Series:
    return Series.from_pylist([encoder.encode(t) for t in texts.to_pylist()])
```

### ❌ Manually Unwrapping Constant Parameters
```python
def encode_batch(texts: Series, encoder: Series) -> Series:
    # Wrong: Trying to unwrap encoder manually
    encoder_obj = encoder.to_pylist()[0]
    ...
```

✅ **Fix:** DaftEngine unwraps automatically!
```python
def encode_batch(texts: Series, encoder: Encoder) -> Series:
    # encoder is already unwrapped!
    return encoder.encode_batch(texts)
```

## Examples

See:
- `notebooks/batch_api.ipynb` - Interactive examples
- `scripts/dual_node_example.py` - Stateless and stateful examples
- `scripts/dual_node_daft_test.py` - Integration tests with DaftEngine

## See Also

- [Daft UDF Documentation](https://www.getdaft.io/projects/docs/en/latest/user_guide/udf.html)
- [DaftEngine Guide](./daft-engine.md)
- [Execution Engines](./execution-engines.md)

