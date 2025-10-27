# DaftBackend Debugging Guide

## Quick Debug Checklist

When you encounter a DaftBackend error or kernel crash:

1. **Enable Debug Mode**
   ```python
   pipeline = Pipeline(
       nodes=[...],
       backend=DaftBackend(show_plan=True, debug=True)  # Enable debugging
   )
   ```

2. **Use the Debug Script**
   ```bash
   uv run python scripts/debug_daft_pipeline.py
   ```

3. **Check the error output** - the wrapper now catches and logs all errors with context

## Common Issues and Solutions

### Issue 1: Kernel Crash in Jupyter

**Symptoms:**
- Kernel crashes when executing Daft pipeline
- No clear error message
- Works in regular Python but not Jupyter

**Causes:**
- Multiprocessing issues with Jupyter
- Memory pressure from large data
- Incompatible library versions

**Solutions:**

#### Solution A: Use Python Script Instead of Jupyter
```bash
# Convert notebook to script
uv run python scripts/debug_daft_pipeline.py
```

#### Solution B: Disable Multiprocessing in Jupyter
```python
# At the top of your notebook
import os
os.environ["DAFT_RUNNER"] = "py"

# Then use DaftBackend
from hypernodes.daft_backend import DaftBackend
pipeline = Pipeline(nodes=[...], backend=DaftBackend())
```

#### Solution C: Reduce Data Size for Testing
```python
# Use smaller data for debugging
inputs = {
    "num_passages": 10,  # Instead of 1000
    "num_queries": 2,    # Instead of 100
    ...
}
```

### Issue 2: `AttributeError: 'dict' object has no attribute 'uuid'`

**Symptom:**
```
AttributeError: 'dict' object has no attribute 'uuid'
```

**Cause:**
A node expects a Pydantic model but receives a dict (after `list_agg()`)

**Solution:**
The DaftBackend should auto-convert, but if it doesn't, make your node flexible:

```python
@node(output_name="result")
def my_node(encoded_passages: List[EncodedPassage]) -> Result:
    """The type hint should trigger auto-conversion."""
    # If auto-conversion fails, add manual handling:
    passages = []
    for p in encoded_passages:
        if isinstance(p, dict):
            passages.append(EncodedPassage(**p))
        else:
            passages.append(p)
    
    # Process passages...
```

### Issue 3: Type Conversion Errors

**Symptoms:**
- Errors like "missing required field"
- Pydantic validation errors
- Type mismatch errors

**Cause:**
Daft may serialize Pydantic models in unexpected ways

**Solution:**
Use Python object storage for complex models:

```python
class EncodedPassage(BaseModel):
    uuid: str
    text: str
    embedding: np.ndarray  # Complex type
    
    # This triggers Python object storage automatically
    model_config = {"frozen": True, "arbitrary_types_allowed": True}
```

## Debugging Tools

### 1. Debug Mode

Enable debug mode to see conversion errors:

```python
backend = DaftBackend(show_plan=True, debug=True)
```

This will print:
- Detailed conversion errors
- Type mismatches
- Function call traces

### 2. Show Execution Plan

See the Daft execution plan:

```python
backend = DaftBackend(show_plan=True)
```

This shows:
- Data flow
- UDF calls
- Aggregations
- Column types

### 3. Step-by-Step Execution

Test nodes individually:

```python
# Test single node
@node(output_name="result")
def my_node(input: MyType) -> MyOutput:
    print(f"Input type: {type(input)}")
    print(f"Input value: {input}")
    result = process(input)
    print(f"Output type: {type(result)}")
    return result

# Test with minimal pipeline
test_pipeline = Pipeline(
    nodes=[my_node],
    backend=DaftBackend(debug=True)
)

result = test_pipeline.run(inputs={"input": test_value})
```

### 4. Validate Data at Boundaries

Add validation nodes between stages:

```python
@node(output_name="validated_passages")
def validate_passages(passages: List[Any]) -> List[EncodedPassage]:
    """Validate and convert passages."""
    print(f"Received {len(passages)} passages")
    print(f"First passage type: {type(passages[0])}")
    
    validated = []
    for i, p in enumerate(passages):
        try:
            if isinstance(p, dict):
                validated.append(EncodedPassage(**p))
            elif isinstance(p, EncodedPassage):
                validated.append(p)
            else:
                print(f"Warning: Unexpected type at index {i}: {type(p)}")
                # Try to convert
                validated.append(EncodedPassage(
                    uuid=getattr(p, 'uuid', f'unknown_{i}'),
                    text=getattr(p, 'text', ''),
                    embedding=getattr(p, 'embedding', np.array([]))
                ))
        except Exception as e:
            print(f"Error validating passage {i}: {e}")
            raise
    
    return validated
```

## Best Practices

### 1. Use Type Hints Consistently

```python
@node(output_name="encoded_passage")
def encode_passage(
    passage: Passage,  # Clear type hint
    encoder: Encoder
) -> EncodedPassage:  # Clear return type
    ...
```

### 2. Handle Both Pydantic and Dict

For nodes that consume aggregated data:

```python
@node(output_name="result")
def process_list(items: List[MyModel]) -> Result:
    """Auto-conversion should handle this, but be defensive."""
    processed = []
    for item in items:
        # Handle both dict and Pydantic
        if isinstance(item, dict):
            item = MyModel(**item)
        processed.append(item)
    return process(processed)
```

### 3. Use `arbitrary_types_allowed` for Complex Types

```python
class EncodedPassage(BaseModel):
    uuid: str
    text: str
    embedding: np.ndarray  # NumPy array
    
    model_config = {
        "frozen": True,
        "arbitrary_types_allowed": True  # Required for numpy
    }
```

### 4. Test with LocalBackend First

```python
# Test with LocalBackend first
pipeline = Pipeline(
    nodes=[...],
    backend=LocalBackend()  # Simpler, easier to debug
)

# Once working, switch to DaftBackend
pipeline = pipeline.with_backend(DaftBackend())
```

## Example: Complete Debug Session

```python
#!/usr/bin/env python3
"""
Complete debugging example.
"""

from hypernodes import Pipeline, node
from hypernodes.daft_backend import DaftBackend
from pydantic import BaseModel
import numpy as np
from typing import List, Any

# 1. Define models with proper config
class MyData(BaseModel):
    id: str
    value: np.ndarray
    model_config = {"frozen": True, "arbitrary_types_allowed": True}

# 2. Define nodes with clear type hints and logging
@node(output_name="processed")
def process_item(item: MyData) -> MyData:
    """Process with logging."""
    print(f"Processing: {item.id} (type: {type(item).__name__})")
    result = MyData(id=item.id, value=item.value * 2)
    print(f"Result: {result.id} (type: {type(result).__name__})")
    return result

# 3. Build pipeline with debug mode
pipeline = Pipeline(
    nodes=[process_item],
    backend=DaftBackend(show_plan=True, debug=True),
    name="debug_example"
)

# 4. Run with minimal data
try:
    result = pipeline.run(
        inputs={"item": MyData(id="test", value=np.array([1, 2, 3]))},
        output_name="processed"
    )
    print(f"✅ Success: {result}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
```

## Getting Help

If you're still stuck after trying these solutions:

1. **Check the error logs** - Look for the detailed error message from the wrapper
2. **Run the debug script** - `uv run python scripts/debug_daft_pipeline.py`
3. **Simplify the pipeline** - Remove nodes one by one to isolate the issue
4. **Check data types** - Print types at each step to see where conversion fails
5. **Try LocalBackend** - If it works there, the issue is Daft-specific

## Known Limitations

1. **Jupyter Multiprocessing**: Daft's multiprocessing may not work well in Jupyter
   - Solution: Run in regular Python script or disable multiprocessing

2. **Large Objects**: Very large numpy arrays may cause memory issues
   - Solution: Use Python object storage (`arbitrary_types_allowed=True`)

3. **Complex Nested Types**: Deeply nested Pydantic models may not serialize well
   - Solution: Flatten the structure or use Python object storage

4. **PyArrow Compatibility**: Some types can't be converted to PyArrow
   - Solution: Automatic fallback to Python object storage is enabled

