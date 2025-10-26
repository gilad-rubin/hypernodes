# PipelineNode Caching Fix

## Problem

**Caching was not working for PipelineNodes** (pipelines wrapped with `.as_node()` for mapping operations).

### Root Cause

In `src/hypernodes/backend.py`, the execution path for `PipelineNode` objects (lines 120-179) completely bypassed the caching system:

1. ❌ Never computed a signature for the PipelineNode
2. ❌ Never checked the cache before execution  
3. ❌ Never stored results in the cache after execution

This meant that any pipeline using `.as_node()` with `map_over` would:
- Re-execute on every run, even with identical inputs
- Not benefit from caching at all
- Waste computation time on repeated operations

### Impact

This affected **any pipeline using mapped operations**, including:
- Text encoding pipelines that process batches
- Retrieval pipelines that map queries
- Any hierarchical pipeline composition with `.as_node()`

## Solution

Added complete caching support for `PipelineNode` objects in `src/hypernodes/backend.py`:

### Changes Made

1. **Compute code hash** for the inner pipeline structure:
   ```python
   inner_code_hashes = []
   for inner_node in inner_pipeline.execution_order:
       if hasattr(inner_node, "pipeline"):
           # Nested PipelineNode - use pipeline ID
           inner_code_hashes.append(inner_node.pipeline.id)
       elif hasattr(inner_node, "func"):
           # Regular node - hash its function
           inner_code_hashes.append(hash_code(inner_node.func))
   code_hash = hashlib.sha256("::".join(inner_code_hashes).encode()).hexdigest()
   ```

2. **Compute inputs hash** from the PipelineNode's inputs

3. **Compute dependencies hash** from upstream node signatures

4. **Check cache** before execution - skip if cache hit

5. **Store results** in cache after execution

6. **Track signatures** for downstream dependency hashing

## Test Results

✅ All 56 existing tests pass  
✅ New caching behavior verified with test scripts  
✅ Supports deeply nested pipelines (3+ levels)  
✅ Partial cache hits work correctly (only computes new items)

### Verification

```bash
# Run the test
uv run python scripts/test_pipeline_cache_fix.py
```

Output shows:
- **First run**: Computes all 3 items (encode + process)
- **Second run**: Uses cache, 0 computations  
- **Third run**: Only computes 1 new item, reuses cache for 2 existing items

## Benefits

1. **Massive speedup** for iterative development with mapped operations
2. **Efficient incremental processing** - only new items are computed
3. **Consistent caching** across all pipeline types (regular nodes, PipelineNodes, nested)
4. **No breaking changes** - all existing code continues to work

## Files Modified

- `src/hypernodes/backend.py` - Added caching logic for PipelineNode execution path

## Example Usage

```python
from hypernodes import Pipeline, node, DiskCache

@node(output_name="encoded")
def encode_text(text: str) -> str:
    # Expensive encoding operation
    return expensive_encode(text)

# Create single-item pipeline
encode_single = Pipeline(nodes=[encode_text])

# Convert to mapped node
encode_mapped = encode_single.as_node(
    input_mapping={"texts": "text"},
    output_mapping={"encoded": "encoded_texts"},
    map_over="texts",
)

# Main pipeline with caching
pipeline = Pipeline(
    nodes=[encode_mapped],
    cache=DiskCache(path=".cache/my_cache"),
)

# First run: encodes all texts
results1 = pipeline.run(inputs={"texts": ["a", "b", "c"]})

# Second run: uses cache, no encoding!
results2 = pipeline.run(inputs={"texts": ["a", "b", "c"]})
```

## Notes

- Cache keys are deterministic based on:
  - Inner pipeline structure (node functions)
  - Input values
  - Upstream dependencies
- Cache persists across kernel restarts (disk-based)
- Works with complex objects (Pydantic models, numpy arrays, etc.)
