# Nested Pipeline Progress Bar Fix - Complete Summary

## Problem
When using `.as_node(map_over=...)` pattern, progress bars showed:
1. ❌ `encode_passages_mapped 0% 0/1` - incorrect node bar with wrong count
2. ❌ No nested map progress bars showing the actual item count
3. ❌ No per-node progress bars for nodes inside the mapped pipeline

## Root Causes

### 1. Duplicate Progress Bars
PipelineNodes were creating regular node bars (total=1) even though they contained nested pipelines that would create their own progress bars.

### 2. Missing Callback Inheritance
Nested pipelines didn't inherit callbacks from parent pipelines because `_parent` wasn't set during execution.

### 3. Missing Context Propagation
When PipelineNode called `pipeline.map()` internally, it created a new CallbackContext instead of using the parent's context, breaking progress tracking hierarchy.

## Solution

### Changes Made

**1. Skip Node Bars for PipelineNodes (`src/hypernodes/telemetry/progress.py`)**
- Added check: `if ctx.get(f"_is_pipeline_node:{node_id}", False): return`
- PipelineNodes no longer create their own progress bars
- Let nested pipelines handle their own progress display

**2. Mark PipelineNodes in Context (`src/hypernodes/backend.py`)**
- Before calling PipelineNode: `ctx.set(f"_is_pipeline_node:{node_id}", True)`
- Progress callback can now identify PipelineNodes and skip creating bars

**3. Callback Inheritance (`src/hypernodes/backend.py`)**
```python
# Temporarily set parent so nested pipeline inherits callbacks/cache/backend
old_parent = inner_pipeline._parent
inner_pipeline._parent = pipeline

# Execute...

# Restore
inner_pipeline._parent = old_parent
```

**4. Context Propagation (`src/hypernodes/pipeline.py` & `src/hypernodes/backend.py`)**
- Added `_ctx` parameter to `Pipeline.map()` method
- Backend stores context in PipelineNode: `node._exec_ctx = ctx`
- PipelineNode passes context to nested calls: `self.pipeline.map(..., _ctx=exec_ctx)`

## Results

### Before Fix
```
hebrew_retrieval → encode_passages_mapped:  47% 8/17
encode_passages_mapped   0% 0/1  ← Wrong! Should show N passages
```

### After Fix
```
hebrew_retrieval → encode_passages_mapped:  47% 8/17
  encode_passage ✓ (1.2s, 8.3 items/s, 0.0% cached): 100% 10/10
  process_single ✓ (1.2s, 8.3 items/s): 100% 10/10
```

## Progress Bar Hierarchy (DRY Approach)

Now all execution modes follow the same pattern:

### 1. Regular Pipeline Execution
```
pipeline_name → node1
pipeline_name → node2
pipeline_name ✓
```

### 2. Pipeline.map() Execution
```
Running pipeline_name with N examples...
  node1 [X items/s, Y% cached]: N/N
  node2 [X items/s, Y% cached]: N/N
pipeline_name ✓ (duration, rate)
```

### 3. Nested Pipeline (as_node with map_over)
```
main_pipeline → mapped_node_name
  nested_pipeline (map): N/N
    inner_node1 [rate, cache%]: N/N
    inner_node2 [rate, cache%]: N/N
  nested_pipeline ✓
main_pipeline → next_node
```

## Key Design Principles

1. **No Duplicate Bars**: Each execution level creates appropriate bars, no overlap
2. **Callback Inheritance**: Nested pipelines inherit parent callbacks via temporary `_parent`
3. **Context Sharing**: Single CallbackContext flows through entire execution hierarchy
4. **Consistent Node IDs**: `_get_node_id()` helper prioritizes explicit names across all node types

## Test Results
✅ All 56 tests pass
✅ Regular pipelines show correct progress
✅ Mapped pipelines show item counts correctly
✅ Nested mapped pipelines show hierarchical progress
✅ No duplicate progress bars

## Files Modified
- `src/hypernodes/telemetry/progress.py` - Skip PipelineNode bars
- `src/hypernodes/backend.py` - Callback inheritance, context passing, node ID helper
- `src/hypernodes/pipeline.py` - Context propagation through map()

## Example Usage

```python
# Single-item pipeline
encode_single = Pipeline(nodes=[encode_text], name="encode_single")

# Mapped version (processed N items)
encode_many = encode_single.as_node(
    input_mapping={"passages": "passage"},
    output_mapping={"encoded": "encoded_passages"},
    map_over="passages",
    name="encode_passages_mapped"  # Shows in progress!
)

# Main pipeline
pipeline = Pipeline(
    nodes=[load_data, encode_many, build_index],
    callbacks=[ProgressCallback()],
    name="retrieval_pipeline"
)

# Progress shows:
# retrieval_pipeline → load_data
# retrieval_pipeline → encode_passages_mapped
#   encode_text (8.3 items/s, 0% cached): 100% 500/500
#   encode_single ✓ (60s, 8.3 items/s)
# retrieval_pipeline → build_index
# retrieval_pipeline ✓ (65s)
```

This creates a clean, hierarchical progress display that accurately reflects the execution structure!
