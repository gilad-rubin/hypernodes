# Progress Bar Node Name Display - Implementation Summary

## Problem
The progress bar was showing `pipeline_XXXXX` instead of showing:
1. The pipeline name
2. The currently running node

This was especially problematic for pipelines using `.as_node(map_over=...)` pattern.

## Solution

### Changes Made

**1. Progress Bar Updates (`src/hypernodes/telemetry/progress.py`)**
- Pipeline bar now shows pipeline name instead of ID
- During execution: shows `{pipeline_name} → {node_name}` 
- Initial map state: `"Running {pipeline_name} with N examples..."`
- Completion: `"{pipeline_name} ✓ (duration, rate)"`
- **Key fix**: Removed condition that only updated bar during map operations - now updates for ALL node executions

**2. Metadata Enhancement (`src/hypernodes/backend.py` & `src/hypernodes/pipeline.py`)**
- Added `pipeline_name` to pipeline metadata
- Pipeline name (or ID fallback) is now accessible to callbacks

**3. Node ID Resolution (`src/hypernodes/backend.py`)**
- Created `_get_node_id()` helper function
- Handles PipelineNodes with explicit names
- Prioritizes: `node.name` → `node.func.__name__` → `node.id` → `node.__name__` → `str(node)`

**4. PipelineNode Callback Support (`src/hypernodes/backend.py`)**
- **Critical fix**: PipelineNode execution now triggers `on_node_start` and `on_node_end` callbacks
- This ensures progress bar updates even for nested mapped pipelines

## Usage

### Simple Pipeline
```python
from hypernodes import Pipeline, node
from hypernodes.telemetry import ProgressCallback

pipeline = Pipeline(
    nodes=[node1, node2, node3],
    callbacks=[ProgressCallback()],
    name="my_pipeline"  # Give it a meaningful name!
)

results = pipeline.run(inputs={...})
```

**Progress display:**
- `my_pipeline → node1`
- `my_pipeline → node2`
- `my_pipeline → node3`
- `my_pipeline ✓ (1.5s)`

### With .as_node(map_over=...)
```python
# Single-item pipeline
process_single = Pipeline(
    nodes=[encode, transform],
    name="process_single"
)

# Mapped version
process_many = process_single.as_node(
    input_mapping={"items": "item"},
    output_mapping={"result": "results"},
    map_over="items",
    name="process_many"  # Name the mapped node!
)

# Main pipeline
pipeline = Pipeline(
    nodes=[load_data, process_many, aggregate],
    callbacks=[ProgressCallback()],
    name="main_pipeline"
)

results = pipeline.run(inputs={})
```

**Progress display:**
- `main_pipeline → load_data`
- `main_pipeline → process_many`  ← Now shows the mapped node!
- `main_pipeline → aggregate`
- `main_pipeline ✓ (2.5s)`

## Test Results
✅ All 56 tests pass
✅ Regular pipelines show node names
✅ Mapped pipelines show node names
✅ Nested pipelines show names at all levels
✅ Backward compatible - unnamed pipelines still work

## Files Modified
- `src/hypernodes/telemetry/progress.py` - Progress bar display logic
- `src/hypernodes/backend.py` - Node ID resolution and PipelineNode callbacks
- `src/hypernodes/pipeline.py` - Metadata generation

## Key Insight
The issue was that `PipelineNode` execution (which happens when using `.as_node(map_over=...)`) wasn't triggering the standard node callbacks. By adding `on_node_start` and `on_node_end` calls around PipelineNode execution, the progress bar now updates correctly for all node types.
