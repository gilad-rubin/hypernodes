# Visualization Fixes - Summary

## Issues Fixed

### 1. ✅ Removed Outer Outline for Top-Level Pipeline
**Problem:** The root pipeline had an unnecessary outer cluster/box that cluttered the visualization.

**Solution:** Modified the visualization code to only add cluster boxes for nested pipelines, not the top-level pipeline. The top-level pipeline is now rendered directly without an outer container.

**Code Change:** 
- Removed the conditional cluster creation for the root pipeline
- Changed from wrapping in `cluster_{id}` to directly adding nodes

### 2. ✅ Removed Edge Labels (Text on Arrows)
**Problem:** Edge labels showing parameter names were cluttering the visualization and were redundant since the flow is relatively obvious.

**Solution:** Removed the `xlabel` parameter from edge creation, eliminating text labels on all arrows.

**Code Change:**
```python
# Before:
dot.edge(source_id, target_id, xlabel=edge_label, ...)

# After:
dot.edge(source_id, target_id, label="", ...)
```

### 3. ✅ Increased Arrow Length / Spacing
**Problem:** Arrows between nodes were too short, especially in LR (left-to-right) orientation, making the graph cramped.

**Solution:** Increased `ranksep` from 0.4 to 0.8 and `nodesep` from 0.3 to 0.5 for better spacing between nodes.

**Code Change:**
```python
dot.graph_attr.update({
    "ranksep": "0.8",  # Increased from 0.4
    "nodesep": "0.5",  # Increased from 0.3
    "pad": "0.06",
})
```

### 4. ✅ Fixed AttributeError for PipelineNode
**Problem:** When using `.as_node()` to wrap a pipeline, the visualization would crash with:
```
AttributeError: 'Pipeline' object has no attribute '__name__'
```

This happened because `PipelineNode` has a `func` property that returns a `Pipeline`, which doesn't have `__name__`.

**Solution:** Added proper handling in `_create_node_label()` to check if `func` has `__name__` attribute and handle the Pipeline case specially.

**Code Change:**
```python
# Handle PipelineNode which wraps a Pipeline
if hasattr(node.func, '__name__'):
    func_name = node.func.__name__
elif isinstance(node.func, Pipeline):
    # PipelineNode case
    func_name = "nested_pipeline"
else:
    func_name = str(node.func)
```

### 5. ✅ Cleaned Module Prefixes from Type Names
**Problem:** Type names were showing with module prefixes like `__main__.Passage` instead of just `Passage`, making labels unnecessarily long and cluttered.

**Solution:** Added regex-based cleaning to remove module prefixes from type strings in both `_format_type_hint()` and `_format_return_type()`.

**Code Change:**
```python
# Remove module prefixes like __main__., mymodule., etc.
import re
type_str = re.sub(r'\b[a-zA-Z_][a-zA-Z0-9_]*\.', '', type_str)
```

### 6. ✅ Disabled Grouped Inputs by Default
**Problem:** The grouped inputs feature was creating confusing "group_..." nodes that were not clearly labeled and caused visual clutter. These groups were supposed to combine multiple inputs to the same node, but they were appearing even for inputs that shouldn't be grouped.

**Solution:** Changed the default value of `min_arg_group_size` from `2` to `None`, which disables grouping by default. Users can still enable it by explicitly passing a value.

**Code Change:**
```python
def visualize(
    pipeline: Pipeline,
    ...
    min_arg_group_size: Optional[int] = None,  # Changed from 2
    ...
):
```

### 7. ✅ Fixed Tuple Output Names
**Problem:** When nodes had multiple outputs (tuple of output names), the visualization would crash trying to escape HTML on a tuple object.

**Solution:** Added handling to convert tuple output names to comma-separated strings before escaping.

**Code Change:**
```python
# Handle tuple output names
if isinstance(output_name, tuple):
    output_name = ", ".join(output_name)
```

## Test Results

All fixes have been tested with the provided code examples:
- ✅ Simple pipeline visualization (`single_encode`)
- ✅ Pipeline with `.as_node()` wrapper (`encode_and_index`)
- ✅ Complex nested pipeline (`full_pipeline`)

Test outputs saved to:
- `outputs/test1_single_encode.svg`
- `outputs/test3_encode_and_index.svg`
- `outputs/test4_full_pipeline.svg`

## Impact Summary

**Before:**
- Cluttered outer boxes
- Text labels on every arrow
- Cramped layout
- Crashes with `.as_node()`
- Long type names with module prefixes
- Confusing "group_..." nodes

**After:**
- Clean, minimal design
- Clear arrow flow without labels
- Well-spaced layout
- Full support for `.as_node()`
- Concise type names
- Simple, direct input connections

## Usage

All fixes are applied automatically when calling `.visualize()`:

```python
pipeline.visualize()  # Uses all new defaults
pipeline.visualize(orient="LR")  # Better spacing for LR orientation
pipeline.visualize(min_arg_group_size=3)  # Can still enable grouping if desired
```
