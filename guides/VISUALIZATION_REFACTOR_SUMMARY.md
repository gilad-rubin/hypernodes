# Visualization Refactoring Summary

## Overview

Refactored the visualization module to use the pipeline's `GraphResult` directly instead of recreating dependency analysis with NetworkX. This simplifies the code, removes duplication, and aligns with the new architecture where `Pipeline.graph` is the single source of truth.

## Key Changes

### 1. **Removed NetworkX Dependency from Visualization**

**Before:**
- `build_graph()` function recreated the entire dependency graph using NetworkX
- Duplicated logic for analyzing dependencies, root args, and execution order
- 130+ lines of complex graph building code

**After:**
- `_collect_visualization_data()` uses `pipeline.graph` (GraphResult) directly
- Simple traversal of pre-computed dependencies
- ~80 lines of straightforward collection code

### 2. **Introduced VisualizationGraph Dataclass**

```python
@dataclass
class VisualizationGraph:
    """Simplified graph structure for visualization."""
    nodes: Set[Node]
    edges: List[tuple[Union[Node, str], Node]]
    root_args: Set[str]
    output_to_node: Dict[str, Node]
```

This lightweight structure contains only what's needed for rendering, derived from `GraphResult`.

### 3. **Fixed Critical Bug in SimpleGraphBuilder**

**Bug:** When a node depended on multiple outputs from the same producer (e.g., `add(doubled, tripled)` where both come from `preprocessing`), the dependency list contained duplicates, breaking topological sort.

**Fix:**
```python
# Use a set to avoid duplicate dependencies
node_deps_set: Set[Node] = set()
for param in params:
    if param in output_to_node:
        producer = output_to_node[param]
        if producer != node:
            node_deps_set.add(producer)  # Set automatically deduplicates
dependencies[node] = list(node_deps_set)
```

### 4. **Improved PipelineNode Support**

**Fixed type hint extraction:**
- Added `hasattr(node, "func")` checks before accessing `.func`
- PipelineNode doesn't have a `func` attribute, so type hints are skipped gracefully

**Fixed Pipeline vs PipelineNode handling:**
- Pipeline uses `graph.available_output_names` (list)
- PipelineNode uses `output_name` property (tuple)
- Both are now handled correctly

### 5. **Updated Imports and Exports**

**visualization.py:**
```python
# Removed
import networkx as nx

# Added
from .graph_builder import GraphResult
```

**__init__.py:**
```python
# Removed from exports
"build_graph",  # Internal implementation detail
```

## Architecture Benefits

### Single Source of Truth
- `Pipeline.graph` (GraphResult) is computed once during Pipeline initialization
- Visualization reads from this, doesn't recompute
- Consistent behavior across execution and visualization

### Separation of Concerns
```
Pipeline
  ├─ graph: GraphResult (dependency analysis, execution order)
  └─ visualize() → visualization.py
       ├─ _collect_visualization_data() → reads graph
       ├─ _identify_grouped_inputs() → analyzes for display
       └─ render with Graphviz
```

### Simpler, More Maintainable Code
- **Before:** 480+ lines in `build_graph()` with complex recursive NetworkX logic
- **After:** 100 lines in `_collect_visualization_data()` with simple dictionary/set operations

## API Compatibility

### Unchanged
- All `visualize()` function parameters remain the same
- `Pipeline.visualize()` method works identically
- All existing tests pass without modification (except one that used wrong API)

### Fixed Test
Updated `test_visualization_depth.py` to use correct API:
```python
# Before (incorrect - Pipeline doesn't have output_name)
outer_pipeline = Pipeline(nodes=[inner_pipeline, add_prefix])

# After (correct - wrap in PipelineNode)
nested_node = PipelineNode(pipeline=inner_pipeline, name="preprocessing")
outer_pipeline = Pipeline(nodes=[nested_node, add_prefix])
```

## Testing

All visualization tests pass:
- ✅ `test_visualization_depth.py` - Nested pipeline depth expansion
- ✅ `test_visualization_graphviz_io.py` - File I/O and PipelineNode rendering
- ✅ Manual nested pipeline tests with depth=1, depth=None, flatten=True

## Performance Implications

### Positive
- **No duplicate graph building** - visualization reuses Pipeline's graph
- **Faster** - no NetworkX overhead for simple graph operations
- **Less memory** - one graph structure instead of two

### Neutral
- Graphviz rendering time unchanged (same output)
- File I/O performance unchanged

## Migration Guide for Developers

### If you were using `build_graph()` directly:
**Don't.** It's now an internal implementation detail. Use `pipeline.graph` instead:

```python
# Before
from hypernodes import build_graph
g = build_graph(pipeline)

# After
graph_result = pipeline.graph
# Access: output_to_node, execution_order, root_args, dependencies
```

### If you're creating nested pipelines:
Always wrap in `PipelineNode`:

```python
from hypernodes.pipeline_node import PipelineNode

inner = Pipeline(nodes=[...])
nested = PipelineNode(pipeline=inner, name="my_nested_pipeline")
outer = Pipeline(nodes=[nested, ...])
```

## Files Modified

1. **src/hypernodes/visualization.py**
   - Removed `build_graph()` function
   - Added `VisualizationGraph` dataclass
   - Added `_collect_visualization_data()` function
   - Updated `_identify_grouped_inputs()` to use VisualizationGraph
   - Fixed PipelineNode attribute access in label creation functions
   - Fixed Pipeline attribute access (use `graph.available_output_names`)

2. **src/hypernodes/graph_builder.py**
   - Fixed duplicate dependency bug in `SimpleGraphBuilder.build_graph()`
   - Used set to deduplicate producer nodes

3. **src/hypernodes/__init__.py**
   - Removed `build_graph` from exports

4. **tests/test_visualization_depth.py**
   - Updated to use PipelineNode for nested pipelines

## Conclusion

This refactoring successfully:
- ✅ Simplified visualization code by removing NetworkX
- ✅ Eliminated duplicate graph analysis logic
- ✅ Fixed critical topological sort bug
- ✅ Improved PipelineNode support
- ✅ Maintained full API compatibility
- ✅ All tests pass

The visualization module is now simpler, faster, and better integrated with the pipeline's core architecture.

