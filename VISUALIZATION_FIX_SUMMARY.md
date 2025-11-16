# Visualization Depth=2 Fix Summary

## Problem
When using `pipeline.visualize(depth=2)` on a pipeline containing nested pipelines, the visualization was not showing edges from inner nodes to outer consuming nodes. 

### Example
```python
# RAG Pipeline (inner)
retrieve -> generate (produces "answer")

# Evaluation Pipeline (outer)
RAG (nested pipeline) -> evaluate_answer (consumes "answer")
```

**Expected behavior with `depth=2`:** 
- Expand RAG pipeline to show `retrieve` and `generate` nodes
- Show edge from `generate` to `evaluate_answer` (since `generate` produces `answer` which `evaluate_answer` consumes)

**Actual behavior (before fix):**
- RAG pipeline was expanded correctly
- But the edge from `generate` to `evaluate_answer` was missing
- A disconnected node with the PipelineNode's ID was shown instead

## Root Cause
In `src/hypernodes/visualization.py`, the `_collect_visualization_data()` function was correctly:
1. Expanding nested pipelines when `depth > 1`
2. Updating the `output_to_node` mapping to point to inner nodes

But it was **NOT** creating edges from the inner output-producing nodes to the outer consuming nodes.

## Solution
Modified `_collect_visualization_data()` to:

1. **Track expanded PipelineNodes**: Store which PipelineNodes were expanded along with their visualization data
2. **Resolve dependencies**: When processing outer nodes that depend on an expanded PipelineNode:
   - Look up which parameters the outer node needs
   - Find the inner nodes that produce those outputs (via the updated `output_to_node` mapping)
   - Create edges from those inner nodes to the outer consuming node

### Code Changes
```python
# Track expanded PipelineNodes
expanded_pipeline_nodes: Dict[Node, VisualizationGraph] = {}

# When expanding a PipelineNode, store it
if should_expand and depth is not None:
    # ... expand logic ...
    expanded_pipeline_nodes[node] = nested_viz

# When processing regular nodes, check if dependencies are expanded
for dep_node in graph_result.dependencies.get(node, []):
    if dep_node in expanded_pipeline_nodes:
        # Find which outputs this node uses from the expanded pipeline
        # Create edges from the inner producer nodes
        for param in node.root_args:
            if param in output_to_node:
                inner_producer = output_to_node[param]
                if inner_producer in nested_viz.nodes:
                    edges.append((inner_producer, node))
    else:
        # Regular dependency
        edges.append((dep_node, node))
```

## Testing
Added comprehensive tests in `tests/test_visualization_depth.py`:

1. **`test_depth_2_shows_edges_from_inner_to_outer_nodes`**: Verifies that edges from inner nodes (like `generate`) to outer nodes (like `evaluate_answer`) are correctly shown
2. **`test_depth_1_shows_collapsed_pipeline_node`**: Verifies that `depth=1` (default) still shows the nested pipeline as a single collapsed node
3. **`test_depth_2_shows_all_inner_nodes`**: Verifies that `depth=2` correctly expands and shows all inner nodes

All 109 tests pass, including the 3 new tests.

## Impact
- ✅ Fixes visualization with `depth=2` for nested pipelines
- ✅ No breaking changes - all existing tests pass
- ✅ Backward compatible - `depth=1` (default) behavior unchanged
- ✅ Properly handles complex nested pipeline structures like RAG + Evaluation

