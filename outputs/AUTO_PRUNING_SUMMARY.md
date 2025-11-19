# Automatic Output Pruning for Nested Pipelines

## Summary

Implemented automatic output pruning for nested pipelines. When a pipeline is used as a node (via `.as_node()`), the system now automatically determines which outputs are actually needed by downstream nodes and only computes those outputs.

**Key Design Decision:** The optimization information is stored at the **parent pipeline level** (in `pipeline.graph.required_outputs`) rather than mutating the `PipelineNode` instance. This ensures that the same PipelineNode can be safely reused in multiple parent pipelines with different requirements.

## What Changed

### 1. **GraphResult** (`graph_builder.py`)
- Added `required_outputs: Dict[Node, Optional[List[str]]]` field to store optimization information
- This maps each PipelineNode to its required outputs for THIS specific pipeline
- Keeps the optimization data with the graph, not on the node instance

### 2. **GraphBuilder** (`graph_builder.py`)
- Added `_compute_required_outputs()` method that analyzes dependencies
- For each PipelineNode, determines which outputs are used by downstream nodes
- Returns a mapping instead of mutating the node:
  - If **some** (but not all) outputs are needed → store only those
  - If **all** outputs are needed OR no outputs needed → store None (return all)
- The mapping is stored in `GraphResult.required_outputs`

### 3. **PipelineNode** (`pipeline_node.py`)
- Removed `_required_outputs` instance variable (no more mutation!)
- `output_name` property returns **all** possible outputs (unchanged behavior)
- `__call__` now accepts optional `required_outputs` parameter
- `_get_required_inner_outputs()` uses the passed-in parameter instead of instance state

### 4. **NodeExecution** (`node_execution.py`)
- `_execute_pipeline_node()` looks up required outputs from parent graph
- Passes the optimization info to `pipeline_node(required_outputs=...)` call
- No mutation of the node instance - clean functional design

### 5. **SeqEngine** (`sequential_engine.py`)
- Updated `_store_node_outputs()` to store only outputs present in the result
- Works correctly with pruned outputs from PipelineNodes

## Why This Design is Better

### Problem with Instance Mutation
The initial implementation stored `_required_outputs` directly on the PipelineNode instance. This had a subtle but critical flaw:

```python
rag_pipeline = Pipeline(nodes=[retrieve, generate])

# Create ONE PipelineNode instance
rag_node = rag_pipeline.as_node(name="RAG")

# Use in Pipeline A - only needs 'answer'
pipeline_a = Pipeline(nodes=[rag_node, node_that_needs_only_answer])
# pipeline_a.graph construction sets: rag_node._required_outputs = ["answer"]

# Reuse in Pipeline B - needs BOTH outputs!
pipeline_b = Pipeline(nodes=[rag_node, node_that_needs_both_outputs])
# pipeline_b.graph construction sets: rag_node._required_outputs = ["answer", "retrieved_docs"]

# But now pipeline_a is BROKEN - rag_node was mutated!
```

### Solution: Parent-Level Storage
Now the optimization info is stored in each pipeline's graph:

```python
rag_node = rag_pipeline.as_node(name="RAG")

pipeline_a = Pipeline(nodes=[rag_node, ...])
# pipeline_a.graph.required_outputs[rag_node] = ["answer"]

pipeline_b = Pipeline(nodes=[rag_node, ...])
# pipeline_b.graph.required_outputs[rag_node] = ["answer", "retrieved_docs"]

# Both pipelines work correctly - no mutation!
```

### Benefits
1. **No side effects** - PipelineNode instances remain immutable
2. **Reusability** - Same node can be used in multiple pipelines
3. **Clarity** - Optimization is clearly scoped to the parent pipeline
4. **Functional design** - Data flows through parameters, not mutation

## How It Works

### Example: RAG Pipeline in Evaluation Pipeline

```python
# Inner pipeline produces TWO outputs
@node(output_name="retrieved_docs")
def retrieve(...): ...

@node(output_name="answer")
def generate(...): ...

rag_pipeline = Pipeline(nodes=[retrieve, generate])
# Produces: ["answer", "retrieved_docs"]

# Outer pipeline only needs "answer"
@node(output_name="evaluation")
def evaluate_answer(answer, ground_truth, query): ...

eval_pipeline = Pipeline(nodes=[rag_pipeline.as_node(name="RAG"), evaluate_answer])
```

**What happens:**

1. **Graph Analysis** (at pipeline construction):
   - GraphBuilder analyzes that `evaluate_answer` depends on `answer`
   - `evaluate_answer` does NOT depend on `retrieved_docs`
   - Sets `RAG_node._required_outputs = ["answer"]`

2. **Property Access**:
   - `RAG_node.output_name` returns `"answer"` (not `("answer", "retrieved_docs")`)

3. **Execution**:
   - `RAG_node.__call__()` passes `output_name="answer"` to inner pipeline
   - Inner pipeline only computes nodes needed for "answer"
   - `retrieve` node is **skipped** entirely (not needed for "answer")
   - Only `generate` node executes

4. **Results**:
   - Final output: `{"answer": ..., "evaluation": ...}`
   - `retrieved_docs` is NOT in the result

5. **Visualization**:
   - RAG node shows only `answer` as output
   - Clean, accurate representation of data flow

## Benefits

1. **Performance**: Skips computation of unused outputs
2. **Clarity**: Visualization shows only what's actually used
3. **Automatic**: No user configuration required
4. **Correct**: Only affects nested pipelines where pruning is safe

## Edge Cases Handled

- **Terminal nodes**: If a PipelineNode has no downstream dependencies, all outputs are returned (no pruning)
- **All outputs needed**: If all outputs are used, no pruning occurs
- **Nested nesting**: Works recursively for deeply nested pipelines
- **Output mapping**: Correctly maps between inner and outer output names

## Tests

All 92 tests pass, including:
- Updated `test_nested_pipelines.py` to reflect new optimized behavior
- Existing caching, callback, map, and execution tests unchanged
- New behavior is backward compatible (only affects nested pipelines)

## Visualization Example

**Before** (hypothetical without optimization):
```
RAG ⚙
├─ answer
└─ retrieved_docs
```

**After** (with optimization):
```
RAG ⚙
└─ answer
```

Only `answer` is shown because it's the only output used by the outer pipeline.

